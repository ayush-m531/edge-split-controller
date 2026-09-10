"""
Cloud half of the split-inference system.

Loads layers 1..31 plus the final norm and lm_head, and waits. When a packed
activation arrives it unpacks the quantized groups back to bf16, reads the
split index from the header, and resumes from split+1 - so it can serve ANY
split the edge chooses, including one the edge did not plan for.

WHY THE SUPERSET: an earlier design fixed the split at launch and had the cloud
load only layers k+1..31. That is broken. If the edge aborts early - say it
stops at L3 because the device overheated when it had planned L6 - the
activation arrives and layers 4, 5, 6 never run. The output would be silently
wrong, not an error. So the cloud holds every layer it might need and reads the
actual split from the wire.

The asymmetry matches reality: the cloud is a server with memory to spare, the
edge is the constrained device. In a deployment they are separate machines and
the duplicated layers cost nothing.

UNPACKING: the edge packs 4-bit codes two per byte and 8-bit codes one per
byte, and sends the per-group scale in the header. The cloud reverses that -
unpack the integers, subtract the offset, multiply by the scale, and scatter
each group back to its original channel positions using the per-layer ordering.

THE CHANNEL ORDERING is read from a file, standing in for the precomputed table
that ships with the model in the real design. Transmitting it per request would
cost 4096 indices at 12 bits, about 6,144 bytes for a list that never changes.
"""
import argparse
import json
import socket
import struct
import sys
import time
import traceback

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAGIC = b"SPLT"
DEFAULT_PORT = 50007


def recv_exactly(conn, n):
    """Read exactly n bytes or raise. A socket read can return short."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(
                f"connection closed after {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def unpack_codes(blob, n_values, n_bits):
    """Reverse the edge's packing. Returns signed integer codes."""
    offset = 2 ** (n_bits - 1)
    if n_bits == 8:
        u = np.frombuffer(blob, dtype=np.uint8)[:n_values]
    elif n_bits == 4:
        b = np.frombuffer(blob, dtype=np.uint8)
        lo = b & 0x0F
        hi = (b >> 4) & 0x0F
        u = np.empty(b.size * 2, dtype=np.uint8)
        u[0::2] = lo
        u[1::2] = hi
        u = u[:n_values]
    else:
        raise ValueError(f"unsupported bit width {n_bits}")
    if u.size != n_values:
        raise ValueError(f"unpacked {u.size} values, expected {n_values}")
    return u.astype(np.int32) - offset


def reconstruct(header, blocks, order):
    """Rebuild the [seq, hidden] activation from the packed groups."""
    seq = header["seq_len"]
    hidden = header["hidden"]
    tk, g2e, g3e = header["top_keep"], header["g2_end"], header["g3_end"]
    b2, b3, b4 = header["bits"]
    scales = header["scales"]

    out = np.zeros((seq, hidden), dtype=np.float32)

    # top group: float16, never quantized
    top_idx = order[:tk]
    top = np.frombuffer(blocks[0], dtype=np.float16).astype(np.float32)
    if top.size != seq * tk:
        raise ValueError(f"top block has {top.size} values, expected {seq*tk}")
    out[:, top_idx] = top.reshape(seq, tk)

    # the three quantized groups
    for blk, idx, nb, key in [(blocks[1], order[tk:g2e], b2, "g2"),
                              (blocks[2], order[g2e:g3e], b3, "g3"),
                              (blocks[3], order[g3e:], b4, "g4")]:
        n_ch = idx.size
        codes = unpack_codes(blk, seq * n_ch, nb)
        vals = codes.astype(np.float32) * float(scales[key])
        out[:, idx] = vals.reshape(seq, n_ch)

    return out


def send_message(conn, header):
    h = json.dumps(header).encode("utf-8")
    conn.sendall(MAGIC + struct.pack("<I", len(h)) + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    print("=" * 74)
    print("CLOUD PROCESS")
    print("=" * 74)
    print(f"Loading {MODEL_NAME} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # The edge holds the embedding table; the cloud never needs it.
    try:
        model.model.embed_tokens.to("meta")
        freed_embed = True
    except Exception:
        freed_embed = False
    torch.cuda.empty_cache()

    print(f"  loaded in {time.time() - t0:.1f}s | {n_layers} layers | "
          f"hidden {hidden} | "
          f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB on GPU")
    print(f"  embedding table freed: {freed_embed}")
    print(f"  ready to resume from ANY split index 0..{n_layers - 2}")
    print()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((args.host, args.port))
    except OSError as e:
        print(f"FATAL: cannot bind {args.host}:{args.port} - {e}")
        print(f"Another cloud process may be running. Check:  "
              f"ss -ltnp | grep {args.port}")
        sys.exit(1)
    srv.listen(4)
    print(f"Listening on {args.host}:{args.port}. Ctrl-C to stop.\n",
          flush=True)

    served = 0
    try:
        while True:
            conn, _ = srv.accept()
            conn.settimeout(300)
            try:
                t_recv = time.time()
                if recv_exactly(conn, 4) != MAGIC:
                    raise ValueError("bad magic - not a split-inference message")
                (hlen,) = struct.unpack("<I", recv_exactly(conn, 4))
                if hlen == 0 or hlen > 1 << 20:
                    raise ValueError(f"implausible header length {hlen}")
                header = json.loads(recv_exactly(conn, hlen).decode("utf-8"))

                sizes = header["block_sizes"]
                blocks = [recv_exactly(conn, n) for n in sizes]
                total = sum(sizes)
                recv_ms = (time.time() - t_recv) * 1000

                split = header["split"]
                seq_len = header["seq_len"]

                # ---- validate before touching the GPU ----
                if header.get("hidden") != hidden:
                    raise ValueError(
                        f"hidden size mismatch: edge {header.get('hidden')}, "
                        f"cloud {hidden}")
                if not (0 <= split < n_layers - 1):
                    raise ValueError(
                        f"split {split} out of range for {n_layers} layers")
                if header.get("n_layers") not in (None, n_layers):
                    raise ValueError(
                        f"layer count mismatch: edge {header['n_layers']}, "
                        f"cloud {n_layers}")

                served += 1
                raw = seq_len * hidden * 2
                print(f"[{served}] split L{split} | seq {seq_len} | "
                      f"{header.get('scheme')} | {total:,} B on the socket "
                      f"({100 * total / raw:.1f}% of bf16) | "
                      f"recv {recv_ms:.1f} ms", flush=True)
                if header.get("aborted"):
                    print(f"      EDGE ABORTED EARLY: "
                          f"{header.get('abort_reason')}", flush=True)
                    print(f"      resuming from L{split + 1} instead of "
                          f"L{header.get('planned_split', split) + 1}",
                          flush=True)

                # ---- unpack ----
                t_un = time.time()
                order = np.load(header["order_path"]).astype(np.int64)
                if order.size != hidden:
                    raise ValueError(
                        f"ordering has {order.size} entries, expected {hidden}")
                arr = reconstruct(header, blocks, order)
                act = torch.from_numpy(arr).unsqueeze(0).to("cuda").to(
                    torch.bfloat16)
                unpack_ms = (time.time() - t_un) * 1000

                # ---- resume from split+1 ----
                t_gen = time.time()
                with torch.no_grad():
                    h = act
                    pos = torch.arange(seq_len).unsqueeze(0).to("cuda")
                    pe = model.model.rotary_emb(h, pos)
                    for i in range(split + 1, n_layers):
                        h = model.model.layers[i](
                            h, attention_mask=None, position_ids=pos,
                            position_embeddings=pe,
                            past_key_values=None, use_cache=False)
                    h = model.model.norm(h)
                    logits = model.lm_head(h)[0, -1, :].float()
                    nxt = int(logits.argmax())
                gen_ms = (time.time() - t_gen) * 1000

                text = tok.decode([nxt], skip_special_tokens=True)
                send_message(conn, {
                    "ok": True,
                    "token": nxt,
                    "text": text,
                    "cloud_layers_run": n_layers - split - 1,
                    "unpack_ms": unpack_ms,
                    "cloud_compute_ms": gen_ms,
                    "recv_ms": recv_ms,
                    "bytes_received": total,
                    "note": ("one token per request: continuing generation "
                             "needs the next token's activation from the "
                             "edge, which requires a return path not "
                             "implemented here"),
                })
                print(f"      -> unpacked in {unpack_ms:.1f} ms | ran layers "
                      f"{split+1}..{n_layers-1} ({n_layers - split - 1}) in "
                      f"{gen_ms:.1f} ms | token {text!r}\n", flush=True)

            except Exception as e:
                traceback.print_exc()
                try:
                    send_message(conn, {"ok": False, "error": str(e),
                                        "type": type(e).__name__})
                except Exception:
                    pass
            finally:
                conn.close()
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print(f"\nShutting down. Served {served} request(s).")
    finally:
        srv.close()


if __name__ == "__main__":
    main()

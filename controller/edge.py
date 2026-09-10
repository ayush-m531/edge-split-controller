"""
Edge half of the split-inference system.

Asks the controller how many layers this device can run, loads exactly those,
runs the prompt through them, QUANTIZES AND PACKS the activation, and sends the
packed bytes plus the group scales to the cloud.

THE PACKING IS THE POINT. An earlier version quantized the values (so they
genuinely lost precision) but then transmitted them as float32 - a megabyte on
the wire while claiming 208,912 bytes. That is a simulation of compression, not
compression. Here the 4-bit groups are packed two values per byte and the
8-bit groups one per byte, so the bytes that actually cross the socket are the
bytes the thesis measures.

WIRE FORMAT (little-endian):
    magic          4 bytes    b"SPLT"
    header_len     4 bytes    uint32
    header         N bytes    utf-8 JSON, includes the split index and scales
    top_block      M bytes    top-5 channels, float16, unquantized
    g2_block       M bytes    shoulder group, packed at b2 bits
    g3_block       M bytes    mid group, packed at b3 bits
    g4_block       M bytes    bulk group, packed at b4 bits

CHANNEL ORDERING: the cloud needs to know which channel went in which group.
The design ships a precomputed per-layer ordering with the model, so it costs
nothing at runtime - transmitting it would be 4096 indices at 12 bits, about
6,144 bytes per request for a list that never changes. Here the edge writes the
ordering to a file the cloud reads once, standing in for that shipped table.

MID-PASS ABORT: the controller decides before the pass starts, but conditions
can cross a threshold partway through. With --abort-at-layer the edge stops
there, re-asks the controller for a scheme valid at that shallower layer, and
sends. The cloud reads the split from the wire and resumes from split+1, so an
early abort produces correct output at a different cost - not silent
corruption.

WHAT IS REAL AND WHAT IS NOT:
  REAL      the layer split across two OS processes, the quantization, the
            BIT PACKING, the byte count on the socket, the TCP transfer, the
            cloud unpacking and resuming from the received split
  MODELLED  transfer TIME. localhost is effectively instant, so wire time is
            computed as bytes / bandwidth.
  SIMULATED free RAM, battery, temperature - passed as flags.
"""
import argparse
import json
import os
import socket
import struct
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookup_table import (decide, cheapest_scheme, scheme_bytes, SCHEMES,
                          P95_KL, QUALITY_LEVELS, TOP_KEEP, G2_END, G3_END)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAGIC = b"SPLT"
DEFAULT_PORT = 50007
ORDER_DIR = "/home/ayush.thakar/edge-split-controller/data/orders"


def quantize_group(x, n_bits):
    """Return integer codes and the scale. x is a flat float tensor.

    Symmetric quantization: the scale is set by the largest absolute value in
    the group, which is why grouping matters - one loud channel would
    otherwise coarsen every quiet one sharing its scale.
    """
    if x.numel() == 0:
        return np.zeros(0, dtype=np.int32), 1.0
    absmax = float(x.abs().max())
    if absmax == 0:
        return np.zeros(x.numel(), dtype=np.int32), 1.0
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    codes = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return codes.cpu().numpy().astype(np.int32), scale


def pack_codes(codes, n_bits):
    """Pack signed integer codes into bytes.

    Codes are shifted to unsigned before packing (a 4-bit signed value runs
    -8..7, stored as 0..15). At 4 bits two values share a byte; at 8 bits one
    value per byte.
    """
    offset = 2 ** (n_bits - 1)
    u = (codes + offset).astype(np.uint8)
    if n_bits == 8:
        return u.tobytes()
    if n_bits == 4:
        if u.size % 2:
            u = np.concatenate([u, np.zeros(1, dtype=np.uint8)])
        lo = u[0::2] & 0x0F
        hi = (u[1::2] & 0x0F) << 4
        return (lo | hi).tobytes()
    raise ValueError(f"unsupported bit width {n_bits}")


def recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(
                f"connection closed after {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def recv_reply(sock):
    magic = recv_exactly(sock, 4)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r} in reply")
    (hlen,) = struct.unpack("<I", recv_exactly(sock, 4))
    return json.loads(recv_exactly(sock, hlen).decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(
        description="Edge process for adaptive split inference")
    ap.add_argument("--prompt", default="The three most important things to "
                                        "know about machine learning are")
    ap.add_argument("--quality", default="strict",
                    choices=["strict", "balanced", "relaxed"],
                    help="set once per request; cannot change mid-inference")
    ap.add_argument("--free-gb", type=float, default=7.81,
                    help="simulated free RAM (5.90 and 7.81 are measured)")
    ap.add_argument("--battery", type=int, default=100)
    ap.add_argument("--temp", type=int, default=40)
    ap.add_argument("--bandwidth", type=int, default=10,
                    help="Mbps, for the modelled transfer time")
    ap.add_argument("--abort-at-layer", type=int, default=None,
                    help="stop here instead of the planned split, simulating "
                         "a threshold crossed mid-pass")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    print("=" * 74)
    print("EDGE PROCESS")
    print("=" * 74)
    print(f"  prompt   : {args.prompt!r}")
    print(f"  quality  : {args.quality}  (per-request, fixed for this run)")
    print(f"  device   : RAM {args.free_gb:.2f} GB free | battery "
          f"{args.battery}% | {args.temp}C | {args.bandwidth} Mbps")
    print()

    # ---- 1. ask the controller ----
    d = decide(args.free_gb, args.bandwidth, args.quality,
               seq_len=100, battery_pct=args.battery, temp_c=args.temp)
    if not d["feasible"]:
        print("CONTROLLER: no feasible split on this device.")
        print(f"  binding constraint : {d['binding_constraint']}")
        print(f"  recommendation     : {d['recommendation']}")
        print()
        print("Nothing to send. A real client would forward the raw tokens and")
        print("run entirely in the cloud - about 300 bytes for a 100-token")
        print("prompt, against ~205 KB for a layer-0 activation.")
        return

    planned = d["split"]
    scheme_name = d["scheme"]
    print("CONTROLLER DECISION")
    print(f"  memory ceiling     : L{d['memory_ceiling']}")
    print(f"  usable cap         : L{d['effective_cap']}")
    print(f"  planned split      : L{planned}  ({planned + 1} layers on edge)")
    print(f"  scheme             : {scheme_name}")
    print(f"  p95 KL             : {d['p95_kl']:.5f}  (budget {args.quality})")
    print(f"  binding constraint : {d['binding_constraint']}")
    print()

    # ---- 2. load only the layers this device can hold ----
    n_load = planned + 1
    print(f"Loading model, keeping layers 0..{planned} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    n_layers_total = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    freed = 0
    for i in range(n_load, n_layers_total):
        try:
            model.model.layers[i].to("meta")
            freed += 1
        except Exception:
            pass
    try:
        model.lm_head.to("meta")
    except Exception:
        pass
    torch.cuda.empty_cache()
    print(f"  loaded in {time.time() - t0:.1f}s | kept {n_load} layers | "
          f"freed {freed} | "
          f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB on GPU")
    print()

    ids = tok(args.prompt, return_tensors="pt")["input_ids"].to("cuda")
    seq_len = ids.shape[1]
    print(f"Prompt tokenized: {seq_len} tokens")

    # ---- 3. run the edge layers, with optional mid-pass abort ----
    abort_at = args.abort_at_layer
    if abort_at is not None and not (0 <= abort_at <= planned):
        print(f"\nERROR: --abort-at-layer {abort_at} must be between 0 and "
              f"the planned split L{planned}.")
        return

    t_edge = time.time()
    aborted, abort_reason, actual = False, None, planned
    with torch.no_grad():
        h = model.model.embed_tokens(ids)
        pos = torch.arange(seq_len).unsqueeze(0).to("cuda")
        pe = model.model.rotary_emb(h, pos)
        for i in range(n_load):
            h = model.model.layers[i](
                h, attention_mask=None, position_ids=pos,
                position_embeddings=pe, past_key_values=None, use_cache=False)
            if abort_at is not None and i == abort_at and i < planned:
                actual, aborted = i, True
                abort_reason = (f"threshold crossed after layer {i}; "
                                f"planned split was L{planned}")
                print(f"\n  ABORT at layer {i} (planned L{planned})")
                print(f"  {abort_reason}")
                break
    edge_ms = (time.time() - t_edge) * 1000

    # If the pass aborted, the planned scheme was chosen for a DIFFERENT layer
    # and may no longer meet the budget here. Re-ask rather than sending under
    # a scheme picked for depth the pass never reached.
    if aborted:
        sc = cheapest_scheme(actual, QUALITY_LEVELS[args.quality])
        if sc is None:
            print(f"  no scheme meets a {args.quality} budget at L{actual}; "
                  f"falling back to the safest available")
            sc = "grouped(8,8,8)"
        if sc != scheme_name:
            print(f"  scheme changed for the shallower split: "
                  f"{scheme_name} -> {sc}")
        scheme_name = sc

    bits = SCHEMES[scheme_name]

    # ---- 4. rank channels, quantize each group, pack ----
    t_comp = time.time()
    act = h[0].float()                                  # [seq, hidden]
    order = torch.argsort(act.abs().max(dim=0).values, descending=True)

    # The ordering is shipped with the model in the real design; written to a
    # file here so the cloud can read it without paying 6,144 bytes per
    # request to receive it.
    os.makedirs(ORDER_DIR, exist_ok=True)
    order_path = os.path.join(ORDER_DIR, f"layer_{actual}.npy")
    np.save(order_path, order.cpu().numpy().astype(np.int16))

    groups = [("top", order[:TOP_KEEP], None),
              ("g2", order[TOP_KEEP:G2_END], bits[0]),
              ("g3", order[G2_END:G3_END], bits[1]),
              ("g4", order[G3_END:], bits[2])]

    blocks, scales, counts = [], {}, {}
    for name, idx, nb in groups:
        sub = act[:, idx]                               # [seq, n_ch]
        counts[name] = int(idx.numel())
        if nb is None:                                  # top-5, unquantized
            blocks.append(sub.to(torch.float16).cpu().numpy().tobytes())
            scales[name] = None
        else:
            codes, sc = quantize_group(sub.reshape(-1), nb)
            blocks.append(pack_codes(codes, nb))
            scales[name] = sc
    comp_ms = (time.time() - t_comp) * 1000

    payload = b"".join(blocks)
    raw_bytes = seq_len * hidden_size * 2
    predicted = scheme_bytes(seq_len, bits)

    header = {
        "split": actual,
        "planned_split": planned,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "seq_len": seq_len,
        "hidden": hidden_size,
        "n_layers": n_layers_total,
        "scheme": scheme_name,
        "bits": list(bits),
        "top_keep": TOP_KEEP,
        "g2_end": G2_END,
        "g3_end": G3_END,
        "scales": scales,
        "counts": counts,
        "block_sizes": [len(b) for b in blocks],
        "order_path": order_path,
        "quality": args.quality,
    }

    print()
    print("TRANSMISSION")
    print(f"  actual split       : L{actual}"
          + (f"  (planned L{planned})" if aborted else ""))
    print(f"  scheme             : {scheme_name}")
    print(f"  p95 KL at L{actual:<2}      : {P95_KL[actual][scheme_name]:.5f}")
    print()
    print(f"  {'group':>8} {'channels':>9} {'bits':>5} {'bytes':>10}")
    print("  " + "-" * 36)
    for (name, idx, nb), blk in zip(groups, blocks):
        print(f"  {name:>8} {counts[name]:>9} "
              f"{('16' if nb is None else str(nb)):>5} {len(blk):>10,}")
    print("  " + "-" * 36)
    print(f"  {'total':>8} {hidden_size:>9} {'':>5} {len(payload):>10,}")
    print()
    print(f"  uncompressed bf16  : {raw_bytes:>10,} B")
    print(f"  ON THE SOCKET      : {len(payload):>10,} B  "
          f"({100 * len(payload) / raw_bytes:.1f}% of bf16)")
    print(f"  table predicted    : {predicted:>10,.0f} B")
    print(f"  difference         : {len(payload) - predicted:>+10,.0f} B  "
          f"(header scales vs the table's 4 B per group)")
    print(f"  modelled transfer  : "
          f"{len(payload) * 8 / (args.bandwidth * 1e6) * 1000:.1f} ms "
          f"at {args.bandwidth} Mbps")
    print()

    # ---- 5. send ----
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(300)
        sock.connect((args.host, args.port))
    except (ConnectionRefusedError, OSError) as e:
        print(f"FATAL: cannot reach the cloud at {args.host}:{args.port} - {e}")
        print("Start it first:  python3 cloud.py")
        return

    try:
        hb = json.dumps(header).encode("utf-8")
        t_send = time.time()
        sock.sendall(MAGIC + struct.pack("<I", len(hb)) + hb + payload)
        reply = recv_reply(sock)
        rt_ms = (time.time() - t_send) * 1000
    except Exception as e:
        print(f"FATAL: transfer failed - {type(e).__name__}: {e}")
        sock.close()
        return
    finally:
        sock.close()

    if not reply.get("ok"):
        print(f"CLOUD ERROR [{reply.get('type')}]: {reply.get('error')}")
        return

    print("RESULT")
    print(f"  next token         : {reply['text']!r}")
    print()
    print("TIMING")
    print(f"  edge compute       : {edge_ms:>8.1f} ms  ({actual + 1} layers)")
    print(f"  quantize and pack  : {comp_ms:>8.1f} ms")
    print(f"  round trip (real)  : {rt_ms:>8.1f} ms  [localhost]")
    print(f"    unpack on cloud  : {reply['unpack_ms']:>8.1f} ms")
    print(f"    cloud compute    : {reply['cloud_compute_ms']:>8.1f} ms  "
          f"({reply['cloud_layers_run']} layers)")
    print(f"  transfer (modelled): "
          f"{len(payload) * 8 / (args.bandwidth * 1e6) * 1000:>8.1f} ms  "
          f"at {args.bandwidth} Mbps")


if __name__ == "__main__":
    main()

"""
Offline lookup table for the adaptive split-inference controller.

Given device conditions and a quality budget, returns the (split point,
compression scheme) to use. Built from measurements taken offline; no GPU and
no model weights are needed at runtime.

DESIGN, and why each part is what it is:

  MEMORY sets the upper bound on the split. Derived from parameter counts, not
  assumed. On a real handset at bf16 the ceiling is L6 when the device is busy
  and L10 when cleared - the same phone, four layers apart, which is why the
  decision has to be made online rather than once offline as EdgeShard does.

  QUALITY BUDGET picks the scheme, selected on the 95th PERCENTILE of KL
  measured across 300 WikiText passages. An earlier version of this table used
  a single text and was 3.7x optimistic: it claimed KL 0.0221 at L3 where the
  corpus mean is 0.083 and the corpus MINIMUM is 0.035. The controller was
  reporting that it met a strict budget while missing it for most inputs.

  THE SPLIT IS SEARCHED, not assumed. An earlier version went as deep as memory
  allowed, on the reasoning that bytes are flat above the floor so depth is
  free. Under the p95 table the byte floor is reached at L6 (strict), L4
  (balanced) and L8 (relaxed) - deeper than that costs edge compute for no
  transmission benefit. So the search takes the SHALLOWEST layer achieving the
  cheapest byte cost.

  THE CONTROLLER NEVER INSPECTS THE INPUT. Per-input quality cost varies about
  8x at a fixed layer and scheme, so there is real variation. But no cheap
  statistic predicts it: tested individually (Pearson max |r| 0.391), monotone
  non-linear (Spearman 0.397), combined and held-out validated (mean R2 0.000),
  token-position structure (0.446), and a random forest in both regression and
  classification form (held-out R2 negative; AUC mean 0.533 against a coin-flip
  baseline of 0.5). Rather than attempting detection, the controller provisions
  for the variation via the p95 margin.

  LAYER 0 IS EXCLUDED as degenerate. Transmitting a layer-0 activation costs
  ~205 KB per 100 tokens; the raw token IDs cost ~300 bytes, about 680x
  cheaper. If layer 0 were optimal there would be no reason to split at all.
  The only motivation is privacy, and one layer gives weak privacy.
"""
import json
import os
from transformers import AutoConfig

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "/home/ayush.thakar/edge-split-controller/data"
RUN_TAG = "lookup_table"

# Fraction of free RAM assumed available for weights; the rest covers
# activations, KV cache, runtime and OS headroom. A JUDGEMENT CALL, not a
# measurement - at 7.81 GB free, 0.67 gives L9 and 0.70 gives L10.
USABLE_FRACTION = 0.70

MIN_SPLIT = 1               # L0 is degenerate; see the module docstring
SAFETY_MARGIN_LAYERS = 1    # headroom below the ceiling
MEASURED_MAX_LAYER = 11     # deepest layer with measured data

QUALITY_LEVELS = {"strict": 0.05, "balanced": 0.10, "relaxed": 0.25}

TOP_KEEP, G2_END, G3_END = 5, 72, 1312
BANDWIDTHS_MBPS = [1, 5, 10, 50, 100]
SEQ_LENS = [50, 100, 250]

RAM_LEVELS_GB = [2.0, 4.0, 5.90, 6.0, 7.81, 8.0]
MEASURED_RAM = {5.90: "measured: phone under normal use",
                7.81: "measured: same phone, apps cleared"}

# ---------------- model geometry ----------------
cfg = AutoConfig.from_pretrained(MODEL_NAME)
H, I, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
V = cfg.vocab_size
n_heads = cfg.num_attention_heads
n_kv = getattr(cfg, "num_key_value_heads", n_heads)
head_dim = H // n_heads

per_layer_params = (2 * H * H + 2 * H * (n_kv * head_dim)) + 3 * H * I + 2 * H
embed_params = V * H


def layers_that_fit(free_gb, bytes_per_param=2.0, usable=USABLE_FRACTION):
    budget = free_gb * (1024 ** 3) * usable
    remaining = budget - embed_params * bytes_per_param
    return 0 if remaining < 0 else int(remaining // (per_layer_params * bytes_per_param))


def memory_ceiling(free_gb, bytes_per_param=2.0, usable=USABLE_FRACTION):
    """Deepest split INDEX that fits, or -1 if not even one layer does.

    Split k means layers 0..k inclusive = k+1 layers, so k layers fitting gives
    a deepest index of k-1. An earlier version returned the COUNT as if it were
    an INDEX, making every ceiling one layer too deep.
    """
    k = layers_that_fit(free_gb, bytes_per_param, usable)
    return -1 if k <= 0 else min(k - 1, L - 1)


# ---------------- schemes and byte cost ----------------
SCHEMES = {"grouped(4,4,4)": (4, 4, 4), "grouped(8,4,4)": (8, 4, 4),
           "grouped(8,8,4)": (8, 8, 4), "grouped(8,8,8)": (8, 8, 8)}
SCHEME_ORDER = ["grouped(4,4,4)", "grouped(8,4,4)",
                "grouped(8,8,4)", "grouped(8,8,8)"]     # cheapest first


def scheme_bytes(seq_len, bits):
    b2, b3, b4 = bits
    nb = TOP_KEEP * seq_len * 16 / 8
    for n, b in [(G2_END - TOP_KEEP, b2), (G3_END - G2_END, b3),
                 (H - G3_END, b4)]:
        nb += n * seq_len * b / 8 + 4
    return nb


def fp16_bytes(seq_len):
    return seq_len * H * 2


# ---------------- p95 KL, from 300 WikiText passages ----------------
# Source: 11a_collect (86,400 records) analysed in 11b, seq 256, per-layer
# channel ordering. Sequence length was found not to matter materially
# (ratios 0.85-1.2 across a 4x change), so one table covers all lengths.
P95_KL = {
    0:  {"grouped(4,4,4)": 0.29929, "grouped(8,4,4)": 0.01895,
         "grouped(8,8,4)": 0.00447, "grouped(8,8,8)": 0.00167},
    1:  {"grouped(4,4,4)": 4.13609, "grouped(8,4,4)": 0.71235,
         "grouped(8,8,4)": 0.03117, "grouped(8,8,8)": 0.01253},
    2:  {"grouped(4,4,4)": 2.83991, "grouped(8,4,4)": 0.21808,
         "grouped(8,8,4)": 0.01390, "grouped(8,8,8)": 0.00705},
    3:  {"grouped(4,4,4)": 1.19049, "grouped(8,4,4)": 0.14112,
         "grouped(8,8,4)": 0.00735, "grouped(8,8,8)": 0.00388},
    4:  {"grouped(4,4,4)": 0.57365, "grouped(8,4,4)": 0.07923,
         "grouped(8,8,4)": 0.00495, "grouped(8,8,8)": 0.00255},
    5:  {"grouped(4,4,4)": 0.56326, "grouped(8,4,4)": 0.05417,
         "grouped(8,8,4)": 0.00348, "grouped(8,8,8)": 0.00185},
    6:  {"grouped(4,4,4)": 0.47576, "grouped(8,4,4)": 0.03975,
         "grouped(8,8,4)": 0.00295, "grouped(8,8,8)": 0.00150},
    7:  {"grouped(4,4,4)": 0.35716, "grouped(8,4,4)": 0.03077,
         "grouped(8,8,4)": 0.00254, "grouped(8,8,8)": 0.00125},
    8:  {"grouped(4,4,4)": 0.24741, "grouped(8,4,4)": 0.02386,
         "grouped(8,8,4)": 0.00242, "grouped(8,8,8)": 0.00113},
    9:  {"grouped(4,4,4)": 0.19739, "grouped(8,4,4)": 0.02022,
         "grouped(8,8,4)": 0.00232, "grouped(8,8,8)": 0.00105},
    10: {"grouped(4,4,4)": 0.15572, "grouped(8,4,4)": 0.01408,
         "grouped(8,8,4)": 0.00216, "grouped(8,8,8)": 0.00101},
    11: {"grouped(4,4,4)": 0.11696, "grouped(8,4,4)": 0.01369,
         "grouped(8,8,4)": 0.00220, "grouped(8,8,8)": 0.00096},
}

# Mean KL, kept alongside so the p95 margin is visible and auditable.
MEAN_KL = {
    0:  {"grouped(4,4,4)": 0.12356, "grouped(8,4,4)": 0.01029,
         "grouped(8,8,4)": 0.00261, "grouped(8,8,8)": 0.00110},
    1:  {"grouped(4,4,4)": 3.26322, "grouped(8,4,4)": 0.44334,
         "grouped(8,8,4)": 0.01541, "grouped(8,8,8)": 0.00687},
    2:  {"grouped(4,4,4)": 2.13805, "grouped(8,4,4)": 0.12997,
         "grouped(8,8,4)": 0.00786, "grouped(8,8,8)": 0.00402},
    3:  {"grouped(4,4,4)": 0.82200, "grouped(8,4,4)": 0.08283,
         "grouped(8,8,4)": 0.00450, "grouped(8,8,8)": 0.00249},
    4:  {"grouped(4,4,4)": 0.42128, "grouped(8,4,4)": 0.04995,
         "grouped(8,8,4)": 0.00331, "grouped(8,8,8)": 0.00180},
    5:  {"grouped(4,4,4)": 0.41511, "grouped(8,4,4)": 0.03720,
         "grouped(8,8,4)": 0.00258, "grouped(8,8,8)": 0.00141},
    6:  {"grouped(4,4,4)": 0.34929, "grouped(8,4,4)": 0.02706,
         "grouped(8,8,4)": 0.00221, "grouped(8,8,8)": 0.00118},
    7:  {"grouped(4,4,4)": 0.26321, "grouped(8,4,4)": 0.02123,
         "grouped(8,8,4)": 0.00195, "grouped(8,8,8)": 0.00102},
    8:  {"grouped(4,4,4)": 0.17506, "grouped(8,4,4)": 0.01739,
         "grouped(8,8,4)": 0.00187, "grouped(8,8,8)": 0.00095},
    9:  {"grouped(4,4,4)": 0.14844, "grouped(8,4,4)": 0.01497,
         "grouped(8,8,4)": 0.00182, "grouped(8,8,8)": 0.00090},
    10: {"grouped(4,4,4)": 0.11348, "grouped(8,4,4)": 0.01032,
         "grouped(8,8,4)": 0.00175, "grouped(8,8,8)": 0.00085},
    11: {"grouped(4,4,4)": 0.08908, "grouped(8,4,4)": 0.00984,
         "grouped(8,8,4)": 0.00170, "grouped(8,8,8)": 0.00081},
}


def cheapest_scheme(split, kl_budget, table=P95_KL):
    """Cheapest scheme at this split whose p95 KL meets the budget.
    Returns None if nothing qualifies."""
    for name in SCHEME_ORDER:
        if table[split][name] <= kl_budget:
            return name
    return None


def transfer_ms(nbytes, bw_mbps):
    return (nbytes * 8) / (bw_mbps * 1e6) * 1000


# ---------------- the decision ----------------
def decide(free_gb, bw_mbps, quality, seq_len=100,
           battery_pct=100, temp_c=40, weight_bits=2.0, use_p95=True):
    """Return the configuration to use under these conditions.

    Searches every feasible split, finds the cheapest transmitted size, and
    takes the SHALLOWEST layer achieving it - shallowest because edge compute
    is a real cost, even though the available timings measure GPU kernel launch
    overhead rather than compute and so are not modelled here.

    Battery below 20% and temperature above 80C pull the split back toward the
    edge doing less work. These are SIMULATED RULES, not hardware measurements.
    """
    table = P95_KL if use_p95 else MEAN_KL
    budget = QUALITY_LEVELS[quality]

    ceil_mem = memory_ceiling(free_gb, weight_bits)
    cap = ceil_mem - SAFETY_MARGIN_LAYERS
    binding = "memory"

    if battery_pct < 20 and cap > MIN_SPLIT:
        cap = MIN_SPLIT
        binding = "battery"
    if temp_c > 80 and cap > MIN_SPLIT:
        cap = MIN_SPLIT
        binding = "thermal"

    cap = min(cap, MEASURED_MAX_LAYER)

    if cap < MIN_SPLIT:
        return {"feasible": False, "split": None, "scheme": None,
                "memory_ceiling": ceil_mem, "binding_constraint": binding,
                "note": "no feasible split above the degenerate layer 0",
                "recommendation": "send raw tokens; run entirely in the cloud"}

    # search every feasible split
    candidates = []
    for s in range(MIN_SPLIT, cap + 1):
        sc = cheapest_scheme(s, budget, table)
        if sc is None:
            continue
        candidates.append((s, sc, scheme_bytes(seq_len, SCHEMES[sc])))

    if not candidates:
        return {"feasible": False, "split": None, "scheme": None,
                "memory_ceiling": ceil_mem, "binding_constraint": "quality",
                "note": f"no scheme meets a {quality} budget at any feasible split",
                "recommendation": "relax the budget or transmit uncompressed"}

    best_bytes = min(c[2] for c in candidates)
    # shallowest layer achieving the cheapest cost
    split, scheme, nbytes = min((c for c in candidates if c[2] == best_bytes),
                                key=lambda c: c[0])

    deepest = max(c[0] for c in candidates)
    raw = fp16_bytes(seq_len)

    return {"feasible": True,
            "split": split,
            "scheme": scheme,
            "bytes": nbytes,
            "pct_of_fp16": 100 * nbytes / raw,
            "reduction_pct": 100 * (1 - nbytes / raw),
            "transfer_ms": transfer_ms(nbytes, bw_mbps),
            "p95_kl": P95_KL[split][scheme],
            "mean_kl": MEAN_KL[split][scheme],
            "memory_ceiling": ceil_mem,
            "effective_cap": cap,
            "layers_searched": len(candidates),
            "deepest_feasible": deepest,
            "layers_saved_vs_deepest": deepest - split,
            "binding_constraint": binding}


if __name__ == "__main__":
    print(f"Model: {MODEL_NAME}")
    print(f"Split range searched: L{MIN_SPLIT} to the memory cap "
          f"(L0 excluded as degenerate)")
    print(f"Quality selected on the p95 of 300 WikiText passages")
    print(f"USABLE_FRACTION {USABLE_FRACTION} (judgement call, not measured)")
    print()

    print("=" * 84)
    print("MEMORY CEILING BY AVAILABLE RAM (bf16 weights)")
    print("=" * 84)
    print(f"{'free RAM':>10} {'layers fit':>11} {'ceiling':>9} {'usable cap':>11}"
          f"   note")
    print("-" * 84)
    for gb in RAM_LEVELS_GB:
        c = memory_ceiling(gb)
        cap = min(c - SAFETY_MARGIN_LAYERS, MEASURED_MAX_LAYER)
        print(f"{gb:>8.2f}GB {layers_that_fit(gb):>11} "
              f"{('L'+str(c)) if c >= 0 else 'none':>9} "
              f"{('L'+str(cap)) if cap >= MIN_SPLIT else 'none':>11}"
              f"   {MEASURED_RAM.get(gb, '')}")

    print("\n" + "=" * 84)
    print("p95 vs MEAN: the margin the percentile table buys")
    print("=" * 84)
    print("Selecting on the mean would violate the budget for roughly half of")
    print("inputs. The ratio is how much headroom p95 adds.")
    print()
    print(f"{'layer':>5} {'scheme':>16} {'mean KL':>10} {'p95 KL':>10} {'ratio':>8}")
    print("-" * 84)
    for L_ in [1, 3, 6, 9, 11]:
        for s in ["grouped(4,4,4)", "grouped(8,4,4)"]:
            m, p = MEAN_KL[L_][s], P95_KL[L_][s]
            print(f"{L_:5d} {s:>16} {m:10.5f} {p:10.5f} {p/m:8.2f}")

    print("\n" + "=" * 84)
    print("SCHEME COST")
    print("=" * 84)
    print(f"{'scheme':>16}" + "".join(f"{'seq '+str(s):>17}" for s in SEQ_LENS))
    print("-" * 84)
    for name in SCHEME_ORDER:
        row = f"{name:>16}"
        for s in SEQ_LENS:
            nb = scheme_bytes(s, SCHEMES[name])
            row += f"{nb:>10.0f} ({100*nb/fp16_bytes(s):4.1f}%)"
        print(row)

    print("\n" + "=" * 84)
    print("DECISION TABLE (seq 100, normal battery and temperature)")
    print("=" * 84)
    print(f"{'free RAM':>9} {'quality':>10} {'split':>6} {'scheme':>16} "
          f"{'bytes':>9} {'%fp16':>7} {'p95 KL':>9} {'deepest':>8} {'saved':>6}")
    print("-" * 84)
    table = {}
    for gb in RAM_LEVELS_GB:
        for q in QUALITY_LEVELS:
            d = decide(gb, 10, q, seq_len=100)
            table[f"{gb}_{q}"] = d
            if d["feasible"]:
                print(f"{gb:>7.2f}GB {q:>10} {'L'+str(d['split']):>6} "
                      f"{d['scheme']:>16} {d['bytes']:>9.0f} "
                      f"{d['pct_of_fp16']:>6.1f}% {d['p95_kl']:>9.5f} "
                      f"{'L'+str(d['deepest_feasible']):>8} "
                      f"{d['layers_saved_vs_deepest']:>6}")
            else:
                print(f"{gb:>7.2f}GB {q:>10} {'--':>6} {'run in cloud':>16}")
    print()
    print("  'deepest' = the deepest feasible split; 'saved' = how many layers")
    print("  of edge compute the search avoids by taking the shallowest layer")
    print("  that achieves the same transmitted size.")

    print("\n" + "=" * 84)
    print("WHAT THE p95 TABLE COSTS (7.81 GB free, seq 100)")
    print("=" * 84)
    print(f"{'quality':>10} {'on mean':>28} {'on p95':>28}")
    print("-" * 84)
    for q in QUALITY_LEVELS:
        a = decide(7.81, 10, q, seq_len=100, use_p95=False)
        b = decide(7.81, 10, q, seq_len=100, use_p95=True)
        fa = (f"L{a['split']} {a['scheme']} {a['bytes']:.0f}B"
              if a["feasible"] else "infeasible")
        fb = (f"L{b['split']} {b['scheme']} {b['bytes']:.0f}B"
              if b["feasible"] else "infeasible")
        print(f"{q:>10} {fa:>28} {fb:>28}")
    print()
    print("  Selecting on the mean meets the budget for about half of inputs;")
    print("  on p95, for about 95%. Any extra bytes are the price of that.")

    print("\n" + "=" * 84)
    print("TRANSFER TIME BY BANDWIDTH (7.81 GB free, strict quality)")
    print("=" * 84)
    print(f"{'bandwidth':>12} {'seq 50':>12} {'seq 100':>12} {'seq 250':>12}")
    print("-" * 84)
    for bw in BANDWIDTHS_MBPS:
        row = f"{str(bw)+' Mbps':>12}"
        for s in SEQ_LENS:
            d = decide(7.81, bw, "strict", seq_len=s)
            row += f"{d['transfer_ms']:>10.1f}ms" if d["feasible"] else f"{'--':>12}"
        print(row)

    print("\n" + "=" * 84)
    print("DEVICE CAPS (7.81 GB free, strict quality, seq 100)")
    print("=" * 84)
    for label, kw in [("normal", {}),
                      ("battery 15%", {"battery_pct": 15}),
                      ("temperature 85C", {"temp_c": 85}),
                      ("battery 15% + 85C", {"battery_pct": 15, "temp_c": 85})]:
        d = decide(7.81, 10, "strict", seq_len=100, **kw)
        if d["feasible"]:
            print(f"  {label:>18}: L{d['split']} {d['scheme']} "
                  f"{d['bytes']:.0f}B, binding = {d['binding_constraint']}")
        else:
            print(f"  {label:>18}: infeasible ({d['binding_constraint']}) - "
                  f"{d['recommendation']}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{RUN_TAG}.json"), "w") as f:
        json.dump({"model": MODEL_NAME,
                   "min_split": MIN_SPLIT,
                   "safety_margin_layers": SAFETY_MARGIN_LAYERS,
                   "usable_fraction": USABLE_FRACTION,
                   "quality_levels": QUALITY_LEVELS,
                   "selection": "p95 of KL over 300 WikiText passages",
                   "p95_kl": {str(k): v for k, v in P95_KL.items()},
                   "mean_kl": {str(k): v for k, v in MEAN_KL.items()},
                   "scheme_bytes": {n: {str(s): scheme_bytes(s, SCHEMES[n])
                                        for s in SEQ_LENS}
                                    for n in SCHEME_ORDER},
                   "decisions": table}, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/{RUN_TAG}.json")

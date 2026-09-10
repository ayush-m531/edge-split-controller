# Adaptive Split-Inference Controller

Runtime controller and working two-process system for splitting LLM inference
between a memory-constrained edge device and a cloud server, with outlier-aware
compression of the activation that crosses the network.

Built for Llama 3.1 8B. Measurements throughout come from that model on an
NVIDIA A100.

---

## The problem

Split inference runs the first *k* layers of a model on a device and the rest
in the cloud. The intermediate activation has to cross the network. For Llama
3.1 8B that tensor is 4096 values wide per token — 8 KB per token at bf16, so
a 1000-token prompt means 8 MB before the model produces anything.

Compressing it naively destroys the output. A handful of channels carry values
thousands of times larger than the rest, and a single shared quantization scale
stretches to fit them, rounding almost everything else to zero.

Two questions follow, and this repository answers both:

1. **How do you compress the activation without wrecking the output?**
2. **Where do you split, given that the answer changes as the device's memory,
   battery and temperature change?**

---

## What this does

**`lookup_table.py`** — the offline decision table. Given free RAM, a quality
budget, bandwidth, battery and temperature, it returns the split point and
compression scheme to use. Memory ceilings are derived from the model's
parameter counts; quality is selected on the 95th percentile of KL divergence
measured across 300 WikiText passages.

**`controller.py`** — the runtime loop. Watches conditions, re-decides, and
reconfigures only when the decision actually changes. Ships with a 21-step
scenario trace.

**`edge.py` / `cloud.py`** — a working two-process system. The edge loads the
layers its memory allows, runs the prompt through them, quantizes and
bit-packs the activation, and sends it over TCP. The cloud unpacks it, resumes
from the received split index, and returns the next token.

---

## Quick start

```bash
# terminal 1 — the cloud, loads layers 1..31 and waits
python3 controller/cloud.py

# terminal 2 — the edge
python3 controller/edge.py --quality strict --free-gb 7.81
```

The offline pieces need no GPU:

```bash
python3 controller/lookup_table.py     # decision table across conditions
python3 controller/controller.py       # 21-step scenario trace
```

---

## Measured results

All five cases below were run end to end. "Socket bytes" is what actually
crossed the TCP connection, not a modelled figure.

| Case | Split | Scheme | Socket bytes | % of bf16 |
|---|---|---|---|---|
| Normal, 7.81 GB free | L6 | grouped(8,4,4) | 25,068 | 25.5% |
| Memory pressure, 5.90 GB | L1 | grouped(8,8,4) | 32,508 | 33.1% |
| Relaxed budget, 7.81 GB | L8 | grouped(4,4,4) | 24,666 | 25.1% |
| Mid-pass abort at L3 | L3 | grouped(8,8,4) | 32,508 | 33.1% |
| 2 GB free | — | cloud only | — | — |

The byte model is accurate: 25,068 measured against 25,080 predicted, a 12-byte
gap accounted for by header scales versus the table's assumed 4 bytes per
group.

### Memory pressure changes the answer

The same handset, minutes apart:

```
5.90 GB free (apps running)  ->  L1, grouped(8,8,4), 32,508 B
7.81 GB free (apps cleared)  ->  L6, grouped(8,4,4), 25,068 B
```

23% fewer bytes because the user closed some applications. At 5.90 GB the
reachable layers are L1–L5, and none of them meets a strict quality budget with
the cheaper scheme, so the controller falls back. Both values are real
measurements from a six-year-old handset with 12 GB installed.

This is the case for deciding online rather than once offline.

### Mid-pass abort

The controller decides before the forward pass starts, but a device can cross a
thermal or battery threshold partway through. `--abort-at-layer` simulates
that:

```
planned split    L6, grouped(8,4,4)
aborted at       L3
scheme re-chosen grouped(8,4,4) -> grouped(8,8,4)
bytes            25,068 -> 32,508  (+30%)
cloud resumed    from L4, ran 28 layers instead of 25
output           unchanged and correct
```

Three things have to work for that. The split index travels with the payload,
so the cloud resumes from the right place — a system with a fixed split would
have started at L7 and silently skipped layers 4, 5 and 6. The scheme is
re-chosen for the shallower layer, because grouped(8,4,4) has a p95 KL of 0.141
at L3, well over the 0.05 budget. And the 30% byte penalty is the correct trade
against overheating the device.

---

## How the compression works

Channels are ranked by magnitude and split into four bands, each quantized with
its own scale:

| Group | Channels | Typical bits |
|---|---|---|
| top | 5 | 16 (unquantized) |
| shoulder | 6–72 | 8 |
| mid | 73–1312 | 4 or 8 |
| bulk | 1313–4096 | 4 |

Per-group scales are the point. A quantization scale is set by the largest
value in its group, so isolating the loud channels stops them coarsening the
thousands of quiet ones. Against per-tensor uniform quantization at the same
size this is 13–99× lower KL divergence; per-tensor uniform at 8 bits produces
degenerate output at any split point tested.

The channel ordering is computed **per layer** and precomputed offline. Using
one ordering everywhere costs up to 749× higher KL at layer 0, because the
outlier channels there are a different set entirely. Orderings derived from
unrelated calibration text reproduce the scheme selection of a same-text oracle
in 88–94% of cases, so they can be shipped with the model rather than
transmitted — sending 4096 indices at 12 bits would cost ~6 KB per request for
a list that never changes.

---

## How the controller decides

```
memory       ->  ceiling on the split, from parameter counts
quality      ->  scheme, selected on the p95 of the calibration distribution
battery/temp ->  caps that pull the split back toward less edge work
bandwidth    ->  affects transfer time only, not which configuration is correct
```

**The split is searched, not assumed.** Every feasible layer is evaluated, the
cheapest transmitted size found, and the *shallowest* layer achieving it taken
— shallowest because edge compute is a real cost. An earlier version went as
deep as memory allowed and paid up to 5 extra layers of compute for no
reduction in bytes.

**Layer 0 is excluded as degenerate.** A layer-0 activation costs ~205 KB per
100 tokens; the raw token IDs cost ~300 bytes, about 680× cheaper. If layer 0
were optimal there would be no reason to split at all.

**Reconfiguration is suppressed unless the decision changes.** Over the 21-step
scenario that is 8 reconfigurations rather than 21. Each one means the edge
loading or dropping layer weights at 416 MB per layer, so this is not cosmetic.

**The quality budget is per-request, not adapted.** The application declares it
when the request is submitted and it holds for that inference. It is a
user-facing knob because per-input quality cost cannot be predicted from the
activation — see below.

---

## Why the controller never looks at the prompt

Per-input quality cost varies by about 8× at a fixed layer and scheme. There is
real variation to detect. Whether any cheap statistic tracks it was tested five
ways across 300 texts:

| Approach | Best result |
|---|---|
| 9 activation statistics, individually, Pearson | \|r\| = 0.391 |
| Same, monotone non-linear (Spearman) | \|r\| = 0.397 |
| Same, combined, held-out validated | mean R² = 0.000 |
| Token-position structure and effective rank | \|r\| = 0.446 |
| Random forest, regression and classification | held-out R² negative; AUC 0.533 |

Nothing usable. Adding the activation features to a global model that already
knows the layer and scheme changes held-out R² by −0.002.

So the controller does not attempt detection. It provisions for the variation
by selecting schemes on the 95th percentile of the calibration distribution,
which meets the budget for roughly 95% of inputs without measuring anything at
runtime.

---

## What is real and what is not

**Real:** the layer split across two OS processes; the quantization; the bit
packing; the byte count on the socket; the TCP transfer; the cloud unpacking
and resuming from the received split index; the memory ceilings, derived from
parameter counts; the quality table, from 300 WikiText passages.

**Modelled:** transfer *time*. Both processes run on one machine over
localhost, which is effectively instant, so wire time is computed as
bytes ÷ bandwidth. The bytes are real; the milliseconds are arithmetic.

**Simulated:** free RAM, battery and temperature are passed as flags rather
than read from hardware. The two RAM figures used throughout — 5.90 and 7.81 GB
— are real measurements from a handset, taken by hand rather than over a
socket.

**Assumption:** 70% of free RAM is treated as available for weights, the
remainder covering activations, KV cache, runtime and OS headroom. This is a
judgement call, and the deep end of the range is sensitive to it: 0.67 gives a
ceiling one layer shallower than 0.70.

---

## Known limitations

**One token per request.** Continuing generation needs the next token's
activation at the split layer, which only the edge can produce. That requires a
return path and a round trip per token, not implemented here. The demo covers
the prefill phase.

**Prefill only.** Decode — token-by-token generation with a KV cache — has a
different transmission profile: many small messages rather than one large one,
latency-bound rather than bandwidth-bound. Not modelled.

**Hysteresis acts on decisions, not inputs.** It suppresses a repeat of the
same answer but would not suppress genuine oscillation between two valid
answers. Free memory hovering across a layer-fit boundary would thrash. A
deployed system needs input smoothing or a dwell time.

**The quality table is calibrated on WikiText.** English encyclopaedic prose.
Whether it holds for code, other languages or unusual formatting is untested —
two attempts to test it failed because the selected passages turned out to be
memorised training data, scoring 5–8× *easier* than the calibration corpus.

**p95 is a probabilistic guarantee.** The worst 5% of inputs still exceed the
budget. Nothing reaches 100%: the next text could be worse than anything
measured.

**Timing is not modelled.** Available per-layer timings measured GPU kernel
launch overhead rather than compute — a 0.5B model cost the same at 250 tokens
as at 5 — so edge compute cost does not enter the decision. The controller
returns the most transmission-efficient feasible configuration and leaves
compute as future work.

---

## Layout

```
controller/
  lookup_table.py    offline decision table; p95 quality data; memory ceilings
  controller.py      runtime loop, hysteresis, 21-step scenario trace
  edge.py            edge process: loads layers, quantizes, packs, sends
  cloud.py           cloud process: unpacks, resumes from the received split
data/
  lookup_table.json      the decision table
  controller_trace.json  the scenario trace
  orders/                per-layer channel orderings
```

---

## Requirements

Measured and run on:

| | |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| transformers | 5.12.1 |
| numpy | 2.4.4 |
| GPU | NVIDIA A100-PCIE-40GB, driver 570.172.08 |

```bash
pip install torch transformers numpy
```

Versions matter here. Several APIs this code touches have
changed recently:

- `model.model.rotary_emb(...)` and the `position_embeddings=` argument to a
  decoder layer — the signature differs across Transformers versions, and the
  edge and cloud processes both call layers directly rather than going through
  `model.forward()`
- `torch_dtype=` is deprecated in favour of `dtype=` in transformers 5.x; the
  code still uses the old name and emits a warning

**Model access.** `meta-llama/Llama-3.1-8B-Instruct` is gated on Hugging Face.
Accept the licence on the model page, then `hf auth login`. The weights are
about 16 GB and download once.

**GPU memory.** The cloud process holds layers 1–31 plus the head (~14 GB); the
edge holds the embedding table plus up to 12 layers (~4–6 GB). Both fit
together on a 40 GB card with room, but nothing else should be running.

The offline pieces — `lookup_table.py` and `controller.py` — need no GPU and no
model weights. They read the model's config only.

---

## Where the numbers come from

The quality table, channel orderings and byte model are derived from the
experiments in the companion repository, which profiles activation structure
and compression sensitivity across 86,400 measurements on 300 texts.

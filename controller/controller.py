"""
Runtime controller for adaptive split inference.

The lookup table answers "given these conditions, what configuration?". This
is the loop that watches conditions change, re-asks, and reports when the
answer moves.

TWO KINDS OF INPUT, and the distinction matters:

  THE QUALITY BUDGET is a property of the REQUEST. The application declares it
  when the request is submitted and it holds for that inference - the split is
  chosen from it and the edge loads layers accordingly, so it cannot change
  partway through. It is a user-facing knob because per-input quality cost
  cannot be predicted from the activation, so the system cannot infer how much
  a request matters.

  DEVICE CONDITIONS drift continuously within a session. Free memory, battery
  and temperature change as the user opens applications, the device heats, the
  battery drains. These are what the controller adapts to.

WHAT IS REAL AND WHAT IS SIMULATED - state this plainly whenever the results
are presented:
  REAL       the memory ceiling (derived from Llama 3.1 8B parameter counts),
             the p95 quality table (300 WikiText passages), the byte costs,
             and the decision logic itself
  SIMULATED  the condition stream. Bandwidth, free RAM, battery and
             temperature are fed from a scenario rather than read from
             hardware. Reading them live needs a physical device and is
             future work.

The controller is INPUT-AGNOSTIC: it never inspects the prompt. Per-input
quality cost varies about 8x at a fixed layer and scheme, but no cheap
statistic predicts it (tested five ways; strongest correlation 0.446, mean
held-out R2 of a learned model 0.000). So it provisions for that variation via
the p95 margin rather than attempting to detect it.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookup_table import decide, memory_ceiling, QUALITY_LEVELS

OUT_DIR = "/home/ayush.thakar/edge-split-controller/data"
RUN_TAG = "controller_trace"


class Conditions:
    """Device and network state at one instant.

    The quality budget is carried here for convenience, but note it is NOT a
    device condition - it is fixed per request. The session trace holds it
    constant for exactly that reason.
    """

    def __init__(self, free_gb, bw_mbps, battery_pct, temp_c,
                 quality="strict", seq_len=100):
        self.free_gb = free_gb
        self.bw_mbps = bw_mbps
        self.battery_pct = battery_pct
        self.temp_c = temp_c
        self.quality = quality
        self.seq_len = seq_len

    def __str__(self):
        return (f"RAM {self.free_gb:.2f}GB | {self.bw_mbps:>3}Mbps | "
                f"bat {self.battery_pct:>3}% | {self.temp_c:>2}C")

    def as_dict(self):
        return {"free_gb": self.free_gb, "bw_mbps": self.bw_mbps,
                "battery_pct": self.battery_pct, "temp_c": self.temp_c,
                "quality": self.quality, "seq_len": self.seq_len}


class Controller:
    """Re-decides when conditions change, and reconfigures only when the
    decision actually differs.

    WHY THE HYSTERESIS MATTERS: a naive controller recomputes on every reading
    and reconfigures whenever anything moves. In a split system, changing the
    split point means the edge must load or drop layer weights - 416 MB per
    layer at bf16. Reconfiguring for an unchanged decision is pure cost. So
    the controller compares the new decision against the current one and acts
    only on a genuine change.
    """

    def __init__(self):
        self.current = None
        self.history = []
        self.n_decisions = 0
        self.n_changes = 0

    def update(self, cond, label=""):
        self.n_decisions += 1
        d = decide(cond.free_gb, cond.bw_mbps, cond.quality,
                   seq_len=cond.seq_len, battery_pct=cond.battery_pct,
                   temp_c=cond.temp_c)
        changed = self._differs(self.current, d)
        if changed:
            self.n_changes += 1
        self.history.append({"step": self.n_decisions, "label": label,
                             "conditions": cond.as_dict(), "decision": d,
                             "changed": changed})
        self.current = d
        return d, changed

    @staticmethod
    def _differs(old, new):
        if old is None:
            return True
        if old.get("feasible") != new.get("feasible"):
            return True
        if not new.get("feasible"):
            return False
        return old["split"] != new["split"] or old["scheme"] != new["scheme"]

    def summary(self):
        return {"decisions": self.n_decisions,
                "reconfigurations": self.n_changes,
                "reconfiguration_rate": (self.n_changes / self.n_decisions
                                         if self.n_decisions else 0)}


def fmt(d):
    if not d["feasible"]:
        return f"{'CLOUD ONLY':>16}  {d['binding_constraint']:>8}"
    return (f"L{d['split']:<2} {d['scheme']:>14}  "
            f"{d['bytes']:>7.0f}B  {d['transfer_ms']:>7.1f}ms  "
            f"{d['binding_constraint']:>8}")


# ---------------- session scenario ----------------
# ONE session, ONE quality budget (strict) held throughout, so that every
# reconfiguration is attributable to device conditions alone. Each step
# changes as few things as possible. The set exercises specific behaviours
# rather than looking busy.
SCENARIO = [
    # cold start: no current configuration, nothing loaded
    ("cold start, idle wifi",     Conditions(7.81, 50, 95, 38)),

    # memory pressure: the online-decision argument
    ("user opens more apps",      Conditions(5.90, 50, 90, 40)),

    # network degrades; the decision should NOT change. Bandwidth affects
    # transfer TIME, not which configuration is correct.
    ("moves to mobile data",      Conditions(5.90, 10, 88, 41)),
    ("weak signal",               Conditions(5.90,  1, 86, 42)),

    # long stable period: nothing changes, nothing should happen
    ("stable",                    Conditions(5.90,  1, 84, 43)),
    ("stable",                    Conditions(5.90,  1, 82, 43)),

    # thermal becomes ACTIVE but not BINDING - over 80C the thermal cap pulls
    # the split to L1, but memory had already forced L1. The label changes,
    # the decision does not. Comparing outcomes rather than causes gets this
    # right.
    ("sustained load, heating",   Conditions(5.90,  1, 79, 63)),
    ("thermal limit reached",     Conditions(5.90,  1, 76, 84)),
    ("cools, network recovers",   Conditions(5.90, 10, 73, 55)),

    # memory freed: the cheaper scheme becomes reachable again
    ("closes background apps",    Conditions(7.81, 10, 70, 48)),

    # OSCILLATION TEST. Free memory hovering either side of a layer-fit
    # boundary, as real devices do when background processes allocate and
    # release. Included deliberately to see whether the controller thrashes;
    # if it does, that is a limitation worth naming rather than hiding.
    ("memory hovering (a)",       Conditions(6.40, 10, 68, 47)),
    ("memory hovering (b)",       Conditions(6.55, 10, 67, 47)),
    ("memory hovering (c)",       Conditions(6.40, 10, 66, 47)),
    ("memory hovering (d)",       Conditions(6.55, 10, 65, 47)),

    # battery cap: a different constraint takes over
    ("battery falling",           Conditions(7.81, 10, 24, 46)),
    ("battery critical",          Conditions(7.81, 10, 14, 45)),

    # two constraints active at once; which one binds?
    ("critical battery AND hot",  Conditions(7.81, 10, 12, 86)),

    ("plugged in, back on wifi",  Conditions(7.81, 50, 31, 40)),

    # degradation to infeasible, and recovery
    ("heavy multitasking",        Conditions(4.00, 50, 45, 44)),
    ("memory exhausted",          Conditions(2.00, 50, 50, 44)),
    ("recovers",                  Conditions(7.81, 50, 60, 40)),
]


if __name__ == "__main__":
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 104)
    out("ADAPTIVE SPLIT-INFERENCE CONTROLLER")
    out("=" * 104)
    out("REAL      : the memory ceiling (from Llama 3.1 8B parameter counts),")
    out("            the p95 quality table (300 WikiText passages), byte costs,")
    out("            and the decision logic itself.")
    out("SIMULATED : the condition stream. Bandwidth, free RAM, battery and")
    out("            temperature are fed from a scenario rather than read from")
    out("            hardware. Reading them live needs a physical device and")
    out("            is future work.")
    out()

    # ---------------- A. per-request budget ----------------
    out("=" * 104)
    out("A. QUALITY BUDGET: SET ONCE PER REQUEST, NOT ADAPTED")
    out("=" * 104)
    out("The budget is a property of the REQUEST, not of the session. The")
    out("application declares it when the request is submitted and it holds")
    out("for that inference - the split is chosen from it and the edge loads")
    out("layers accordingly, so it cannot change partway through.")
    out()
    out("It is a user-facing knob because per-input quality cost cannot be")
    out("predicted from the activation (tested five ways; strongest")
    out("correlation 0.446, mean held-out R2 of a learned model 0.000). The")
    out("system cannot infer how much a request matters, so the application")
    out("states it.")
    out()
    out("Three separate requests, identical device conditions:")
    out()
    out(f"{'request':>28} {'budget':>8}  {'decision':>46}")
    out("-" * 104)
    per_request = {}
    for q, b in QUALITY_LEVELS.items():
        d = decide(7.81, 50, q, seq_len=100)
        per_request[q] = d
        out(f"{q + ' request':>28} {b:>8.2f}  {fmt(d):>46}")
    out()
    out("  Three budgets, three different splits, two different schemes.")
    out("  Within any one request the budget is FIXED; only device conditions")
    out("  vary, which is what section B traces.")
    out()

    # ---------------- B. session trace ----------------
    out("=" * 104)
    out("B. DEVICE CONDITIONS: ADAPTED WITHIN A SESSION")
    out("=" * 104)
    out("Budget held at 'strict' throughout, so every reconfiguration is")
    out("attributable to device conditions alone.")
    out()

    ctrl = Controller()
    out(f"{'step':>4} {'event':>26}  {'conditions':>42}")
    out(f"{'':>4} {'':>26}  {'decision':>42}")
    out("-" * 104)
    for i, (label, cond) in enumerate(SCENARIO, 1):
        d, changed = ctrl.update(cond, label)
        mark = " *" if changed else "  "
        out(f"{i:>4} {label:>26}  {str(cond):>42}")
        out(f"{mark:>4} {'':>26}  {fmt(d):>42}")
    out()
    out("  * = the decision CHANGED and the system reconfigured")
    out()

    s = ctrl.summary()
    out("=" * 104)
    out("SUMMARY")
    out("=" * 104)
    out(f"  decisions evaluated : {s['decisions']}")
    out(f"  reconfigurations    : {s['reconfigurations']}")
    out(f"  reconfiguration rate: {100*s['reconfiguration_rate']:.0f}%")
    out()
    out("  Without hysteresis the controller would reconfigure on every")
    out(f"  reading - {s['decisions']} times rather than "
        f"{s['reconfigurations']}. Each reconfiguration means")
    out("  the edge loads or drops layer weights (416 MB per layer at bf16),")
    out("  so suppressing unchanged decisions is not cosmetic.")
    out()

    configs = {}
    for h in ctrl.history:
        d = h["decision"]
        k = "cloud only" if not d["feasible"] else f"L{d['split']} {d['scheme']}"
        configs[k] = configs.get(k, 0) + 1
    out("=" * 104)
    out("CONFIGURATIONS VISITED IN THE SESSION")
    out("=" * 104)
    for k, n in sorted(configs.items(), key=lambda kv: -kv[1]):
        out(f"  {k:>26} : {n:>2} steps")
    out()
    out("  Fewer distinct configurations than section A shows, because the")
    out("  budget is fixed here. L4 and L8 appear only under a different")
    out("  per-request budget, not through adaptation.")
    out()

    out("=" * 104)
    out("WHAT DROVE EACH RECONFIGURATION")
    out("=" * 104)
    prev = None
    for h in ctrl.history:
        if not h["changed"]:
            prev = h
            continue
        d = h["decision"]
        if prev is None:
            out(f"  step {h['step']:>2} {h['label']:>26}: initial configuration")
        else:
            pd = prev["decision"]
            a = f"L{pd['split']} {pd['scheme']}" if pd["feasible"] else "cloud only"
            b = f"L{d['split']} {d['scheme']}" if d["feasible"] else "cloud only"
            db = (d["bytes"] - pd["bytes"]) if (d["feasible"] and pd["feasible"]) else None
            delta = f"  ({db:+.0f} B)" if db is not None else ""
            out(f"  step {h['step']:>2} {h['label']:>26}: {a} -> {b}{delta}")
            out(f"     {'':>29} binding: {d.get('binding_constraint')}")
        prev = h
    out()

    out("=" * 104)
    out("ACTIVE vs BINDING CONSTRAINTS")
    out("=" * 104)
    out("A constraint can become ACTIVE without being BINDING - it demands")
    out("something the system is already doing. The controller compares")
    out("OUTCOMES, not causes, so these produce no reconfiguration. A naive")
    out("implementation reacting to 'thermal limit crossed' would reconfigure")
    out("for nothing.")
    out()
    any_ab = False
    for h in ctrl.history:
        d, c = h["decision"], h["conditions"]
        hot = c["temp_c"] > 80
        low = c["battery_pct"] < 20
        if (hot or low) and not h["changed"]:
            which = " and ".join((["thermal"] if hot else []) +
                                 (["battery"] if low else []))
            out(f"  step {h['step']:>2} {h['label']:>26}: {which} active, "
                f"binding = {d.get('binding_constraint')}, no reconfiguration")
            any_ab = True
    if not any_ab:
        out("  (none in this scenario)")
    out()

    out("=" * 104)
    out("OSCILLATION UNDER HOVERING MEMORY")
    out("=" * 104)
    out("Free memory bouncing either side of a layer-fit boundary, as real")
    out("devices do when background processes allocate and release.")
    out()
    osc = [h for h in ctrl.history if "hovering" in h["label"]]
    flips = sum(1 for h in osc if h["changed"])
    for h in osc:
        d = h["decision"]
        cfg = f"L{d['split']} {d['scheme']}" if d["feasible"] else "cloud only"
        out(f"  step {h['step']:>2}  RAM {h['conditions']['free_gb']:.2f}GB  "
            f"-> {cfg}{'   * reconfigured' if h['changed'] else ''}")
    out()
    if flips == 0:
        out("  No thrashing: the hovering does not cross a decision boundary,")
        out("  so the configuration holds. This does NOT prove the controller")
        out("  is immune - it means these particular values fall on the same")
        out("  side of the boundary.")
    else:
        out(f"  THRASHING: {flips} reconfigurations across {len(osc)} readings")
        out("  of essentially unchanged memory.")
    out()
    out("  LIMITATION EITHER WAY: the hysteresis acts on DECISIONS, not on")
    out("  INPUTS. It suppresses a repeat of the same answer but would not")
    out("  suppress genuine oscillation between two valid answers. A deployed")
    out("  system would need input smoothing or a dwell time before acting.")
    out()

    out("=" * 104)
    out("EFFECT OF PROMPT LENGTH (7.81GB / 10Mbps / strict)")
    out("=" * 104)
    out("Length changes the transmitted SIZE but not the scheme choice -")
    out("quality varies little with length (ratios 0.85-1.2 across a 4x")
    out("change), so one quality table covers all lengths.")
    out()
    out(f"{'tokens':>8} {'split':>6} {'scheme':>16} {'bytes':>11} "
        f"{'transfer':>12} {'vs bf16':>9}")
    out("-" * 104)
    for n in [50, 100, 250, 500, 1000, 2000]:
        d = decide(7.81, 10, "strict", seq_len=n)
        if d["feasible"]:
            out(f"{n:>8} {'L'+str(d['split']):>6} {d['scheme']:>16} "
                f"{d['bytes']:>11.0f} {d['transfer_ms']:>10.1f}ms "
                f"{d['pct_of_fp16']:>8.1f}%")
    out()
    raw2000 = 2000 * 4096 * 2
    out(f"  At 2000 tokens, uncompressed bf16 is {raw2000:,} B and takes")
    out(f"  {raw2000 * 8 / 1e7 * 1000:.0f} ms at 10 Mbps. The controller's")
    out(f"  choice is about a quarter of that.")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{RUN_TAG}.json"), "w") as f:
        json.dump({"per_request_budgets": per_request,
                   "scenario": [{"label": l, **c.as_dict()}
                                for l, c in SCENARIO],
                   "history": ctrl.history, "summary": s,
                   "configurations_visited": configs,
                   "oscillation_flips": flips}, f, indent=2)
    with open(os.path.join(OUT_DIR, f"{RUN_TAG}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {OUT_DIR}/{RUN_TAG}.{{json,txt}}")

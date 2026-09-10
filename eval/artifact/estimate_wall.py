#!/usr/bin/env python3
"""How long does reproducing the campaign take? Read off the campaign's OWN rows.

    eval/artifact/estimate_wall.py                       # every tier, on the reference card
    eval/artifact/estimate_wall.py --mode remeasure      # without the recorded-failure pin
    eval/artifact/estimate_wall.py --write-subset        # regenerate eval/artifact/subset.json

The artifact tells a reviewer what each tier costs, and a number in a README rots. This derives
it from the frozen rows, so `reproduce.sh` prints an estimate the data can be held to, and
ARTIFACT.md quotes this script rather than a memory of a run.

WHAT THE ESTIMATE IS. A reproduction pinned by `--grants` pays, per cell:
  * the rival's ONE run at its recorded rung        -> the row's own `wall_s`
  * our arm at three seeds                          -> `n_models / sps_e2e_with_delivery` each,
                                                       plus the offline dedup (`distinct_s`)
  * the CMSGen arm                                  -> `cms_wall_s`
  * a recorded rival FAILURE                        -> nothing under `--mode honour` (the
                                                       shortcut), the full init budget under
                                                       `--mode remeasure`
No ladder walk and no start_point probe: that is what the grants file removes.

WHAT IT DOES NOT COVER, and why two overhead terms are added: the rows do not time the per-cell
ABC simplify and equivalence check, the 262k-sequence rare-pattern simulation, the miter build,
the `sim_check` replay of every kept row through the original netlist, the discarded warm-up per
(tool, process), CNF encoding, or subprocess and CUDA start-up.

Measured on real cells, that is NOT a fraction of the measured wall: on the smallest cells it
dwarfs it (a DEMOTIC cell whose recorded parts sum to 6 s takes ~26 s). It has two parts, and
both are modelled:
  * a per-cell CONSTANT (`--overhead-per-cell`) — process and CUDA start-up, the discarded
    warm-up, the CNF encode;
  * a term LINEAR IN THE NETLIST (`--overhead-per-gate`) — the ABC simplify and its
    equivalence check, measured at a few tenths of a millisecond per gate, which is seconds on
    b03 and over a minute on b19. Omitting it underestimates precisely the cells whose rival
    side is free (a pinned failure), which the reviewer tier is full of.
A `--overhead` fraction carries what is left (chiefly the `sim_check` replay, which scales with
what the cell delivered). All three were measured on one host and are CPU-bound, so a machine
whose CPU differs pays a different constant: `--cpu-scale` carries them across.

It is an estimate FOR THE CARD THE ROWS WERE MEASURED ON. Another GPU scales it: `--scale`.
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

#: stage stem -> (the row key holding the rival's numbers, a human label)
STAGES = {
    "rivals_demotic_native": ("demotic", "DEMOTIC, their ISCAS-85 pins, T=1"),
    "rivals_demotic_comb_rare": ("demotic", "DEMOTIC, ISCAS-85, our rare pin, T=1"),
    "rivals_recurrent_table": ("rcs", "Recurrent, their target table, T=25"),
    "rivals_recurrent_rare": ("rcs", "Recurrent, our rare pin, T=25"),
    "rivals_recurrent_table_t50": ("rcs", "Recurrent, their target table, T=50"),
    "rivals_recurrent_rare_t50": ("rcs", "Recurrent, our rare pin, T=50"),
}
T25_STAGES = [s for s in STAGES if not s.endswith("_t50")]

#: The initialization budget a re-measured recorded failure spends before it is one
#: (`rivals_recurrent.py --init-budget`, protocol v1.0).
INIT_BUDGET_S = 1200.0
#: What the rows do not time (see the module docstring), split into the two terms that fit it:
#: a per-cell constant (process and CUDA start-up, the ABC pass, the equivalence check, the
#: rare-pattern simulation, the warm-up) and a fraction of the measured wall (the `sim_check`
#: replay, which scales with what the cell delivered).
DEFAULT_OVERHEAD = 0.10
DEFAULT_OVERHEAD_PER_CELL = 12.0
DEFAULT_OVERHEAD_PER_GATE = 0.0003
#: Where the gate counts come from — the same files the harnesses hand the rivals.
BENCH_DIRS = [os.path.join(REPO, "benchmarks", "circuits", "bench", d) for d in ("seq", "comb")]
#: The subset rule: keep a sequential instance when BOTH its tasks together cost no more than
#: this at the reference card, under the artifact's default (failures honoured). Mechanical and
#: about the clock alone — it cannot select for cells we win, which a hand-picked list could.
SUBSET_MINUTES = 10.0


def gate_count(name):
    """Gates in the committed netlist, or 0 when it is not installed (b17-b22 are fetched, not
    committed). A missing count makes the estimate LOW, and the caller is told which."""
    for d in BENCH_DIRS:
        path = os.path.join(d, f"{name}.bench")
        if os.path.isfile(path):
            return sum(1 for ln in open(path)
                       if "=" in ln and not ln.lstrip().startswith("#"))
    return 0


def _rows(path):
    payload = json.load(open(path))
    return (payload.get("rows") if isinstance(payload, dict) else payload) or []


def cell_cost(row, rival_key, mode):
    """(seconds, is_pinned_failure) for one cell of a grants-pinned reproduction."""
    rv = row.get(rival_key) or {}
    g = row.get("gpap") or {}
    ours = 0.0
    for s in g.get("seeds") or []:
        rate = s.get("sps_e2e_with_delivery") or s.get("sps_e2e")
        n = s.get("n_models") or 0
        if rate and n:
            ours += n / rate
        ours += float(s.get("distinct_s") or 0.0)
    failed = bool(rv.get("error"))
    rival = float(rv.get("wall_s") or 0.0)          # a failure records no successful run
    if failed and mode == "remeasure":
        rival += INIT_BUDGET_S
    return rival + ours + float(row.get("cms_wall_s") or 0.0), failed


def campaign_costs(data_dir, mode):
    """{stage: {instance: (seconds, failed)}} over whatever stage files are present."""
    out = {}
    for stem, (key, _label) in STAGES.items():
        path = os.path.join(data_dir, f"{stem}.json")
        if not os.path.isfile(path):
            continue
        out[stem] = {r["name"]: cell_cost(r, key, mode) for r in _rows(path) if r.get("name")}
    return out


#: The committed selection `reproduce.sh subset` actually runs. Read it when it exists, so an
#: estimate cannot describe a different cell set from the one the script will run: the rule is
#: evaluated once, under the artifact's own defaults, and frozen there.
SUBSET_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subset.json")


def committed_subset():
    if os.path.isfile(SUBSET_JSON):
        return json.load(open(SUBSET_JSON)).get("instances") or None
    return None


def subset_instances(costs, minutes=SUBSET_MINUTES):
    """The sequential instances the reviewer tier keeps, by the stated clock rule at T=25."""
    per = {}
    for stem in T25_STAGES:
        if "recurrent" not in stem:
            continue
        for name, (sec, _f) in costs.get(stem, {}).items():
            per[name] = per.get(name, 0.0) + sec
    return sorted(n for n, sec in per.items() if sec <= minutes * 60.0), per


def tier_cells(costs, tier, subset):
    """{stage: [instance...]} — which cells a tier runs."""
    if tier == "full":
        return {s: sorted(c) for s, c in costs.items()}
    if tier == "subset":
        return {s: (sorted(c) if "demotic" in s else sorted(set(c) & set(subset)))
                for s, c in costs.items() if s in T25_STAGES}
    if tier == "smoke":                      # one cheap instance per stage, as reproduce.sh runs it
        want = {"rivals_demotic_native": ["c432"], "rivals_demotic_comb_rare": ["c432"],
                "rivals_recurrent_table": ["b03"], "rivals_recurrent_rare": ["b03"]}
        return {s: [n for n in v if n in costs.get(s, {})] for s, v in want.items()
                if s in costs}
    raise ValueError(tier)


def tier_wall(costs, cells, overhead, per_cell=DEFAULT_OVERHEAD_PER_CELL,
              per_gate=DEFAULT_OVERHEAD_PER_GATE, cpu_scale=1.0):
    """(total seconds, per-stage seconds, cell count, instances with no gate count)."""
    per, n, unknown = {}, 0, set()
    for stem, names in cells.items():
        sec = sum(costs[stem][x][0] for x in names)
        head = 0.0
        for x in names:
            g = gate_count(x)
            if not g:
                unknown.add(x)
            head += (per_cell + per_gate * g) * cpu_scale
        per[stem] = sec * (1.0 + overhead) + head
        n += len(names)
    return sum(per.values()), per, n, sorted(unknown)


def _hms(sec):
    return f"{sec/3600:.2f} h" if sec >= 3600 else f"{sec/60:.1f} min"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", nargs="?",
                    default=os.path.join(os.environ.get("GPAP_RESULTS",
                                                        os.path.join(REPO, "..", "gpap-results")),
                                         "paper", "data"),
                    help="the REFERENCE campaign rows (default: $GPAP_RESULTS/paper/data)")
    ap.add_argument("--mode", choices=("honour", "remeasure"), default="honour",
                    help="honour (default, the artifact's own default): a cell the grants file "
                         "records as a rival failure is pinned, not re-derived. remeasure: it "
                         "costs the full initialization budget again")
    ap.add_argument("--overhead", type=float, default=DEFAULT_OVERHEAD,
                    help=f"fraction of the measured wall added for what the rows do not time "
                         f"(default {DEFAULT_OVERHEAD:.2f}; see the module docstring)")
    ap.add_argument("--overhead-per-cell", type=float, default=DEFAULT_OVERHEAD_PER_CELL,
                    metavar="SECONDS",
                    help=f"per-cell constant added for the same reason — start-up, warm-up, "
                         f"encode (default {DEFAULT_OVERHEAD_PER_CELL:.0f}s)")
    ap.add_argument("--overhead-per-gate", type=float, default=DEFAULT_OVERHEAD_PER_GATE,
                    metavar="SECONDS",
                    help=f"per-cell seconds PER GATE of the netlist — the ABC pass and its "
                         f"equivalence check (default {DEFAULT_OVERHEAD_PER_GATE})")
    ap.add_argument("--cpu-scale", type=float, default=1.0,
                    help="scale both overhead terms for a host whose CPU differs from the one "
                         "they were measured on — they are CPU-bound, so > 1 for a slower host "
                         "and < 1 for a faster one")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply by this for a card other than the one measured")
    ap.add_argument("--tier", choices=("smoke", "subset", "full", "all"), default="all")
    ap.add_argument("--write-subset", metavar="FILE", nargs="?", const=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "subset.json"),
                    help="write the subset selection (with the walls that produced it)")
    ap.add_argument("--json", action="store_true", help="machine-readable, for reproduce.sh")
    args = ap.parse_args()

    data = os.path.abspath(args.data_dir)
    if not glob.glob(os.path.join(data, "rivals_*.json")):
        sys.exit(f"no reference rows in {data} — pass the campaign's data directory")
    costs = campaign_costs(data, args.mode)
    derived, per_inst = subset_instances(costs)
    # What the script runs beats what this run of the rule would pick: `--write-subset` is how
    # the selection changes, deliberately and visibly.
    subset = (committed_subset() or derived) if not args.write_subset else derived
    gpu = None
    for stem in costs:                                    # the card the reference was measured on
        rows = _rows(os.path.join(data, f"{stem}.json"))
        gpu = next((r.get("gpu", {}).get("name") for r in rows if r.get("gpu")), None) or gpu
        if gpu:
            break

    tiers = {}
    for tier in (("smoke", "subset", "full") if args.tier == "all" else (args.tier,)):
        cells = tier_cells(costs, tier, subset)
        total, per, n, unknown = tier_wall(costs, cells, args.overhead, args.overhead_per_cell,
                                           args.overhead_per_gate, args.cpu_scale)
        tiers[tier] = {"seconds": total * args.scale, "cells": n, "no_gate_count": unknown,
                       "per_stage": {k: v * args.scale for k, v in per.items()},
                       "instances": cells}

    if args.json:
        print(json.dumps({"reference_gpu": gpu, "mode": args.mode, "overhead": args.overhead,
                          "overhead_per_cell_s": args.overhead_per_cell,
                          "scale": args.scale, "subset_rule_minutes": SUBSET_MINUTES,
                          "subset": subset,
                          "tiers": {k: {"seconds": v["seconds"], "cells": v["cells"]}
                                    for k, v in tiers.items()}}, indent=1))
    else:
        print(f"reference rows : {data}")
        print(f"reference GPU  : {gpu}   (scale {args.scale:g} for another card)")
        print(f"failure policy : {args.mode}"
              + ("  (recorded rival failures pinned, not re-derived)" if args.mode == "honour"
                 else f"  (each recorded failure re-pays {INIT_BUDGET_S:.0f}s)"))
        print(f"overhead       : +{args.overhead:.0%} of the measured wall, "
              f"+{args.overhead_per_cell:.0f}s per cell and "
              f"+{args.overhead_per_gate * 1000:.2f}ms per gate"
              + (f", all x{args.cpu_scale:g} for this host's CPU" if args.cpu_scale != 1 else "")
              + "\n")
        for tier, v in tiers.items():
            print(f"== {tier}: {v['cells']} cells, {_hms(v['seconds'])}")
            if v["no_gate_count"]:
                print(f"     (no netlist for {', '.join(v['no_gate_count'])} — their ABC term is "
                      f"missing, so this is LOW; run scripts/fetch_benchmarks.sh itc99)")
            for stem, sec in sorted(v["per_stage"].items()):
                if v["instances"].get(stem):
                    print(f"     {STAGES[stem][1]:<42s} {len(v['instances'][stem]):>3d} cells "
                          f"{_hms(sec):>9s}")
        src = ("eval/artifact/subset.json, what reproduce.sh runs"
               if committed_subset() and not args.write_subset else "the rule, evaluated now")
        print(f"\nsubset ({src}):\n            a sequential instance is kept when both its T=25 "
              f"tasks together cost <= {SUBSET_MINUTES:.0f} min on the\n            reference "
              f"card with recorded failures pinned. Kept {len(subset)}/{len(per_inst)}:")
        print("            " + " ".join(subset))
        print("            dropped: " + " ".join(sorted(set(per_inst) - set(subset))))

    if args.write_subset:
        payload = {
            "rule": f"a sequential instance is kept when its two T=25 tasks together cost "
                    f"<= {SUBSET_MINUTES:.0f} min on {gpu} with recorded rival failures pinned "
                    f"(--mode honour). Mechanical and about the clock alone: it cannot select "
                    f"for cells we win. Regenerate with eval/artifact/estimate_wall.py "
                    f"--write-subset.",
            "reference_gpu": gpu, "minutes": SUBSET_MINUTES,
            "instances": subset,
            "reference_seconds": {n: round(s, 1) for n, s in sorted(per_inst.items())},
        }
        with open(args.write_subset, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nwrote {args.write_subset}: {len(subset)} instances")


if __name__ == "__main__":
    main()

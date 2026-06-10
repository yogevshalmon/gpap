#!/usr/bin/env python3
"""Cheap-trial PORTFOLIO selector for GPU clause injection — cashing the per-instance oracle.

The heuristic sweep showed no single variable-selection rule wins universally (oracle
best-per-instance ≈ 1.45× vs ~1.1× for any fixed rule), and that injection sometimes HURTS — so
the win lives in *per-instance selection*, with `plain` (no injection) as an essential fallback.

This evaluator collects, per instance and per config, the FULL solve (ground truth) and a cheap
TRIAL solve (bounded conflict budget), then computes several selector policies OFFLINE from that
data so many policies are compared from one expensive pass:

  configs  : plain | units | jw-bin | rare-bin | random-bin   (binaries = units + anchor×all
             over the jw / rare(low-degree) / random top-Kx variables; static scores → no warmup)
  oracle   : best full solve per instance (upper bound)
  global-X : always config X (no selection)
  trial    : sequential cheap-trial race — try configs in priority order with a small conflict
             budget; if one SOLVES in trial, take it; else commit to the best global rule. Cost =
             Σ trial times + (committed full solve). The honest "1 CPU core" selector.
  parallel : run all configs concurrently (K cores); wall ≈ oracle. The "spare cores" selector.

speedup is reported NO-GPU (discovery time recorded but excluded — assume a fast GPU). All clauses
are RUP so every config must return the same verdict.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/inprocess_portfolio.py --insts /tmp/hardpool.json \\
        --device cuda --Kx 24 --trial-frac 0.4 --trial-cap 30000 --max-solve-s 45 \\
        --output .experiments/inproc_portfolio.json
"""
import argparse
import json
import os
import random
import statistics
import sys
from threading import Timer

import torch

sys.path.insert(0, ".")
from pysat.formula import CNF
from pysat.solvers import Cadical195

from gpap.solver.inprocess import GPUInprocessor
from scripts.inprocess_heuristics import static_scores, select

CONFIGS = ["plain", "units", "jw-bin", "rare-bin", "random-bin"]
TRIAL_ORDER = ["units", "jw-bin", "rare-bin", "random-bin", "plain"]  # try cheap/promising first
GLOBAL_FALLBACK = "jw-bin"  # best single global rule from the sweep (most consistent)


def _solve(clauses, budget, max_solve_s):
    """Solve with a conflict budget + wall cap. Returns (result, solve_ms, conflicts, solved)."""
    cnf = CNF(from_clauses=clauses)
    s = Cadical195(bootstrap_with=cnf, use_timer=True)
    timer = Timer(max_solve_s, s.interrupt)
    timer.start()
    s.conf_budget(budget)
    r = s.solve_limited(expect_interrupt=True)
    timer.cancel()
    ms = s.time_accum() * 1e3
    cf = s.accum_stats()["conflicts"]
    s.delete()
    res = "SAT" if r else "UNSAT" if r is False else "UNK"
    return res, ms, cf, (r is not None)


def eval_instance(rec, device, Kx, trial_frac, trial_cap, max_solve_s, seed):
    path = rec["path"]
    cnf = CNF(from_file=path)
    orig = [list(c) for c in cnf.clauses if c]
    N = cnf.nv

    res0, full0, cf0, solved0 = _solve(orig, 10_000_000, max_solve_s)
    if not solved0:
        return {"name": os.path.basename(path)[:34], "skip": "plain UNK"}

    deg, jw = static_scores(cnf)
    rng = random.Random(seed)
    inp = GPUInprocessor.from_cnf_file(path, device)

    import time as _t
    t0 = _t.perf_counter(); units, _ = inp.probe_failed_literals()
    disc = {"units": (_t.perf_counter() - t0) * 1e3}

    def anchored(H):
        anchors = [v - 1 for v in select(H, Kx, {}, deg, jw, N, rng)]
        s = _t.perf_counter(); inp.reset_emitted()
        b = inp.discover_binaries_anchored(anchors)[0]
        disc[H + "-bin"] = (_t.perf_counter() - s) * 1e3 + disc["units"]
        return b
    bins = {"jw-bin": anchored("jw"), "rare-bin": anchored("rare"), "random-bin": anchored("random")}
    del inp
    if device.type == "cuda":
        torch.cuda.empty_cache()

    clause_sets = {
        "plain": orig,
        "units": orig + list(units),
        "jw-bin": orig + list(units) + bins["jw-bin"],
        "rare-bin": orig + list(units) + bins["rare-bin"],
        "random-bin": orig + list(units) + bins["random-bin"],
    }
    trial_budget = min(trial_cap, max(2000, int(trial_frac * cf0)))

    cfg = {}
    for name, cls in clause_sets.items():
        fr, fms, fcf, fs = _solve(cls, 10_000_000, max_solve_s)
        tr, tms, tcf, ts = _solve(cls, trial_budget, max_solve_s)
        cfg[name] = {
            "full_ms": fms, "full_cf": fcf, "full_result": fr, "full_solved": fs,
            "trial_ms": tms, "trial_cf": tcf, "trial_solved": ts,
            "sound": fr in (res0, "UNK"),
            "disc_ms": disc.get(name, 0.0),
            "n_clauses_added": len(cls) - len(orig),
        }
    return {"name": os.path.basename(path)[:34], "family": rec.get("family"), "N": N, "M": len(orig),
            "result": res0, "plain_full_ms": full0, "plain_cf": cf0,
            "trial_budget": trial_budget, "cfg": cfg}


# ----------------------------------------------------------------- selector policies (offline)
def _full(r, name):
    c = r["cfg"][name]
    return c["full_ms"] if c["full_solved"] else float("inf")


def policy_oracle(r):
    return min(_full(r, n) for n in CONFIGS)


def policy_global(r, name):
    return _full(r, name)


def policy_trial_1core(r):
    """Sequential cheap-trial race: pay each trial in order; if one solves in trial, stop and use
    it; else commit to the best global rule (pay its full solve too)."""
    cost = 0.0
    for name in TRIAL_ORDER:
        c = r["cfg"][name]
        cost += c["trial_ms"]
        if c["trial_solved"]:
            return cost                      # solved during the trial — done
    return cost + _full(r, GLOBAL_FALLBACK)  # nobody solved in trial → commit to global rule


def policy_parallel(r, names=CONFIGS):
    """Parallel portfolio: with len(names) cores, wall ≈ fastest config's full solve
    (discovery assumed free/overlapped). Including 'plain' makes it never worse than plain."""
    return min(_full(r, n) for n in names)


# parallel sub-portfolios (always include plain → a "safe" portfolio that can't lose to plain)
SAFE2 = ["plain", "rare-bin"]                       # 2 cores
SAFE3 = ["plain", "units", "rare-bin"]              # 3 cores
SAFE4 = ["plain", "units", "jw-bin", "rare-bin"]    # 4 cores


def _gm(xs):
    xs = [x for x in xs if x and x > 0 and x != float("inf")]
    return statistics.geometric_mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--Kx", type=int, default=24)
    ap.add_argument("--trial-frac", type=float, default=0.4, help="trial conflict budget = frac × plain conflicts")
    ap.add_argument("--trial-cap", type=int, default=30000, help="max trial conflict budget")
    ap.add_argument("--max-solve-s", type=float, default=45.0)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    insts = json.load(open(args.insts))[:args.limit]
    print(f"Device {device} | {len(insts)} instances | Kx={args.Kx} trial=min({args.trial_cap},"
          f"{args.trial_frac}×plain_cf) | speedup vs plain (no-GPU)\n")
    hdr = (f"{'instance':<34}{'plain':>7}{'Btrial':>7}" + "".join(f"{n[:8]:>9}" for n in CONFIGS)
           + f"{'oracle':>8}{'trial1c':>8}")
    print(hdr); print("-" * len(hdr))

    results = []
    for rec in insts:
        try:
            r = eval_instance(rec, device, args.Kx, args.trial_frac, args.trial_cap, args.max_solve_s, args.seed)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(rec['path'])[:34]:<34}  ERROR: {e}"); continue
        results.append(r)
        if "skip" in r:
            print(f"{r['name']:<34}  skip ({r['skip']})"); continue
        p = r["plain_full_ms"]
        sp = lambda name: (f"{p/_full(r,name):.2f}" if _full(r, name) != float('inf') else "to")
        row = "".join(f"{sp(n):>9}" for n in CONFIGS)
        orc = f"{p/policy_oracle(r):.2f}"
        t1 = f"{p/policy_trial_1core(r):.2f}"
        flag = "" if all(c["sound"] for c in r["cfg"].values()) else " !UNSOUND"
        print(f"{r['name']:<34}{p:>7.0f}{r['trial_budget']:>7}{row}{orc:>8}{t1:>8}{flag}")

    ev = [r for r in results if "cfg" in r]
    if ev:
        print("\n" + "=" * 72)
        print(f"instances: {len(ev)}   geomean speedup vs plain (no-GPU):")
        for name in CONFIGS:
            xs = [r["plain_full_ms"] / _full(r, name) for r in ev]
            print(f"  global {name:<11}: {_gm(xs):.3f}  improved {sum(1 for x in xs if x>1)}/{len(ev)}")
        def gms(fn):
            return _gm([r["plain_full_ms"] / fn(r) for r in ev])
        orc = [r["plain_full_ms"] / policy_oracle(r) for r in ev]
        t1c = [r["plain_full_ms"] / policy_trial_1core(r) for r in ev]
        n_solved_trial = sum(1 for r in ev if any(r["cfg"][n]["trial_solved"] and not r["cfg"]["plain"]["trial_solved"] for n in TRIAL_ORDER[:-1]))
        worst_safe2 = min(r["plain_full_ms"] / policy_parallel(r, SAFE2) for r in ev)
        print(f"  ----- parallel portfolios (plain always included → never worse than plain):")
        print(f"  SAFE-2 cores (plain|rare)        : {gms(lambda r: policy_parallel(r, SAFE2)):.3f}  "
              f"worst-case {worst_safe2:.2f}")
        print(f"  SAFE-3 cores (plain|units|rare)  : {gms(lambda r: policy_parallel(r, SAFE3)):.3f}")
        print(f"  SAFE-4 cores (plain|units|jw|rare): {gms(lambda r: policy_parallel(r, SAFE4)):.3f}")
        print(f"  PARALLEL-5 (all) = ORACLE        : {_gm(orc):.3f}  median {statistics.median(orc):.2f}")
        print(f"  ----- single core:")
        print(f"  TRIAL-1core (cheap race)         : {_gm(t1c):.3f}  median {statistics.median(t1c):.2f}  "
              f"[an injected config solved in-trial on {n_solved_trial}/{len(ev)}]")
        unsound = sum(1 for r in ev for c in r["cfg"].values() if not c["sound"])
        print(f"  unsound: {unsound} (MUST be 0)")
        print("=" * 72)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "Kx": args.Kx, "trial_frac": args.trial_frac,
                   "trial_cap": args.trial_cap, "results": results}, open(args.output, "w"), indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

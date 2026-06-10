#!/usr/bin/env python3
"""Batch evaluation: does GPU-discovered clause injection speed up CaDiCaL, across many instances?

For each instance: (1) plain solve; (2) GPU discovery — failed-literal units + binary clauses over
a WIDE variable net (exhaustive when feasible, else a sampled pool — the M3 finding is that the
*important*/most-constrained variables' binaries are already subsumed, so the useful ones live over
the wider set); (3) augmented solve at several injection budgets. Every injected clause is RUP, so
the verdict must match.

Reports speedup TWO ways, because the GPU discovery cost is hardware/kernel-improvable (A100, fused
kernels) while the solving benefit is not:
  • speedup_no_gpu  = plain_solve / aug_solve          (the ceiling — GPU discovery assumed free)
  • speedup_with_gpu = plain_solve / (aug_solve + gpu)  (realistic on this box — discovery still costs)

Usage:
    PYTHONPATH=. .venv/bin/python scripts/inprocess_suite.py --insts /tmp/good_insts.json \\
        --device cuda --limit 25 --budgets 0,1000,5000,20000 --output .experiments/inproc_suite.json
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from threading import Timer

import torch

sys.path.insert(0, ".")
from pysat.formula import CNF
from pysat.solvers import Cadical195

from gpap.solver.inprocess import GPUInprocessor
from scripts.profile_engine import count_all_pairs

MAX_EXHAUSTIVE_PAIRS = 8_000_000   # above this, sample a variable pool instead of all-pairs


import multiprocessing as _mp


def _solve_worker(clauses, q):
    try:
        from pysat.formula import CNF as _CNF
        from pysat.solvers import Cadical195 as _Cad
        cnf = _CNF(from_clauses=clauses)
        s = _Cad(bootstrap_with=cnf, use_timer=True)
        s.conf_budget(80_000_000)            # backstop only; the wall-clock kill is the real cap
        r = s.solve_limited(expect_interrupt=True)
        ms = s.time_accum() * 1e3
        cf = s.accum_stats()["conflicts"]
        s.delete()
        q.put(("SAT" if r else "UNSAT" if r is False else "UNK", ms, cf))
    except Exception:  # noqa: BLE001
        q.put(("UNK", 0.0, 0))


_SPAWN = _mp.get_context("spawn")   # spawn (not fork) → safe with a CUDA-initialised parent


def timed_solve(clauses: list[list[int]], nv: int, max_solve_s: float):
    """Solve in a child process with a HARD wall-clock kill.

    pysat's thread-based Timer→interrupt() is NOT a reliable cutoff (it can fail to stop a hard
    solve, hanging for hours), so the only dependable timeout is terminating the child process.
    Returns (result, solve_ms, conflicts); UNK with solve_ms=max_solve_s*1e3 if killed.
    """
    q = _SPAWN.Queue()
    p = _SPAWN.Process(target=_solve_worker, args=(clauses, q))
    t0 = time.perf_counter()
    p.start()
    p.join(max_solve_s)
    if p.is_alive():
        p.terminate(); p.join(3)
        if p.is_alive():
            p.kill(); p.join()
        return "UNK", max_solve_s * 1e3, 0
    try:
        return q.get(timeout=3)
    except Exception:  # noqa: BLE001
        return "UNK", (time.perf_counter() - t0) * 1e3, 0


def discover(path: str, device: torch.device, pairs_pool: int, seed: int):
    """Wide GPU discovery: probe units + binaries. Returns (units, binaries, gpu_ms, build_ms, N)."""
    t_b = time.perf_counter()
    inp = GPUInprocessor.from_cnf_file(path, device)
    build_ms = (time.perf_counter() - t_b) * 1e3
    N = inp.N
    t_g = time.perf_counter()
    units, _ = inp.probe_failed_literals()
    if count_all_pairs(N) <= MAX_EXHAUSTIVE_PAIRS:
        bvars = list(range(N))                      # exhaustive over all variables
    else:
        rng = random.Random(seed)
        bvars = rng.sample(range(N), min(N, pairs_pool))
    bins, _ = inp.discover_binaries(bvars)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t_g) * 1e3
    del inp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return units, bins, gpu_ms, build_ms, N


def eval_instance(rec, device, pairs_pool, budgets, max_solve_s, seed):
    path = rec["path"]
    cnf = CNF(from_file=path)
    # Drop empty clauses (pysat keeps them; they crash Cadical195 bootstrap on clause[0], and
    # GPAP drops them anyway — keeps plain and augmented on the same clause base).
    orig = [list(c) for c in cnf.clauses if c]
    nv = cnf.nv

    res0, solve0, cf0 = timed_solve(orig, nv, max_solve_s)
    if res0 == "UNK":
        return {"name": os.path.basename(path)[:50], "skip": "plain UNK"}

    units, bins, gpu_ms, build_ms, N = discover(path, device, pairs_pool, seed)

    configs = []
    for B in budgets:
        extra = list(units) + bins[:B]              # units always; binaries up to budget B
        resB, solveB, cfB = timed_solve(orig + extra, nv, max_solve_s)
        configs.append({
            "budget": B, "n_inject": len(extra), "result": resB,
            "solve_ms": solveB, "conflicts": cfB,
            "sound": (resB == res0 or resB == "UNK"),
            "speedup_no_gpu": (solve0 / solveB) if solveB > 0 else None,
            "speedup_with_gpu": (solve0 / (solveB + gpu_ms)) if (solveB + gpu_ms) > 0 else None,
        })

    finished = [c for c in configs if c["result"] == res0]
    best = max(finished, key=lambda c: c["speedup_no_gpu"]) if finished else None
    return {
        "name": os.path.basename(path)[:50], "family": rec.get("family"),
        "N": N, "M": len(orig), "result": res0,
        "plain_solve_ms": solve0, "plain_conflicts": cf0,
        "n_units": len(units), "n_bins": len(bins),
        "gpu_ms": gpu_ms, "build_ms": build_ms,
        "configs": configs,
        "best": best,
        "unsound": [c["budget"] for c in configs if not c["sound"]],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insts", required=True, help="JSON list of {path,...} instance records")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--pairs-pool", type=int, default=1000, help="Var sample size when not exhaustive")
    ap.add_argument("--budgets", default="0,1000,5000,20000",
                    help="Comma list of #binaries to inject (units always added)")
    ap.add_argument("--max-solve-s", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    budgets = [int(b) for b in args.budgets.split(",")]
    insts = json.load(open(args.insts))[:args.limit]

    print(f"Device {device}  |  {len(insts)} instances  |  budgets(#bins)={budgets}\n")
    hdr = f"{'instance':<34}{'M':>7}{'plainms':>9}{'units':>6}{'bins':>7}{'gpums':>7}{'bestB':>7}{'×noGPU':>8}{'×wGPU':>7}"
    print(hdr); print("-" * len(hdr))

    results = []
    for rec in insts:
        try:
            r = eval_instance(rec, device, args.pairs_pool, budgets, args.max_solve_s, args.seed)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(rec['path'])[:34]:<34}  ERROR: {e}")
            continue
        results.append(r)
        if "skip" in r:
            print(f"{r['name']:<34}  skip ({r['skip']})"); continue
        b = r["best"]
        sn = f"{b['speedup_no_gpu']:.2f}" if b else "-"
        sw = f"{b['speedup_with_gpu']:.2f}" if b else "-"
        bb = b["budget"] if b else "-"
        flag = "  !UNSOUND" if r["unsound"] else ""
        print(f"{r['name']:<34}{r['M']:>7}{r['plain_solve_ms']:>9.0f}{r['n_units']:>6}"
              f"{r['n_bins']:>7}{r['gpu_ms']:>7.0f}{str(bb):>7}{sn:>8}{sw:>7}{flag}")

    # ---- summary ----
    ev = [r for r in results if "best" in r and r["best"]]
    if ev:
        def gm(xs):
            xs = [x for x in xs if x and x > 0]
            return statistics.geometric_mean(xs) if xs else float("nan")
        best_ng = [r["best"]["speedup_no_gpu"] for r in ev]
        best_wg = [r["best"]["speedup_with_gpu"] for r in ev]
        improved_ng = sum(1 for x in best_ng if x > 1.0)
        improved_wg = sum(1 for x in best_wg if x > 1.0)
        any_unsound = sum(1 for r in results if r.get("unsound"))
        print("\n" + "=" * 60)
        print(f"instances evaluated      : {len(ev)}")
        print(f"best-budget ×no-GPU      : geomean {gm(best_ng):.2f}  median {statistics.median(best_ng):.2f}  "
              f"(improved {improved_ng}/{len(ev)})")
        print(f"best-budget ×with-GPU    : geomean {gm(best_wg):.2f}  median {statistics.median(best_wg):.2f}  "
              f"(improved {improved_wg}/{len(ev)})")
        # fixed-budget view (no per-instance oracle): use the middle budget
        for B in budgets:
            xs_ng = [next((c["speedup_no_gpu"] for c in r["configs"] if c["budget"] == B and c["result"] == r["result"]), None) for r in ev]
            xs_wg = [next((c["speedup_with_gpu"] for c in r["configs"] if c["budget"] == B and c["result"] == r["result"]), None) for r in ev]
            print(f"fixed B={B:<6} ×no-GPU {gm(xs_ng):.2f} (imp {sum(1 for x in xs_ng if x and x>1)}/{len(ev)})  "
                  f"×with-GPU {gm(xs_wg):.2f}")
        print(f"unsound instances        : {any_unsound}  (MUST be 0)")
        print("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "budgets": budgets, "results": results},
                  open(args.output, "w"), indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

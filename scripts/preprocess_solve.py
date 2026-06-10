#!/usr/bin/env python3
"""Does GPU *preprocessing* (vivification + unit-closure + subsumption) beat plain CaDiCaL?

This is the reduction counterpart to the (failed) clause-*addition* experiment. Adding entailed
clauses perturbs CDCL chaotically; here we instead REDUCE the formula — shorten clauses by BCP
(vivify), assert failed-literal units and propagate, and let discovered short clauses subsume
longer ones — all equivalence-preserving. The question: does a more-exhaustive-than-the-solver
GPU reduction pass make the resulting (smaller) CNF solve faster, on top of CaDiCaL's own
inprocessing which is already on in every config?

GPU discovery (units, binaries) + vivification run ONCE per instance; the configs below are then
assembled on the CPU so they share that GPU work:
    plain   original formula (baseline; CaDiCaL still vivifies/subsumes internally)
    viv     vivify → subsume/dedup
    viv+U   vivify → unit-closure → subsume/dedup
    full    vivify → unit-closure → learned-binary subsumption → subsume/dedup

Every reduced formula is logically equivalent to the original, so the SAT/UNSAT verdict MUST
match plain — a mismatch flags an unsound reduction and is reported loudly.

Two speedups (GPU discovery is hardware-improvable; the solve benefit is not):
    ×no-GPU  = plain_solve / config_solve
    ×wGPU    = plain_solve / (config_solve + gpu_ms)
Plus a SAFE portfolio = min(plain, full) per instance (run both, keep the winner) — never regresses.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/preprocess_solve.py --insts /tmp/bigpool2.json \\
        --device cuda --max-solve-s 30 --vivify-max-len 12 --output .experiments/preproc.json
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, ".")

from gpap.solver.inprocess import GPUInprocessor
from gpap.solver.preprocess import (
    _simplify_by_assignment,
    apply_learned_subsumption,
    subsume_all,
    unit_closure,
)
from scripts.inprocess_suite import timed_solve
from scripts.profile_engine import count_all_pairs

CONFIGS = ["plain", "viv", "viv+U", "full"]
_UNSAT = [[1], [-1]]


def _read_clauses(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "cp%":
                continue
            lits = [x for x in (int(t) for t in line.split()) if x != 0]
            if lits:
                out.append(lits)
    return out


def _bin_vars(N, mode, pool, deg, seed):
    if mode == "none":
        return []
    if mode == "all" or count_all_pairs(N) <= 4_000_000:
        return list(range(N))
    # degree-weighted: the highest-occurrence variables are likeliest to yield subsuming binaries
    if deg is not None:
        return [v - 1 for v in sorted(range(1, N + 1), key=lambda v: -deg.get(v, 0))[:pool]]
    return random.Random(seed).sample(range(N), min(N, pool))


def discover(path, device, orig, N, *, bin_mode, bin_pool, vmax_len, vmax_rounds, deg, seed):
    """Run the GPU phase once; return (units, bins, vivified, gpu_ms)."""
    inp = GPUInprocessor.from_cnf_file(path, device)
    t0 = time.perf_counter()
    units, _ = inp.probe_failed_literals()
    bins = []
    bvars = _bin_vars(N, bin_mode, bin_pool, deg, seed)
    if len(bvars) >= 2:
        inp.reset_emitted()
        bins, _ = inp.discover_binaries(bvars)
    vivified, vstats = inp.vivify_clauses(orig, max_len=vmax_len, max_rounds=vmax_rounds)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1e3
    del inp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return units, bins, vivified, gpu_ms, vstats


def build_configs(orig, units, bins, vivified):
    """Assemble the CPU configs from the shared GPU output. Returns {name: clauses}."""
    cfg = {"plain": orig, "viv": subsume_all(vivified)}
    uc, assigned, status = unit_closure(vivified, units)
    if status == "unsat":
        cfg["viv+U"] = list(_UNSAT)
        cfg["full"] = list(_UNSAT)
        return cfg
    cfg["viv+U"] = subsume_all(uc)
    full, _ = apply_learned_subsumption(uc, _simplify_by_assignment(bins, assigned))
    cfg["full"] = subsume_all(full)
    return cfg


def eval_instance(rec, device, args, deg_fn):
    path = rec["path"]
    orig = _read_clauses(path)
    if not orig:
        return {"name": os.path.basename(path)[:38], "skip": "empty"}
    nv = max(abs(l) for c in orig for l in c)

    res0, ms0, cf0 = timed_solve(orig, nv, args.max_solve_s)
    if res0 == "UNK":
        return {"name": os.path.basename(path)[:38], "skip": "plain UNK"}

    deg = deg_fn(path) if deg_fn else None
    units, bins, vivified, gpu_ms, vstats = discover(
        path, device, orig, nv, bin_mode=args.bin_mode, bin_pool=args.bin_pool,
        vmax_len=args.vivify_max_len, vmax_rounds=args.vivify_max_rounds, deg=deg, seed=args.seed)
    cfgs = build_configs(orig, units, bins, vivified)

    out = {"name": os.path.basename(path)[:38], "family": rec.get("family"), "nv": nv,
           "M": len(orig), "lits": sum(len(c) for c in orig), "result": res0,
           "plain_ms": ms0, "gpu_ms": gpu_ms,
           "n_units": len(units), "n_bins": len(bins),
           "n_vivified": vstats.get("n_vivified", 0), "viv_lits_removed": vstats.get("lits_removed", 0),
           "cfg": {}}
    for name in CONFIGS:
        if name == "plain":
            out["cfg"]["plain"] = {"solve_ms": ms0, "result": res0, "M": len(orig),
                                   "lits": sum(len(c) for c in orig), "speedup": 1.0,
                                   "speedup_wgpu": 1.0, "sound": True}
            continue
        cls = cfgs[name]
        r, ms, cf = timed_solve(cls, nv, args.max_solve_s)
        sound = r in (res0, "UNK")
        out["cfg"][name] = {
            "solve_ms": ms, "result": r, "M": len(cls), "lits": sum(len(c) for c in cls),
            "speedup": (ms0 / ms) if (ms > 0 and r == res0) else None,
            "speedup_wgpu": (ms0 / (ms + gpu_ms)) if (ms + gpu_ms) > 0 and r == res0 else None,
            "sound": sound,
        }
    return out


def _gm(xs):
    xs = [x for x in xs if x and x > 0]
    return statistics.geometric_mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-solve-s", type=float, default=30.0)
    ap.add_argument("--vivify-max-len", type=int, default=12)
    ap.add_argument("--vivify-max-rounds", type=int, default=None)
    ap.add_argument("--bin-mode", default="degree", choices=["none", "degree", "all"])
    ap.add_argument("--bin-pool", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    insts = json.load(open(args.insts))[:args.limit]

    deg_fn = None
    if args.bin_mode == "degree":
        from scripts.inprocess_heuristics import static_scores
        from pysat.formula import CNF

        def deg_fn(path):
            return static_scores(CNF(from_file=path))[0]

    print(f"Device {device} | {len(insts)} instances | vivify_max_len={args.vivify_max_len} "
          f"bin_mode={args.bin_mode} | speedup = plain / config (CaDiCaL inprocessing ON in all)\n")
    hdr = (f"{'instance':<34}{'M':>7}{'plainms':>8}{'gpums':>7}{'viv':>6}{'units':>6}"
           f"{'Δlit%':>6}" + "".join(f"{n:>8}" for n in CONFIGS[1:]))
    print(hdr); print("-" * len(hdr))

    results = []
    for rec in insts:
        try:
            r = eval_instance(rec, device, args, deg_fn)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(rec['path'])[:34]:<34}  ERROR: {type(e).__name__}: {e}")
            continue
        results.append(r)
        if "skip" in r:
            print(f"{r['name']:<34}  skip ({r['skip']})")
            continue
        dlit = 100.0 * (1 - r["cfg"]["full"]["lits"] / r["lits"]) if r["lits"] else 0.0

        def cell(n):
            c = r["cfg"][n]
            if not c["sound"]:
                return "UNSND"
            return f"{c['speedup']:.2f}" if c["speedup"] else "-"
        print(f"{r['name']:<34}{r['M']:>7}{r['plain_ms']:>8.0f}{r['gpu_ms']:>7.0f}"
              f"{r['n_vivified']:>6}{r['n_units']:>6}{dlit:>6.1f}"
              + "".join(f"{cell(n):>8}" for n in CONFIGS[1:]))

    ev = [r for r in results if "cfg" in r]
    if ev:
        print("\n" + "=" * 70)
        print(f"instances: {len(ev)}   geomean speedup vs plain (CaDiCaL inprocessing on in all):")
        for n in CONFIGS:
            ng = [r["cfg"][n]["speedup"] for r in ev]
            wg = [r["cfg"][n]["speedup_wgpu"] for r in ev]
            imp = sum(1 for x in ng if x and x > 1.0)
            print(f"  {n:<7} ×no-GPU {_gm(ng):.3f}  ×wGPU {_gm(wg):.3f}  "
                  f"median {statistics.median([x for x in ng if x] or [0]):.2f}  improved {imp}/{len(ev)}")
        # SAFE portfolio: per instance keep the faster of plain and full (run both → never worse)
        safe = []
        for r in ev:
            f = r["cfg"]["full"]
            safe.append(r["plain_ms"] / min(r["plain_ms"], f["solve_ms"])
                        if f["result"] == r["result"] else 1.0)
        print(f"  SAFE[plain|full] geomean {_gm(safe):.3f} (min(plain,full) per instance)")
        # mean literal reduction of `full`
        dl = [100.0 * (1 - r["cfg"]["full"]["lits"] / r["lits"]) for r in ev if r["lits"]]
        print(f"  full formula reduction: mean Δliterals {statistics.mean(dl):.1f}%  "
              f"median {statistics.median(dl):.1f}%")
        unsound = sum(1 for r in ev for c in r["cfg"].values() if not c["sound"])
        print(f"  unsound configs: {unsound}  (MUST be 0)")
        print("=" * 70)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "args": vars(args), "results": results},
                  open(args.output, "w"), indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

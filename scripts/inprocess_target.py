#!/usr/bin/env python3
"""Does TARGETING the right variables make GPU clause discovery help solving? (no-GPU potential)

Tests variable-targeting strategies for discovering units/binaries/triples, measuring
speedup_no_gpu = plain_solve / augmented_solve (GPU discovery time intentionally ignored — the
question is whether a powerful-enough GPU + the *right* targeting would help at all). Variables are
scored by a VSIDS proxy: CaDiCaL branch-decision counts from a warm-start solve.

Configs compared (all include the failed-literal UNITS, which are selection-independent):
    units     : units only (baseline)
    vsCliq    : + binaries WITHIN the top-K VSIDS variables (clique) — tends to be subsumed
    vsCross   : + binaries between top-Kx VSIDS anchors and EVERY variable (anchor×all) — each
                binary touches a hot variable but isn't subsumed  ← the targeted hypothesis
    rndCross  : + binaries between Kx RANDOM anchors and every variable (control for vsCross)
    vsTri     : + vsCliq binaries + triples over the top-Kt VSIDS variables

Every injected clause is RUP, so the verdict must match the plain solve.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/inprocess_target.py --insts /tmp/comp_insts.json \\
        --device cuda --K 256 --Kx 32 --Kt 48 --warmup 8000 --limit 12 --output .experiments/target.json
"""
import argparse
import json
import os
import random
import statistics
import sys

import torch

sys.path.insert(0, ".")
from pysat.formula import CNF

from gpap.solver.inprocess import GPUInprocessor
from gpap.solver.cadical_hook import score_variables_by_decisions
from scripts.inprocess_suite import timed_solve

CONFIGS = ["units", "vsCliq", "vsCross", "rndCross", "vsTri"]


def select(strategy, K, dec, degree, N, rng):
    """Top-K variables (1-based) under a selection strategy."""
    if strategy == "vsids":
        return sorted(range(1, N + 1), key=lambda v: (dec.get(v, 0), degree.get(v, 0)), reverse=True)[:K]
    if strategy == "degree":
        return sorted(range(1, N + 1), key=lambda v: degree.get(v, 0), reverse=True)[:K]
    vs = list(range(1, N + 1)); rng.shuffle(vs); return vs[:K]


def eval_instance(rec, device, K, Kx, Kt, warmup, max_solve_s, seed):
    path = rec["path"]
    cnf = CNF(from_file=path)
    orig = [list(c) for c in cnf.clauses if c]
    N = cnf.nv

    res0, solve0, cf0 = timed_solve(orig, N, max_solve_s)
    if res0 == "UNK":
        return {"name": os.path.basename(path)[:40], "skip": "plain UNK"}

    # observe_cap bounds warm-start callback overhead on big formulas (scoring, not counted vs speedup)
    dec, _assign, degree, _ = score_variables_by_decisions(
        path, warmup_conflicts=warmup, observe_cap=min(N, 5000))
    rng = random.Random(seed)
    vs_K  = [v - 1 for v in select("vsids",  K,  dec, degree, N, rng)]
    vs_Kx = [v - 1 for v in select("vsids",  Kx, dec, degree, N, rng)]
    rn_Kx = [v - 1 for v in select("random", Kx, dec, degree, N, rng)]
    vs_Kt = [v - 1 for v in select("vsids",  Kt, dec, degree, N, rng)]

    inp = GPUInprocessor.from_cnf_file(path, device)
    units, _ = inp.probe_failed_literals()

    def disc(fn, arg):
        inp.reset_emitted()
        return fn(arg)[0]

    bins_cliq  = disc(inp.discover_binaries, vs_K)
    bins_cross = disc(inp.discover_binaries_anchored, vs_Kx)
    bins_rcross = disc(inp.discover_binaries_anchored, rn_Kx)
    inp.reset_emitted(); tris = inp.discover_ternaries(vs_Kt)[0]
    del inp
    if device.type == "cuda":
        torch.cuda.empty_cache()

    extras = {
        "units": [], "vsCliq": bins_cliq, "vsCross": bins_cross,
        "rndCross": bins_rcross, "vsTri": bins_cliq + tris,
    }
    cfgs = {}
    for name, extra in extras.items():
        r, s, _ = timed_solve(orig + list(units) + extra, N, max_solve_s)
        cfgs[name] = {"speedup": (solve0 / s) if (s > 0 and r == res0) else None,
                      "n_inject": len(units) + len(extra), "sound": r in (res0, "UNK")}

    return {
        "name": os.path.basename(path)[:40], "family": rec.get("family"),
        "N": N, "M": len(orig), "result": res0, "plain_ms": solve0, "plain_cf": cf0,
        "n_units": len(units),
        "counts": {"cliq": len(bins_cliq), "cross": len(bins_cross),
                   "rcross": len(bins_rcross), "tri": len(tris)},
        "cfgs": cfgs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--K", type=int, default=256, help="top-K vars for clique binary discovery")
    ap.add_argument("--Kx", type=int, default=32, help="top-Kx VSIDS anchors for cross (anchor×all)")
    ap.add_argument("--Kt", type=int, default=48, help="top-Kt vars for triple discovery")
    ap.add_argument("--warmup", type=int, default=8000, help="warm-start conflicts for VSIDS proxy")
    ap.add_argument("--max-solve-s", type=float, default=40.0)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    insts = json.load(open(args.insts))[:args.limit]
    print(f"Device {device} | {len(insts)} instances | K={args.K} Kx={args.Kx} Kt={args.Kt} "
          f"warmup={args.warmup} | speedup = plain/augmented (GPU ignored)\n")
    hdr = (f"{'instance':<32}{'M':>7}{'plain':>7}{'unit':>5}{'cross':>7}"
           f"{'×U':>6}{'×vsCliq':>9}{'×vsCross':>9}{'×rndX':>7}{'×vsTri':>8}")
    print(hdr); print("-" * len(hdr))

    results = []
    for rec in insts:
        try:
            r = eval_instance(rec, device, args.K, args.Kx, args.Kt, args.warmup, args.max_solve_s, args.seed)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(rec['path'])[:32]:<32}  ERROR: {e}"); continue
        results.append(r)
        if "skip" in r:
            print(f"{r['name']:<32}  skip ({r['skip']})"); continue
        c = r["cfgs"]
        sp = lambda n: (f"{c[n]['speedup']:.2f}" if c[n]["speedup"] else "-")
        unsound = "" if all(v["sound"] for v in c.values()) else " !UNSOUND"
        print(f"{r['name']:<32}{r['M']:>7}{r['plain_ms']:>7.0f}{r['n_units']:>5}{r['counts']['cross']:>7}"
              f"{sp('units'):>6}{sp('vsCliq'):>9}{sp('vsCross'):>9}{sp('rndCross'):>7}{sp('vsTri'):>8}{unsound}")

    ev = [r for r in results if "cfgs" in r]
    if ev:
        def gm(xs):
            xs = [x for x in xs if x and x > 0]
            return statistics.geometric_mean(xs) if xs else float("nan")
        print("\n" + "=" * 66)
        print(f"instances evaluated: {len(ev)}   (speedup_no_gpu = plain / augmented)")
        for name in CONFIGS:
            xs = [r["cfgs"][name]["speedup"] for r in ev]
            print(f"  {name:<9}: geomean {gm(xs):.2f}  median {statistics.median([x for x in xs if x] or [0]):.2f}  "
                  f"improved {sum(1 for x in xs if x and x > 1)}/{len(ev)}")
        unsound = sum(1 for r in ev if not all(v["sound"] for v in r["cfgs"].values()))
        print(f"  unsound  : {unsound}  (MUST be 0)")
        print("=" * 66)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "K": args.K, "Kx": args.Kx, "Kt": args.Kt,
                   "warmup": args.warmup, "results": results}, open(args.output, "w"), indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

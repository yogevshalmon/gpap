#!/usr/bin/env python3
"""Search for a GLOBAL variable-selection heuristic for GPU clause discovery (no-GPU potential).

On harder instances (solve > a few seconds), test which variable-scoring rule — used to pick the
variables for binary (anchor×all) and triple (clique) discovery — gives the best AVERAGE solve
speedup across instances. The failed-literal UNITS (probe) are heuristic-independent and always
included; we measure speedup_no_gpu = plain_solve / augmented_solve (GPU discovery time ignored:
the question is whether the right targeting helps at all, given a fast enough GPU).

Variable-scoring heuristics:
    vsids   warm-start CaDiCaL branch-decision counts (activity)
    degree  static occurrence count (high)
    jw      Jeroslow-Wang: Σ_clauses∋v 2^-|clause|  (favors vars in short clauses)
    rare    low occurrence degree (peripheral variables)
    random  uniform

Every injected clause is RUP, so the verdict must match the plain solve.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/inprocess_heuristics.py --insts /tmp/hard_insts.json \\
        --device cuda --Kx 32 --Kt 48 --warmup 10000 --max-solve-s 60 --output .experiments/heur.json
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

HEURS = ["vsids", "degree", "jw", "rare", "random"]


def static_scores(cnf):
    """Per-variable degree and Jeroslow-Wang scores from the clause set."""
    deg, jw = {}, {}
    for c in cnf.clauses:
        if not c:
            continue
        w = 2.0 ** (-len(c))
        for lit in c:
            v = abs(lit)
            deg[v] = deg.get(v, 0) + 1
            jw[v] = jw.get(v, 0.0) + w
    return deg, jw


def select(H, K, dec, deg, jw, N, rng):
    """Top-K variables (1-based) under heuristic H."""
    if H == "vsids":
        return sorted(range(1, N + 1), key=lambda v: (dec.get(v, 0), deg.get(v, 0)), reverse=True)[:K]
    if H == "degree":
        return sorted(range(1, N + 1), key=lambda v: deg.get(v, 0), reverse=True)[:K]
    if H == "jw":
        return sorted(range(1, N + 1), key=lambda v: jw.get(v, 0.0), reverse=True)[:K]
    if H == "rare":
        return sorted(range(1, N + 1), key=lambda v: deg.get(v, 0))[:K]      # low degree first
    vs = list(range(1, N + 1)); rng.shuffle(vs); return vs[:K]


def eval_instance(rec, device, Kx, Kt, warmup, max_solve_s, seed, do_bin, do_tri):
    path = rec["path"]
    cnf = CNF(from_file=path)
    orig = [list(c) for c in cnf.clauses if c]
    N = cnf.nv

    res0, solve0, cf0 = timed_solve(orig, N, max_solve_s)
    if res0 == "UNK":
        return {"name": os.path.basename(path)[:34], "skip": "plain UNK"}

    dec, _a, _deg0, _ = score_variables_by_decisions(path, warmup_conflicts=warmup, observe_cap=min(N, 5000))
    deg, jw = static_scores(cnf)
    rng = random.Random(seed)

    inp = GPUInprocessor.from_cnf_file(path, device)
    units, _ = inp.probe_failed_literals()
    ru, su, _ = timed_solve(orig + list(units), N, max_solve_s)
    units_sp = (solve0 / su) if (su > 0 and ru == res0) else None

    out = {"name": os.path.basename(path)[:34], "family": rec.get("family"), "N": N, "M": len(orig),
           "result": res0, "plain_ms": solve0, "n_units": len(units), "units_speedup": units_sp,
           "bin": {}, "tri": {}}

    for H in HEURS:
        if do_bin:
            anchors = [v - 1 for v in select(H, Kx, dec, deg, jw, N, rng)]
            inp.reset_emitted()
            b = inp.discover_binaries_anchored(anchors)[0]
            rb, sb, _ = timed_solve(orig + list(units) + b, N, max_solve_s)
            out["bin"][H] = {"n": len(b), "speedup": (solve0 / sb) if (sb > 0 and rb == res0) else None,
                             "sound": rb in (res0, "UNK")}
        if do_tri:
            tv = [v - 1 for v in select(H, Kt, dec, deg, jw, N, rng)]
            inp.reset_emitted()
            t = inp.discover_ternaries(tv)[0]
            rt, st, _ = timed_solve(orig + list(units) + t, N, max_solve_s)
            out["tri"][H] = {"n": len(t), "speedup": (solve0 / st) if (st > 0 and rt == res0) else None,
                             "sound": rt in (res0, "UNK")}
    del inp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _gm(xs):
    xs = [x for x in xs if x and x > 0]
    return statistics.geometric_mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--Kx", type=int, default=32, help="anchors for binary (anchor×all) discovery")
    ap.add_argument("--Kt", type=int, default=48, help="top-Kt vars for triple (clique) discovery")
    ap.add_argument("--warmup", type=int, default=10000)
    ap.add_argument("--max-solve-s", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-binaries", action="store_true")
    ap.add_argument("--no-triples", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    insts = json.load(open(args.insts))[:args.limit]
    do_bin, do_tri = not args.no_binaries, not args.no_triples
    print(f"Device {device} | {len(insts)} instances | Kx={args.Kx} Kt={args.Kt} warmup={args.warmup} "
          f"| bin={do_bin} tri={do_tri} | speedup = plain/augmented (GPU ignored)\n")
    hdr = f"{'instance':<36}{'plain':>7}{'×U':>6}" + "".join(f"{('b:'+H)[:7]:>8}" for H in HEURS) \
          + "".join(f"{('t:'+H)[:7]:>8}" for H in HEURS)
    print(hdr); print("-" * len(hdr))

    results = []
    for rec in insts:
        try:
            r = eval_instance(rec, device, args.Kx, args.Kt, args.warmup, args.max_solve_s,
                              args.seed, do_bin, do_tri)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(rec['path'])[:36]:<36}  ERROR: {e}"); continue
        results.append(r)
        if "skip" in r:
            print(f"{r['name']:<36}  skip ({r['skip']})"); continue
        def cell(d, H):
            sp = d.get(H, {}).get("speedup")
            return f"{sp:.2f}" if sp else "-"
        u = f"{r['units_speedup']:.2f}" if r["units_speedup"] else "-"
        bcols = "".join(f"{cell(r['bin'], H):>8}" for H in HEURS)
        tcols = "".join(f"{cell(r['tri'], H):>8}" for H in HEURS)
        print(f"{r['name']:<36}{r['plain_ms']:>7.0f}{u:>6}{bcols}{tcols}")

    ev = [r for r in results if "bin" in r]
    if ev:
        print("\n" + "=" * 70)
        print(f"instances: {len(ev)}   GLOBAL geomean speedup_no_gpu per heuristic:")
        print(f"  units-only : {_gm([r['units_speedup'] for r in ev]):.3f}  "
              f"(improved {sum(1 for r in ev if (r['units_speedup'] or 0)>1)}/{len(ev)})")
        if do_bin:
            print("  binaries (anchor×all):")
            for H in HEURS:
                xs = [r['bin'].get(H, {}).get('speedup') for r in ev]
                print(f"    {H:<8}: geomean {_gm(xs):.3f}  improved {sum(1 for x in xs if x and x>1)}/{len(ev)}")
        if do_tri:
            print("  triples (clique):")
            for H in HEURS:
                xs = [r['tri'].get(H, {}).get('speedup') for r in ev]
                print(f"    {H:<8}: geomean {_gm(xs):.3f}  improved {sum(1 for x in xs if x and x>1)}/{len(ev)}")
        # oracle: best of {units, all bin, all tri} per instance
        def best(r):
            cand = [r["units_speedup"]] + [v.get("speedup") for v in r["bin"].values()] \
                   + [v.get("speedup") for v in r["tri"].values()]
            cand = [c for c in cand if c]
            return max(cand) if cand else None
        orc = [best(r) for r in ev]
        print(f"  ORACLE best-per-instance: geomean {_gm(orc):.3f}  median {statistics.median([o for o in orc if o] or [0]):.2f}")
        unsnd = sum(1 for r in ev for v in list(r['bin'].values())+list(r['tri'].values()) if not v.get('sound', True))
        print(f"  unsound configs: {unsnd} (MUST be 0)")
        print("=" * 70)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "Kx": args.Kx, "Kt": args.Kt, "warmup": args.warmup,
                   "results": results}, open(args.output, "w"), indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

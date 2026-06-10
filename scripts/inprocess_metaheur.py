#!/usr/bin/env python3
"""One combined meta-heuristic: inject ALL learned clauses (units + binaries + triples) under the
peripheral/low-degree ("rare") rule, vs the single clause-types — does combining win, and at what
budget (bloat control)?

Per instance, speedup_no_gpu = plain_solve / augmented_solve (GPU discovery ignored). Configs:
    plain        no injection (baseline / fallback)
    U            units only (failed-literal probing)
    U+B          units + rare-anchored binaries            (= the prior "rare-bin")
    U+B+T        units + rare binaries + rare-clique triples  (combine ALL learned clauses)
    U+B+T@<cap>  same, but shortest-first capped to <cap> clauses (units always kept)

All clauses are RUP → verdicts must match plain.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/inprocess_metaheur.py --insts /tmp/struct_insts.json \\
        --device cuda --Kx 24 --Kt 44 --cap 5000 --max-solve-s 30 --output .experiments/metaheur.json
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
from scripts.inprocess_heuristics import static_scores, select
from scripts.inprocess_suite import timed_solve


def eval_instance(rec, device, Kx, Kt, cap, max_solve_s, seed):
    path = rec["path"]
    cnf = CNF(from_file=path)
    orig = [list(c) for c in cnf.clauses if c]
    N = cnf.nv

    res0, solve0, cf0 = timed_solve(orig, N, max_solve_s)
    if res0 == "UNK":
        return {"name": os.path.basename(path)[:40], "skip": "plain UNK"}

    deg, jw = static_scores(cnf)
    rng = random.Random(seed)
    rare_anchors = [v - 1 for v in select("rare", Kx, {}, deg, jw, N, rng)]
    rare_clique = [v - 1 for v in select("rare", Kt, {}, deg, jw, N, rng)]

    if device.type == "cuda":
        torch.cuda.empty_cache()
    inp = GPUInprocessor.from_cnf_file(path, device)
    units, _ = inp.probe_failed_literals()
    inp.reset_emitted(); bins, _ = inp.discover_binaries_anchored(rare_anchors)
    inp.reset_emitted(); tris, _ = inp.discover_ternaries(rare_clique)
    del inp
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Bound injection: huge binary counts (e.g. schup 612k) bloat the formula → slow solves
    # and GPU pressure with no benefit (the useful budget is ≤ a few thousand).
    bins = bins[:6000]
    tris = tris[:3000]
    u = list(units)
    configs = {
        "plain": orig,
        "U": orig + u,                              # units only (prior winner)
        "U+B.5k": orig + u + bins[:500],            # units + small binary budget
        "U+B2k": orig + u + bins[:2000],
        "U+B6k": orig + u + bins,                   # units + up to 6k rare-anchored binaries
        "U+B+T": orig + u + bins + tris,            # combine ALL learned clauses (bounded)
    }
    out = {"name": os.path.basename(path)[:40], "family": rec.get("family"), "N": N, "M": len(orig),
           "result": res0, "plain_ms": solve0,
           "n": {"units": len(units), "bins": len(bins), "tris": len(tris)}, "cfg": {}}
    for name, cls in configs.items():
        r, s, cf = timed_solve(cls, N, max_solve_s)
        out["cfg"][name] = {"speedup": (solve0 / s) if (s > 0 and r == res0) else None,
                            "added": len(cls) - len(orig), "sound": r in (res0, "UNK")}
    return out


CFG_NAMES = ["plain", "U", "U+B.5k", "U+B2k", "U+B6k", "U+B+T"]
COMBO = ["U+B.5k", "U+B2k", "U+B6k", "U+B+T"]   # configs that add binaries/triples on top of units


def _gm(xs):
    xs = [x for x in xs if x and x > 0]
    return statistics.geometric_mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--Kx", type=int, default=24)
    ap.add_argument("--Kt", type=int, default=44)
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--max-solve-s", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    insts = json.load(open(args.insts))[:args.limit]
    names = CFG_NAMES
    print(f"Device {device} | {len(insts)} instances | Kx={args.Kx} Kt={args.Kt} "
          f"| speedup = plain/augmented (no-GPU)\n")
    hdr = f"{'instance':<38}{'plain':>7}{'units':>6}{'bins':>7}{'tris':>7}" + "".join(f"{n:>11}" for n in names[1:])
    print(hdr); print("-" * len(hdr))

    results = []
    for rec in insts:
        try:
            r = eval_instance(rec, device, args.Kx, args.Kt, args.cap, args.max_solve_s, args.seed)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(rec['path'])[:38]:<38}  ERROR: {e}"); continue
        results.append(r)
        if "skip" in r:
            print(f"{r['name']:<38}  skip ({r['skip']})"); continue
        def c(n):
            sp = r["cfg"].get(n, {}).get("speedup")
            return f"{sp:.2f}" if sp else "-"
        flag = "" if all(v["sound"] for v in r["cfg"].values()) else " !UNSOUND"
        print(f"{r['name']:<38}{r['plain_ms']:>7.0f}{r['n']['units']:>6}{r['n']['bins']:>7}{r['n']['tris']:>7}"
              + "".join(f"{c(n):>11}" for n in names[1:]) + flag)

    ev = [r for r in results if "cfg" in r]
    if ev:
        print("\n" + "=" * 64)
        print(f"instances: {len(ev)}   geomean speedup vs plain (no-GPU):")
        for n in names:
            xs = [r["cfg"][n]["speedup"] for r in ev]
            print(f"  {n:<12}: geomean {_gm(xs):.3f}  median "
                  f"{statistics.median([x for x in xs if x] or [0]):.2f}  "
                  f"improved {sum(1 for x in xs if x and x>1)}/{len(ev)}")
        # how often does ANY binary/triple-adding config beat units-only?
        wins_combine = sum(1 for r in ev
                           if max((r["cfg"][n]["speedup"] or 0) for n in COMBO)
                           > (r["cfg"]["U"]["speedup"] or 0) + 1e-9)
        print(f"  best-combine beats units-only on {wins_combine}/{len(ev)}")
        unsound = sum(1 for r in ev for v in r["cfg"].values() if not v["sound"])
        print(f"  unsound: {unsound} (MUST be 0)")
        print("=" * 64)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "Kx": args.Kx, "Kt": args.Kt, "cap": args.cap,
                   "results": results}, open(args.output, "w"), indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

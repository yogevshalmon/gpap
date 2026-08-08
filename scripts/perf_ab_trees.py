#!/usr/bin/env python3
"""Throughput A/B between TWO CHECKOUTS — "did my refactor cost performance?".

Complements the no-change gate, which answers the different question "are the models
bit-identical?" — pinned K, one config, one seed, compare model COUNTS (docs/BENCHMARKING.md
§3). Use both: identical counts prove correctness, this proves you did not pay for them.

    git worktree add /tmp/base HEAD
    export GPAP_LIB=$(readlink -f build/libgpap.so)      # ONE lib for both arms
    scripts/perf_ab_trees.py --base /tmp/base --head . --sk-glob 'benchmarks/sampling/3*.cnf'

Arms select the package by cwd + PYTHONPATH, never `pip install`, so the two trees cannot shadow
each other. Point GPAP_LIB at a single build when no C++ changed — otherwise you are also
A/B-ing two compilations.

Method, and it is not optional (docs/BENCHMARKING.md §1b): SM clocks on this box swing 615-1710
MHz *within* a run, so two sequential sweeps cannot resolve anything smaller than ~2x. Both arms
therefore run back-to-back in the same session with the arm ORDER ALTERNATED per rep, and each
arm is scored by its best rep.

`--k-cap` is pinned identically on both arms so this measures the code change alone, not the
separate VRAM-proportional-cap change (whose effect is a different question, already answered).

Cells deliberately span all three code paths the refactor touched:
  * sk_ CNFs        - small M, the fused BOOST is on  (batch_size == K_fused)
  * circuit unrolls - large M, boost inactive         (batch_size == K)
  * an --n-cov cell - the COVERAGE path, where `smp.K = min(...)` became `pin_batch(...)`
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

CELLS = [
    # (label, extra gpap-bench argv)
    ("sk", ["--glob", "SKGLOB", "--n-time", "20000", "--time-limit", "20"]),
    ("sk_cov", ["--glob", "SKGLOB", "--n-time", "20000", "--time-limit", "20", "--n-cov", "2000"]),
    ("b03_c25", ["--glob", "benchmarks/circuits/bench/seq/b03.bench", "--cycles", "25",
                 "--encoder", "tseitin", "--target", "rare", "--n-time", "20000", "--time-limit", "20"]),
    ("b12_c25", ["--glob", "benchmarks/circuits/bench/seq/b12.bench", "--cycles", "25",
                 "--encoder", "tseitin", "--target", "rare", "--n-time", "20000", "--time-limit", "20"]),
]


def run(tree, cell_argv, out, k_cap, seed, py):
    argv = [py, "-m", "gpap.benchmark"] + cell_argv + [
        "--k-cap", str(k_cap), "--seed", str(seed), "--output", out]
    env = dict(os.environ, PYTHONPATH=tree, GPAP_LIB=os.environ["GPAP_LIB"])
    p = subprocess.run(argv, cwd=tree, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        return None, p.stderr[-800:]
    try:
        rows = json.load(open(out))["results"]
    except Exception as e:                                   # noqa: BLE001
        return None, f"unreadable results: {e}"
    return {r["name"]: r for r in rows}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--sk-glob", required=True)
    ap.add_argument("--k-cap", type=int, default=49152)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp()
    got = {}          # (cell, instance) -> {arm: [sps,...]}
    meta = {}
    for rep in range(a.reps):
        # ALTERNATE: rep 0 runs base first, rep 1 head first, ...
        arms = [("base", a.base), ("head", a.head)]
        if rep % 2:
            arms.reverse()
        for cell, argv in CELLS:
            argv = [x.replace("SKGLOB", a.sk_glob) for x in argv]
            for arm, tree in arms:
                out = os.path.join(tmp, f"{cell}_{arm}_{rep}.json")
                rows, err = run(tree, argv, out, a.k_cap, a.seed, a.python)
                if rows is None:
                    print(f"  [FAIL] {cell:<9} {arm:<5} rep{rep}: {err}", file=sys.stderr)
                    continue
                for name, r in rows.items():
                    key = (cell, name)
                    got.setdefault(key, {}).setdefault(arm, []).append(r.get("gpap_sps") or 0.0)
                    e = r.get("engine") or {}
                    meta.setdefault(key, {})[arm] = dict(
                        models=r.get("n_models"), K=e.get("K"), Kf=e.get("K_fused"),
                        batch=e.get("batch"), mode=e.get("mode"),
                        distinct=r.get("n_distinct"))
                print(f"  ran {cell:<9} {arm:<5} rep{rep}")

    print(f"\n{'cell':<9}{'instance':<17}{'base':>13}{'head':>13}{'head/base':>11}"
          f"   {'batch b/h':>17}  models b/h")
    ratios = []
    for key in sorted(got):
        cell, name = key
        arms = got[key]
        if "base" not in arms or "head" not in arms:
            continue
        b, h = max(arms["base"]), max(arms["head"])
        if b <= 0 or h <= 0:
            print(f"{cell:<9}{name:<17}{b:>13,.0f}{h:>13,.0f}{'n/a':>11}")
            continue
        ratios.append(h / b)
        mb, mh = meta[key]["base"], meta[key]["head"]
        print(f"{cell:<9}{name:<17}{b:>13,.0f}{h:>13,.0f}{h / b:>11.3f}"
              f"   {str(mb.get('Kf')) + '/' + str(mh.get('batch')):>17}"
              f"  {mb.get('models')}/{mh.get('models')}")
    if ratios:
        g = statistics.geometric_mean(ratios)
        print(f"\nGEOMEAN head/base = {g:.4f}   (n={len(ratios)}; "
              f"min {min(ratios):.3f}, max {max(ratios):.3f})")
        print("1.00 = no change. Below ~0.95 on several cells would be a real regression;")
        print("single-cell scatter of +-10% is draw noise on low-success instances.")
    if a.out:
        json.dump({"ratios": ratios, "meta": {f"{c}|{n}": m for (c, n), m in meta.items()}},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

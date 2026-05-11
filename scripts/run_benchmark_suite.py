#!/usr/bin/env python3
"""Run the GPAP BCP benchmark suite and optionally compare against a baseline.

For each instance in the manifest:
  - Generates K random assumption sets of size L (from build_benchmark_suite.py config)
  - Runs K=1 baseline to measure GPU parallelism speedup
  - Runs full K batches and records timing + BCP stats
  - Runs PySAT serially on min(K, pysat_k_cap) assumptions and extrapolates to K
    to compute the real-world speedup vs a serial BCP baseline

Two speedup metrics reported:
  speedup_vs_pysat  — extrapolated PySAT(K) / GPAP(K): the headline real-world speedup
  speedup_vs_k1     — (K × GPAP(K=1)) / GPAP(K): GPU parallelism efficiency

Usage:
    # Profile and save results
    PYTHONPATH=. .venv/bin/python scripts/run_benchmark_suite.py \\
        --suite benchmarks/suite.json \\
        --output benchmarks/results_20260511.json

    # Regression check against a previous run
    PYTHONPATH=. .venv/bin/python scripts/run_benchmark_suite.py \\
        --suite benchmarks/suite.json \\
        --baseline benchmarks/results_20260101.json \\
        --threshold 1.10
"""
import argparse
import datetime
import json
import os
import sys

import torch

sys.path.insert(0, ".")
from scripts.profile_engine import (
    compare_results,
    generate_random_assumptions,
    profile_external,
    profile_gpap,
)


def _run_instance(
    entry: dict,
    device: torch.device,
    warmup: int,
    runs: int,
    seed: int,
    no_external: bool,
    pysat_k_cap: int,
    verify: bool,
    override_l: int | None = None,
    max_rounds: int | None = None,
) -> dict:
    path = entry["path"]
    K = entry["K"]
    L = override_l if override_l is not None else entry["L"]
    N = entry["N"]

    assumptions = generate_random_assumptions(N, K, L, seed)

    # K=1 baseline for GPU parallelism speedup
    k1 = profile_gpap(path, assumptions[:1], device, warmup, max(runs, 3), max_rounds=max_rounds)
    result = profile_gpap(path, assumptions, device, warmup, runs, max_rounds=max_rounds)

    gpu_speedup = (k1["mean_ms"] * K) / result["mean_ms"] if result["mean_ms"] > 0 else 0.0

    out = {
        **entry,
        "L":                L,   # actual L used (may differ from manifest if --override-l set)
        "mean_ms":          round(result["mean_ms"], 3),
        "std_ms":           round(result["std_ms"], 3),
        "bcp_rounds_mean":  round(result["bcp_rounds_mean"], 2),
        "conflict_rate":    round(result["conflict_rate"], 4),
        "avg_propagations": round(result["avg_propagations"], 2),
        "speedup_vs_k1":    round(gpu_speedup, 2),
        "k1_mean_ms":       round(k1["mean_ms"], 3),
        "completion_rate":  round(result["completion_rate"], 4),
    }

    if not no_external:
        # Run PySAT on min(K, cap) batches, extrapolate to K for apples-to-apples comparison.
        # PySAT is serial so time scales linearly — extrapolation is exact.
        K_ext = min(K, pysat_k_cap)
        ext = profile_external(path, assumptions[:K_ext], min(runs, 3))
        pysat_per_batch_ms = ext["mean_ms"] / K_ext
        pysat_extrap_ms = pysat_per_batch_ms * K
        out["pysat_per_batch_ms"] = round(pysat_per_batch_ms, 3)
        out["pysat_extrap_ms"]    = round(pysat_extrap_ms, 3)
        out["pysat_k_ext"]        = K_ext
        out["speedup_vs_pysat"]   = round(pysat_extrap_ms / result["mean_ms"], 2) if result["mean_ms"] > 0 else 0.0

        if verify:
            cmp = compare_results(result["conflict"][:K_ext], ext["conflict"])
            out["agreement_rate"] = round(cmp["agreement_rate"], 4)
            out["mismatches"] = cmp["mismatches"]

    return out


def _print_table(results: list[dict]) -> None:
    has_pysat    = any("speedup_vs_pysat" in r for r in results)
    has_verify   = any("agreement_rate"   in r for r in results)
    has_capped   = any(r.get("completion_rate", 1.0) < 1.0 for r in results)

    cols = (f"{'Instance':<36} {'N':>7} {'M':>8} {'K':>5} {'L':>3} "
            f"{'ms':>8} {'±':>6}")
    if has_pysat:
        cols += f" {'vs_pysat':>10}"
    cols += f" {'vs_k1':>8} {'rounds':>7} {'conflict':>9}"
    if has_capped:
        cols += f" {'done%':>6}"
    if has_verify:
        cols += f"  {'agree':>6}"

    print(cols)
    print("-" * (len(cols) + 2))
    for r in results:
        name = os.path.basename(r["path"])
        if len(name) > 36:
            name = "…" + name[-35:]
        line = (f"{name:<36} {r['N']:>7} {r['M']:>8} {r['K']:>5} {r['L']:>3} "
                f"{r['mean_ms']:>8.2f} {r['std_ms']:>6.2f}")
        if has_pysat:
            vs_pysat = r.get("speedup_vs_pysat", float("nan"))
            line += f" {vs_pysat:>9.1f}x"
        line += (f" {r['speedup_vs_k1']:>7.1f}x"
                 f" {r['bcp_rounds_mean']:>7.1f}"
                 f" {r['conflict_rate']:>8.1%}")
        if has_capped:
            done = r.get("completion_rate", 1.0)
            line += f" {done:>5.1%}"
        if has_verify:
            agree = r.get("agreement_rate", float("nan"))
            line += f"  {agree:>5.1%}"
        print(line)


def _check_baseline(
    results: list[dict], baseline_path: str, threshold: float
) -> list[dict]:
    with open(baseline_path) as f:
        data = json.load(f)
    baseline_list = data if isinstance(data, list) else data.get("instances", [])
    baseline = {r["path"]: r for r in baseline_list if "path" in r and "error" not in r}

    regressions = []
    for r in results:
        b = baseline.get(r.get("path", ""))
        if b is None or b.get("mean_ms", 0) <= 0:
            continue
        ratio = r["mean_ms"] / b["mean_ms"]
        if ratio > threshold:
            regressions.append({
                "path":   r["path"],
                "ratio":  round(ratio, 3),
                "old_ms": b["mean_ms"],
                "new_ms": r["mean_ms"],
            })
    return regressions


def main():
    ap = argparse.ArgumentParser(description="Run GPAP BCP benchmark suite")
    ap.add_argument("--suite", default="benchmarks/suite.json",
                    help="Suite manifest from build_benchmark_suite.py (default: benchmarks/suite.json)")
    ap.add_argument("--output", default=None,
                    help="Results JSON path (default: benchmarks/results_YYYYMMDD.json)")
    ap.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
    ap.add_argument("--warmup", type=int, default=3, help="Warmup runs per instance (default: 3)")
    ap.add_argument("--runs",   type=int, default=10, help="Timed runs per instance (default: 10)")
    ap.add_argument("--seed",   type=int, default=42, help="RNG seed for assumption generation")
    ap.add_argument("--no-external", action="store_true",
                    help="Skip PySAT timing entirely (faster, no speedup_vs_pysat metric)")
    ap.add_argument("--pysat-k-cap", type=int, default=50,
                    help="Max K batches to run through PySAT; extrapolated to full K (default: 50)")
    ap.add_argument("--verify", action="store_true",
                    help="Also check GPAP/PySAT conflict agreement (requires external enabled)")
    ap.add_argument("--baseline", default=None,
                    help="Previous results JSON to compare against for regressions")
    ap.add_argument("--threshold", type=float, default=1.10,
                    help="Regression threshold: flag if new/old > this (default: 1.10)")
    ap.add_argument("--override-l", type=int, default=None,
                    help="Override L (assumptions per batch) for all instances, ignoring manifest")
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="Cap BCP rounds per batch; incomplete batches are counted as non-finished "
                         "(completion_rate < 1). None = run to fixpoint (default)")
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with open(args.suite) as f:
        suite = json.load(f)

    if args.output is None:
        args.output = f"benchmarks/results_{datetime.date.today():%Y%m%d}.json"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    l_note = f"  L={args.override_l} (override)" if args.override_l is not None else ""
    r_note = f"  max_rounds={args.max_rounds}" if args.max_rounds is not None else ""
    print(f"Running {len(suite)} instances  device={device}  "
          f"warmup={args.warmup}  runs={args.runs}  seed={args.seed}{l_note}{r_note}")
    if not args.no_external:
        print(f"  PySAT comparison enabled  pysat_k_cap={args.pysat_k_cap}"
              + ("  +agreement check" if args.verify else ""))
    print()

    results = []
    for i, entry in enumerate(suite, 1):
        name = os.path.basename(entry["path"])
        actual_l = args.override_l if args.override_l is not None else entry['L']
        print(f"[{i:3d}/{len(suite)}] {name}  "
              f"N={entry['N']} M={entry['M']} K={entry['K']} L={actual_l} ...",
              end=" ", flush=True)
        try:
            r = _run_instance(entry, device, args.warmup, args.runs,
                              args.seed, args.no_external, args.pysat_k_cap, args.verify,
                              override_l=args.override_l, max_rounds=args.max_rounds)
            results.append(r)
            status = f"{r['mean_ms']:.2f} ms"
            if "speedup_vs_pysat" in r:
                status += f"  vs_pysat={r['speedup_vs_pysat']:.1f}x"
            status += f"  vs_k1={r['speedup_vs_k1']:.1f}x"
            done = r.get("completion_rate", 1.0)
            if done < 1.0:
                status += f"  done={done:.1%}"
            if "agreement_rate" in r:
                status += f"  agree={r['agreement_rate']:.1%}"
            print(status)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({**entry, "error": str(e)})

    print()
    good = [r for r in results if "error" not in r]
    _print_table(good)

    output_data = {
        "date":     datetime.datetime.now().isoformat(),
        "device":   str(device),
        "warmup":   args.warmup,
        "runs":     args.runs,
        "seed":     args.seed,
        "instances": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved → {args.output}")

    if args.baseline:
        regressions = _check_baseline(good, args.baseline, args.threshold)
        if regressions:
            print(f"\nREGRESSIONS ({len(regressions)} instances >{args.threshold:.0%} slower):")
            for reg in regressions:
                print(f"  {os.path.basename(reg['path']):<40}  "
                      f"{reg['old_ms']:.2f} ms → {reg['new_ms']:.2f} ms  "
                      f"({reg['ratio']:.2f}x)")
            sys.exit(1)
        else:
            print(f"\nNo regressions detected  "
                  f"(threshold={args.threshold:.0%}, checked {len(good)} instances)")


if __name__ == "__main__":
    main()

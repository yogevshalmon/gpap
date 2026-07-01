#!/usr/bin/env python3
"""Benchmark the GPU BCP-completion sampler (gpap.solver.sampler.GPUSampler) against CMSGen.

GPAP runs CMSGen's UniformLikeWitness algorithm (random var + random polarity + unit propagation)
massively in parallel on the GPU. This harness compares them with the SAT-sampling literature's
own metrics (Golia et al., FMCAD'21):

  THROUGHPUT  — samples/sec (wall time to produce --n-time samples, including setup).
  QUALITY     — 2-wise coverage %: for each pair of sampling-set vars, the fraction of pairs whose
                all four (±,±) combinations appear across the samples (the combinatorial-testing
                quality metric CMSGen reports). Higher = better solution-space spread.
  SOUNDNESS   — GPAP: GPU check (every model's clauses all have sat_count>0) + optional CPU verify.

GPAP uses unit-propagation simplification (--simplify, on by default): strictly model-preserving
(unlike the VE/BCE that CMSGen disables), validated in tests/test_sampler.py — it shrinks M so a
much larger K fits, which is where the GPU throughput win comes from.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/sample_benchmark.py \
      --glob 'benchmarks/sampling/*.cnf' --device cuda \
      --cmsgen /path/to/cmsgen --n-time 50000 --n-cov 5000 --output .experiments/sample.json
"""
import argparse, glob, json, os, subprocess, sys, time

import numpy as np
import torch
sys.path.insert(0, ".")
from pysat.formula import CNF
from gpap.solver.sampler import GPUSampler, parse_independent_support


def two_wise_coverage(models, support):
    """% of sampling-var pairs whose all 4 (±,±) combos appear. support: 1-based vars."""
    S = len(support)
    if S < 2 or not models:
        return 0.0
    idx = {v: i for i, v in enumerate(support)}
    M = np.zeros((len(models), S), dtype=np.int8)
    for r, m in enumerate(models):
        s = set(m)
        for v in support:
            if v in s:
                M[r, idx[v]] = 1
    covered = total = 0
    for i in range(S):
        ci = M[:, i].astype(np.int16) << 1
        for j in range(i + 1, S):
            total += 1
            if np.unique(ci | M[:, j]).size == 4:
                covered += 1
    return 100.0 * covered / total if total else 0.0


# Default config portfolio for the adaptive bandit: the two extremes that cover the niche
# ((random order, B=8) — fast, BCP-determined) and the search-heavy "service" family
# ((structured order, B=1) — completes via the gate DAG). The bandit measures each at full K and
# spends the budget on the winner per formula (dynamic, robust — no probe mis-estimate / commit).
DEFAULT_PORTFOLIO = [(True, 8), (False, 1)]


def gpap_sample(path, device, *, simplify, n_time, n_cov, b, k_cap, max_chunks=8,
                cov_chunks=4, max_rounds=300, lookahead=False, verify_clauses=None,
                cov_k_cap=2048, els=False, random_var="auto", configs="default"):
    """Throughput run timed end-to-end (construction + simplify + sample, to match CMSGen's
    'produce N samples' wall clock) + a small collect run for coverage/soundness. The timed run
    is bounded by `max_chunks` (rate = models/wall is valid regardless of how many it collects —
    avoids spinning on low-success formulas).

    `configs` (default = adaptive portfolio DEFAULT_PORTFOLIO): a throughput bandit that runs the
    best config per chunk at full K, dynamically switching to the winner — cracks the search-heavy
    "service" family (structured order + B=1) AND keeps the niche fast (random order + B=8) in one
    run, more robustly than the small-K `random_var="auto"` probe. Pass configs=None to fall back to
    `random_var` (e.g. "auto"). The dominant config is REUSED for the coverage run (focused).

    The coverage run COLLECTS models to host RAM (one chunk = K models × N ints reconstructed), so
    its K is capped to `cov_k_cap` independently of the (full-K) timing run — otherwise a large
    auto-K on a low-RAM host (12 GB Colab) OOM-kills the session collecting a full chunk."""
    if configs == "default":
        configs = DEFAULT_PORTFOLIO
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    smp = GPUSampler.from_cnf_file(path, device, simplify=simplify, els=els, k_cap=k_cap, seed=5)
    if configs:
        t = smp.sample(n_time, configs=configs, max_decision_rounds=max_rounds,
                       max_chunks=max_chunks, collect=False, lookahead=lookahead)
    else:
        t = smp.sample(n_time, decisions_per_round=b, random_var=random_var,
                       max_decision_rounds=max_rounds, max_chunks=max_chunks, collect=False,
                       lookahead=lookahead)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0                  # includes construct+simplify+bandit (fair vs CMSGen)
    sps = t["n_models"] / wall if wall > 0 else 0.0
    tuned_rv, tuned_b = t["random_var"], t["B"]      # dominant config (or autotuned); reuse for coverage
    if n_cov:
        smp.K = min(smp.K, cov_k_cap)                # bound host RAM from collected/reconstructed models
    c = (smp.sample(n_cov, decisions_per_round=tuned_b, random_var=tuned_rv, max_decision_rounds=max_rounds,
                    max_chunks=cov_chunks, collect=True, verify_clauses=verify_clauses,
                    lookahead=lookahead) if n_cov else {})
    del smp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"samples_per_s": sps, "success_rate": t["success_rate"],
            "support_size": t["support_size"], "gpu_sound": c.get("gpu_sound", True),
            "cpu_sound": c.get("all_valid"), "models": c.get("models", []),
            "n_distinct": c.get("n_distinct"), "tuned_rv": tuned_rv, "tuned_b": tuned_b,
            "config_breakdown": t.get("config_breakdown")}


def cmsgen_sample(path, cmsgen_bin, n, scratch, keep=None):
    """Run CMSGen to produce n samples; return (samp/s, parsed models).

    `keep` caps how many parsed models are retained in memory (the rest are only COUNTED, for the
    rate). This is essential on low-RAM hosts (e.g. a 12 GB Colab): retaining all `n` full models
    is `n × N` Python ints (~8-9 GB at n=50k, N=5k) and OOM-kills the session. None = keep all."""
    out = os.path.join(scratch, "cms_" + os.path.basename(path) + ".txt")
    t0 = time.perf_counter()
    try:
        subprocess.run([cmsgen_bin, "--samples", str(n), "--samplefile", out, path],
                       capture_output=True, timeout=1200)
    except Exception:  # noqa: BLE001
        return 0.0, []
    dt = time.perf_counter() - t0
    models = []
    count = 0
    if os.path.exists(out):
        for line in open(out):
            lits = [int(x) for x in line.split() if x.lstrip("-").isdigit()]
            if lits and lits[-1] == 0:
                lits = lits[:-1]
            if not lits:
                continue
            count += 1                                   # rate uses the full count, not just kept
            if keep is None or len(models) < keep:
                models.append(lits)
        try:
            os.remove(out)                               # the samplefile can be >1 GB; don't accumulate
        except OSError:
            pass
    return (count / dt if dt else 0.0), models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="benchmarks/sampling/*.cnf")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cmsgen", default=None, help="path to the cmsgen binary (enables comparison)")
    ap.add_argument("--n-time", type=int, default=50_000, help="samples for the throughput timing")
    ap.add_argument("--n-cov", type=int, default=5_000, help="samples used for 2-wise coverage")
    ap.add_argument("--b", type=int, default=8,
                    help="decisions per round (8 is the throughput sweet spot: fewer rounds while "
                         "success barely drops; B>=16 over-constrains and collapses success)")
    ap.add_argument("--k-cap", type=int, default=49152)
    ap.add_argument("--no-simplify", dest="simplify", action="store_false", default=True)
    ap.add_argument("--els", action="store_true",
                    help="also do equivalent-literal substitution (removes vars: faster propagate + "
                         "bigger K). Sound + model-preserving; ~1.5x throughput on the niche")
    ap.add_argument("--also-plain", action="store_true", help="also run GPAP without simplification")
    ap.add_argument("--verify-cpu", action="store_true", help="also CPU-verify a few models vs clauses")
    ap.add_argument("--scratch", default="/tmp")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    files = sorted(glob.glob(args.glob))
    print(f"device={device} | simplify={args.simplify} | n_time={args.n_time:,} n_cov={args.n_cov:,} "
          f"| cmsgen={'yes' if args.cmsgen else 'no'}\n{len(files)} benchmarks\n")
    hdr = (f"{'instance':<24}{'N':>6}{'M':>7}{'|sup|':>6}{'GPAP/s':>10}{'succ%':>6}{'2wiseG':>7}"
           f"{'snd':>4}")
    if args.cmsgen:
        hdr += f"{'CMS/s':>9}{'2wiseC':>7}{'GPAP/CMS':>9}"
    print(hdr); print("-" * len(hdr))
    results, ratios = [], []
    for path in files:
        name = os.path.basename(path)
        cnf = CNF(from_file=path)
        sup = parse_independent_support(path)
        sup1 = [v + 1 for v in (sup or range(cnf.nv))]
        vc = [list(c) for c in cnf.clauses] if args.verify_cpu else None
        g = gpap_sample(path, device, simplify=args.simplify, els=args.els, n_time=args.n_time,
                        n_cov=args.n_cov, b=args.b, k_cap=args.k_cap, verify_clauses=vc)
        twg = two_wise_coverage(g["models"], sup1)
        snd = "OK" if g["gpu_sound"] and (g["cpu_sound"] in (None, True)) else "BAD"
        row = {"name": name, "nv": cnf.nv, "M": len(cnf.clauses), "support": len(sup1),
               "gpap_sps": g["samples_per_s"], "succ": g["success_rate"], "twise_gpap": twg,
               "gpu_sound": g["gpu_sound"], "n_distinct": g["n_distinct"]}
        line = (f"{name[:22]:<24}{cnf.nv:>6}{len(cnf.clauses):>7}{len(sup1):>6}"
                f"{g['samples_per_s']:>10,.0f}{100*g['success_rate']:>5.0f}%{twg:>6.1f}%{snd:>4}")
        plain = None
        if args.also_plain:
            plain = gpap_sample(path, device, simplify=False, n_time=args.n_time, n_cov=0,
                                b=args.b, k_cap=args.k_cap)
            row["gpap_plain_sps"] = plain["samples_per_s"]
        if args.cmsgen:
            # single CMSGen run at n_time (fair samp/s); use a subset of its samples for coverage
            crate_t, cmodels = cmsgen_sample(path, args.cmsgen, args.n_time, args.scratch,
                                             keep=args.n_cov)
            twc = two_wise_coverage(cmodels[:args.n_cov], sup1)
            ratio = (g["samples_per_s"] / crate_t) if crate_t else float("inf")
            row.update(cms_sps=crate_t, twise_cms=twc, ratio=ratio)
            ratios.append(ratio)
            line += f"{crate_t:>9,.0f}{twc:>6.1f}%{ratio:>8.2f}x"
        print(line)
        results.append(row)

    if ratios:
        import statistics as st
        gm = st.geometric_mean([r for r in ratios if r and r > 0 and r != float("inf")])
        print(f"\ngeomean GPAP/CMSGen throughput = {gm:.2f}x  "
              f"(GPAP=GPU, CMSGen=single-core CPU; GPAP scales with K -> A100)")
        # quality: mean 2-wise coverage delta
        dq = [r["twise_gpap"] - r["twise_cms"] for r in results if "twise_cms" in r]
        if dq:
            print(f"mean 2-wise coverage: GPAP {np.mean([r['twise_gpap'] for r in results]):.1f}% vs "
                  f"CMSGen {np.mean([r['twise_cms'] for r in results if 'twise_cms' in r]):.1f}% "
                  f"(Δ {np.mean(dq):+.1f}pp)")
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "args": vars(args), "results": results},
                  open(args.output, "w"), indent=1)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

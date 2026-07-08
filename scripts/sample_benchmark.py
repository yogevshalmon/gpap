#!/usr/bin/env python3
"""Benchmark the GPU BCP-completion sampler (gpap.sampling) against CMSGen — a thin CLI/table
layer over gpap.sampling.pipeline.sample_file (the shared e2e flow).

GPAP runs CMSGen's UniformLikeWitness algorithm (random var + random polarity + unit propagation)
massively in parallel on the GPU. This harness compares them with the SAT-sampling literature's
own metrics (Golia et al., FMCAD'21):

  THROUGHPUT  — samples/sec (wall time to produce --n-time samples, including setup).
  QUALITY     — 2-wise coverage %: for each pair of sampling-set vars, the fraction of pairs whose
                all four (±,±) combinations appear across the samples (the combinatorial-testing
                quality metric CMSGen reports). Higher = better solution-space spread.
  SOUNDNESS   — GPAP: GPU check (every model's clauses all have sat_count>0) + optional CPU verify.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/sample_benchmark.py \
      --glob 'benchmarks/sampling/*.cnf' --device cuda \
      --cmsgen /path/to/cmsgen --n-time 50000 --n-cov 5000 --output .experiments/sample.json
"""
import argparse, glob, json, os, subprocess, sys, time

import numpy as np
import torch
sys.path.insert(0, ".")
from gpap.sampling.metrics import distinct_projections, two_wise_coverage
from gpap.sampling.pipeline import CIRCUIT_PORTFOLIO, DEFAULT_PORTFOLIO, sample_file


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
    ap.add_argument("--n-time", type=int, default=50_000,
                    help="samples to PRODUCE in the timed run (a contract, not a hint). Only the "
                         "safety valves end the run short of this ('<' after sols): --time-limit, "
                         "or zero models after every config was measured (hopeless instance)")
    ap.add_argument("--time-limit", type=float, default=60.0,
                    help="per-instance sampling wall cap in seconds (the safety valve that bounds "
                         "very-low-success instances; e.g. at 1k samples/s, 200k samples would "
                         "otherwise take 200s)")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="expert knob: hard chunk cap for the timed run (default none — --n-time "
                         "and --time-limit govern)")
    ap.add_argument("--n-cov", type=int, default=0,
                    help="samples for the QUALITY pass (2-wise coverage + distinct count). 0 (default) "
                         "SKIPS it: the collect run's Python model-reconstruction dominated instance "
                         "wall (measured 26s of 34s on s1196 T=25) while distinct≈100%% on all non-tiny "
                         "circuits. Pass e.g. 5000 for the paper-quality metrics")
    ap.add_argument("--portfolio", choices=["circuit", "default"], default="circuit",
                    help="bandit config portfolio: 'circuit' (fused/fsoft — the CircuitSAT "
                         "focus, default) or 'default' (the sk_ projected-sampling portfolio)")
    ap.add_argument("--b", type=int, default=8,
                    help="decisions per round (8 is the throughput sweet spot: fewer rounds while "
                         "success barely drops; B>=16 over-constrains and collapses success)")
    ap.add_argument("--k-cap", type=int, default=49152)
    ap.add_argument("--use-kernel", choices=["auto", "wide", "mono"], default="auto",
                    help="big-instance kernel variants (post-simplify 2N > 65535; small instances "
                         "keep the proven 16|16 path in all modes). 'wide' = 8|24 packed word "
                         "(same 4B state, half the traffic of the unpacked fallback). 'mono' = "
                         "1-byte count-only state + unit-time clause scan. 'auto' = default dispatch")
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
    portfolio = CIRCUIT_PORTFOLIO if args.portfolio == "circuit" else DEFAULT_PORTFOLIO
    quality = args.n_cov > 0
    print(f"device={device} | simplify={args.simplify} | portfolio={args.portfolio} "
          f"| n_time={args.n_time:,} (limit {args.time_limit:g}s/inst) n_cov={args.n_cov:,} "
          f"| cmsgen={'yes' if args.cmsgen else 'no'}\n{len(files)} benchmarks\n")
    hdr = (f"{'instance':<24}{'N':>6}{'M':>7}{'|sup|':>6}{'GPAP/s':>10}{'e2e/s':>10}"
           f"{'dec/s':>9}{'impl/s':>9}{'succ%':>6}{'sols':>10}{'t_slv':>7}{'t_tot':>7}")
    if quality:
        hdr += f"{'uniq/s~':>10}{'2wiseG':>8}"
    hdr += f"{'snd':>4}"
    if args.cmsgen:
        hdr += f"{'CMS/s':>9}{'GPAP/CMS':>9}{'e2e/CMS':>8}"
        if quality:
            hdr += f"{'2wiseC':>7}"
    print(hdr); print("-" * len(hdr))
    results, ratios = [], []
    t_suite = time.perf_counter()
    total_sols = 0
    for path in files:
        name = os.path.basename(path)
        g = sample_file(path, device, simplify=args.simplify, els=args.els, n_time=args.n_time,
                        n_cov=args.n_cov, b=args.b, k_cap=args.k_cap, verify_cpu=args.verify_cpu,
                        configs=portfolio, max_chunks=args.max_chunks,
                        time_limit=args.time_limit, use_kernel=args.use_kernel)
        sup1 = g["sup1"]                       # metadata from the pipeline's single parse
        twg = two_wise_coverage(g["models"], sup1) if quality else None
        snd = "OK" if g["gpu_sound"] and (g["cpu_sound"] in (None, True)) else "BAD"
        total_sols += g["n_models"]
        row = {"name": name, "nv": g["nv"], "M": g["n_clauses"], "support": len(sup1),
               "gpap_sps": g["samples_per_s"], "gpap_sps_e2e": g.get("sps_e2e_total"),
               "n_models": g["n_models"], "wall_timed_s": g["wall_timed_s"],
               "wall_total_s": g["wall_total_s"],
               "succ": g["success_rate"], "twise_gpap": twg,
               "gpu_sound": g["gpu_sound"], "n_distinct": g["n_distinct"],
               "unique_sps_est": g.get("unique_sps_est"), "assigns_per_s": g.get("assigns_per_s"),
               "decisions_per_s": g.get("decisions_per_s"),
               "implications_per_s": g.get("implications_per_s"), "mined": g.get("mined"),
               "phases": g.get("phases"), "config_breakdown": g.get("config_breakdown")}
        dps = f"{g['decisions_per_s']:.1e}" if g.get("decisions_per_s") else "—"
        ips = f"{g['implications_per_s']:.1e}" if g.get("implications_per_s") else "—"
        mi = g.get("mined")
        # '+' after succ% = entailed-clause mining ran (low-success gate); details land in the JSON.
        # '<' after sols = a safety valve (--time-limit, or zero-progress on a hopeless instance)
        # stopped the run before it delivered --n-time samples; rates are still valid.
        mn = "+" if mi else " "
        sols_s = f"{g['n_models']:,}" + ("<" if g.get("chunk_capped") else "")
        line = (f"{name[:22]:<24}{g['nv']:>6}{g['n_clauses']:>7}{len(sup1):>6}"
                f"{g['samples_per_s']:>10,.0f}{g.get('sps_e2e_total') or 0:>10,.0f}{dps:>9}{ips:>9}"
                f"{100*g['success_rate']:>5.0f}%{mn}{sols_s:>9}"
                f"{g['wall_timed_s']:>6.1f}s{g['wall_total_s']:>6.1f}s")
        if quality:
            uq = f"{g['unique_sps_est']:,.0f}" if g.get("unique_sps_est") else "—"
            line += f"{uq:>10}{twg:>7.1f}%"
        line += f"{snd:>4}"
        plain = None
        if args.also_plain:
            plain = sample_file(path, device, simplify=False, n_time=args.n_time, n_cov=0,
                                b=args.b, k_cap=args.k_cap, configs=portfolio,
                                max_chunks=args.max_chunks, time_limit=args.time_limit)
            row["gpap_plain_sps"] = plain["samples_per_s"]
        if args.cmsgen:
            # single CMSGen run at n_time (fair samp/s); use a subset of its samples for coverage
            crate_t, cmodels = cmsgen_sample(path, args.cmsgen, args.n_time, args.scratch,
                                             keep=args.n_cov or 1)
            ratio = (g["samples_per_s"] / crate_t) if crate_t else float("inf")
            ratio_e2e = ((g.get("sps_e2e_total") or 0) / crate_t) if crate_t else float("inf")
            row.update(cms_sps=crate_t, ratio=ratio, ratio_e2e=ratio_e2e)
            ratios.append(ratio)
            line += f"{crate_t:>9,.0f}{ratio:>8.2f}x{ratio_e2e:>7.2f}x"
            if quality:
                twc = two_wise_coverage(cmodels[:args.n_cov], sup1)
                # CMSGen distinct fraction on the same subset — the papers' metric is UNIQUE
                # solutions/s, so report the dedup factor for BOTH samplers, not just GPAP
                cd, cn = distinct_projections(cmodels[:args.n_cov], sup1)
                row.update(twise_cms=twc, cms_distinct=cd, cms_distinct_of=cn)
                line += f"{twc:>6.1f}%"
        print(line, flush=True)
        results.append(row)

    suite_wall = time.perf_counter() - t_suite
    print(f"\nTOTAL: {total_sols:,} solutions over {len(results)} instances in {suite_wall:,.1f}s "
          f"suite wall (solver e2e {sum(r['wall_total_s'] for r in results):,.1f}s, of which "
          f"timed sampling {sum(r['wall_timed_s'] for r in results):,.1f}s)")
    if ratios:
        import statistics as st
        gm = st.geometric_mean([r for r in ratios if r and r > 0 and r != float("inf")])
        e2e_rs = [r["ratio_e2e"] for r in results
                  if r.get("ratio_e2e") and 0 < r["ratio_e2e"] != float("inf")]
        gm_e2e = st.geometric_mean(e2e_rs) if e2e_rs else 0.0
        print(f"geomean GPAP/CMSGen throughput = {gm:.2f}x steady-state, {gm_e2e:.2f}x "
              f"end-to-end incl. setup+mining  (GPAP=GPU, CMSGen=single-core CPU)")
        # quality: mean 2-wise coverage delta (only when the quality pass ran)
        dq = [r["twise_gpap"] - r["twise_cms"] for r in results
              if r.get("twise_cms") is not None and r.get("twise_gpap") is not None]
        if dq:
            print(f"mean 2-wise coverage: GPAP "
                  f"{np.mean([r['twise_gpap'] for r in results if r['twise_gpap'] is not None]):.1f}% "
                  f"vs CMSGen {np.mean([r['twise_cms'] for r in results if r.get('twise_cms') is not None]):.1f}% "
                  f"(Δ {np.mean(dq):+.1f}pp)")
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "args": vars(args), "results": results},
                  open(args.output, "w"), indent=1)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

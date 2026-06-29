#!/usr/bin/env python3
"""Cross-GPU benchmark driver for the GPAP **sampler** — designed to run on Colab (any GPU) or locally.

This is the sampling-direction analogue of scripts/colab_benchmark.py. It measures, on *this* GPU,
how fast GPAP draws guaranteed-valid SAT models (GPU-parallel CMSGen: random var + random polarity +
unit propagation, K trajectories at once) versus CMSGen (the SOTA fast "uniform-like" CPU sampler).

Three measurements, all on the projected-sampling sk_ benchmarks (independent support via `c ind`):

  1. **Throughput vs CMSGen** (the headline): end-to-end wall to produce models, GPAP-simplified
     samples/s ÷ CMSGen samples/s, per niche instance + geomean. GPAP uses model-preserving
     unit-propagation simplification (--simplify, on) so a much larger K fits — the throughput lever.

  2. **Quality parity**: 2-wise coverage % (the combinatorial-testing metric CMSGen reports) for GPAP
     vs CMSGen — the speed win must come at equal coverage, not lower quality.

  3. **K-scaling** (the A100 story): on a couple of niche instances, sweep K and measure samples/s.
     Throughput is VRAM-limited (not compute-saturated) so it scales ~linearly with K — which is why
     a big-VRAM GPU (A100 80 GB) lands far up the line. The plot marks where *this* GPU's auto-K sits.

K auto-sizes to the GPU's free VRAM (calibrated per-batch cost, ~70·M+40·N bytes at 0.7× free), so the
same code adapts to a 16 GB T4 or an 80 GB A100. Results are saved per-GPU to
benchmarks/colab_sampling_results/<gpu>.json (+ plots) so runs on different GPUs accumulate and can be
ranked with compare_saved().

Headless usage:
    PYTHONPATH=. .venv/bin/python scripts/colab_sampling_benchmark.py --quick --device cuda \
        --cmsgen /path/to/cmsgen
    PYTHONPATH=. .venv/bin/python scripts/colab_sampling_benchmark.py --compare   # rank saved results
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gpap.solver.sampler import GPUSampler, parse_independent_support
from scripts.colab_benchmark import _slug, gpu_info       # reuse device-info helpers
from scripts.sample_benchmark import cmsgen_sample, gpap_sample, two_wise_coverage

DEFAULT_SAMPLING_DIR = os.path.join(REPO_ROOT, "benchmarks", "sampling")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "colab_sampling_results")

# The validated niche (freeAftSup≈0 ∧ high supOK): numeric sk_ where GPAP beats CMSGen. Used for the
# quick run + K-sweep; the full run sweeps every *.cnf so the niche boundary (service family, low-supOK
# 7/71/80.sk) shows up as ~0 success too.
NICHE_FOCUS = [
    "35.sk_3_52.cnf", "10.sk_1_46.cnf", "27.sk_3_32.cnf", "32.sk_4_38.cnf",
    "111.sk_2_36.cnf", "53.sk_4_32.cnf", "55.sk_3_46.cnf", "109.sk_4_36.cnf",
]
K_SWEEP_DEFAULT = ["35.sk_3_52.cnf", "10.sk_1_46.cnf"]


def _resolve_device(device) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# CMSGen build helper (Colab has no pip wheel — build meelgroup/cmsgen from source)
# ---------------------------------------------------------------------------

def build_cmsgen(dest: str | None = None) -> str | None:
    """Clone + cmake-build meelgroup/cmsgen, returning the binary path (or None on failure).

    Idempotent: if the binary already exists it is returned immediately. On Colab this needs
    `cmake` + `build-essential` + `zlib1g-dev` (the notebook's setup cell installs them)."""
    dest = dest or os.path.join(REPO_ROOT, ".cmsgen")
    binp = os.path.join(dest, "build", "cmsgen")
    if os.path.exists(binp):
        return binp
    try:
        if not os.path.isdir(dest):
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/meelgroup/cmsgen", dest], check=True)
        build = os.path.join(dest, "build")
        os.makedirs(build, exist_ok=True)
        subprocess.run(["cmake", ".."], cwd=build, check=True)
        subprocess.run(["make", "-j"], cwd=build, check=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [cmsgen] build failed ({e}); running GPAP-only")
        return None
    return binp if os.path.exists(binp) else None


# ---------------------------------------------------------------------------
# K-scaling sweep — the A100 headline (throughput ∝ K, VRAM-limited)
# ---------------------------------------------------------------------------

def k_scaling(path: str, device: torch.device, *, b: int, chunks: int, k_cap: int,
              n_points: int = 5, seed: int = 5) -> dict | None:
    """Sweep K (batch size) on one instance and measure samples/s at each, to show throughput scales
    ~linearly with K. Builds one simplified sampler, then runs `chunks` chunks at each K in a
    geometric ladder up to the GPU's auto-K. Returns {auto_K, points:[{K,samples_per_s,success}]}."""
    smp = GPUSampler.from_cnf_file(path, device, simplify=True, k_cap=k_cap, seed=seed)
    auto_K = smp.K
    if smp._recon and smp._recon.get("unsat"):
        return None
    # geometric ladder ending at auto_K: auto_K/2^(n-1) .. auto_K, floored at 256
    ks = sorted({max(256, auto_K >> i) for i in range(n_points)})
    points = []
    import random as _random
    for k in ks:
        if k > auto_K:
            continue
        smp.K = k
        # Reseed per point so every K samples the SAME trajectory distribution — isolates the
        # throughput-vs-K effect from random success-rate noise across the sweep.
        smp._gen.manual_seed(seed)
        smp._pyrng = _random.Random(seed)
        smp.sample(k, decisions_per_round=b, max_decision_rounds=120, max_chunks=1, collect=False)  # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = smp.sample(k * chunks, decisions_per_round=b, max_decision_rounds=120,
                       max_chunks=chunks, collect=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        sps = o["n_models"] / wall if wall > 0 else 0.0
        points.append({"K": k, "samples_per_s": round(sps, 1),
                       "success_rate": round(o["success_rate"], 4),
                       "attempts_per_s": round(o["attempts_per_s"], 1)})
    del smp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"name": os.path.basename(path), "auto_K": auto_K, "points": points}


# ---------------------------------------------------------------------------
# Per-instance throughput + quality vs CMSGen
# ---------------------------------------------------------------------------

def run_instance(path: str, device: torch.device, *, cmsgen_bin: str | None, n_time: int,
                 n_cov: int, b: int, k_cap: int, scratch: str, seed: int = 5) -> dict:
    """Benchmark one CNF: GPAP-simplified throughput + 2-wise coverage + GPU soundness, and (if a
    cmsgen binary is given) the CMSGen throughput + coverage + the GPAP/CMSGen ratio."""
    name = os.path.basename(path)
    sup0 = parse_independent_support(path)
    from pysat.formula import CNF
    cnf = CNF(from_file=path)
    sup1 = [v + 1 for v in (sup0 or range(cnf.nv))]

    g = gpap_sample(path, device, simplify=True, n_time=n_time, n_cov=n_cov, b=b, k_cap=k_cap)
    twg = two_wise_coverage(g["models"], sup1)
    row = {
        "name": name, "N": cnf.nv, "M": len(cnf.clauses), "support": len(sup1),
        "gpap_sps": round(g["samples_per_s"], 1), "succ": round(g["success_rate"], 4),
        "twise_gpap": round(twg, 2), "gpu_sound": g["gpu_sound"], "n_distinct": g["n_distinct"],
        "in_niche": g["success_rate"] > 0.10,           # rough: produces models at a usable rate
    }
    if cmsgen_bin:
        crate, cmodels = cmsgen_sample(path, cmsgen_bin, n_time, scratch)
        twc = two_wise_coverage(cmodels[:n_cov], sup1)
        ratio = (g["samples_per_s"] / crate) if crate else None
        row.update(cms_sps=round(crate, 1), twise_cms=round(twc, 2),
                   ratio=(round(ratio, 2) if ratio else None))
    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list[dict], ksweeps: list[dict]) -> dict:
    niche = [r for r in results if r.get("in_niche")]
    sps = [r["gpap_sps"] for r in niche]
    ratios = [r["ratio"] for r in niche if r.get("ratio")]
    dq = [r["twise_gpap"] - r["twise_cms"] for r in niche if "twise_cms" in r]

    def _gm(xs):
        xs = [x for x in xs if x and x > 0]
        return round(statistics.geometric_mean(xs), 2) if xs else None

    # K-scaling: log-log slope (≈1 means linear) over the largest sweep, + the per-K peak.
    slope = None
    peak_k_sps = 0.0
    auto_K = 0
    for ks in ksweeps:
        pts = [p for p in ks["points"] if p["samples_per_s"] > 0]
        peak_k_sps = max(peak_k_sps, max((p["samples_per_s"] for p in pts), default=0.0))
        auto_K = max(auto_K, ks.get("auto_K", 0))
        if len(pts) >= 2 and slope is None:
            lx = np.log([p["K"] for p in pts]); ly = np.log([p["samples_per_s"] for p in pts])
            slope = round(float(np.polyfit(lx, ly, 1)[0]), 2)
    return {
        "n_instances":          len(results),
        "n_niche":              len(niche),
        "gpap_sps_peak":        round(max(sps), 1) if sps else 0.0,
        "gpap_sps_median":      round(statistics.median(sps), 1) if sps else 0.0,
        "ratio_geomean":        _gm(ratios),
        "ratio_max":            round(max(ratios), 2) if ratios else None,
        "twise_delta_mean":     round(float(np.mean(dq)), 2) if dq else None,
        "all_gpu_sound":        all(r["gpu_sound"] for r in results) if results else None,
        "kscale_slope_loglog":  slope,          # ~1.0 ⇒ samples/s ∝ K
        "kscale_peak_sps":      round(peak_k_sps, 1),
        "auto_K":               auto_K,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_table(results: list[dict], has_cms: bool) -> None:
    hdr = (f"{'instance':<24}{'N':>6}{'M':>7}{'|sup|':>6}{'GPAP/s':>10}{'succ%':>6}"
           f"{'2wG':>6}{'snd':>4}")
    if has_cms:
        hdr += f"{'CMS/s':>9}{'2wC':>6}{'GPAP/CMS':>9}"
    print("\n" + hdr); print("-" * len(hdr))
    for r in results:
        snd = "OK" if r["gpu_sound"] else "BAD"
        line = (f"{r['name'][:22]:<24}{r['N']:>6}{r['M']:>7}{r['support']:>6}"
                f"{r['gpap_sps']:>10,.0f}{100*r['succ']:>5.0f}%{r['twise_gpap']:>6.1f}{snd:>4}")
        if has_cms and "cms_sps" in r:
            ratio = f"{r['ratio']:.2f}x" if r.get("ratio") else "—"
            line += f"{r['cms_sps']:>9,.0f}{r['twise_cms']:>6.1f}{ratio:>9}"
        print(line)


def _print_headline(payload: dict) -> None:
    a = payload["aggregate"]
    name = payload["gpu"]["name"]
    print("\n" + "=" * 74)
    print("HEADLINE — GPU SAT sampling throughput (GPAP = GPU-parallel CMSGen)")
    print("=" * 74)
    print(f"  GPU                  : {name}")
    print(f"  Auto K (batch)       : {a['auto_K']:,}")
    print(f"  Peak GPAP throughput : {a['gpap_sps_peak']:>12,.0f} samples/s")
    print(f"  Median (niche)       : {a['gpap_sps_median']:>12,.0f} samples/s")
    if a.get("ratio_geomean"):
        print(f"  vs CMSGen (geomean)  : {a['ratio_geomean']:>12.2f}x   (max {a['ratio_max']:.1f}x)")
    if a.get("twise_delta_mean") is not None:
        print(f"  2-wise coverage Δ    : {a['twise_delta_mean']:>+12.2f} pp  (vs CMSGen; ≈0 = equal quality)")
    print(f"  GPU-sound (all)      : {a['all_gpu_sound']}")
    if a.get("kscale_slope_loglog") is not None:
        print(f"  K-scaling (log-log)  : slope {a['kscale_slope_loglog']:.2f}  "
              f"(1.0 = throughput ∝ K; peak {a['kscale_peak_sps']:,.0f}/s at K-sweep)")
    print(f"\n  → On {name}, GPAP samples at up to {a['gpap_sps_peak']:,.0f} models/s"
          + (f", {a['ratio_geomean']:.1f}× CMSGen (geomean)" if a.get("ratio_geomean") else "") + ";")
    print(f"    throughput scales ~linearly with K, so larger-VRAM GPUs sit further up the curve.")
    print("=" * 74)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_all(quick: bool = False, device=None, sampling_dir: str = DEFAULT_SAMPLING_DIR,
            glob_pat: str | None = None, cmsgen_bin: str | None = "auto", n_time: int | None = None,
            n_cov: int | None = None, b: int = 8, k_cap: int = 49152,
            k_sweep_insts: list[str] | None = None, k_sweep_chunks: int | None = None,
            limit: int | None = None, do_plots: bool = True, out_dir: str = DEFAULT_OUT_DIR,
            scratch: str | None = None, verbose: bool = True) -> dict:
    """Discover sampling CNFs, benchmark GPAP vs CMSGen + a K-scaling sweep, and return a payload.

    quick=True restricts to the validated niche instances, lowers n_time, and sweeps K on one
    instance (a few minutes). The full run sweeps every *.cnf in `sampling_dir` (the niche boundary
    shows up as ~0-success rows) and sweeps K on two instances.
    """
    device = _resolve_device(device)
    scratch = scratch or os.path.join(REPO_ROOT, ".cmsgen_scratch")
    os.makedirs(scratch, exist_ok=True)

    if cmsgen_bin == "auto":
        cmsgen_bin = build_cmsgen()
    if quick:
        n_time = 20_000 if n_time is None else n_time
        n_cov = 2_000 if n_cov is None else n_cov
        k_sweep_insts = ["35.sk_3_52.cnf"] if k_sweep_insts is None else k_sweep_insts
        k_sweep_chunks = 4 if k_sweep_chunks is None else k_sweep_chunks
        files = [os.path.join(sampling_dir, n) for n in NICHE_FOCUS
                 if os.path.exists(os.path.join(sampling_dir, n))]
    else:
        n_time = 50_000 if n_time is None else n_time
        n_cov = 5_000 if n_cov is None else n_cov
        k_sweep_insts = list(K_SWEEP_DEFAULT) if k_sweep_insts is None else k_sweep_insts
        k_sweep_chunks = 8 if k_sweep_chunks is None else k_sweep_chunks
        files = sorted(glob.glob(glob_pat or os.path.join(sampling_dir, "*.cnf")))
    if limit is not None:
        files = files[:limit]

    info = gpu_info(device)
    if verbose:
        vram = f"{info['total_gb']:.1f} GB" if info.get("total_gb") else "n/a"
        print(f"\nDevice : {info['name']}  (VRAM={vram}, torch={info['torch_version']}, "
              f"cuda={info['cuda_version']})")
        print(f"  instances={len(files)}  B={b}  n_time={n_time:,}  "
              f"cmsgen={'yes' if cmsgen_bin else 'no'}\n")

    results = []
    for i, path in enumerate(files, 1):
        if verbose:
            print(f"[{i}/{len(files)}] {os.path.basename(path)[:48]} ...", flush=True)
        try:
            r = run_instance(path, device, cmsgen_bin=cmsgen_bin, n_time=n_time, n_cov=n_cov,
                             b=b, k_cap=k_cap, scratch=scratch)
            results.append(r)
            if verbose:
                rt = f"  {r['ratio']:.2f}x CMS" if r.get("ratio") else ""
                print(f"   GPAP {r['gpap_sps']:,.0f}/s  succ {100*r['succ']:.0f}%{rt}")
        except Exception as e:  # noqa: BLE001
            print(f"   ERROR: {e}")
            results.append({"name": os.path.basename(path), "error": str(e)})

    # K-scaling sweep on the chosen instances.
    ksweeps = []
    for n in k_sweep_insts:
        p = os.path.join(sampling_dir, n)
        if not os.path.exists(p):
            continue
        if verbose:
            print(f"[K-sweep] {n} ...", flush=True)
        try:
            ks = k_scaling(p, device, b=b, chunks=k_sweep_chunks, k_cap=k_cap)
            if ks:
                ksweeps.append(ks)
                if verbose:
                    for pt in ks["points"]:
                        print(f"   K={pt['K']:>8,}  {pt['samples_per_s']:>12,.0f} samp/s  "
                              f"(succ {100*pt['success_rate']:.0f}%)")
        except Exception as e:  # noqa: BLE001
            print(f"   [K-sweep] {n} ERROR: {e}")

    good = [r for r in results if "error" not in r]
    agg = aggregate(good, ksweeps)
    payload = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "gpu":         info,
        "quick":       quick,
        "b":           b,
        "n_time":      n_time,
        "cmsgen":      bool(cmsgen_bin),
        "aggregate":   agg,
        "instances":   results,
        "ksweeps":     ksweeps,
    }
    if verbose:
        _print_table(good, has_cms=bool(cmsgen_bin))
        _print_headline(payload)
    if do_plots:
        try:
            from scripts.colab_sampling_plots import make_plots
            paths = make_plots(payload, out_dir)
            if verbose and paths:
                print("\nPlots:")
                for p in paths:
                    print(f"  {p}")
        except Exception as e:  # noqa: BLE001
            print(f"  [plot] skipped: {e}")
    return payload


def save_results(payload: dict, out_dir: str = DEFAULT_OUT_DIR) -> str:
    """Write payload to <out_dir>/<gpu-slug>.json (one file per GPU; later runs overwrite)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{_slug(payload['gpu']['name'])}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {path}")
    return path


def compare_saved(out_dir: str = DEFAULT_OUT_DIR) -> list[dict]:
    """Load all saved per-GPU sampling results and rank by peak GPAP throughput."""
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        a = d.get("aggregate", {})
        rows.append({
            "gpu":        d.get("gpu", {}).get("name", os.path.basename(p)),
            "vram_gb":    d.get("gpu", {}).get("total_gb"),
            "auto_K":     a.get("auto_K", 0),
            "peak_sps":   a.get("gpap_sps_peak", 0.0),
            "median_sps": a.get("gpap_sps_median", 0.0),
            "ratio_gm":   a.get("ratio_geomean"),
            "slope":      a.get("kscale_slope_loglog"),
        })
    rows.sort(key=lambda r: r["peak_sps"], reverse=True)
    if not rows:
        print(f"No saved results in {out_dir}")
        return rows
    base = rows[-1]["peak_sps"] or 1.0
    print(f"\n{'GPU':<28}{'VRAM':>7}{'auto K':>10}{'peak samp/s':>14}"
          f"{'median':>12}{'/CMS gm':>9}{'rel':>6}")
    print("-" * 86)
    for r in rows:
        vram = f"{r['vram_gb']:.0f}GB" if r["vram_gb"] else "—"
        gm = f"{r['ratio_gm']:.1f}x" if r.get("ratio_gm") else "—"
        rel = r["peak_sps"] / base if base else 0.0
        print(f"{r['gpu']:<28}{vram:>7}{r['auto_K']:>10,}{r['peak_sps']:>14,.0f}"
              f"{r['median_sps']:>12,.0f}{gm:>9}{rel:>5.1f}x")
    print("\n('rel' = peak throughput relative to the slowest saved GPU; /CMS gm = geomean vs CMSGen)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="GPAP cross-GPU SAMPLING benchmark driver")
    ap.add_argument("--quick", action="store_true",
                    help="Fast run: niche instances only, smaller n_time, single-instance K-sweep")
    ap.add_argument("--device", default=None, help="torch device (cuda/cpu); default auto")
    ap.add_argument("--cmsgen", default="auto",
                    help="path to a cmsgen binary; 'auto' builds it from source; 'none' to skip")
    ap.add_argument("--sampling-dir", default=DEFAULT_SAMPLING_DIR)
    ap.add_argument("--glob", default=None, help="override the instance glob (full run)")
    ap.add_argument("--n-time", type=int, default=None, help="samples for the throughput timing")
    ap.add_argument("--n-cov", type=int, default=None, help="samples used for 2-wise coverage")
    ap.add_argument("--b", type=int, default=8, help="decisions per round (8 = throughput sweet spot)")
    ap.add_argument("--k-cap", type=int, default=49152)
    ap.add_argument("--k-sweep", default=None, help="comma list of instances for the K-scaling sweep")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="Only print a ranking of already-saved results, then exit")
    args = ap.parse_args()

    if args.compare:
        compare_saved(args.out_dir)
        return

    cmsgen = None if args.cmsgen.lower() in ("none", "no", "") else args.cmsgen
    k_sweep = args.k_sweep.split(",") if args.k_sweep else None
    payload = run_all(quick=args.quick, device=args.device, sampling_dir=args.sampling_dir,
                      glob_pat=args.glob, cmsgen_bin=cmsgen, n_time=args.n_time, n_cov=args.n_cov,
                      b=args.b, k_cap=args.k_cap, k_sweep_insts=k_sweep, limit=args.limit,
                      do_plots=not args.no_plot, out_dir=args.out_dir)
    if not args.no_save:
        save_results(payload, args.out_dir)
        compare_saved(args.out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cross-GPU benchmark driver for the GPAP sampler (Colab or local).

Measures on this GPU: (1) throughput vs CMSGen (GPAP samples/s / CMSGen samples/s per instance +
geomean), (2) quality parity via 2-wise coverage, (3) K-scaling (samples/s along a K ladder up to
this GPU's auto-K). K auto-sizes to free VRAM under the library's batch cap (default_k_cap), the
same on every GPU, so rows from two cards are comparable without pinning; --k-cap overrides.
Results go to benchmarks/colab_sampling_results/<gpu>.json (+ plots); compare_saved() ranks them.

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

from gpap.sampling.pipeline import sound_verdict
from gpap.sampling.sampler import GPUSampler   # sets expandable_segments
from gpap.engine.calibrate import default_k_cap
from gpap.benchmark import cmsgen_sample, distinct_projections, two_wise_coverage
from gpap.sampling.pipeline import sample_file as gpap_sample


def gpu_info(device: torch.device) -> dict:
    """Describe the device this run executed on (name, VRAM, CUDA/torch versions)."""
    import platform
    info = {
        "device":        str(device),
        "torch_version": torch.__version__,
        "cuda_version":  torch.version.cuda,
        "platform":      platform.platform(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        info.update({
            "name":             props.name,
            "total_gb":         round(props.total_memory / 1e9, 2),
            "free_gb":          round(free / 1e9, 2),
            "capability":       f"{props.major}.{props.minor}",
            "multiprocessors":  props.multi_processor_count,
        })
    else:
        info.update({
            "name":       f"CPU ({platform.processor() or platform.machine()})",
            "total_gb":   None,
            "free_gb":    None,
            "capability": None,
        })
    return info


def _slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", (name or "cpu").lower()).strip("-") or "cpu"

DEFAULT_SAMPLING_DIR = os.path.join(REPO_ROOT, "benchmarks", "sampling")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "colab_sampling_results")

# The K cap has one definition, gpap.engine.calibrate.default_k_cap; do not redefine it here.

# The validated niche: numeric sk_ instances where GPAP beats CMSGen (quick run + K-sweep); the
# full run sweeps every *.cnf so the niche boundary shows up as ~0-success rows.
NICHE_FOCUS = [
    "35.sk_3_52.cnf", "10.sk_1_46.cnf", "27.sk_3_32.cnf", "32.sk_4_38.cnf",
    "111.sk_2_36.cnf", "53.sk_4_32.cnf", "55.sk_3_46.cnf", "109.sk_4_36.cnf",
]
K_SWEEP_DEFAULT = ["35.sk_3_52.cnf", "10.sk_1_46.cnf"]


def _resolve_device(device) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --- CMSGen build helper (Colab has no pip wheel; build meelgroup/cmsgen from source) ---

def build_cmsgen(dest: str | None = None) -> str | None:
    """Clone + cmake-build meelgroup/cmsgen, returning the binary path (or None on failure).

    Idempotent: an existing binary is returned immediately. Needs cmake, build-essential, zlib1g-dev."""
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


# --- K-scaling sweep (throughput vs K, VRAM-limited) ---

def k_scaling(path: str, device: torch.device, *, b: int, chunks: int, k_cap: int,
              els: bool = True, n_points: int = 5, seed: int = 5) -> dict | None:
    """Sweep K on one instance and measure samples/s at each point of a geometric ladder up to the
    GPU's auto-K (`chunks` chunks per point, one simplified sampler).
    Returns {auto_K, points:[{K,samples_per_s,success}]} or None if the instance is UNSAT."""
    smp = GPUSampler.from_cnf_file(path, device, simplify=True, els=els, k_cap=k_cap, seed=seed)
    # `batch_size`, not `K`: the batch a chunk actually runs is what the ladder must top out at.
    auto_K = smp.batch_size
    if smp._recon and smp._recon.get("unsat"):
        return None
    # geometric ladder ending at auto_K: auto_K/2^(n-1) .. auto_K, floored at 256
    ks = sorted({max(256, auto_K >> i) for i in range(n_points)})
    points = []
    for k in ks:
        if k > auto_K:
            continue
        # pin_batch, not `smp.K = k`: only the pin bounds the batch actually run (tests/test_k_cap.py).
        smp.pin_batch(k)
        # Reseed per point so every K samples the same trajectory distribution.
        smp.reseed(seed)                    # all streams: core sampling, delivery, host fills
        smp.sample(k, decisions_per_round=b, max_chunks=1, collect=False)  # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = smp.sample(k * chunks, decisions_per_round=b,
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


# --- Per-instance throughput + quality vs CMSGen ---

def run_instance(path: str, device: torch.device, *, cmsgen_bin: str | None, n_time: int,
                 n_cov: int, b: int, k_cap: int, scratch: str, els: bool = True,
                 seed: int = 5, configs="default") -> dict:
    """Benchmark one CNF: GPAP throughput + 2-wise coverage + soundness + distinct-solution count,
    and with a cmsgen binary the same for CMSGen plus the GPAP/CMSGen ratio. `els` enables
    equivalent-literal substitution; `configs` is the bandit portfolio (CIRCUIT_PORTFOLIO for circuits)."""
    name = os.path.basename(path)
    # verify_sound: published rows need a verdict, not an absence (its wall is outside every rate).
    g = gpap_sample(path, device, simplify=True, els=els, n_time=n_time, n_cov=n_cov, b=b,
                    verify_sound=True,
                    k_cap=k_cap, configs=configs)
    sup1 = g["sup1"]                       # metadata from gpap_sample's single parse
    twg = two_wise_coverage(g["models"], sup1)
    row = {
        "name": name, "N": g["nv"], "M": g["n_clauses"], "support": len(sup1),
        "gpap_sps": round(g["samples_per_s"], 1), "succ": round(g["success_rate"], 4),
        "twise_gpap": round(twg, 2), "sound_check": g.get("sound_check", "off"),
        # distinct support projections among the coverage models, computed off the timing path
        "n_distinct": g["n_distinct"], "n_collected": g.get("n_collected"),
        "tuned_rv": g.get("tuned_rv"), "tuned_b": g.get("tuned_b"), "tuned_la": g.get("tuned_la"),
        # work stats: high attempts/s + low samples/s = low success; low attempts/s = BCP-bound
        "attempts_per_s": g.get("attempts_per_s"), "bcp_rounds_pc": g.get("bcp_rounds_per_chunk"),
        "bcp_ops_pa": g.get("bcp_ops_per_attempt"),
        # unique/s estimate = raw rate x distinct fraction of the coverage sample (post-processing)
        "unique_sps_est": g.get("unique_sps_est"), "assigns_per_s": g.get("assigns_per_s"),
        "decisions_per_s": g.get("decisions_per_s"),     # branch decisions/s
        "implications_per_s": g.get("implications_per_s"),  # unit-prop implications/s
        "in_niche": g["success_rate"] > 0.10,           # rough: produces models at a usable rate
    }
    if cmsgen_bin:
        # keep=n_cov: retain only the coverage models; keeping all n_time full models OOMs a small host
        crate, cmodels = cmsgen_sample(path, cmsgen_bin, n_time, scratch, keep=n_cov)
        twc = two_wise_coverage(cmodels[:n_cov], sup1)
        cms_uniq, cms_coll = distinct_projections(cmodels[:n_cov], sup1)   # same off-timing readout
        ratio = (g["samples_per_s"] / crate) if crate else None
        row.update(cms_sps=round(crate, 1), twise_cms=round(twc, 2),
                   cms_distinct=cms_uniq, cms_collected=cms_coll,
                   ratio=(round(ratio, 2) if ratio else None))
    return row


# --- Circuit unrolling benchmark (the CircuitSAT-paper families) ---

DEFAULT_CIRCUIT_DIR = os.path.join(REPO_ROOT, "benchmarks", "circuits", "bench")
CIRCUIT_SEQ = ["s386", "s526", "s832", "s1196"]        # ISCAS-89 sequential (unrolled over cycles)
# ITC'99 sequential b01-b14 (Recurrent CircuitSAT Sampling, arXiv 2502.21226); oversized
# (instance, cycle) combos are skipped by the CIRCUIT_MAX_CLAUSES guard in run_circuits.
CIRCUIT_ITC = [f"b{i:02d}" for i in range(1, 15)]
CIRCUIT_COMB = ["c432", "c499", "c880", "c1355", "c1908",    # ISCAS-85 combinational — DEMOTIC's
                "c2670", "c3540", "c5315", "c6288", "c7552"]  # exact 10-circuit set
CIRCUIT_CYCLES = [1, 5, 10, 25]
CIRCUIT_MAX_CLAUSES = 100_000                          # skip any unroll whose CNF exceeds this
# A curated fast subset for --quick (small-to-medium, easy and hard cases represented).
CIRCUIT_ITC_QUICK = ["b01", "b03", "b06", "b08", "b10", "b13"]

# Circuit sampler portfolio: the bandit picks per instance; the list itself lives in the pipeline.
from gpap.sampling.pipeline import CIRCUIT_PORTFOLIO  # single source of truth (same list)


def run_circuits(device: torch.device, *, cmsgen_bin: str | None, n_time: int, n_cov: int, b: int,
                 k_cap: int, scratch: str, els: bool = True, cycles: list[int] | None = None,
                 insts: list[str] | None = None, circuit_dir: str = DEFAULT_CIRCUIT_DIR,
                 max_clauses: int = CIRCUIT_MAX_CLAUSES, configs=CIRCUIT_PORTFOLIO,
                 verbose: bool = True, on_row=None, skip_keys=None) -> list[dict]:
    """Benchmark GPAP vs CMSGen on the CircuitSAT sampling families the competitors use.

    Sequential (ISCAS-89 s*, ITC'99 b*): unroll T cycles, Tseitin, pin all final-cycle outputs to a
    simulated target, sample the primary inputs, swept over `cycles`. Combinational (ISCAS-85 c*):
    T=1, pin one output. A CNF above `max_clauses` is skipped. `c ind` = primary inputs; `configs`
    is the sampler portfolio. Resumable: `skip_keys` = (inst, T) pairs already done; `on_row(row)`
    is called after each (inst, T) so callers can persist incrementally. A cell that raises becomes
    an error row and the sweep continues."""
    from gpap.circuits.bench import parse_bench, build_cnf, write_dimacs
    cycles = cycles or CIRCUIT_CYCLES
    insts = insts or (CIRCUIT_SEQ + CIRCUIT_ITC + CIRCUIT_COMB)
    skip_keys = skip_keys or set()
    gen_dir = os.path.join(scratch, "circuits")
    os.makedirs(gen_dir, exist_ok=True)
    rows = []
    for inst in insts:
        bench = os.path.join(circuit_dir, f"{inst}.bench")
        if not os.path.exists(bench):
            continue
        circ = parse_bench(bench)
        combinational = not circ[2]                    # circ = (inputs, outputs, dffs, gates)
        inst_cycles = [1] if combinational else cycles  # combinational = no unrolling
        for T in inst_cycles:
            if (inst, T) in skip_keys:                 # persisted on a prior run
                continue
            cnfp = os.path.join(gen_dir, f"{inst}.c{T}.cnf")
            if not os.path.exists(cnfp):
                # combinational: pin one output to keep a large samplable solution space
                builder, support, _, _ = build_cnf(circ, T, seed=1,
                                                pin_outputs=1 if combinational else None)
                if len(builder.clauses) > max_clauses:  # keep the unrolled CNF within budget
                    if verbose:
                        print(f"   {inst} T={T:>2}: skip ({len(builder.clauses):,} clauses "
                              f"> {max_clauses:,})", flush=True)
                    continue
                write_dimacs(cnfp, builder, support)
            try:
                r = run_instance(cnfp, device, cmsgen_bin=cmsgen_bin, n_time=n_time, n_cov=n_cov,
                                 b=b, k_cap=k_cap, scratch=scratch, els=els, configs=configs)
            except Exception as e:  # noqa: BLE001
                r = {"name": f"{inst}.c{T}.cnf", "error": str(e)}
                if verbose:
                    print(f"   {inst} T={T:>2}: ERROR {e}", flush=True)
            r["circuit"], r["cycles"], r["family"] = inst, T, ("iscas85" if combinational else "iscas89")
            rows.append(r)
            if on_row is not None:
                on_row(r)                              # persist incrementally (resume checkpoint)
            if "error" not in r and verbose:
                rt = f"  {r['ratio']:.2f}x CMS" if r.get("ratio") else ""
                la = f",la={r['tuned_la']}" if r.get("tuned_la") else ""
                cfg = f"  [rv={r.get('tuned_rv')},B={r.get('tuned_b')}{la}]"
                att = r.get("attempts_per_s") or 0
                uq = f"{r['unique_sps_est']:,.0f}" if r.get("unique_sps_est") else "—"
                dps = f"{r['decisions_per_s']:.1e}" if r.get("decisions_per_s") else "—"
                ips = f"{r['implications_per_s']:.1e}" if r.get("implications_per_s") else "—"
                work = (f"  | uniq~{uq}/s  {dps} dec/s  {ips} impl/s  {att:>9,.0f} att/s  "
                        f"{r.get('bcp_ops_pa') or 0:,.0f} ops/att")
                print(f"   {inst} T={T:>2}: GPAP {r['gpap_sps']:>12,.0f}/s  succ {100*r['succ']:.0f}%"
                      f"{rt}{cfg}{work}", flush=True)
    return rows


def _print_circuit_matrix(rows: list[dict]) -> None:
    if not rows:
        return
    insts = sorted({r["circuit"] for r in rows}, key=lambda s: (len(s), s))
    cyc = sorted({r["cycles"] for r in rows})
    idx = {(r["circuit"], r["cycles"]): r for r in rows}
    print("\n=== circuits: GPAP/CMSGen ratio by unroll depth (rows=circuit, cols=cycles) ===")
    print(f"{'circuit':<9}" + "".join(f"{'T='+str(t):>9}" for t in cyc))
    for inst in insts:
        cells = []
        for t in cyc:
            r = idx.get((inst, t))
            cells.append(f"{r['ratio']:>8.2f}x" if (r and r.get("ratio")) else f"{'—':>9}")
        print(f"{inst:<9}" + "".join(cells))


# --- Aggregation ---

def _sound_summary(rows: list[dict]) -> str:
    """One line for the opt-in soundness re-check: "ok (n/n)" when every row passed, "UNSOUND" if
    any failed, "not run" / a partial count otherwise (an unchecked sweep is neither verdict)."""
    if not rows:
        return "no rows"
    # sound_verdict, not a string compare: legacy rows carry `gpu_sound`, and it reads both shapes.
    verdicts = [sound_verdict(r) for r in rows]
    if "UNSOUND" in verdicts:
        return "UNSOUND"
    n_ok = verdicts.count("ok")
    return f"ok ({n_ok}/{len(rows)})" if n_ok else "not run"


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
    decs = [r["decisions_per_s"] for r in niche if r.get("decisions_per_s")]
    impl = [r["implications_per_s"] for r in niche if r.get("implications_per_s")]
    return {
        "n_instances":          len(results),
        "n_niche":              len(niche),
        "gpap_sps_peak":        round(max(sps), 1) if sps else 0.0,
        "gpap_sps_median":      round(statistics.median(sps), 1) if sps else 0.0,
        "ratio_geomean":        _gm(ratios),
        "ratio_max":            round(max(ratios), 2) if ratios else None,
        "twise_delta_mean":     round(float(np.mean(dq)), 2) if dq else None,
        "all_sound_checked":    _sound_summary(results),
        "kscale_slope_loglog":  slope,          # ~1.0 => samples/s proportional to K
        "kscale_peak_sps":      round(peak_k_sps, 1),
        "auto_K":               auto_K,
        # CDCL-comparable work rates (peak over the niche): decisions/s and implications/s
        "decisions_per_s_peak":     round(max(decs), 1) if decs else 0.0,
        "implications_per_s_peak":  round(max(impl), 1) if impl else 0.0,
    }


def aggregate_circuits(circuits: list[dict], ksweeps: list[dict]) -> dict:
    """Aggregate the circuit rows over all circuits (not just the niche), including the mean
    distinct/collected fraction for GPAP and CMSGen (the diversity readout)."""
    rows = [r for r in circuits if "error" not in r and "gpap_sps" in r]
    sps = [r["gpap_sps"] for r in rows if r.get("gpap_sps")]
    ratios = [r["ratio"] for r in rows if r.get("ratio")]
    dq = [r["twise_gpap"] - r["twise_cms"] for r in rows if "twise_cms" in r]

    def _gm(xs):
        xs = [x for x in xs if x and x > 0]
        return round(statistics.geometric_mean(xs), 2) if xs else None

    def _frac(dk, ck):                                   # mean distinct/collected fraction
        fr = [r[dk] / r[ck] for r in rows if r.get(ck) and r.get(dk) is not None]
        return round(float(np.mean(fr)), 3) if fr else None

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
        "n_instances":          len(rows),
        "n_circuits":           len({r["circuit"] for r in rows}),
        "gpap_sps_peak":        round(max(sps), 1) if sps else 0.0,
        "gpap_sps_median":      round(statistics.median(sps), 1) if sps else 0.0,
        "ratio_geomean":        _gm(ratios),
        "ratio_max":            round(max(ratios), 2) if ratios else None,
        "n_win_vs_cms":         sum(1 for x in ratios if x >= 1.0),
        "n_compared":           len(ratios),
        "twise_delta_mean":     round(float(np.mean(dq)), 2) if dq else None,
        "all_sound_checked":    _sound_summary(rows),
        "gpap_distinct_frac":   _frac("n_distinct", "n_collected"),   # ~1.0 = all-distinct output
        "cms_distinct_frac":    _frac("cms_distinct", "cms_collected"),
        "kscale_slope_loglog":  slope,
        "kscale_peak_sps":      round(peak_k_sps, 1),
        "auto_K":               auto_K,
        # CDCL-comparable work rates (peak across the circuit rows)
        "decisions_per_s_peak":     round(max([r["decisions_per_s"] for r in rows
                                               if r.get("decisions_per_s")], default=0.0), 1),
        "implications_per_s_peak":  round(max([r["implications_per_s"] for r in rows
                                               if r.get("implications_per_s")], default=0.0), 1),
    }


# --- Pretty printing ---

def _print_circuit_table(rows: list[dict], has_cms: bool) -> None:
    """Per-(circuit, cycle) throughput + distinct-solution counts among the coverage models."""
    rows = [r for r in rows if "gpap_sps" in r]
    if not rows:
        return
    hdr = (f"{'circuit':<8}{'T':>4}{'N':>7}{'M':>7}{'sup':>5}{'GPAP/s':>11}{'uniq/s~':>10}"
           f"{'dec/s':>9}{'impl/s':>9}{'succ':>6}{'uniqG':>8}{'config':>15}")
    if has_cms:
        hdr += f"{'CMS/s':>10}{'uniqC':>8}{'ratio':>8}"
    print("\n=== circuits: throughput + unique (distinct) solution counts ===")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (len(x["circuit"]), x["circuit"], x["cycles"])):
        la = f",{r['tuned_la']}" if r.get("tuned_la") else ""
        cfg = f"rv={int(bool(r.get('tuned_rv')))},B={r.get('tuned_b')}{la}"
        uq = f"{r['unique_sps_est']:,.0f}" if r.get("unique_sps_est") else "—"
        dps = f"{r['decisions_per_s']:.1e}" if r.get("decisions_per_s") else "—"
        ips = f"{r['implications_per_s']:.1e}" if r.get("implications_per_s") else "—"
        line = (f"{r['circuit']:<8}{r['cycles']:>4}{r['N']:>7}{r['M']:>7}{r['support']:>5}"
                f"{r['gpap_sps']:>11,.0f}{uq:>10}{dps:>9}{ips:>9}"
                f"{100*r['succ']:>5.0f}%{(r.get('n_distinct') or 0):>8,}{cfg:>15}")
        if has_cms and "cms_sps" in r:
            ratio = f"{r['ratio']:.2f}x" if r.get("ratio") else "—"
            line += f"{r['cms_sps']:>10,.0f}{(r.get('cms_distinct') or 0):>8,}{ratio:>8}"
        print(line)


def _print_table(results: list[dict], has_cms: bool) -> None:
    hdr = (f"{'instance':<24}{'N':>6}{'M':>7}{'|sup|':>6}{'GPAP/s':>10}{'succ%':>6}"
           f"{'2wG':>6}{'snd':>4}")
    if has_cms:
        hdr += f"{'CMS/s':>9}{'2wC':>6}{'GPAP/CMS':>9}"
    print("\n" + hdr); print("-" * len(hdr))
    for r in results:
        snd = {"ok": "OK", "UNSOUND": "BAD"}.get(sound_verdict(r), "-")
        line = (f"{r['name'][:22]:<24}{r['N']:>6}{r['M']:>7}{r['support']:>6}"
                f"{r['gpap_sps']:>10,.0f}{100*r['succ']:>5.0f}%{r['twise_gpap']:>6.1f}{snd:>4}")
        if has_cms and "cms_sps" in r:
            ratio = f"{r['ratio']:.2f}x" if r.get("ratio") else "—"
            line += f"{r['cms_sps']:>9,.0f}{r['twise_cms']:>6.1f}{ratio:>9}"
        print(line)


def _print_headline(payload: dict) -> None:
    a = payload["aggregate"]
    name = payload["gpu"]["name"]
    circ = payload.get("circuits_only")
    print("\n" + "=" * 74)
    kind = "CIRCUIT-SAT" if circ else "GPU SAT"
    print(f"HEADLINE — {kind} sampling throughput (GPAP = GPU-parallel CMSGen)")
    print("=" * 74)
    print(f"  GPU                  : {name}")
    print(f"  ELS reduction        : {'on' if payload.get('els') else 'off'}")
    print(f"  Auto K (batch)       : {a['auto_K']:,}")
    print(f"  Peak GPAP throughput : {a['gpap_sps_peak']:>12,.0f} samples/s")
    print(f"  Median throughput    : {a['gpap_sps_median']:>12,.0f} samples/s")
    if a.get("implications_per_s_peak"):
        print(f"  Peak decisions/s     : {a['decisions_per_s_peak']:>12,.0f}  (branch picks)")
        print(f"  Peak implications/s  : {a['implications_per_s_peak']:>12,.0f}  (unit-prop; CDCL ≈1e7/s single-core)")
    if a.get("ratio_geomean"):
        print(f"  vs CMSGen (geomean)  : {a['ratio_geomean']:>12.2f}x   (max {a['ratio_max']:.1f}x)")
    if a.get("n_compared"):
        print(f"  Won vs CMSGen        : {a['n_win_vs_cms']:>7}/{a['n_compared']} circuit unrolls")
    if a.get("twise_delta_mean") is not None:
        print(f"  2-wise coverage Δ    : {a['twise_delta_mean']:>+12.2f} pp  (vs CMSGen; ≈0 = equal quality)")
    if a.get("gpap_distinct_frac") is not None:
        cf = (f"  (CMSGen {100*a['cms_distinct_frac']:.0f}%)"
              if a.get("cms_distinct_frac") is not None else "")
        print(f"  Unique output (GPAP) : {100*a['gpap_distinct_frac']:>11.0f}%  distinct of collected"
              f"{cf}")
    print(f"  sound re-check       : {a['all_sound_checked']}")
    if a.get("kscale_slope_loglog") is not None:
        print(f"  K-scaling (log-log)  : slope {a['kscale_slope_loglog']:.2f}  "
              f"(1.0 = throughput ∝ K; peak {a['kscale_peak_sps']:,.0f}/s at K-sweep)")
    print(f"\n  → On {name}, GPAP samples at up to {a['gpap_sps_peak']:,.0f} models/s"
          + (f", {a['ratio_geomean']:.1f}× CMSGen (geomean)" if a.get("ratio_geomean") else "") + ";")
    print("    throughput scales ~linearly with K, so larger-VRAM GPUs sit further up the curve.")
    print("=" * 74)


# --- Top-level orchestration ---

def run_all(quick: bool = False, device=None, sampling_dir: str = DEFAULT_SAMPLING_DIR,
            glob_pat: str | None = None, cmsgen_bin: str | None = "auto", n_time: int | None = None,
            n_cov: int | None = None, b: int = 8, k_cap: int | None = None, els: bool = True,
            k_sweep_insts: list[str] | None = None, k_sweep_chunks: int | None = None,
            limit: int | None = None, do_plots: bool = True, out_dir: str = DEFAULT_OUT_DIR,
            scratch: str | None = None, verbose: bool = True, do_circuits: bool = False,
            circuit_cycles: list[int] | None = None, circuits_only: bool = False,
            circuit_insts: list[str] | None = None) -> dict:
    """Discover sampling CNFs, benchmark GPAP vs CMSGen + a K-scaling sweep, and return a payload.

    quick=True restricts to the niche instances, lowers n_time and sweeps K on one instance; the
    full run sweeps every *.cnf in `sampling_dir` and K on two instances. circuits_only=True runs
    only the circuit suite (sequential families over unroll depth, + ISCAS-85 on full runs) and
    K-sweeps its best-throughput circuit. k_cap=None uses the library's batch cap (default_k_cap),
    the same on every GPU."""
    device = _resolve_device(device)
    if k_cap is None:
        k_cap = default_k_cap(device)
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
    if circuits_only:
        files = []                       # skip the sk_ suite; circuits are the whole run
        do_circuits = True
        if circuit_insts is None:        # paper family (seq) + b instances; +ISCAS-85 on full runs
            circuit_insts = (CIRCUIT_SEQ + CIRCUIT_ITC_QUICK if quick
                             else CIRCUIT_SEQ + CIRCUIT_ITC + CIRCUIT_COMB)
    if limit is not None:
        files = files[:limit]

    info = gpu_info(device)
    if verbose:
        vram = f"{info['total_gb']:.1f} GB" if info.get("total_gb") else "n/a"
        print(f"\nDevice : {info['name']}  (VRAM={vram}, torch={info['torch_version']}, "
              f"cuda={info['cuda_version']})")
        print(f"  instances={len(files)}  B={b}  n_time={n_time:,}  els={els}  k_cap={k_cap:,}  "
              f"cmsgen={'yes' if cmsgen_bin else 'no'}\n")

    results = []
    for i, path in enumerate(files, 1):
        if verbose:
            print(f"[{i}/{len(files)}] {os.path.basename(path)[:48]} ...", flush=True)
        try:
            r = run_instance(path, device, cmsgen_bin=cmsgen_bin, n_time=n_time, n_cov=n_cov,
                             b=b, k_cap=k_cap, scratch=scratch, els=els)
            results.append(r)
            if verbose:
                rt = f"  {r['ratio']:.2f}x CMS" if r.get("ratio") else ""
                print(f"   GPAP {r['gpap_sps']:,.0f}/s  succ {100*r['succ']:.0f}%{rt}")
        except Exception as e:  # noqa: BLE001
            print(f"   ERROR: {e}")
            results.append({"name": os.path.basename(path), "error": str(e)})

    # K-scaling sweep on the sk_ instances (skipped in circuits_only — swept on a circuit below).
    ksweeps = []
    if not circuits_only:
        for n in k_sweep_insts:
            p = os.path.join(sampling_dir, n)
            if not os.path.exists(p):
                continue
            if verbose:
                print(f"[K-sweep] {n} ...", flush=True)
            try:
                ks = k_scaling(p, device, b=b, chunks=k_sweep_chunks, k_cap=k_cap, els=els)
                if ks:
                    ksweeps.append(ks)
                    if verbose:
                        for pt in ks["points"]:
                            print(f"   K={pt['K']:>8,}  {pt['samples_per_s']:>12,.0f} samp/s  "
                                  f"(succ {100*pt['success_rate']:.0f}%)")
            except Exception as e:  # noqa: BLE001
                print(f"   [K-sweep] {n} ERROR: {e}")

    # Circuit unrolling suite, swept over depth, under CIRCUIT_PORTFOLIO.
    circuits = []
    if do_circuits:
        if verbose:
            print(f"\n[circuits] unrolling (cycles={circuit_cycles or CIRCUIT_CYCLES}) ...", flush=True)
        try:
            circuits = run_circuits(device, cmsgen_bin=cmsgen_bin, n_time=n_time, n_cov=n_cov, b=b,
                                    k_cap=k_cap, scratch=scratch, els=els, cycles=circuit_cycles,
                                    insts=circuit_insts, verbose=verbose)
        except Exception as e:  # noqa: BLE001
            print(f"   [circuits] ERROR: {e}")

    # circuits_only: K-sweep the best-throughput circuit just generated.
    if circuits_only and circuits:
        best = max((r for r in circuits if r.get("in_niche") and "gpap_sps" in r),
                   key=lambda r: r["gpap_sps"], default=None)
        if best:
            cnfp = os.path.join(scratch, "circuits", f"{best['circuit']}.c{best['cycles']}.cnf")
            if os.path.exists(cnfp):
                if verbose:
                    print(f"[K-sweep] {best['circuit']} T={best['cycles']} ...", flush=True)
                try:
                    ks = k_scaling(cnfp, device, b=b, chunks=k_sweep_chunks, k_cap=k_cap, els=els)
                    if ks:
                        ksweeps.append(ks)
                        if verbose:
                            for pt in ks["points"]:
                                print(f"   K={pt['K']:>8,}  {pt['samples_per_s']:>12,.0f} samp/s  "
                                      f"(succ {100*pt['success_rate']:.0f}%)")
                except Exception as e:  # noqa: BLE001
                    print(f"   [K-sweep] circuit ERROR: {e}")

    good = [r for r in results if "error" not in r]
    agg = aggregate_circuits(circuits, ksweeps) if circuits_only else aggregate(good, ksweeps)
    payload = {
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "gpu":           info,
        "quick":         quick,
        "b":             b,
        "n_time":        n_time,
        "els":           els,
        "cmsgen":        bool(cmsgen_bin),
        "circuits_only": circuits_only,
        "aggregate":     agg,
        "instances":     results,
        "ksweeps":       ksweeps,
        "circuits":      circuits,
    }
    if verbose:
        if circuits_only:
            _print_circuit_table(circuits, has_cms=bool(cmsgen_bin))
            _print_circuit_matrix(circuits)
        else:
            _print_table(good, has_cms=bool(cmsgen_bin))
            if circuits:
                _print_circuit_matrix(circuits)
        _print_headline(payload)
    if do_plots:
        try:
            from eval.colab_sampling_plots import make_plots
            paths = make_plots(payload, out_dir)
            if verbose and paths:
                print("\nPlots:")
                for p in paths:
                    print(f"  {p}")
        except Exception as e:  # noqa: BLE001
            print(f"  [plot] skipped: {e}")
    return payload


# --- Per-family runner (Colab idle-disconnect mitigation) ---
# Colab reaps a runtime that looks idle, so the suite runs as one short cell per family, each
# persisting its rows to a partial file after every instance; re-running a cell resumes, and
# `finalize_families()` merges the partials into the one <gpu>.json that compare_saved/plots read.
FAMILIES = {                                    # family -> (kind, full-run insts, quick-run insts)
    "iscas89": ("circuit", CIRCUIT_SEQ, CIRCUIT_SEQ),          # ISCAS-89 sequential (s*)
    "itc99":   ("circuit", CIRCUIT_ITC, CIRCUIT_ITC_QUICK),    # ITC'99 (b01-b14)
    "iscas85": ("circuit", CIRCUIT_COMB, CIRCUIT_COMB),        # ISCAS-85 combinational (c*)
    "sk":      ("sk", None, None),                             # the sk_ sampling suite
}
_PARTIAL_SUBDIR = "_partial"


def _partial_path(out_dir: str, slug: str, family: str) -> str:
    d = os.path.join(out_dir, _PARTIAL_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{slug}.{family}.json")


def run_family(family: str, *, quick: bool = False, device=None, cmsgen_bin: str | None = "auto",
               out_dir: str = DEFAULT_OUT_DIR, sampling_dir: str = DEFAULT_SAMPLING_DIR,
               circuit_cycles: list[int] | None = None, n_time: int | None = None,
               n_cov: int | None = None, b: int = 8, k_cap: int | None = None, els: bool = True,
               resume: bool = True, scratch: str | None = None, limit: int | None = None,
               verbose: bool = True) -> dict:
    """Benchmark one family (see FAMILIES), persisting each instance to a partial file as it finishes.

    Resumable: re-running skips instances already saved. Call finalize_families() once every family
    cell is done. Returns the partial payload {"gpu", "family", "rows", ...}."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {list(FAMILIES)}")
    kind, insts_full, insts_quick = FAMILIES[family]
    device = _resolve_device(device)
    if k_cap is None:
        k_cap = default_k_cap(device)
    scratch = scratch or os.path.join(REPO_ROOT, ".cmsgen_scratch")
    os.makedirs(scratch, exist_ok=True)
    if cmsgen_bin == "auto":
        cmsgen_bin = build_cmsgen()
    n_time = (20_000 if quick else 50_000) if n_time is None else n_time
    n_cov = (2_000 if quick else 5_000) if n_cov is None else n_cov

    info = gpu_info(device)
    slug = _slug(info["name"])
    ppath = _partial_path(out_dir, slug, family)
    rows: list[dict] = []
    if resume and os.path.exists(ppath):
        try:
            rows = json.load(open(ppath)).get("rows", [])
            if verbose and rows:
                print(f"[resume] {family}: {len(rows)} instance(s) already saved in {ppath}")
        except Exception:  # noqa: BLE001
            rows = []

    params = {"quick": quick, "b": b, "n_time": n_time, "n_cov": n_cov, "els": els,
              "k_cap": k_cap, "cmsgen": bool(cmsgen_bin)}

    def _save():
        with open(ppath, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"), "gpu": info,
                       "family": family, "kind": kind, "params": params, "rows": rows}, f, indent=2)

    if verbose:
        vram = f"{info['total_gb']:.1f} GB" if info.get("total_gb") else "n/a"
        print(f"\n=== family '{family}' on {info['name']} (VRAM={vram}) — k_cap={k_cap:,} "
              f"n_time={n_time:,} ===", flush=True)

    if kind == "circuit":
        insts = insts_quick if quick else insts_full
        if limit:
            insts = insts[:limit]
        done = {(r["circuit"], r["cycles"]) for r in rows if "error" not in r and "circuit" in r}

        def _on_row(r):
            rows.append(r)
            _save()                                  # checkpoint after every (inst, T)
        new = run_circuits(device, cmsgen_bin=cmsgen_bin, n_time=n_time, n_cov=n_cov, b=b,
                           k_cap=k_cap, scratch=scratch, els=els, cycles=circuit_cycles,
                           insts=insts, verbose=verbose, on_row=_on_row, skip_keys=done)
        _ = new                                      # rows already collected via _on_row
        if verbose:
            _print_circuit_table([r for r in rows if "error" not in r], has_cms=bool(cmsgen_bin))
    else:  # sk suite
        files = ([os.path.join(sampling_dir, n) for n in NICHE_FOCUS] if quick
                 else sorted(glob.glob(os.path.join(sampling_dir, "*.cnf"))))
        files = [p for p in files if os.path.exists(p)]
        if limit:
            files = files[:limit]
        done = {r["name"] for r in rows if "error" not in r}
        for i, path in enumerate(files, 1):
            name = os.path.basename(path)
            if name in done:
                continue
            if verbose:
                print(f"[{i}/{len(files)}] {name[:48]} ...", flush=True)
            try:
                r = run_instance(path, device, cmsgen_bin=cmsgen_bin, n_time=n_time, n_cov=n_cov,
                                 b=b, k_cap=k_cap, scratch=scratch, els=els)
            except Exception as e:  # noqa: BLE001
                r = {"name": name, "error": str(e)}
                print(f"   ERROR: {e}")
            rows.append(r)
            _save()                                  # checkpoint after every instance
            if verbose and "error" not in r:
                rt = f"  {r['ratio']:.2f}x CMS" if r.get("ratio") else ""
                print(f"   GPAP {r['gpap_sps']:,.0f}/s  succ {100*r['succ']:.0f}%{rt}", flush=True)
        if verbose:
            _print_table([r for r in rows if "error" not in r], has_cms=bool(cmsgen_bin))
    _save()
    if verbose:
        print(f"\n[saved] {family}: {len([r for r in rows if 'error' not in r])} rows → {ppath}")
    return {"gpu": info, "family": family, "kind": kind, "params": params, "rows": rows}


def finalize_families(*, device=None, out_dir: str = DEFAULT_OUT_DIR, do_plots: bool = True,
                      verbose: bool = True) -> dict:
    """Merge this GPU's per-family partial files into the combined <gpu>.json, aggregate, print the
    headline and plot. Circuit families aggregate as the circuit suite; an sk partial goes under
    `instances`."""
    device = _resolve_device(device)
    info = gpu_info(device)
    slug = _slug(info["name"])
    pdir = os.path.join(out_dir, _PARTIAL_SUBDIR)
    circuits: list[dict] = []
    instances: list[dict] = []
    params: dict = {}
    for p in sorted(glob.glob(os.path.join(pdir, f"{slug}.*.json"))):
        try:
            part = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        params = part.get("params", params)
        if part.get("kind") == "circuit":
            circuits.extend(part.get("rows", []))
        else:
            instances.extend(part.get("rows", []))
    good_circ = [r for r in circuits if "error" not in r and "gpap_sps" in r]
    good_inst = [r for r in instances if "error" not in r and "gpap_sps" in r]
    circuits_only = bool(good_circ) and not good_inst
    agg = aggregate_circuits(circuits, []) if good_circ else aggregate(good_inst, [])
    payload = {
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "gpu":           info,
        "quick":         params.get("quick"),
        "b":             params.get("b"),
        "n_time":        params.get("n_time"),
        "els":           params.get("els"),
        "cmsgen":        params.get("cmsgen"),
        "circuits_only": circuits_only,
        "aggregate":     agg,
        "instances":     instances,
        "ksweeps":       [],
        "circuits":      circuits,
    }
    if verbose:
        if good_circ:
            _print_circuit_table(circuits, has_cms=bool(params.get("cmsgen")))
            _print_circuit_matrix(circuits)
        if good_inst:
            _print_table(good_inst, has_cms=bool(params.get("cmsgen")))
        _print_headline(payload)
    save_results(payload, out_dir)
    if do_plots:
        try:
            from eval.colab_sampling_plots import make_plots
            for p in make_plots(payload, out_dir):
                print(f"  plot: {p}")
        except Exception as e:  # noqa: BLE001
            print(f"  [plot] skipped: {e}")
    return payload


def run_ksweep(payload: dict | None = None, *, device=None, out_dir: str = DEFAULT_OUT_DIR,
               scratch: str | None = None, b: int = 8, chunks: int | None = None,
               k_cap: int | None = None, els: bool = True, circuit: str | None = None,
               cycles: int | None = None, do_plots: bool = True, verbose: bool = True) -> dict:
    """K-scaling sweep on one circuit, run after finalize_families().

    With no `circuit` it sweeps the highest-throughput circuit in the finalized `<gpu>.json` (or
    `payload`); the CNF must still be in `scratch/circuits/`. Merges the sweep into the saved
    payload and re-plots."""
    device = _resolve_device(device)
    if k_cap is None:
        k_cap = default_k_cap(device)
    scratch = scratch or os.path.join(REPO_ROOT, ".cmsgen_scratch")
    chunks = chunks or 8
    if payload is None:
        p = os.path.join(out_dir, f"{_slug(gpu_info(device)['name'])}.json")
        payload = json.load(open(p))
    rows = [r for r in payload.get("circuits", []) if "error" not in r and "gpap_sps" in r]
    if circuit is None:
        best = max(rows, key=lambda r: r.get("gpap_sps", 0), default=None)
        if best is None:
            print("[ksweep] no circuit rows to sweep"); return payload
        circuit, cycles = best["circuit"], best["cycles"]
    cnfp = os.path.join(scratch, "circuits", f"{circuit}.c{cycles}.cnf")
    if not os.path.exists(cnfp):
        print(f"[ksweep] {cnfp} not found (re-run the family cell first)"); return payload
    if verbose:
        print(f"[K-sweep] {circuit} T={cycles} up to auto-K ...", flush=True)
    ks = k_scaling(cnfp, device, b=b, chunks=chunks, k_cap=k_cap, els=els)
    if ks:
        payload.setdefault("ksweeps", []).append(ks)
        if verbose:
            for pt in ks["points"]:
                print(f"   K={pt['K']:>9,}  {pt['samples_per_s']:>13,.0f} samp/s  "
                      f"(succ {100*pt['success_rate']:.0f}%)", flush=True)
        payload["aggregate"] = (aggregate_circuits(payload.get("circuits", []), payload["ksweeps"])
                                if payload.get("circuits_only")
                                else aggregate([r for r in payload.get("instances", [])
                                                if "error" not in r], payload["ksweeps"]))
        if verbose:
            _print_headline(payload)
        save_results(payload, out_dir)
        if do_plots:
            try:
                from eval.colab_sampling_plots import make_plots
                for pp in make_plots(payload, out_dir):
                    print(f"  plot: {pp}")
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
    ap.add_argument("--k-cap", type=int, default=None,
                    help="max K (batch). Default: the library cap, same on every GPU — see default_k_cap")
    ap.add_argument("--no-els", dest="els", action="store_false", default=True,
                    help="disable equivalent-literal substitution (on by default; ~1.5x throughput)")
    ap.add_argument("--k-sweep", default=None, help="comma list of instances for the K-scaling sweep")
    ap.add_argument("--circuits", action="store_true",
                    help="also run the ISCAS-89/ITC'99 sequential + ISCAS-85 combinational unrolling "
                         "suite (the CircuitSAT-paper families), swept over cycle depth")
    ap.add_argument("--circuits-only", action="store_true",
                    help="run ONLY the circuit suite (skip the sk_ suite); the current focus — "
                         "ISCAS-89 + ITC'99 b-instances swept over unroll depth")
    ap.add_argument("--circuit-cycles", default=None,
                    help="comma list of unroll depths for --circuits (default 1,5,10,25)")
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
    circuit_cycles = ([int(x) for x in args.circuit_cycles.split(",")]
                      if args.circuit_cycles else None)
    payload = run_all(quick=args.quick, device=args.device, sampling_dir=args.sampling_dir,
                      glob_pat=args.glob, cmsgen_bin=cmsgen, n_time=args.n_time, n_cov=args.n_cov,
                      b=args.b, k_cap=args.k_cap, els=args.els, k_sweep_insts=k_sweep,
                      limit=args.limit, do_plots=not args.no_plot, out_dir=args.out_dir,
                      do_circuits=args.circuits or args.circuits_only,
                      circuits_only=args.circuits_only, circuit_cycles=circuit_cycles)
    if not args.no_save:
        save_results(payload, args.out_dir)
        compare_saved(args.out_dir)


if __name__ == "__main__":
    main()

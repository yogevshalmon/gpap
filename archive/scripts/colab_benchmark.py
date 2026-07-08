#!/usr/bin/env python3
"""Cross-GPU benchmark driver for GPAP — designed to run on Colab (any GPU) or locally.

Two metrics, both measured end-to-end (full generate → reset → apply → propagate → extract
pipeline over many K-sized chunks), GPU vs a sequential CPU PySAT baseline:

  1. **Pairs anchor** (deterministic): exhaustive (±v1,±v2) sets, no soft-stop. The headline
     "On <GPU>, GPAP propagates assumption sets at X sets/s, Y× faster than CPU", plus a
     conflict-count sanity check against PySAT.

  2. **Random-L × soft-stop grid** (the core sweep): for each L in l_values, random L-literal
     assumption sets are streamed in chunks until target_chunks×K sets are processed. Each L is
     run with no cutoff (baseline) and with several soft_stop_frac values (min_rounds=5). We
     report both raw `sets/s` and `successful_sets/s` — finished sets (¬timed_out) ÷ wall — so
     you can see whether an aggressive soft-stop trades a few cut-off batches for more useful
     throughput. CPU is timed on a bounded sample per L and extrapolated to the GPU workload.

K (batch size) auto-sizes to the detected GPU's free VRAM via compute_k_safe (cap=None — use all
the VRAM), so the same code adapts to a 16 GB T4 or an 80 GB A100. The workload then scales as
target_chunks × K, keeping the GPU busy regardless of how large K is.

Results are saved per-GPU to benchmarks/colab_results/<gpu>.json (+ plots) so runs on different
GPUs accumulate and can be ranked with compare_saved().

Headless usage:
    PYTHONPATH=. .venv/bin/python scripts/colab_benchmark.py --quick --device cuda
    PYTHONPATH=. .venv/bin/python scripts/colab_benchmark.py --compare   # rank saved results
"""
import argparse
import glob
import json
import os
import platform
import re
import statistics
import sys
from collections import Counter
from datetime import datetime

os.environ.setdefault("TQDM_DISABLE", "1")  # silence CnfFormula's per-file parse bars

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gpap.clsdata.cnf_formula import CnfFormula
from gpap.utils.gpu_budget import compute_k_safe, gpu_mem_available
from scripts.profile_engine import (
    count_all_pairs,
    run_e2e_cpu,
    run_e2e_cpu_random,
    run_e2e_gpu,
)

DEFAULT_CNF_DIR = os.path.join(REPO_ROOT, "benchmarks", "cnf")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "colab_results")

CPU_SAMPLE_CAP = 50_000     # max PySAT calls used to estimate the CPU per-set cost (then extrapolated)


# ---------------------------------------------------------------------------
# Device / GPU info
# ---------------------------------------------------------------------------

def _resolve_device(device: "str | torch.device | None") -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def gpu_info(device: torch.device) -> dict:
    """Describe the device this run executed on (name, VRAM, CUDA/torch versions)."""
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
    return re.sub(r"[^a-z0-9]+", "-", (name or "cpu").lower()).strip("-") or "cpu"


# ---------------------------------------------------------------------------
# Instance discovery
# ---------------------------------------------------------------------------

def _parse_header(path: str):
    """Read `p cnf N M` from a DIMACS header. Returns (N, M) or None."""
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("p cnf"):
                    parts = s.split()
                    if len(parts) >= 4:
                        return int(parts[2]), int(parts[3])
                if s and not s.startswith("c"):
                    return None
    except Exception:
        return None
    return None


def discover_instances(cnf_dir: str = DEFAULT_CNF_DIR) -> list[dict]:
    """Glob *.cnf in cnf_dir, read headers, return [{path,name,N,M}] sorted by (M, N)."""
    insts = []
    for path in sorted(glob.glob(os.path.join(cnf_dir, "*.cnf"))):
        hdr = _parse_header(path)
        if hdr is None:
            print(f"  [skip] no header: {path}", file=sys.stderr)
            continue
        N, M = hdr
        insts.append({"path": path, "name": os.path.basename(path), "N": N, "M": M})
    insts.sort(key=lambda e: (e["M"], e["N"]))
    return insts


# ---------------------------------------------------------------------------
# Auto-K + GPU run with OOM/spill retry
# ---------------------------------------------------------------------------

def _is_oom(e: BaseException) -> bool:
    """True if an exception is a CUDA out-of-memory error (across torch versions)."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(e, oom_cls) or "out of memory" in str(e).lower()


def _auto_k(M: int, N: int, max_cls: int, total_lits: int,
            device: torch.device) -> tuple[int, float]:
    """Return (K, budget_gb): the largest K this GPU's free VRAM allows (no max_sets cap).

    Uses the full compute_k_safe(cap=None) — high-VRAM GPUs get larger K (more parallelism); the
    workload is scaled separately as target_chunks × K. budget_gb is the allocatable VRAM K was
    sized against; if a run's measured peak exceeds it, K is retried smaller (see _gpu_with_retry).
    """
    budget_bytes = gpu_mem_available(device)
    K = compute_k_safe(M, N, budget_bytes, cap=None,
                       max_clause_size=max_cls, total_lits=total_lits,
                       deduct_formula=False)
    if device.type != "cuda":
        K = min(K, 4096)                   # keep CPU-fallback allocations modest
    return max(1, K), budget_bytes / 1e9


def _gpu_with_retry(mode: str, path: str, formula: CnfFormula, K: int,
                    device: torch.device, budget_gb: float, *, N: int,
                    max_sets: int | None = None, L: int | None = None,
                    target_chunks: int | None = None,
                    soft_stop_frac: float | None = None, min_rounds: int = 0,
                    warmup: int = 2, seed: int = 42, max_retries: int = 5) -> tuple[dict, int]:
    """Run run_e2e_gpu, halving K and retrying on CUDA OOM *or* a measured budget overflow.

    For mode="random" the total workload is target_chunks × K, recomputed each attempt so a
    halved K still does the same number of chunks. A measured peak above budget_gb means the
    memory model under-sized this formula (hard OOM on Linux, silent spill on WSL); either way we
    shrink K. Returns (gpu_result, K_actually_used).
    """
    # Start every performance test from a clean allocator state: return the previous test's
    # freed-but-cached blocks to the driver so each measured run is isolated (no fragmentation
    # carried across the L × soft-stop grid). run_e2e_gpu warms up before timing, so the
    # re-allocation cost is excluded from the measurement.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    res = None
    for attempt in range(max_retries + 1):
        target_sets = target_chunks * K if mode == "random" else None
        try:
            res = run_e2e_gpu(path, mode, K, device, formula=formula, num_warmup=warmup,
                              max_sets=max_sets, L=L, target_sets=target_sets,
                              soft_stop_frac=soft_stop_frac, min_rounds=min_rounds, seed=seed)
        except Exception as e:  # noqa: BLE001 — only OOM is retried; everything else re-raised
            if not _is_oom(e) or K <= 1 or attempt == max_retries:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            K = max(1, K // 2)
            print(f"   [OOM] {mode}: retrying at K={K:,}")
            continue
        # Success — but did we blow the budget (spill risk)? Allow 5% slack: the budget already
        # bakes in headroom (0.75 fraction + 0.8 GB reserve), so a marginal overage is safe.
        if (device.type == "cuda" and K > 1 and attempt < max_retries
                and res["peak_overall_gb"] > budget_gb * 1.05):
            torch.cuda.empty_cache()
            new_K = max(1, K // 2)
            print(f"   [spill] {mode}: peak {res['peak_overall_gb']:.1f} GB > budget "
                  f"{budget_gb:.1f} GB — retrying at K={new_K:,}")
            K = new_K
            continue
        return res, K
    return res, K  # exhausted retries; return last result


# ---------------------------------------------------------------------------
# Per-instance benchmark
# ---------------------------------------------------------------------------

def _pairs_result(gpu: dict, cpu: dict, N: int, K_used: int) -> dict:
    """Build the deterministic pairs-anchor comparison from a GPU + sequential-CPU run."""
    gpu_ms, cpu_ms = gpu["total_ms"], cpu["total_ms"]
    gpu_sps = gpu["total_sets"] / (gpu_ms / 1e3) if gpu_ms else 0.0
    cpu_sps = cpu["total_sets"] / (cpu_ms / 1e3) if cpu_ms else 0.0
    return {
        "total_sets":      gpu["total_sets"],
        "total_sets_cpu":  cpu["total_sets"],
        "total_available": count_all_pairs(N),
        "gpu_ms":          round(gpu_ms, 2),
        "cpu_ms":          round(cpu_ms, 2),
        "gpu_sets_s":      round(gpu_sps, 1),
        "cpu_sets_s":      round(cpu_sps, 1),
        "speedup":         round(cpu_ms / gpu_ms, 2) if gpu_ms else None,
        "gpu_conflicts":   gpu["conflicts_found"],   # real_conflict (BCP-derived) only
        "cpu_conflicts":   cpu["conflicts_found"],   # PySAT: all conflicts
        # GPU's BCP counts only non-trivial (round>1) conflicts; CPU counts all, so GPU <= CPU is
        # expected — the gap is trivial round-1 conflicts, not a disagreement.
        "trivial_conflicts": cpu["conflicts_found"] - gpu["conflicts_found"],
        "conflicts_sane":    gpu["conflicts_found"] <= cpu["conflicts_found"],
        "n_chunks":          gpu["n_chunks"],
        "t_reset_ms":        round(gpu["t_reset_ms"], 2),
        "t_prop_ms":         round(gpu["t_prop_ms"], 2),
        "peak_overall_gb":   round(gpu["peak_overall_gb"], 3),
        "K_used":            K_used,
    }


def _random_setting_result(gpu: dict, cpu_sets_s: float, soft_stop_frac, min_rounds: int,
                           K_used: int) -> dict:
    """Build one row of the random-L × soft-stop grid from a GPU run + the per-L CPU rate."""
    gpu_ms = gpu["total_ms"]
    total = gpu["total_sets"]
    succ = gpu["successful_sets"]
    gpu_sps = total / (gpu_ms / 1e3) if gpu_ms else 0.0
    succ_sps = succ / (gpu_ms / 1e3) if gpu_ms else 0.0
    return {
        "soft_stop_frac":    soft_stop_frac,
        "min_rounds":        min_rounds,
        "total_sets":        total,
        "successful_sets":   succ,
        "timed_out_sets":    gpu["timed_out_sets"],
        "timed_out_frac":    round(gpu["timed_out_sets"] / total, 4) if total else 0.0,
        "gpu_ms":            round(gpu_ms, 2),
        "gpu_sets_s":        round(gpu_sps, 1),
        "successful_sets_s": round(succ_sps, 1),
        # Speedups vs the (soft-stop-independent) CPU per-set rate for this L.
        "speedup":            round(gpu_sps / cpu_sets_s, 2) if cpu_sets_s else None,
        "speedup_successful": round(succ_sps / cpu_sets_s, 2) if cpu_sets_s else None,
        "gpu_conflicts":     gpu["conflicts_found"],
        "conflict_rate":     round(gpu["conflicts_found"] / total, 4) if total else 0.0,
        "n_chunks":          gpu["n_chunks"],
        "peak_overall_gb":   round(gpu["peak_overall_gb"], 3),
        "K_used":            K_used,
    }


def run_instance(inst: dict, device: torch.device, l_values: list[int],
                 soft_stop_fracs: list[float], target_chunks: int,
                 warmup: int, seed: int = 42) -> dict:
    """Benchmark a single CNF: deterministic pairs anchor + random-L × soft-stop e2e grid."""
    if device.type == "cuda":
        # Return the previous instance's freed-but-cached blocks to the driver so
        # gpu_mem_available() (hence auto-K) reflects true free VRAM, not a shrinking pool.
        torch.cuda.empty_cache()
    formula = CnfFormula(device)
    formula.parse_cnf_file(inst["path"])
    N, M = formula.num_vars, formula.num_clauses
    max_cls = int(formula.clause_sizes.max().item())
    total_lits = int(formula.clause_sizes.sum().item())
    K_auto, budget_gb = _auto_k(M, N, max_cls, total_lits, device)
    K = K_auto

    # --- Pairs anchor (deterministic, no soft-stop). GPU + sequential CPU over the same cap. ---
    pairs_cap = min(count_all_pairs(N), target_chunks * K)
    gpu_pairs, K = _gpu_with_retry("pairs", inst["path"], formula, K, device, budget_gb,
                                   N=N, max_sets=pairs_cap, warmup=warmup, seed=seed)
    cpu_pairs = run_e2e_cpu(inst["path"], "pairs", N, max_sets=pairs_cap)
    pairs = _pairs_result(gpu_pairs, cpu_pairs, N, K)

    # --- Random-L × soft-stop grid. CPU once per L (soft-stop independent), then each setting. ---
    settings_list = [None] + list(soft_stop_fracs)   # None = no-cutoff baseline
    l_sweep = []
    for L in l_values:
        if L > N:
            continue
        cpu_sample = min(K, CPU_SAMPLE_CAP)
        cpu = run_e2e_cpu_random(inst["path"], N, L, cpu_sample, seed)
        cpu_sets_s = cpu["total_sets"] / (cpu["total_ms"] / 1e3) if cpu["total_ms"] else 0.0

        rows = []
        for ss in settings_list:
            mr = 5 if ss is not None else 0
            gpu, K = _gpu_with_retry("random", inst["path"], formula, K, device, budget_gb,
                                     N=N, L=L, target_chunks=target_chunks,
                                     soft_stop_frac=ss, min_rounds=mr, warmup=warmup, seed=seed)
            rows.append(_random_setting_result(gpu, cpu_sets_s, ss, mr, K))
        l_sweep.append({
            "L": L,
            "cpu_sample_sets": cpu["total_sets"],
            "cpu_sets_s":      round(cpu_sets_s, 1),
            "cpu_conflicts":   cpu["conflicts_found"],
            "settings":        rows,
        })

    return {
        "name": inst["name"], "path": inst["path"], "N": N, "M": M,
        "max_cls": max_cls, "total_lits": total_lits,
        "K": K, "K_auto": K_auto, "budget_gb": round(budget_gb, 2),
        "pairs": pairs, "l_sweep": l_sweep,
    }


# ---------------------------------------------------------------------------
# Aggregation + headline
# ---------------------------------------------------------------------------

def _best_successful(l_entry: dict) -> dict:
    """Return the setting row with the highest successful_sets_s for one L entry."""
    return max(l_entry["settings"], key=lambda r: r["successful_sets_s"])


def aggregate(results: list[dict]) -> dict:
    """Summarize the pairs anchor (headline) + random-L grid (baseline vs best soft-stop)."""
    pairs = [r["pairs"] for r in results if r.get("pairs")]
    sps     = [p["gpu_sets_s"] for p in pairs]
    cpu_sps = [p["cpu_sets_s"] for p in pairs]
    spd     = [p["speedup"] for p in pairs if p["speedup"] is not None]

    base_spd, best_succ_spd, best_fracs = [], [], []
    for r in results:
        for le in r.get("l_sweep", []):
            base = next((s for s in le["settings"] if s["soft_stop_frac"] is None), None)
            if base and base["speedup"] is not None:
                base_spd.append(base["speedup"])
            best = _best_successful(le)
            if best["speedup_successful"] is not None:
                best_succ_spd.append(best["speedup_successful"])
            best_fracs.append(best["soft_stop_frac"])  # None if baseline won

    frac_counts = Counter(best_fracs)
    best_overall = frac_counts.most_common(1)[0][0] if frac_counts else None
    return {
        "n_instances":                  len(results),
        "pairs_gpu_sets_s_peak":        round(max(sps), 1) if sps else 0.0,
        "pairs_gpu_sets_s_median":      round(statistics.median(sps), 1) if sps else 0.0,
        "pairs_cpu_sets_s_median":      round(statistics.median(cpu_sps), 1) if cpu_sps else 0.0,
        "pairs_speedup_median":         round(statistics.median(spd), 2) if spd else 0.0,
        "pairs_speedup_max":            round(max(spd), 2) if spd else 0.0,
        "lsweep_baseline_speedup_median": round(statistics.median(base_spd), 2) if base_spd else 0.0,
        "lsweep_baseline_speedup_max":    round(max(base_spd), 2) if base_spd else 0.0,
        "lsweep_best_successful_speedup_median": round(statistics.median(best_succ_spd), 2) if best_succ_spd else 0.0,
        "lsweep_best_successful_speedup_max":    round(max(best_succ_spd), 2) if best_succ_spd else 0.0,
        "softstop_best_frac_overall":   best_overall,   # None means "no cutoff usually wins"
        "all_conflicts_sane":           all(p["conflicts_sane"] for p in pairs) if pairs else None,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_gpu_info(info: dict) -> None:
    vram = (f"{info['total_gb']:.1f} GB ({info['free_gb']:.1f} free)"
            if info.get("total_gb") else "n/a")
    print(f"\nDevice : {info['name']}")
    print(f"  VRAM={vram}  cc={info.get('capability')}  "
          f"torch={info['torch_version']}  cuda={info['cuda_version']}")


def _fmt_frac(f) -> str:
    return "off" if f is None else f"{f:g}"


def _print_instance_line(r: dict) -> None:
    p = r["pairs"]
    spd = f"{p['speedup']:.1f}x" if p["speedup"] is not None else "—"
    kflag = f"(↓{r['K_auto']:,})" if r.get("K_auto", r["K"]) > r["K"] else ""
    # Best soft-stop result across the L grid (by successful_sets/s speedup).
    best_txt = ""
    best = None
    for le in r.get("l_sweep", []):
        b = _best_successful(le)
        if best is None or b["successful_sets_s"] > best[1]["successful_sets_s"]:
            best = (le["L"], b)
    if best is not None:
        L, b = best
        best_txt = (f"  L-grid best: L={L} ss={_fmt_frac(b['soft_stop_frac'])} "
                    f"{b['successful_sets_s']:,.0f} succ-sets/s ({b['speedup_successful']:.1f}x)")
    print(f"   K={r['K']:>8}{kflag}  pairs: GPU={p['gpu_sets_s']:>12,.0f} sets/s  "
          f"speedup={spd:>6}  peak={p['peak_overall_gb']:.2f}GB{best_txt}")


def _print_table(results: list[dict]) -> None:
    good = [r for r in results if "error" not in r]
    if good:
        print(f"\nPairs anchor (L=2 exhaustive, GPU vs sequential CPU PySAT):")
        print(f"{'Instance':<34} {'N':>5} {'M':>6} {'K':>8} "
              f"{'GPU sets/s':>13} {'CPU sets/s':>12} {'speedup':>8} {'sane':>5}")
        print("-" * 100)
        for r in good:
            p = r["pairs"]
            name = r["name"]
            name = ("…" + name[-31:]) if len(name) > 32 else name
            spd = f"{p['speedup']:.1f}x" if p["speedup"] is not None else "—"
            sane = "ok" if p["conflicts_sane"] else "!!"
            print(f"{name:<34} {r['N']:>5} {r['M']:>6} {r['K']:>8} "
                  f"{p['gpu_sets_s']:>13,.0f} {p['cpu_sets_s']:>12,.0f} {spd:>8} {sane:>5}")

        # Random-L grid: baseline vs best soft-stop, median across instances, per L.
        all_L = sorted({le["L"] for r in good for le in r.get("l_sweep", [])})
        if all_L:
            print(f"\nRandom-L e2e — median speedup vs CPU (baseline → best soft-stop succ.), per L:")
            for L in all_L:
                base, best = [], []
                for r in good:
                    for le in r["l_sweep"]:
                        if le["L"] != L:
                            continue
                        b = next((s for s in le["settings"] if s["soft_stop_frac"] is None), None)
                        if b and b["speedup"] is not None:
                            base.append(b["speedup"])
                        bs = _best_successful(le)
                        if bs["speedup_successful"] is not None:
                            best.append(bs["speedup_successful"])
                if base:
                    bm = statistics.median(base)
                    sm = statistics.median(best) if best else bm
                    print(f"   L={L:<3}  baseline {bm:5.1f}x   →   best soft-stop {sm:5.1f}x (successful)")

    for r in (r for r in results if "error" in r):
        print(f"{r.get('name','?'):<34} ERROR: {r['error']}")


def _print_headline(payload: dict) -> None:
    a = payload["aggregate"]
    name = payload["gpu"]["name"]
    print("\n" + "=" * 72)
    print("HEADLINE — set-propagation throughput (exhaustive pairs, GPU vs CPU)")
    print("=" * 72)
    print(f"  GPU                : {name}")
    print(f"  Peak GPU throughput: {a['pairs_gpu_sets_s_peak']:>14,.0f} sets/s")
    print(f"  Median GPU         : {a['pairs_gpu_sets_s_median']:>14,.0f} sets/s")
    print(f"  Median CPU (PySAT) : {a['pairs_cpu_sets_s_median']:>14,.0f} sets/s")
    print(f"  Median speedup     : {a['pairs_speedup_median']:>13.1f}x   "
          f"(max {a['pairs_speedup_max']:.1f}x)")
    print(f"  Random-L baseline  : {a['lsweep_baseline_speedup_median']:>13.1f}x   "
          f"(max {a['lsweep_baseline_speedup_max']:.1f}x)")
    print(f"  Random-L soft-stop : {a['lsweep_best_successful_speedup_median']:>13.1f}x   "
          f"(max {a['lsweep_best_successful_speedup_max']:.1f}x, successful_sets/s; "
          f"best frac={_fmt_frac(a['softstop_best_frac_overall'])})")
    print(f"\n  → On {name}, GPAP propagates assumption sets at up to "
          f"{a['pairs_gpu_sets_s_peak']:,.0f} sets/s,")
    print(f"    a median {a['pairs_speedup_median']:.1f}× over a sequential CPU baseline.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_all(quick: bool = False, device=None, cnf_dir: str = DEFAULT_CNF_DIR,
            l_values: list[int] | None = None, soft_stop_fracs: list[float] | None = None,
            target_chunks: int | None = None, limit: int | None = None,
            warmup: int = 2, seed: int = 42, do_plots: bool = True,
            out_dir: str = DEFAULT_OUT_DIR, verbose: bool = True) -> dict:
    """Discover bundled CNFs, benchmark each, aggregate, and return a result payload.

    quick=True trims the instance set, lowers target_chunks, and shrinks the L/soft-stop grid so a
    run finishes in a few minutes. The full run sweeps every instance, L in {2,5,10,20}, and
    soft_stop_frac in {0.9,0.7,0.5,0.3,0.1} (+ baseline) — minutes to ~1 h depending on the GPU.
    """
    device = _resolve_device(device)
    insts = discover_instances(cnf_dir)

    if quick:
        insts = insts[::2]                       # every other instance (still size-spread)
        l_values        = [2, 10]            if l_values        is None else l_values
        soft_stop_fracs = [0.5, 0.1]         if soft_stop_fracs is None else soft_stop_fracs
        target_chunks   = 6                  if target_chunks   is None else target_chunks
        warmup          = max(1, warmup - 1)
    else:
        l_values        = [2, 5, 10, 20]              if l_values        is None else l_values
        soft_stop_fracs = [0.9, 0.7, 0.5, 0.3, 0.1]   if soft_stop_fracs is None else soft_stop_fracs
        target_chunks   = 32                          if target_chunks   is None else target_chunks
    if limit is not None:
        insts = insts[:limit]

    info = gpu_info(device)
    if verbose:
        _print_gpu_info(info)
        print(f"  instances={len(insts)}  target_chunks={target_chunks}  "
              f"L-values={l_values}  soft_stop_fracs={soft_stop_fracs}\n")

    results = []
    for i, inst in enumerate(insts, 1):
        if verbose:
            print(f"[{i}/{len(insts)}] {inst['name'][:50]}  N={inst['N']} M={inst['M']} ...",
                  flush=True)
        try:
            r = run_instance(inst, device, l_values, soft_stop_fracs, target_chunks, warmup, seed)
            results.append(r)
            if verbose:
                _print_instance_line(r)
        except Exception as e:  # keep going; record the failure
            print(f"   ERROR: {e}")
            results.append({**inst, "error": str(e)})

    agg = aggregate([r for r in results if "error" not in r])
    payload = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "gpu":             info,
        "quick":           quick,
        "target_chunks":   target_chunks,
        "l_values":        l_values,
        "soft_stop_fracs": soft_stop_fracs,
        "aggregate":       agg,
        "instances":       results,
    }
    if verbose:
        _print_table(results)
        _print_headline(payload)
    if do_plots:
        try:
            from scripts.colab_plots import make_plots
            paths = make_plots(payload, out_dir)
            if verbose and paths:
                print("\nPlots:")
                for p in paths:
                    print(f"  {p}")
        except Exception as e:  # plotting is best-effort
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
    """Load all saved per-GPU results and print a ranking by peak pairs throughput."""
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        a = d.get("aggregate", {})
        rows.append({
            "gpu":            d.get("gpu", {}).get("name", os.path.basename(p)),
            "vram_gb":        d.get("gpu", {}).get("total_gb"),
            "peak_sets_s":    a.get("pairs_gpu_sets_s_peak", 0.0),
            "median_sets_s":  a.get("pairs_gpu_sets_s_median", 0.0),
            "speedup_median": a.get("pairs_speedup_median", 0.0),
            "timestamp":      d.get("timestamp"),
        })
    rows.sort(key=lambda r: r["peak_sets_s"], reverse=True)

    if not rows:
        print(f"No saved results in {out_dir}")
        return rows

    base = rows[-1]["peak_sets_s"] or 1.0
    print(f"\n{'GPU':<28} {'VRAM':>7} {'peak sets/s':>14} {'median sets/s':>14} "
          f"{'vs CPU':>8} {'rel':>6}")
    print("-" * 84)
    for r in rows:
        vram = f"{r['vram_gb']:.0f}GB" if r["vram_gb"] else "—"
        rel = r["peak_sets_s"] / base if base else 0.0
        print(f"{r['gpu']:<28} {vram:>7} {r['peak_sets_s']:>14,.0f} "
              f"{r['median_sets_s']:>14,.0f} {r['speedup_median']:>7.1f}x {rel:>5.1f}x")
    print(f"\n('rel' = peak throughput relative to the slowest saved GPU)")
    return rows


def _parse_fracs(s: str) -> list[float]:
    """Parse a comma list of soft_stop_fracs; 'none'/'off' tokens are dropped (baseline is implicit)."""
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok or tok in ("none", "off", "baseline"):
            continue
        out.append(float(tok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="GPAP cross-GPU benchmark driver")
    ap.add_argument("--quick", action="store_true",
                    help="Fast run: fewer instances, smaller target_chunks, trimmed L/soft-stop grid")
    ap.add_argument("--device", default=None, help="torch device (cuda/cpu); default auto")
    ap.add_argument("--target-chunks", type=int, default=None,
                    help="Total sets per random-L run = target_chunks × K (default 32 full / 6 quick)")
    ap.add_argument("--l-values", default=None,
                    help="Comma list of assumption-set sizes L (e.g. 2,5,10,20)")
    ap.add_argument("--soft-stop-fracs", default=None,
                    help="Comma list of soft_stop_frac values (baseline is always included)")
    ap.add_argument("--cnf-dir", default=DEFAULT_CNF_DIR, help="Directory of bundled .cnf files")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Where per-GPU result JSON/plots go")
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N instances")
    ap.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    ap.add_argument("--no-save", action="store_true", help="Do not write a result JSON")
    ap.add_argument("--compare", action="store_true",
                    help="Only print a ranking of already-saved results, then exit")
    args = ap.parse_args()

    if args.compare:
        compare_saved(args.out_dir)
        return

    l_values = [int(x) for x in args.l_values.split(",")] if args.l_values else None
    soft_stop_fracs = _parse_fracs(args.soft_stop_fracs) if args.soft_stop_fracs else None

    payload = run_all(quick=args.quick, device=args.device, cnf_dir=args.cnf_dir,
                      l_values=l_values, soft_stop_fracs=soft_stop_fracs,
                      target_chunks=args.target_chunks, limit=args.limit,
                      do_plots=not args.no_plot, out_dir=args.out_dir)
    if not args.no_save:
        save_results(payload, args.out_dir)
        compare_saved(args.out_dir)


if __name__ == "__main__":
    main()

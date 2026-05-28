#!/usr/bin/env python3
"""Cross-GPU benchmark driver for GPAP — designed to run on Colab (any GPU) or locally.

Headline metric: **set-propagation throughput, GPU vs CPU.** For each bundled CNF we run
the full GPAP pipeline over all (capped) ± variable *pairs* and compare wall time against a
sequential CPU PySAT loop over the identical set of pairs:

    speedup = CPU_wall_ms / GPU_wall_ms        (GPU sets/s ÷ CPU sets/s)

This is the number to push: "On <GPU>, GPAP propagates assumption sets at X sets/s,
Y× faster than CPU." Triplets and a random-L assumption-size sweep are reported as supporting
context. Results are saved per-GPU to benchmarks/colab_results/<gpu>.json so runs on different
GPUs accumulate and can be ranked with compare_saved().

K (batch size) is auto-sized to the detected GPU's free VRAM via compute_k_safe, so the same
code adapts to a 16 GB T4 or an 80 GB A100 with no changes.

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
    count_all_triplets,
    generate_random_assumptions,
    profile_external,
    profile_gpap,
    run_e2e_cpu,
    run_e2e_gpu,
)

DEFAULT_CNF_DIR = os.path.join(REPO_ROOT, "benchmarks", "cnf")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "colab_results")


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
# Per-instance benchmark
# ---------------------------------------------------------------------------

def _is_oom(e: BaseException) -> bool:
    """True if an exception is a CUDA out-of-memory error (across torch versions)."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(e, oom_cls) or "out of memory" in str(e).lower()


def _auto_k(M: int, N: int, max_cls: int, total_lits: int,
            device: torch.device, max_sets: int | None) -> tuple[int, float]:
    """Return (K, budget_gb): largest K for this GPU's free VRAM and the budget it targets.

    budget_gb is the allocatable VRAM the K was sized against; if a run's measured peak exceeds
    it, the formula's prop-peak was under-modelled and K is retried smaller (see _e2e_with_retry).
    """
    budget_bytes = gpu_mem_available(device)
    K = compute_k_safe(M, N, budget_bytes, cap=None,
                       max_clause_size=max_cls, total_lits=total_lits,
                       deduct_formula=False)
    if max_sets is not None:
        K = min(K, max_sets)               # no point allocating beyond the capped workload
    if device.type != "cuda":
        K = min(K, 4096)                   # keep CPU-fallback allocations modest
    return max(1, K), budget_bytes / 1e9


def _e2e_with_retry(mode: str, path: str, formula: CnfFormula, K: int,
                    device: torch.device, max_sets: int | None, warmup: int, N: int,
                    budget_gb: float, max_retries: int = 5) -> tuple[dict, int]:
    """Run _e2e_compare, halving K and retrying on CUDA OOM *or* a measured budget overflow.

    A measured peak above budget_gb means the memory model under-sized this formula; on Linux
    that path OOMs, on WSL it silently spills to system RAM (≈10× slower). Either way we shrink
    K so the run fits in VRAM. Returns (result, K_actually_used).
    """
    for attempt in range(max_retries + 1):
        try:
            res = _e2e_compare(mode, path, formula, K, device, max_sets, warmup, N)
        except Exception as e:  # noqa: BLE001 — only OOM is retried; everything else re-raised
            if not _is_oom(e) or K <= 1 or attempt == max_retries:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            K = max(1, K // 2)
            print(f"   [OOM] {mode}: retrying at K={K:,}")
            continue
        # Success — but did we blow the budget (spill risk)? Allow 5% slack: the budget
        # already bakes in headroom (0.75 fraction + 0.8 GB reserve), so a marginal overage
        # is safe and shouldn't force a full K halving — only real overflows retry.
        if (device.type == "cuda" and K > 1 and attempt < max_retries
                and res["peak_overall_gb"] > budget_gb * 1.05):
            torch.cuda.empty_cache()
            new_K = max(1, K // 2)
            print(f"   [spill] {mode}: peak {res['peak_overall_gb']:.1f} GB > budget "
                  f"{budget_gb:.1f} GB — retrying at K={new_K:,}")
            K = new_K
            continue
        res["K_used"] = K
        return res, K
    return res, K  # exhausted retries; return last result


def _e2e_compare(mode: str, path: str, formula: CnfFormula, K: int,
                 device: torch.device, max_sets: int | None, warmup: int, N: int) -> dict:
    """Run all (capped) pairs/triplets on GPU and on a sequential CPU PySAT loop; compare."""
    total_avail = count_all_pairs(N) if mode == "pairs" else count_all_triplets(N)
    gpu = run_e2e_gpu(path, mode, K, device, formula=formula,
                      num_warmup=warmup, max_sets=max_sets)
    cpu = run_e2e_cpu(path, mode, N, max_sets=max_sets)

    gpu_ms, cpu_ms = gpu["total_ms"], cpu["total_ms"]
    gpu_sps = gpu["total_sets"] / (gpu_ms / 1e3) if gpu_ms else 0.0
    cpu_sps = cpu["total_sets"] / (cpu_ms / 1e3) if cpu_ms else 0.0
    return {
        "total_sets":      gpu["total_sets"],
        "total_sets_cpu":  cpu["total_sets"],
        "total_available": total_avail,
        "gpu_ms":          round(gpu_ms, 2),
        "cpu_ms":          round(cpu_ms, 2),
        "gpu_sets_s":      round(gpu_sps, 1),
        "cpu_sets_s":      round(cpu_sps, 1),
        "speedup":         round(cpu_ms / gpu_ms, 2) if gpu_ms else None,
        "gpu_conflicts":   gpu["conflicts_found"],   # real_conflict (BCP-derived) only
        "cpu_conflicts":   cpu["conflicts_found"],   # PySAT: all conflicts
        # GPU's BCP counts only non-trivial (round>1) conflicts; CPU counts all, so
        # GPU <= CPU is expected — the gap is trivial round-1 conflicts, not a disagreement.
        "trivial_conflicts": cpu["conflicts_found"] - gpu["conflicts_found"],
        "conflicts_sane":    gpu["conflicts_found"] <= cpu["conflicts_found"],
        "n_chunks":          gpu["n_chunks"],
        "t_reset_ms":        round(gpu["t_reset_ms"], 2),
        "t_prop_ms":         round(gpu["t_prop_ms"], 2),
        "peak_overall_gb":   round(gpu["peak_overall_gb"], 3),
    }


def _l_sweep(path: str, N: int, K: int, device: torch.device,
             l_values: list[int], warmup: int, seed: int,
             pysat_cap: int = 50, runs: int = 5) -> list[dict]:
    """Random-L assumptions: GPAP bulk vs PySAT serial (run on <=pysat_cap, extrapolated to K)."""
    K_sweep = min(K, 65536)
    rows = []
    for L in l_values:
        if L > N:
            continue
        assumptions = generate_random_assumptions(N, K_sweep, L, seed)
        gpap = profile_gpap(path, assumptions, device, warmup, runs)
        K_ext = min(K_sweep, pysat_cap)
        ext = profile_external(path, assumptions[:K_ext], min(runs, 3))
        pysat_per_batch = ext["mean_ms"] / K_ext
        pysat_extrap = pysat_per_batch * K_sweep
        rows.append({
            "L":                  L,
            "K":                  K_sweep,
            "gpap_ms":            round(gpap["mean_ms"], 3),
            "gpap_per_batch_us":  round(gpap["mean_ms"] / K_sweep * 1e3, 3),
            "pysat_per_batch_ms": round(pysat_per_batch, 4),
            "pysat_extrap_ms":    round(pysat_extrap, 2),
            "speedup_vs_pysat":   round(pysat_extrap / gpap["mean_ms"], 2) if gpap["mean_ms"] else None,
            "conflict_rate":      round(gpap["conflict_rate"], 4),
        })
    return rows


def run_instance(inst: dict, device: torch.device, max_sets: int | None,
                 l_sweep: list[int], do_triplets: bool, warmup: int, seed: int = 42) -> dict:
    """Benchmark a single CNF: headline pairs (+ optional triplets) + random-L sweep."""
    if device.type == "cuda":
        # Return the previous instance's freed-but-cached blocks to the driver so
        # gpu_mem_available() (hence auto-K) reflects true free VRAM, not a shrinking pool.
        torch.cuda.empty_cache()
    formula = CnfFormula(device)
    formula.parse_cnf_file(inst["path"])
    N, M = formula.num_vars, formula.num_clauses
    max_cls = int(formula.clause_sizes.max().item())
    total_lits = int(formula.clause_sizes.sum().item())
    K_auto, budget_gb = _auto_k(M, N, max_cls, total_lits, device, max_sets)

    # Headline pairs first; K may shrink to fit VRAM, and the reduced K carries into the
    # later phases so they don't repeat the same OOM/spill.
    pairs, K = _e2e_with_retry("pairs", inst["path"], formula, K_auto, device,
                               max_sets, warmup, N, budget_gb)
    triplets = None
    if do_triplets:
        triplets, K = _e2e_with_retry("triplets", inst["path"], formula, K, device,
                                      max_sets, warmup, N, budget_gb)
    l_sweep = _l_sweep(inst["path"], N, K, device, l_sweep, warmup, seed)

    return {
        "name": inst["name"], "path": inst["path"], "N": N, "M": M,
        "max_cls": max_cls, "total_lits": total_lits,
        "K": K, "K_auto": K_auto, "budget_gb": round(budget_gb, 2),
        "pairs": pairs, "triplets": triplets, "l_sweep": l_sweep,
    }


# ---------------------------------------------------------------------------
# Aggregation + headline
# ---------------------------------------------------------------------------

def aggregate(results: list[dict]) -> dict:
    """Summarize per-instance pairs results (headline) + L-sweep (supporting) into one dict."""
    pairs = [r["pairs"] for r in results if r.get("pairs")]
    sps     = [p["gpu_sets_s"] for p in pairs]
    cpu_sps = [p["cpu_sets_s"] for p in pairs]
    spd     = [p["speedup"] for p in pairs if p["speedup"] is not None]
    lswp    = [row["speedup_vs_pysat"] for r in results for row in r.get("l_sweep", [])
               if row.get("speedup_vs_pysat") is not None]
    return {
        "n_instances":             len(results),
        "pairs_gpu_sets_s_peak":   round(max(sps), 1) if sps else 0.0,
        "pairs_gpu_sets_s_median": round(statistics.median(sps), 1) if sps else 0.0,
        "pairs_cpu_sets_s_median": round(statistics.median(cpu_sps), 1) if cpu_sps else 0.0,
        "pairs_speedup_median":    round(statistics.median(spd), 2) if spd else 0.0,
        "pairs_speedup_max":       round(max(spd), 2) if spd else 0.0,
        "lsweep_speedup_median":   round(statistics.median(lswp), 2) if lswp else 0.0,
        "lsweep_speedup_max":      round(max(lswp), 2) if lswp else 0.0,
        "all_conflicts_sane":      all(p["conflicts_sane"] for p in pairs) if pairs else None,
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


def _print_instance_line(r: dict) -> None:
    p = r["pairs"]
    spd = f"{p['speedup']:.1f}x" if p["speedup"] is not None else "—"
    lbest = max((row["speedup_vs_pysat"] for row in r.get("l_sweep", [])
                 if row.get("speedup_vs_pysat") is not None), default=None)
    lbest_s = f"  L-sweep≤{lbest:.1f}x" if lbest is not None else ""
    kflag = f"(↓{r['K_auto']:,})" if r.get("K_auto", r["K"]) > r["K"] else ""
    print(f"   K={r['K']:>7}{kflag}  pairs: GPU={p['gpu_sets_s']:>12,.0f} sets/s  "
          f"CPU={p['cpu_sets_s']:>10,.0f} sets/s  speedup={spd:>6}{lbest_s}  "
          f"chunks={p['n_chunks']}  peak={p['peak_overall_gb']:.2f}GB")


def _print_table(results: list[dict]) -> None:
    good = [r for r in results if "error" not in r]
    if good:
        print(f"\nPairs (L=2, exhaustive, GPU vs sequential CPU PySAT loop):")
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

        # L-sweep: GPAP bulk vs extrapolated PySAT, by assumption-set size.
        all_L = sorted({row["L"] for r in good for row in r.get("l_sweep", [])})
        if all_L:
            print(f"\nRandom-L sweep — median GPAP speedup vs PySAT, by assumptions/set:")
            cells = []
            for L in all_L:
                vals = [row["speedup_vs_pysat"] for r in good for row in r["l_sweep"]
                        if row["L"] == L and row.get("speedup_vs_pysat") is not None]
                if vals:
                    cells.append(f"L={L}: {statistics.median(vals):.1f}x")
            print("   " + "   ".join(cells))

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
    print(f"  L-sweep speedup    : {a['lsweep_speedup_median']:>13.1f}x   "
          f"(max {a['lsweep_speedup_max']:.1f}x, larger assumption sets)")
    print(f"\n  → On {name}, GPAP propagates assumption sets at up to "
          f"{a['pairs_gpu_sets_s_peak']:,.0f} sets/s,")
    print(f"    a median {a['pairs_speedup_median']:.1f}× over a sequential CPU baseline "
          f"(up to {a['lsweep_speedup_max']:.1f}× for larger sets).")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_all(quick: bool = False, device=None, cnf_dir: str = DEFAULT_CNF_DIR,
            max_sets: int | None = None, do_triplets: bool | None = None,
            limit: int | None = None, warmup: int = 2, seed: int = 42,
            verbose: bool = True) -> dict:
    """Discover bundled CNFs, benchmark each, aggregate, and return a result payload.

    quick=True trims the instance set, lowers max_sets, drops triplets, and shrinks the L-sweep
    so a full run finishes in well under a minute on a modern GPU.
    """
    device = _resolve_device(device)
    insts = discover_instances(cnf_dir)

    if quick:
        insts = insts[::2]                       # every other instance (still size-spread)
        max_sets    = 200_000 if max_sets    is None else max_sets
        do_triplets = False   if do_triplets is None else do_triplets
        l_sweep     = [2, 10]
        warmup      = max(1, warmup - 1)
    else:
        max_sets    = 500_000 if max_sets    is None else max_sets
        do_triplets = True    if do_triplets is None else do_triplets
        l_sweep     = [2, 5, 10, 20]
    if limit is not None:
        insts = insts[:limit]

    info = gpu_info(device)
    if verbose:
        _print_gpu_info(info)
        print(f"  instances={len(insts)}  max_sets={max_sets:,}  "
              f"triplets={do_triplets}  L-sweep={l_sweep}\n")

    results = []
    for i, inst in enumerate(insts, 1):
        if verbose:
            print(f"[{i}/{len(insts)}] {inst['name'][:50]}  N={inst['N']} M={inst['M']} ...",
                  flush=True)
        try:
            r = run_instance(inst, device, max_sets, l_sweep, do_triplets, warmup, seed)
            results.append(r)
            if verbose:
                _print_instance_line(r)
        except Exception as e:  # keep going; record the failure
            print(f"   ERROR: {e}")
            results.append({**inst, "error": str(e)})

    agg = aggregate([r for r in results if "error" not in r])
    payload = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "gpu":         info,
        "quick":       quick,
        "max_sets":    max_sets,
        "do_triplets": do_triplets,
        "l_sweep":     l_sweep,
        "aggregate":   agg,
        "instances":   results,
    }
    if verbose:
        _print_table(results)
        _print_headline(payload)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="GPAP cross-GPU benchmark driver")
    ap.add_argument("--quick", action="store_true",
                    help="Fast run: fewer instances, smaller caps, no triplets")
    ap.add_argument("--device", default=None, help="torch device (cuda/cpu); default auto")
    ap.add_argument("--max-sets", type=int, default=None,
                    help="Cap on pairs/triplets processed per instance (GPU and CPU alike)")
    ap.add_argument("--cnf-dir", default=DEFAULT_CNF_DIR, help="Directory of bundled .cnf files")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Where per-GPU result JSON is written")
    ap.add_argument("--no-triplets", action="store_true", help="Skip the triplets comparison")
    ap.add_argument("--triplets", action="store_true", help="Force the triplets comparison on")
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N instances")
    ap.add_argument("--no-save", action="store_true", help="Do not write a result JSON")
    ap.add_argument("--compare", action="store_true",
                    help="Only print a ranking of already-saved results, then exit")
    args = ap.parse_args()

    if args.compare:
        compare_saved(args.out_dir)
        return

    do_triplets = None
    if args.no_triplets:
        do_triplets = False
    if args.triplets:
        do_triplets = True

    payload = run_all(quick=args.quick, device=args.device, cnf_dir=args.cnf_dir,
                      max_sets=args.max_sets, do_triplets=do_triplets, limit=args.limit)
    if not args.no_save:
        save_results(payload, args.out_dir)
        compare_saved(args.out_dir)


if __name__ == "__main__":
    main()

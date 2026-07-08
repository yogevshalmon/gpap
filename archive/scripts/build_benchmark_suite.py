#!/usr/bin/env python3
"""Scan CNF benchmark directories, compute per-instance K/L parameters, and write a suite manifest.

total_lits_est = AVG_LITS_PER_CLAUSE * M  (key driver for both K and L)

K (batch size) is memory-bounded by peak allocation (static tensors + intermediates):
  formula_bytes   = 8*M + 16*N + 8*total_lits_est   (one-time device allocation)
  per_batch_bytes = 6*M + 10*N + 3                  (static 5M+2N + reset temp M + apply temps 8N)
  K = min(floor((GPU_MEM - formula_bytes) / per_batch_bytes), 2048)

L (assumptions per batch) = round(log2(total_lits_est)) — scales with formula complexity
rather than variable count, capturing both clause count and average clause size.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/build_benchmark_suite.py \\
        --dirs /home/yshalmon/sat_benchmarks/sat24_benchmarks \\
               /home/yshalmon/sat_benchmarks/sat2006_benchmarks \\
               /home/yshalmon/sat_benchmarks/RTI_k3_n100_m429 \\
               /home/yshalmon/sat_benchmarks/uf20-91 \\
        --output benchmarks/suite.json --tier fast
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, ".")
from gpap.utils.gpu_budget import (
    AVG_LITS_PER_CLAUSE as _AVG_LITS_PER_CLAUSE,
    compute_k_safe,
    formula_mem_bytes,
    per_batch_mem_bytes,
)

# Per-tier selection limits: how many instances and max file size per family.
# "_default" applies to any family not explicitly listed.
_TIER_CONFIG: dict[str, dict[str, dict]] = {
    "fast": {
        "uf20":       {"max_files": None, "max_mb": None},
        "pret_unsat": {"max_files": None, "max_mb": None},
        "RTI_k3":     {"max_files": 20,   "max_mb": None},
        "blocksworld":{"max_files": 10,   "max_mb": None},
        "pigeon":     {"max_files": 10,   "max_mb": None},
        "sat2006":    {"max_files": 10,   "max_mb": 10},
        "sat24":      {"max_files": 5,    "max_mb": 10},
        "_default":   {"max_files": 5,    "max_mb": 5},
    },
    "full": {
        "uf20":       {"max_files": None, "max_mb": None},
        "pret_unsat": {"max_files": None, "max_mb": None},
        "RTI_k3":     {"max_files": None, "max_mb": None},
        "blocksworld":{"max_files": None, "max_mb": None},
        "pigeon":     {"max_files": None, "max_mb": None},
        "sat2006":    {"max_files": 100,  "max_mb": 100},
        "sat24":      {"max_files": 50,   "max_mb": 100},
        "_default":   {"max_files": 20,   "max_mb": 50},
    },
}

_FAMILY_ALIASES = {
    "sat24_benchmarks":  "sat24",
    "sat2006_benchmarks":"sat2006",
    "RTI_k3_n100_m429":  "RTI_k3",
    "uf20-91":           "uf20",
    "pret_unsat":        "pret_unsat",
    "blocksworld":       "blocksworld",
    "pigeon-hole":       "pigeon",
}


def _parse_header(path: str):
    """Read `p cnf N M` from a DIMACS file header. Returns (N, M) or None."""
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("p cnf"):
                    parts = s.split()
                    if len(parts) >= 4:
                        return int(parts[2]), int(parts[3])
                if s and not s.startswith("c"):
                    return None  # hit clause data before header
    except Exception:
        return None
    return None


def _compute_params(N: int, M: int, gpu_mem_bytes: float):
    """Return (K, L, K_safe, total_lits_est, est_mem_mb) or None if instance doesn't fit."""
    total_lits = int(_AVG_LITS_PER_CLAUSE * M)
    f_bytes = formula_mem_bytes(M, N)
    p_bytes = per_batch_mem_bytes(M, N)

    available = gpu_mem_bytes - f_bytes
    if available <= 0:
        return None

    K_safe = int(available / p_bytes)
    K = compute_k_safe(M, N, gpu_mem_bytes, cap=65536)
    # L scales with log2(total_lits): captures both clause count and average clause size.
    # Gives more assumptions than log2(N) for small dense formulas, same cap for large ones.
    L = max(1, min(15, round(math.log2(max(total_lits, 2)))))
    est_mem_mb = (f_bytes + K * p_bytes) / 1e6
    return K, L, K_safe, total_lits, est_mem_mb


def _family(search_dir: str) -> str:
    base = os.path.basename(os.path.normpath(search_dir))
    return _FAMILY_ALIASES.get(base, base)


def scan(search_dirs: list[str], gpu_mem_bytes: float, min_k: int,
         max_m: int | None = None):
    """Yield qualifying instance dicts from all search directories."""
    for search_dir in search_dirs:
        fam = _family(search_dir)
        for root, dirs, files in os.walk(search_dir):
            dirs.sort()
            for fname in sorted(files):
                if not fname.endswith(".cnf"):
                    continue
                path = os.path.join(root, fname)
                try:
                    size_bytes = os.path.getsize(path)
                except OSError:
                    continue
                header = _parse_header(path)
                if header is None:
                    print(f"  [skip] no header: {path}", file=sys.stderr)
                    continue
                N, M = header
                if max_m is not None and M > max_m:
                    continue
                params = _compute_params(N, M, gpu_mem_bytes)
                if params is None:
                    print(f"  [skip] too large for GPU: {path}", file=sys.stderr)
                    continue
                K, L, K_safe, total_lits, est_mem_mb = params
                if K_safe < min_k:
                    print(f"  [skip] K_safe={K_safe} < {min_k}: {path}", file=sys.stderr)
                    continue
                yield {
                    "path":          path,
                    "family":        fam,
                    "N":             N,
                    "M":             M,
                    "total_lits_est": total_lits,
                    "K":             K,
                    "L":             L,
                    "K_safe":        K_safe,
                    "est_mem_mb":    round(est_mem_mb, 2),
                    "size_mb":       round(size_bytes / 1e6, 2),
                }


def select(candidates: list[dict], tier: str, no_limit: bool = False) -> list[dict]:
    """Apply tier-based per-family limits. Pass no_limit=True to skip file-count caps."""
    if no_limit:
        return list(candidates)
    config = _TIER_CONFIG[tier]
    by_family: dict[str, list[dict]] = {}
    for c in candidates:
        by_family.setdefault(c["family"], []).append(c)

    selected = []
    for family, instances in sorted(by_family.items()):
        cfg = config.get(family, config["_default"])
        kept = instances
        if cfg["max_mb"] is not None:
            kept = [i for i in kept if i["size_mb"] <= cfg["max_mb"]]
        if cfg["max_files"] is not None:
            kept = kept[: cfg["max_files"]]
        selected.extend(kept)
    return selected


def main():
    ap = argparse.ArgumentParser(description="Build a CNF benchmark suite manifest")
    ap.add_argument("--dirs", nargs="+", required=True, help="Directories to scan for .cnf files")
    ap.add_argument("--output", default="benchmarks/suite.json", help="Output manifest path")
    ap.add_argument("--gpu-mem-gb", type=float, default=9.0,
                    help="Usable GPU memory in GB (default: 9.0, i.e. 75%% of 12 GB)")
    ap.add_argument("--min-k", type=int, default=8,
                    help="Skip instances where K_safe < this (default: 8)")
    ap.add_argument("--tier", choices=["fast", "full"], default="fast",
                    help="Selection tier: fast (~30 instances) or full (~200) (default: fast)")
    ap.add_argument("--max-m", type=int, default=None,
                    help="Only include instances with M <= this many clauses")
    ap.add_argument("--no-limit", action="store_true",
                    help="Skip per-family file-count caps from tier config (include everything "
                         "that passes GPU memory and --min-k/--max-m filters)")
    args = ap.parse_args()

    gpu_bytes = args.gpu_mem_gb * 1e9
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    max_m_note = f"  max_M={args.max_m}" if args.max_m is not None else ""
    print(f"Scanning {len(args.dirs)} dir(s)  "
          f"GPU={args.gpu_mem_gb:.1f} GB  min_K={args.min_k}  tier={args.tier}{max_m_note}")

    candidates = list(scan(args.dirs, gpu_bytes, args.min_k, max_m=args.max_m))
    suite = select(candidates, args.tier, no_limit=args.no_limit)

    print(f"\n{'Family':<14} {'N':>8} {'M':>8} {'K':>6} {'L':>3} "
          f"{'mem_MB':>7} {'file_MB':>8}  path")
    print("-" * 110)
    for e in suite:
        print(f"{e['family']:<14} {e['N']:>8} {e['M']:>8} {e['K']:>6} {e['L']:>3} "
              f"{e['est_mem_mb']:>7.1f} {e['size_mb']:>8.2f}  {e['path']}")

    print(f"\n{len(suite)} selected  "
          f"({len(candidates)} passed filters, "
          f"{len(candidates) - len(suite)} dropped by tier limits)")

    with open(args.output, "w") as f:
        json.dump(suite, f, indent=2)
    print(f"Manifest written → {args.output}")


if __name__ == "__main__":
    main()

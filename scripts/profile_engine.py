#!/usr/bin/env python3
"""Profile and compare GPAP BCP engine vs. PySAT external solver.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/profile_engine.py \
        --cnf tests/test_data/unit_prop_chain.cnf --K 50 --assumptions 2
"""
import argparse
import math
import random
import sys
import time

import torch

sys.path.insert(0, ".")
from gpap.engine.formula import CnfFormula
from gpap.engine.formula import ext_lit_to_lit_index
from gpap.engine.bcp_engine import BCPEngine, make_assumption_tensor
from gpap.engine.external_solver import ExternalSolver
from gpap.engine.calibrate import compute_k_safe, gpu_mem_available


# ---------------------------------------------------------------------------
# Assumption generation
# ---------------------------------------------------------------------------

def generate_random_assumptions(
    num_vars: int, K: int, num_per_batch: int, seed: int = 42
) -> list[list[int]]:
    """Generate K assumption sets as DIMACS literals (±integers).

    Variables are sampled without replacement per batch so no intra-batch
    contradictions are introduced.
    """
    rng = random.Random(seed)
    result = []
    for _ in range(K):
        n = min(num_per_batch, num_vars)
        vars_chosen = rng.sample(range(1, num_vars + 1), n)
        lits = [v if rng.random() < 0.5 else -v for v in vars_chosen]
        result.append(lits)
    return result


def count_all_pairs(num_vars: int) -> int:
    """Total (±v1, ±v2) pairs with v1 < v2: C(N,2) × 4."""
    return num_vars * (num_vars - 1) * 2


def count_all_triplets(num_vars: int) -> int:
    """Total (±v1, ±v2, ±v3) triplets with v1 < v2 < v3: C(N,3) × 8."""
    return num_vars * (num_vars - 1) * (num_vars - 2) // 6 * 8


def generate_all_pairs_chunks(num_vars: int, K: int):
    """Yield chunks of ≤K assumption sets covering all (±v1, ±v2) pairs, v1 < v2.

    No intra-batch contradictions by construction (v1 ≠ v2).
    Total assumptions: count_all_pairs(num_vars).
    """
    chunk: list[list[int]] = []
    for v1 in range(1, num_vars + 1):
        for v2 in range(v1 + 1, num_vars + 1):
            for p1 in (1, -1):
                for p2 in (1, -1):
                    chunk.append([p1 * v1, p2 * v2])
                    if len(chunk) == K:
                        yield chunk
                        chunk = []
    if chunk:
        yield chunk


def generate_all_triplets_chunks(num_vars: int, K: int):
    """Yield chunks of ≤K assumption sets covering all (±v1, ±v2, ±v3) triplets, v1 < v2 < v3.

    No intra-batch contradictions by construction (all three variables distinct).
    Total assumptions: count_all_triplets(num_vars).
    """
    chunk: list[list[int]] = []
    for v1 in range(1, num_vars + 1):
        for v2 in range(v1 + 1, num_vars + 1):
            for v3 in range(v2 + 1, num_vars + 1):
                for p1 in (1, -1):
                    for p2 in (1, -1):
                        for p3 in (1, -1):
                            chunk.append([p1 * v1, p2 * v2, p3 * v3])
                            if len(chunk) == K:
                                yield chunk
                                chunk = []
    if chunk:
        yield chunk


def generate_random_assumptions_gpu(
    num_vars: int, K: int, num_per_batch: int, device: torch.device, seed: int = 42
) -> torch.Tensor:
    """Generate a (K, L) int64 lit-index tensor directly on `device` — no Python loop.

    With-replacement sampling of (variable, polarity) per slot. Duplicate variables
    within a row are harmless: a repeated literal collapses, and a (v, -v) pair is a
    trivial conflict filtered by real_conflict. lit_index = 2*var0 + polarity, matching
    ext_lit_to_lit_index (positive -> even, negative -> odd).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    L = min(num_per_batch, num_vars)
    var0 = torch.randint(0, num_vars, (K, L), generator=g, device=device)  # 0-based var
    polarity = torch.randint(0, 2, (K, L), generator=g, device=device)     # 0=pos, 1=neg
    return (2 * var0 + polarity).to(torch.long)


def generate_all_pairs_chunks_gpu(num_vars: int, K: int, device: torch.device):
    """Yield (≤K, 2) int64 lit-index tensors covering all (±v1, ±v2) pairs, v1 < v2.

    Built on the GPU with an O(num_vars) Python loop over v1 only — each iteration
    forms all (v2 > v1) × 4 polarity combos as tensors. Memory-safe (no K-sized list,
    no full C(N,2) materialization). Total rows = count_all_pairs(num_vars).
    """
    pol = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long, device=device)
    buf: list[torch.Tensor] = []
    buf_rows = 0
    for v1 in range(num_vars):  # 0-based variable index
        n2 = num_vars - (v1 + 1)
        if n2 == 0:
            continue
        v2 = torch.arange(v1 + 1, num_vars, dtype=torch.long, device=device)  # (n2,)
        base = torch.empty((n2, 2), dtype=torch.long, device=device)
        base[:, 0] = 2 * v1
        base[:, 1] = 2 * v2
        block = (base.unsqueeze(1) + pol.unsqueeze(0)).reshape(-1, 2)  # (4*n2, 2)
        buf.append(block)
        buf_rows += block.size(0)
        if buf_rows >= K:
            cat = torch.cat(buf, dim=0)
            n_full = cat.size(0) // K
            for c in range(n_full):
                yield cat[c * K:(c + 1) * K]
            buf = []
            rem = cat[n_full * K:].clone()
            del cat
            buf = [rem] if rem.numel() else []
            buf_rows = rem.size(0)
    if buf_rows > 0:
        yield torch.cat(buf, dim=0)


def generate_all_triplets_chunks_gpu(num_vars: int, K: int, device: torch.device):
    """Yield (≤K, 3) int64 lit-index tensors covering all (±v1, ±v2, ±v3) triplets, v1<v2<v3.

    O(N) outer Python loop over v1. For each v1 the inner (v2, v3) body is vectorized on
    GPU using torch.combinations when the resulting tensor fits in < 256 MB; otherwise falls
    back to an O(N) inner v2 loop (needed for large N where torch.combinations per v1 would
    create hundreds of MB). Threshold: N < ~2300.

    This replaces the previous O(N²) Python loop (v1, v2 nested). For N=450: 450 outer
    iterations instead of 101K; each iteration builds the full v1-slice on GPU at once.
    Total rows = count_all_triplets(num_vars).
    """
    _INNER_BYTES_LIMIT = 256 * 1024 * 1024  # 256 MB threshold for vectorised inner body
    pol = torch.tensor(
        [[p1, p2, p3] for p1 in (0, 1) for p2 in (0, 1) for p3 in (0, 1)],
        dtype=torch.long, device=device,
    )  # (8, 3)
    buf: list[torch.Tensor] = []
    buf_rows = 0

    def _flush_buf():
        nonlocal buf, buf_rows
        cat = torch.cat(buf, dim=0)
        buf = []
        n_full = cat.size(0) // K
        for c in range(n_full):
            yield cat[c * K:(c + 1) * K]
        rem = cat[n_full * K:].clone()
        del cat
        buf = [rem] if rem.numel() else []
        buf_rows = rem.size(0)

    for v1 in range(num_vars):
        rest = torch.arange(v1 + 1, num_vars, dtype=torch.long, device=device)
        n2 = rest.numel()
        if n2 < 2:
            continue
        n_pairs = n2 * (n2 - 1) // 2
        if n_pairs * 24 < _INNER_BYTES_LIMIT:
            # Vectorised: build all (v2,v3) pairs for this v1 in one GPU call.
            pairs = torch.combinations(rest, r=2)  # (n_pairs, 2)
            base = torch.stack([
                torch.full((n_pairs,), 2 * v1, dtype=torch.long, device=device),
                2 * pairs[:, 0],
                2 * pairs[:, 1],
            ], dim=1)  # (n_pairs, 3)
            block = (base.unsqueeze(1) + pol.unsqueeze(0)).reshape(-1, 3)
            buf.append(block)
            buf_rows += block.size(0)
            if buf_rows >= K:
                yield from _flush_buf()
        else:
            # Large N: inner v2 loop to keep per-iteration tensor size bounded.
            for v2 in range(v1 + 1, num_vars):
                n3 = num_vars - v2 - 1
                if n3 == 0:
                    continue
                v3 = torch.arange(v2 + 1, num_vars, dtype=torch.long, device=device)
                base = torch.empty((n3, 3), dtype=torch.long, device=device)
                base[:, 0] = 2 * v1
                base[:, 1] = 2 * v2
                base[:, 2] = 2 * v3
                block = (base.unsqueeze(1) + pol.unsqueeze(0)).reshape(-1, 3)
                buf.append(block)
                buf_rows += block.size(0)
                if buf_rows >= K:
                    yield from _flush_buf()
    if buf_rows > 0:
        yield torch.cat(buf, dim=0)


def to_assumption_tensor(
    assumptions_dimacs: list[list[int]], device: torch.device
) -> torch.Tensor:
    """Convert DIMACS literal lists to a padded (K, L) tensor for BCPEngine. Setup cost only."""
    lit_indices = [[ext_lit_to_lit_index(l) for l in batch] for batch in assumptions_dimacs]
    return make_assumption_tensor(lit_indices, device)


# ---------------------------------------------------------------------------
# GPAP profiling
# ---------------------------------------------------------------------------

def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile_gpap(
    cnf_path: str,
    assumptions_dimacs: list[list[int]],
    device: torch.device,
    num_warmup: int,
    num_runs: int,
    max_rounds: int | None = None,
    soft_stop_frac: float | None = None,
    min_rounds: int = 0,
) -> dict:
    K = len(assumptions_dimacs)
    formula = CnfFormula(device)
    formula.parse_cnf_file(cnf_path)

    # Build padded tensor once — outside any timed section
    assumptions_tensor = to_assumption_tensor(assumptions_dimacs, device)

    def run_once(engine):
        _sync(device); t0 = time.perf_counter()
        engine.apply_assumptions_bulk(assumptions_tensor)
        _sync(device); t1 = time.perf_counter()
        engine.propagate(max_rounds=max_rounds, soft_stop_frac=soft_stop_frac, min_rounds=min_rounds)
        _sync(device); t2 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3

    for _ in range(num_warmup):
        engine = BCPEngine(formula)
        engine.init_state(num_batches=K)
        run_once(engine)

    assume_ms, prop_ms, bcp_rounds_list = [], [], []
    for _ in range(num_runs):
        engine = BCPEngine(formula)
        engine.init_state(num_batches=K)
        a, p = run_once(engine)
        assume_ms.append(a)
        prop_ms.append(p)
        bcp_rounds_list.append(engine._bcp_rounds)

    ta = torch.tensor(assume_ms)
    tp = torch.tensor(prop_ms)
    tt = ta + tp
    tr = torch.tensor(bcp_rounds_list, dtype=torch.float)

    conflict_results = engine.conflict.tolist()
    avg_prop = engine.num_propagated.float().mean().item()
    completion_rate = (~engine.timed_out).float().mean().item()

    return {
        "mean_ms":           tt.mean().item(),
        "std_ms":            tt.std().item(),
        "min_ms":            tt.min().item(),
        "max_ms":            tt.max().item(),
        "assume_mean_ms":    ta.mean().item(),
        "assume_std_ms":     ta.std().item(),
        "prop_mean_ms":      tp.mean().item(),
        "prop_std_ms":       tp.std().item(),
        "conflict":          conflict_results,
        "avg_propagations":  avg_prop,
        "conflict_rate":     sum(conflict_results) / K,
        "bcp_rounds_mean":   tr.mean().item(),
        "bcp_rounds_std":    tr.std().item(),
        "completion_rate":   completion_rate,
    }


# ---------------------------------------------------------------------------
# Detailed per-step propagate profiling
# ---------------------------------------------------------------------------

# Step groups for display: each entry is (label, [step_keys])
_PROPAGATE_STEPS = [
    ("frontier_filter (active mask on frontier)",   ["frontier_filter"]),
    ("sc_gather       (sat_count[fb,fc] read)",     ["sc_gather"]),
    ("conflict        (sat==0 detect + mark)",      ["conflict"]),
    ("unit_filter     (B/C/D forced-lit filter)",   ["unit_filter"]),
    ("sparse_dedup    (E: unique + intra-conflict)", ["sparse_dedup"]),
    ("csr_lookup      (occ_offsets slice)",         ["csr_lookup"]),
    ("expand          (repeat_interleave + csi)",   ["expand"]),
    ("scatter_sat     (scatter_add sat_count)",     ["scatter_sat"]),
    ("scatter_lit     (scatter_add lit_sum)",       ["scatter_lit"]),
    ("assign          (assignment + prop count)",   ["assign"]),
]


def profile_propagate_detail(
    cnf_path: str,
    assumptions_dimacs: list[list[int]],
    device: torch.device,
    num_warmup: int,
) -> None:
    """Run one profiled propagate pass and print a per-step breakdown table."""
    K = len(assumptions_dimacs)
    formula = CnfFormula(device)
    formula.parse_cnf_file(cnf_path)
    assumptions_tensor = to_assumption_tensor(assumptions_dimacs, device)

    # Warmup (uses normal propagate — no sync overhead)
    for _ in range(num_warmup):
        engine = BCPEngine(formula)
        engine.init_state(num_batches=K)
        engine.apply_assumptions_bulk(assumptions_tensor)
        engine.propagate()

    # Single profiled run
    engine = BCPEngine(formula)
    engine.init_state(num_batches=K)
    engine.apply_assumptions_bulk(assumptions_tensor)
    _, summary = engine.propagate_profiled()

    total_prop_ms = sum(v["total_ms"] for v in summary.values())
    rounds = summary["frontier_filter"]["n"]

    print(f"\n  propagate detail  ({rounds} BCP rounds, total={total_prop_ms:.2f} ms)")
    print(f"  {'Step':<43}  {'total ms':>9}  {'mean ms':>8}  {'% total':>7}  {'rounds':>6}")
    print(f"  {'-'*43}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*6}")

    for label, keys in _PROPAGATE_STEPS:
        t_ms  = sum(summary[k]["total_ms"] for k in keys)
        m_ms  = sum(summary[k]["mean_ms"]  for k in keys)
        n     = max(summary[k]["n"] for k in keys)
        pct   = 100 * t_ms / total_prop_ms if total_prop_ms else 0
        print(f"  {label:<43}  {t_ms:>9.2f}  {m_ms:>8.3f}  {pct:>6.1f}%  {n:>6}")

    print(f"  {'─'*43}  {'-'*9}")
    print(f"  {'TOTAL':<43}  {total_prop_ms:>9.2f}")


# ---------------------------------------------------------------------------
# External solver (PySAT) profiling
# ---------------------------------------------------------------------------

def profile_external(
    cnf_path: str,
    assumptions_dimacs: list[list[int]],
    num_runs: int,
) -> dict:
    K = len(assumptions_dimacs)
    ext = ExternalSolver(cnf_path)

    # Use propagate_many so the CNF is bootstrapped once per run, not once per batch.
    # Internal solver time excludes even that single bootstrap cost.
    times_ms = []
    for _ in range(num_runs):
        conflict_results, prop_list, solve_s = ext.propagate_many(assumptions_dimacs)
        times_ms.append(solve_s * 1000.0)

    avg_propagations = sum(len(p) for p in prop_list) / K

    t = torch.tensor(times_ms)
    return {
        "mean_ms": t.mean().item(),
        "std_ms": t.std().item(),
        "min_ms": t.min().item(),
        "max_ms": t.max().item(),
        "conflict": conflict_results,        # list[bool], length K
        "conflict_rate": sum(conflict_results) / K,
        "avg_propagations": avg_propagations,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_results(gpap_conflicts: list[bool], ext_conflicts: list[bool]) -> dict:
    """Compute per-batch agreement between GPAP and external solver."""
    assert len(gpap_conflicts) == len(ext_conflicts)
    K = len(gpap_conflicts)
    matches = sum(g == e for g, e in zip(gpap_conflicts, ext_conflicts))
    return {
        "agreement_rate": matches / K,
        "mismatches": K - matches,
        "gpap_only_conflict": sum(g and not e for g, e in zip(gpap_conflicts, ext_conflicts)),
        "ext_only_conflict": sum(e and not g for g, e in zip(gpap_conflicts, ext_conflicts)),
    }


# ---------------------------------------------------------------------------
# K sweep: run GPAP (and optionally PySAT) for multiple batch counts
# ---------------------------------------------------------------------------

def run_k_sweep(
    cnf_path: str,
    all_assumptions: list[list[int]],
    device: torch.device,
    k_values: list[int],
    num_warmup: int,
    num_runs: int,
    run_external: bool = True,
    max_rounds: int | None = None,
    soft_stop_frac: float | None = None,
    min_rounds: int = 0,
) -> list[dict]:
    """Profile GPAP (and optionally PySAT) for each K in k_values.

    all_assumptions must have at least max(k_values) entries; each K slice takes the first K.
    """
    results = []
    for K in k_values:
        assumptions = all_assumptions[:K]
        gpap = profile_gpap(cnf_path, assumptions, device, num_warmup, num_runs,
                            max_rounds=max_rounds, soft_stop_frac=soft_stop_frac,
                            min_rounds=min_rounds)
        entry = {
            "K":               K,
            "gpap_total_ms":   gpap["mean_ms"],
            "gpap_std_ms":     gpap["std_ms"],
            "gpap_per_batch":  gpap["mean_ms"] / K,
            "bcp_rounds_mean": gpap["bcp_rounds_mean"],
            "conflict_rate":   gpap["conflict_rate"],
            "ext_total_ms":    None,
            "ext_per_batch":   None,
        }
        line = (f"  K={K:6d}  GPAP={gpap['mean_ms']:8.2f} ± {gpap['std_ms']:.2f} ms"
                f"  ({gpap['mean_ms']/K:.4f} ms/batch)"
                f"  rounds={gpap['bcp_rounds_mean']:.1f}"
                f"  conflict={gpap['conflict_rate']:.0%}")
        if run_external:
            ext = profile_external(cnf_path, assumptions, min(num_runs, 3))
            entry["ext_total_ms"]  = ext["mean_ms"]
            entry["ext_per_batch"] = ext["mean_ms"] / K
            line += f"  PySAT={ext['mean_ms']:8.2f} ms  speedup={ext['mean_ms']/gpap['mean_ms']:.1f}x"
        print(line)
        results.append(entry)
    return results


def plot_k_sweep(results: list[dict], output_path: str) -> None:
    """Three-panel plot: total time, per-batch time, and parallelism speedup vs K."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plot. Run: pip install matplotlib")
        return

    ks            = [r["K"]              for r in results]
    gpap_total    = [r["gpap_total_ms"]  for r in results]
    gpap_std      = [r["gpap_std_ms"]    for r in results]
    gpap_pb       = [r["gpap_per_batch"] for r in results]  # per-batch
    has_ext       = any(r["ext_total_ms"] is not None for r in results)

    # Reference: per-batch time at smallest K (K=1 if included)
    k1_pb    = gpap_pb[0]
    k1_total = gpap_total[0]

    # Parallelism speedup: how much faster than K independent K=1 GPAP runs
    par_speedup = [k1_pb / pb for pb in gpap_pb]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("GPAP BCP Engine — Batch Size Scaling", fontsize=13, fontweight="bold")

    # --- Panel 1: Total time ---
    ax = axes[0]
    ax.errorbar(ks, gpap_total, yerr=gpap_std, fmt="o-", color="steelblue",
                capsize=3, label="GPAP total")
    # Ideal: total time scales linearly with K (no GPU benefit)
    ideal = [k1_total * k / ks[0] for k in ks]
    ax.plot(ks, ideal, "b--", alpha=0.35, label=f"Ideal linear (K × K={ks[0]} time)")
    if has_ext:
        ext_ks    = [r["K"]            for r in results if r["ext_total_ms"] is not None]
        ext_total = [r["ext_total_ms"] for r in results if r["ext_total_ms"] is not None]
        ax.plot(ext_ks, ext_total, "s--", color="tomato", label="PySAT serial")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Batch count K"); ax.set_ylabel("Total time (ms)")
    ax.set_title("Total Time vs K")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Panel 2: Per-batch time ---
    ax = axes[1]
    ax.plot(ks, gpap_pb, "o-", color="steelblue", label="GPAP / batch")
    ax.axhline(k1_pb, color="steelblue", linestyle=":", alpha=0.5,
               label=f"K={ks[0]} baseline ({k1_pb:.3f} ms)")
    if has_ext:
        ext_pb = [r["ext_per_batch"] for r in results if r["ext_per_batch"] is not None]
        ax.plot(ext_ks, ext_pb, "s--", color="tomato", label="PySAT / batch")
    ax.set_xscale("log")
    ax.set_xlabel("Batch count K"); ax.set_ylabel("Time per batch (ms)")
    ax.set_title("Per-Batch Time vs K")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Panel 3: Speedup vs sequential ---
    ax = axes[2]
    ax.plot(ks, par_speedup, "o-", color="mediumseagreen",
            label="GPU parallelism (vs K sequential GPAP)")
    ax.plot(ks, [k / ks[0] for k in ks], "k--", alpha=0.3, label="Ideal linear")
    if has_ext:
        vs_pysat = [e / g for e, g in zip(ext_total, gpap_total[:len(ext_total)])]
        ax.plot(ext_ks, vs_pysat, "s--", color="tomato", label="vs PySAT serial")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Batch count K"); ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Sequential")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# End-to-end clause generation: GPU vs CPU sequential baseline
# ---------------------------------------------------------------------------

def run_e2e_gpu(
    cnf_path: str,
    mode: str,
    K: int,
    device: torch.device,
    formula: "CnfFormula | None" = None,
    num_warmup: int = 2,
    max_rounds: int | None = None,
    soft_stop_frac: float | None = None,
    min_rounds: int = 0,
    max_sets: int | None = None,
    L: int | None = None,
    target_sets: int | None = None,
    seed: int = 42,
) -> dict:
    """Full GPAP pipeline over assumption-set chunks: generate → reset → propagate → extract.

    mode:
      "pairs"/"triplets" — exhaustive (±v1,±v2[,±v3]) chunks, capped at max_sets.
      "random"           — fresh random-L chunks generated on the fly until target_sets is
                           reached (requires L and target_sets); internally drives the same
                           truncation path by setting max_sets = target_sets.

    Returns timing dict plus per-phase peak GPU memory (CUDA only):
      mem_static_gb   — memory_allocated after init_state (static tensors only)
      peak_reset_gb   — max_memory_allocated during first reset_state
      peak_apply_gb   — max_memory_allocated during first apply_assumptions_bulk
      peak_prop_gb    — max_memory_allocated during first propagate (can be >> static)
      peak_overall_gb — max_memory_allocated across the entire run (warmup excluded)
    and finished-batch counts:
      successful_sets — sets whose BCP reached fixpoint/conflict (not soft-stop timed_out)
      timed_out_sets  — sets cut off by soft_stop_frac/max_rounds (==0 with no cutoff)
    """
    is_cuda = device.type == "cuda"

    def _mem_gb() -> float:
        return torch.cuda.memory_allocated(device) / 1e9 if is_cuda else 0.0

    def _peak_gb() -> float:
        return torch.cuda.max_memory_allocated(device) / 1e9 if is_cuda else 0.0

    def _reset_peak():
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)

    if formula is None:
        formula = CnfFormula(device)
        formula.parse_cnf_file(cnf_path)
    N = formula.num_vars

    if mode == "pairs":
        chunk_gen_fn = lambda: generate_all_pairs_chunks_gpu(N, K, device)
    elif mode == "triplets":
        chunk_gen_fn = lambda: generate_all_triplets_chunks_gpu(N, K, device)
    elif mode == "random":
        if L is None or target_sets is None:
            raise ValueError("mode='random' requires both L and target_sets")
        max_sets = target_sets  # bound the (infinite) generator via the truncation path

        def _random_chunks():
            idx = 0
            while True:
                yield generate_random_assumptions_gpu(N, K, L, device, seed=seed + idx)
                idx += 1

        chunk_gen_fn = _random_chunks
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    engine = BCPEngine(formula)
    warmed_up = False
    t_gen = t_reset = t_apply = t_prop = t_extract = 0.0
    total_sets = 0
    n_chunks = 0
    conflicts_found = 0
    successful_sets = 0
    timed_out_sets = 0

    # Per-phase peak memory (first real chunk only, warmup excluded)
    mem_static_gb = peak_reset_gb = peak_apply_gb = peak_prop_gb = 0.0

    gen_iter = iter(chunk_gen_fn())
    while True:
        _sync(device); g0 = time.perf_counter()
        try:
            chunk_t = next(gen_iter)
        except StopIteration:
            break
        _sync(device); t_gen += (time.perf_counter() - g0) * 1e3

        K_chunk = chunk_t.size(0)
        # Truncate the chunk to hit max_sets exactly on the last iteration
        is_last = False
        if max_sets is not None and total_sets + K_chunk >= max_sets:
            chunk_t = chunk_t[:max_sets - total_sets]
            K_chunk = chunk_t.size(0)
            is_last = True

        if not warmed_up and num_warmup > 0:
            for _ in range(num_warmup):
                engine.reset_state(K_chunk)
                engine.apply_assumptions_bulk(chunk_t)
                engine.propagate(max_rounds=max_rounds, soft_stop_frac=soft_stop_frac,
                                 min_rounds=min_rounds)
            _sync(device)
            warmed_up = True

        # --- phase: reset ---
        _reset_peak()
        _sync(device); r0 = time.perf_counter()
        engine.reset_state(K_chunk)
        _sync(device); t_reset += (time.perf_counter() - r0) * 1e3
        if n_chunks == 0:
            mem_static_gb = _mem_gb()
            peak_reset_gb = _peak_gb()

        # --- phase: apply ---
        _reset_peak()
        _sync(device); a0 = time.perf_counter()
        engine.apply_assumptions_bulk(chunk_t)
        _sync(device); t_apply += (time.perf_counter() - a0) * 1e3
        if n_chunks == 0:
            peak_apply_gb = _peak_gb()

        # --- phase: propagate ---
        _reset_peak()
        _sync(device); p0 = time.perf_counter()
        engine.propagate(max_rounds=max_rounds, soft_stop_frac=soft_stop_frac,
                         min_rounds=min_rounds)
        _sync(device); t_prop += (time.perf_counter() - p0) * 1e3
        if n_chunks == 0:
            peak_prop_gb = _peak_gb()

        # Finished-batch accounting (soft_stop_frac/max_rounds may have cut some off)
        n_timed_out = int(engine.timed_out.sum().item())
        timed_out_sets += n_timed_out
        successful_sets += K_chunk - n_timed_out

        _reset_peak()

        _sync(device); e0 = time.perf_counter()
        n_real = int(engine.real_conflict.sum().item())   # BCP-derived conflicts this chunk
        _sync(device); t_extract += (time.perf_counter() - e0) * 1e3

        conflicts_found += n_real
        total_sets += K_chunk
        n_chunks += 1
        if is_last:
            break

    # Overall peak across entire run (warmup excluded — reset before first real chunk)
    peak_overall_gb = max(peak_reset_gb, peak_apply_gb, peak_prop_gb)

    wall_ms = t_gen + t_reset + t_apply + t_prop + t_extract
    return {
        "total_ms":        wall_ms,
        "t_gen_ms":        t_gen,
        "t_reset_ms":      t_reset,
        "t_apply_ms":      t_apply,
        "t_prop_ms":       t_prop,
        "t_extract_ms":    t_extract,
        "total_sets":      total_sets,
        "successful_sets": successful_sets,
        "timed_out_sets":  timed_out_sets,
        "n_chunks":        n_chunks,
        "conflicts_found": conflicts_found,
        "mem_static_gb":   mem_static_gb,
        "peak_reset_gb":   peak_reset_gb,
        "peak_apply_gb":   peak_apply_gb,
        "peak_prop_gb":    peak_prop_gb,
        "peak_overall_gb": peak_overall_gb,
    }


def run_e2e_cpu(
    cnf_path: str,
    mode: str,
    num_vars: int,
    max_sets: int | None = None,
) -> dict:
    """Sequential CPU baseline: iterate all pairs/triplets in a tight loop, single solver instance.

    Bootstrap MinisatGH once, then call solver.propagate() for each assumption set reusing
    the same buf list in-place — no new allocations in the hot loop.
    """
    from pysat.formula import CNF
    from pysat.solvers import MinisatGH

    class _Done(Exception):
        pass

    cnf = CNF(from_file=cnf_path)
    conflicts_found = 0
    total = 0

    t0 = time.perf_counter()
    try:
        with MinisatGH(bootstrap_with=cnf) as solver:
            if mode == "pairs":
                buf = [0, 0]
                for v1 in range(1, num_vars + 1):
                    for v2 in range(v1 + 1, num_vars + 1):
                        for p1 in (1, -1):
                            for p2 in (1, -1):
                                if max_sets is not None and total >= max_sets:
                                    raise _Done
                                buf[0] = p1 * v1
                                buf[1] = p2 * v2
                                st, _ = solver.propagate(assumptions=buf)
                                if not st:
                                    conflicts_found += 1
                                total += 1
            else:  # triplets
                buf = [0, 0, 0]
                for v1 in range(1, num_vars + 1):
                    for v2 in range(v1 + 1, num_vars + 1):
                        for v3 in range(v2 + 1, num_vars + 1):
                            for p1 in (1, -1):
                                for p2 in (1, -1):
                                    for p3 in (1, -1):
                                        if max_sets is not None and total >= max_sets:
                                            raise _Done
                                        buf[0] = p1 * v1
                                        buf[1] = p2 * v2
                                        buf[2] = p3 * v3
                                        st, _ = solver.propagate(assumptions=buf)
                                        if not st:
                                            conflicts_found += 1
                                        total += 1
    except _Done:
        pass
    wall_ms = (time.perf_counter() - t0) * 1e3
    return {
        "total_ms":        wall_ms,
        "total_sets":      total,
        "conflicts_found": conflicts_found,
    }


def run_e2e_cpu_random(
    cnf_path: str,
    num_vars: int,
    L: int,
    sample_sets: int,
    seed: int = 42,
) -> dict:
    """Sequential CPU baseline for random-L assumptions, on a bounded sample to extrapolate from.

    Generates `sample_sets` random-L assumption lists (same distribution as the GPU random mode)
    and runs them through one MinisatGH instance in a tight loop. The caller extrapolates the
    per-set cost to the GPU's full workload. Returns wall time for the sample + conflict count.
    """
    from pysat.formula import CNF
    from pysat.solvers import MinisatGH

    assumptions = generate_random_assumptions(num_vars, sample_sets, L, seed)
    cnf = CNF(from_file=cnf_path)
    conflicts_found = 0

    t0 = time.perf_counter()
    with MinisatGH(bootstrap_with=cnf) as solver:
        for lits in assumptions:
            st, _ = solver.propagate(assumptions=lits)
            if not st:
                conflicts_found += 1
    wall_ms = (time.perf_counter() - t0) * 1e3
    return {
        "total_ms":        wall_ms,
        "total_sets":      sample_sets,
        "conflicts_found": conflicts_found,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare GPAP BCP engine vs. PySAT propagation")
    ap.add_argument("--cnf", required=True, help="Path to DIMACS CNF file")
    ap.add_argument("--K", type=int, default=100,
                    help="Number of parallel batches (keep small for external solver)")
    ap.add_argument("--K-values", type=int, nargs="+", metavar="K",
                    help="K sweep: profile these batch counts (overrides --K). K=1 always prepended.")
    ap.add_argument("--assumptions", type=int, default=5,
                    help="Number of assumed literals per batch")
    ap.add_argument("--device", default=None, help="torch device (cpu/cuda)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="GPAP warmup runs (not timed)")
    ap.add_argument("--runs", type=int, default=10, help="Timed runs for each solver")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--no-external", action="store_true",
                    help="Skip PySAT in K sweep (much faster for large K)")
    ap.add_argument("--plot", default="k_sweep.png",
                    help="Output path for K-sweep plot (default: k_sweep.png)")
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="Hard cap on BCP rounds; remaining batches tagged timed_out")
    ap.add_argument("--soft-stop-frac", type=float, default=None,
                    help="Soft cutoff: after --min-rounds, stop when a round's unique "
                         "propagations drop below this fraction of peak work seen so far")
    ap.add_argument("--min-rounds", type=int, default=0,
                    help="Minimum rounds before --soft-stop-frac may trigger (default: 0)")
    ap.add_argument("--e2e-mode", choices=["pairs", "triplets"], default=None,
                    help="End-to-end mode: generate all pairs/triplets and compare GPU vs CPU")
    ap.add_argument("--auto-k", action="store_true",
                    help="Compute K from M, N and GPU memory (overrides --K; used in --e2e-mode)")
    ap.add_argument("--e2e-max-sets", type=int, default=None,
                    help="Limit total sets processed in --e2e-mode (sample first N sets)")
    ap.add_argument("--e2e-no-cpu", action="store_true",
                    help="Skip sequential CPU baseline in --e2e-mode (large N only)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load formula once to get num_vars
    formula = CnfFormula(device)
    formula.parse_cnf_file(args.cnf)

    print(f"\nFormula : {args.cnf}  ({formula.num_vars} vars, {formula.num_clauses} clauses)")
    cutoff_note = ""
    if args.max_rounds is not None:
        cutoff_note += f"  max_rounds={args.max_rounds}"
    if args.soft_stop_frac is not None:
        cutoff_note += f"  soft_stop_frac={args.soft_stop_frac}  min_rounds={args.min_rounds}"
    print(f"device={device}  assumptions/batch={args.assumptions}  "
          f"warmup={args.warmup}  runs={args.runs}{cutoff_note}")

    # ---- End-to-end mode: all pairs / all triplets --------------------------
    if args.e2e_mode:
        N = formula.num_vars
        mode = args.e2e_mode
        if args.auto_k:
            max_cls    = int(formula.clause_sizes.max().item())
            total_lits = int(formula.clause_sizes.sum().item())
            K = compute_k_safe(formula.num_clauses, N, gpu_mem_available(device), cap=None,
                               max_clause_size=max_cls, total_lits=total_lits,
                               deduct_formula=False)
            k_note = f"K={K} (auto)"
        else:
            K = args.K
            k_note = f"K={K}"
        total_sets_full = count_all_pairs(N) if mode == "pairs" else count_all_triplets(N)
        max_sets = args.e2e_max_sets
        total_sets = min(total_sets_full, max_sets) if max_sets is not None else total_sets_full
        n_chunks = math.ceil(total_sets / K)
        sets_note = (f"{total_sets:,} / {total_sets_full:,} sets (capped)"
                     if max_sets is not None else f"{total_sets:,} sets")
        print(f"\nE2E {mode}  N={N}  {sets_note}  {k_note}  chunks={n_chunks}")

        print("\nRunning GPU (GPAP) ...")
        gpu = run_e2e_gpu(
            args.cnf, mode, K, device,
            formula=formula,
            num_warmup=args.warmup,
            max_rounds=args.max_rounds,
            soft_stop_frac=args.soft_stop_frac,
            min_rounds=args.min_rounds,
            max_sets=max_sets,
        )
        gpu_sps = gpu["total_sets"] / (gpu["total_ms"] / 1e3) if gpu["total_ms"] else 0
        w = gpu["total_ms"]
        def _pct(ms): return f"{100*ms/w:.1f}%" if w else "—"

        print("  GPU ─────────────────────────────────────────────────────")
        print(f"    generation  : {gpu['t_gen_ms']:>9.1f} ms  {_pct(gpu['t_gen_ms']):>6}  (on-GPU tensor creation)")
        print(f"    reset       : {gpu['t_reset_ms']:>9.1f} ms  {_pct(gpu['t_reset_ms']):>6}  ({gpu['n_chunks']} chunks × {gpu['t_reset_ms']/gpu['n_chunks']:.1f} ms/chunk)")
        print(f"    apply       : {gpu['t_apply_ms']:>9.1f} ms  {_pct(gpu['t_apply_ms']):>6}")
        print(f"    propagate   : {gpu['t_prop_ms']:>9.1f} ms  {_pct(gpu['t_prop_ms']):>6}  (BCP kernel)")
        print(f"    extract     : {gpu['t_extract_ms']:>9.1f} ms  {_pct(gpu['t_extract_ms']):>6}  (device→host)")
        print(f"    {'─'*55}")
        print(f"    wall total  : {gpu['total_ms']:>9.1f} ms          {gpu_sps:,.0f} sets/s")
        print(f"    conflicts   : {gpu['conflicts_found']:,}  ({gpu['conflicts_found']/gpu['total_sets']:.2%} conflict rate)")
        if device.type == "cuda":
            from gpap.engine.calibrate import per_batch_mem_bytes
            M_formula = formula.num_clauses
            max_cls    = int(formula.clause_sizes.max().item())
            tl         = int(formula.clause_sizes.sum().item())
            sat_b = 1 if max_cls <= 127 else (2 if max_cls <= 32767 else 4)
            est_static_gb = K * ((sat_b + 4) * M_formula + 2 * N) / 1e9
            est_budget_gb = K * per_batch_mem_bytes(M_formula, N, max_cls, tl) / 1e9
            print("\n  Memory (first chunk, measured vs. model):")
            print(f"    static tensors    : {gpu['mem_static_gb']:.2f} GB  (model: {est_static_gb:.2f} GB)")
            print(f"    peak after reset  : {gpu['peak_reset_gb']:.2f} GB")
            print(f"    peak after apply  : {gpu['peak_apply_gb']:.2f} GB")
            print(f"    peak after prop   : {gpu['peak_prop_gb']:.2f} GB  ← highest = bottleneck")
            print(f"    peak overall      : {gpu['peak_overall_gb']:.2f} GB  (budget: {est_budget_gb:.2f} GB)")

        cpu: dict | None = None
        if not args.e2e_no_cpu:
            print("\nRunning CPU (sequential, single PySAT instance) ...")
            cpu = run_e2e_cpu(args.cnf, mode, N, max_sets=max_sets)
            cpu_sps = cpu["total_sets"] / (cpu["total_ms"] / 1e3) if cpu["total_ms"] else 0
            print("  CPU ─────────────────────────────────────────────────────")
            print(f"    wall total  : {cpu['total_ms']:>9.1f} ms          {cpu_sps:,.0f} sets/s   (sequential)")
            print(f"    conflicts   : {cpu['conflicts_found']:,}  ({cpu['conflicts_found']/cpu['total_sets']:.2%} conflict rate)")

        print()
        if cpu is not None:
            speedup = cpu["total_ms"] / gpu["total_ms"] if gpu["total_ms"] else float("inf")
            match = gpu["conflicts_found"] == cpu["conflicts_found"]
            print(f"  Speedup (CPU / GPU wall) : {speedup:.1f}x")
            if match:
                print(f"  Conflicts match          : YES ({gpu['conflicts_found']:,})")
            else:
                ratio = cpu["conflicts_found"] / gpu["conflicts_found"] if gpu["conflicts_found"] else float("inf")
                print(f"  Conflicts match          : NO  GPU={gpu['conflicts_found']:,}  CPU={cpu['conflicts_found']:,}  "
                      f"(CPU finds {ratio:.1f}x more — GPU soft-stop too aggressive?)")
        return

    # ---- K sweep mode -------------------------------------------------------
    if args.K_values:
        k_values = sorted(set(args.K_values))
        max_K = max(k_values)
        all_assumptions = generate_random_assumptions(
            formula.num_vars, max_K, args.assumptions, args.seed
        )
        print(f"\nK sweep : {k_values}\n")
        results = run_k_sweep(
            args.cnf, all_assumptions, device, k_values,
            args.warmup, args.runs, run_external=not args.no_external,
            max_rounds=args.max_rounds, soft_stop_frac=args.soft_stop_frac,
            min_rounds=args.min_rounds,
        )
        plot_k_sweep(results, args.plot)
        return

    # ---- Single-K mode ------------------------------------------------------
    assumptions = generate_random_assumptions(
        formula.num_vars, args.K, args.assumptions, args.seed
    )

    print(f"K={args.K}\n")

    # --- GPAP ---
    print("Running GPAP engine...")
    gpap = profile_gpap(args.cnf, assumptions, device, args.warmup, args.runs,
                        max_rounds=args.max_rounds, soft_stop_frac=args.soft_stop_frac,
                        min_rounds=args.min_rounds)

    assume_pct = 100 * gpap["assume_mean_ms"] / gpap["mean_ms"] if gpap["mean_ms"] else 0
    prop_pct   = 100 * gpap["prop_mean_ms"]   / gpap["mean_ms"] if gpap["mean_ms"] else 0

    print(f"  GPAP total  : {gpap['mean_ms']:8.2f} ± {gpap['std_ms']:.2f} ms  "
          f"[{gpap['min_ms']:.2f}, {gpap['max_ms']:.2f}]")
    print(f"  ├ apply_assumptions_bulk : {gpap['assume_mean_ms']:7.2f} ± {gpap['assume_std_ms']:.2f} ms  "
          f"({assume_pct:.1f}%)")
    print(f"  └ propagate              : {gpap['prop_mean_ms']:7.2f} ± {gpap['prop_std_ms']:.2f} ms  "
          f"({prop_pct:.1f}%)  BCP rounds={gpap['bcp_rounds_mean']:.1f}±{gpap['bcp_rounds_std']:.1f}")
    print(f"  conflict rate={gpap['conflict_rate']:.2%}  avg props={gpap['avg_propagations']:.1f}"
          f"  completion={gpap['completion_rate']:.2%}")

    profile_propagate_detail(args.cnf, assumptions, device, args.warmup)

    # --- External solver ---
    print(f"Running external solver (PySAT MinisatGH, {args.K} serial calls per run)...")
    ext = profile_external(args.cnf, assumptions, args.runs)
    print(f"  PySAT : {ext['mean_ms']:8.2f} ± {ext['std_ms']:.2f} ms  "
          f"[{ext['min_ms']:.2f}, {ext['max_ms']:.2f}]  "
          f"conflict rate={ext['conflict_rate']:.2%}  "
          f"avg props={ext['avg_propagations']:.1f}")

    # --- Speedup ---
    speedup = ext["mean_ms"] / gpap["mean_ms"]
    print(f"\nSpeedup : {speedup:.1f}x  (PySAT serial / GPAP bulk)")

    # --- Result agreement ---
    cmp = compare_results(gpap["conflict"], ext["conflict"])
    print(f"Agreement : {cmp['agreement_rate']:.2%}  "
          f"({cmp['mismatches']} mismatches — "
          f"GPAP-only={cmp['gpap_only_conflict']}, "
          f"PySAT-only={cmp['ext_only_conflict']})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Profile and compare GPAP BCP engine vs. PySAT external solver.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/profile_engine.py \
        --cnf tests/test_data/unit_prop_chain.cnf --K 50 --assumptions 2
"""
import argparse
import random
import sys
import time

import torch

sys.path.insert(0, ".")
from gpap.clsdata.cnf_formula import CnfFormula
from gpap.clsdata.occ_list import ext_lit_to_lit_index
from gpap.solver.bcp_engine import BCPEngine, make_assumption_tensor
from gpap.solver.external_solver import ExternalSolver


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
        engine.propagate(max_rounds=max_rounds)
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
) -> list[dict]:
    """Profile GPAP (and optionally PySAT) for each K in k_values.

    all_assumptions must have at least max(k_values) entries; each K slice takes the first K.
    """
    results = []
    for K in k_values:
        assumptions = all_assumptions[:K]
        gpap = profile_gpap(cnf_path, assumptions, device, num_warmup, num_runs)
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
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load formula once to get num_vars
    formula = CnfFormula(device)
    formula.parse_cnf_file(args.cnf)

    print(f"\nFormula : {args.cnf}  ({formula.num_vars} vars, {formula.num_clauses} clauses)")
    print(f"device={device}  assumptions/batch={args.assumptions}  "
          f"warmup={args.warmup}  runs={args.runs}")

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
    gpap = profile_gpap(args.cnf, assumptions, device, args.warmup, args.runs)

    assume_pct = 100 * gpap["assume_mean_ms"] / gpap["mean_ms"] if gpap["mean_ms"] else 0
    prop_pct   = 100 * gpap["prop_mean_ms"]   / gpap["mean_ms"] if gpap["mean_ms"] else 0

    print(f"  GPAP total  : {gpap['mean_ms']:8.2f} ± {gpap['std_ms']:.2f} ms  "
          f"[{gpap['min_ms']:.2f}, {gpap['max_ms']:.2f}]")
    print(f"  ├ apply_assumptions_bulk : {gpap['assume_mean_ms']:7.2f} ± {gpap['assume_std_ms']:.2f} ms  "
          f"({assume_pct:.1f}%)")
    print(f"  └ propagate              : {gpap['prop_mean_ms']:7.2f} ± {gpap['prop_std_ms']:.2f} ms  "
          f"({prop_pct:.1f}%)  BCP rounds={gpap['bcp_rounds_mean']:.1f}±{gpap['bcp_rounds_std']:.1f}")
    print(f"  conflict rate={gpap['conflict_rate']:.2%}  avg props={gpap['avg_propagations']:.1f}")

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

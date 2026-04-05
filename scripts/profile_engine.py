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
from gpap.solver.bcp_engine import BCPEngine
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


def to_lit_indices(assumptions_dimacs: list[list[int]]) -> list[list[int]]:
    """Convert DIMACS literal lists to internal literal-index lists for GPAP."""
    return [[ext_lit_to_lit_index(l) for l in batch] for batch in assumptions_dimacs]


# ---------------------------------------------------------------------------
# GPAP profiling
# ---------------------------------------------------------------------------

def profile_gpap(
    cnf_path: str,
    assumptions_dimacs: list[list[int]],
    device: torch.device,
    num_warmup: int,
    num_runs: int,
) -> dict:
    K = len(assumptions_dimacs)
    formula = CnfFormula(device)
    formula.parse_cnf_file(cnf_path)

    assumptions_idx = to_lit_indices(assumptions_dimacs)

    def run_once():
        engine = BCPEngine(formula)
        engine.init_state(num_batches=K)
        engine.apply_assumptions_bulk(assumptions_idx)
        engine.propagate()
        return engine

    for _ in range(num_warmup):
        run_once()

    times_ms = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng = run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    t = torch.tensor(times_ms)
    conflict_results = eng.conflict.tolist()
    avg_prop = eng.num_propagated.float().mean().item()

    return {
        "mean_ms": t.mean().item(),
        "std_ms": t.std().item(),
        "min_ms": t.min().item(),
        "max_ms": t.max().item(),
        "conflict": conflict_results,        # list[bool], length K
        "avg_propagations": avg_prop,
        "conflict_rate": sum(conflict_results) / K,
    }


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

    def run_once():
        conflicts = []
        for batch_lits in assumptions_dimacs:
            is_conflict, _ = ext.propagate_assumptions(batch_lits)
            conflicts.append(is_conflict)
        return conflicts

    # No warmup: MinisatGH spins up fresh each call, warmup doesn't help
    times_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        conflict_results = run_once()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    t = torch.tensor(times_ms)
    return {
        "mean_ms": t.mean().item(),
        "std_ms": t.std().item(),
        "min_ms": t.min().item(),
        "max_ms": t.max().item(),
        "conflict": conflict_results,        # list[bool], length K
        "conflict_rate": sum(conflict_results) / K,
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
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare GPAP BCP engine vs. PySAT propagation")
    ap.add_argument("--cnf", required=True, help="Path to DIMACS CNF file")
    ap.add_argument("--K", type=int, default=100,
                    help="Number of parallel batches (keep small for external solver)")
    ap.add_argument("--assumptions", type=int, default=5,
                    help="Number of assumed literals per batch")
    ap.add_argument("--device", default=None, help="torch device (cpu/cuda)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="GPAP warmup runs (not timed)")
    ap.add_argument("--runs", type=int, default=10, help="Timed runs for each solver")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load formula once just to get num_vars for generation
    formula = CnfFormula(device)
    formula.parse_cnf_file(args.cnf)

    assumptions = generate_random_assumptions(
        formula.num_vars, args.K, args.assumptions, args.seed
    )

    print(f"\nFormula : {args.cnf}  ({formula.num_vars} vars, {formula.num_clauses} clauses)")
    print(f"Config  : K={args.K}  assumptions/batch={args.assumptions}  "
          f"device={device}  runs={args.runs}\n")

    # --- GPAP ---
    print("Running GPAP engine...")
    gpap = profile_gpap(args.cnf, assumptions, device, args.warmup, args.runs)
    print(f"  GPAP  : {gpap['mean_ms']:8.2f} ± {gpap['std_ms']:.2f} ms  "
          f"[{gpap['min_ms']:.2f}, {gpap['max_ms']:.2f}]  "
          f"conflict rate={gpap['conflict_rate']:.2%}  "
          f"avg props={gpap['avg_propagations']:.1f}")

    # --- External solver ---
    print(f"Running external solver (PySAT MinisatGH, {args.K} serial calls per run)...")
    ext = profile_external(args.cnf, assumptions, args.runs)
    print(f"  PySAT : {ext['mean_ms']:8.2f} ± {ext['std_ms']:.2f} ms  "
          f"[{ext['min_ms']:.2f}, {ext['max_ms']:.2f}]  "
          f"conflict rate={ext['conflict_rate']:.2%}")

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

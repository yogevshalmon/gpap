#!/usr/bin/env python3
"""Use GPAP BCP to learn conflict clauses and solve the augmented formula.

For each conflicting assumption set {a1, ..., aL}, the clause (¬a1 ∨ ... ∨ ¬aL)
is entailed by the original formula (BCP proves {a1,...,aL} → conflict).
Only non-trivial BCP-derived conflicts are learned (round_count > 1), so learned
clauses are never subsumed by an already-existing clause.

Modes:
  random   — K random assumption sets per round (default)
  pairs    — exhaustive enumeration of all (±v1, ±v2) pairs, v1 < v2
  triplets — exhaustive enumeration of all (±v1, ±v2, ±v3) triplets, v1<v2<v3

Workflow:
  1. Solve the original CNF with kissat (plain baseline, optional)
  2. Load formula ONCE; run GPAP in chunks → extract real_conflict clauses
  3. Solve original + learned clauses with kissat → compare

Usage:
    PYTHONPATH=. .venv/bin/python scripts/clause_learn_solve.py \\
        --cnf /path/to/formula.cnf --mode pairs --auto-k --device cuda

    PYTHONPATH=. .venv/bin/python scripts/clause_learn_solve.py \\
        --cnf /path/to/formula.cnf --mode random --K 65536 --L 2 --rounds 5
"""
import argparse
import math
import os
import sys
import time

import torch
from pysat.formula import CNF
from pysat.solvers import Solver

sys.path.insert(0, ".")
from gpap.clsdata.cnf_formula import CnfFormula
from gpap.solver.bcp_engine import BCPEngine
from gpap.solver.clause_learner import extract_conflict_clauses
from scripts.build_benchmark_suite import compute_k_safe
from scripts.profile_engine import (
    count_all_pairs,
    count_all_triplets,
    generate_all_pairs_chunks,
    generate_all_triplets_chunks,
    generate_random_assumptions,
    to_assumption_tensor,
)

_AVG_LITS_PER_CLAUSE = 4.0  # matches build_benchmark_suite default


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _solve(cnf: CNF, solver_name: str) -> tuple[str, float]:
    """Solve a CNF formula. Returns (result, wall_ms)."""
    t0 = time.perf_counter()
    with Solver(name=solver_name, bootstrap_with=cnf) as s:
        sat = s.solve()
    return ("SAT" if sat else "UNSAT"), (time.perf_counter() - t0) * 1e3


def _run_gpap_chunk(
    engine: BCPEngine,
    assumptions_dimacs: list[list[int]],
    device: torch.device,
    max_rounds: int | None = None,
    soft_stop_frac: float | None = None,
    min_rounds: int = 0,
    timed: bool = True,
) -> tuple[list[bool], float]:
    """One BCP pass using a pre-created engine.

    Calls engine.reset_state() in-place (no re-allocation when K is unchanged).
    Uses engine.real_conflict (round_count > 1 only) so learned clauses are never
    subsumed by an existing clause.

    Returns (real_conflict list, wall_ms). wall_ms is 0.0 when timed=False.
    """
    K = len(assumptions_dimacs)
    assumptions_tensor = to_assumption_tensor(assumptions_dimacs, device)
    engine.reset_state(K)           # in-place if K unchanged; init_state() only on K change
    if timed:
        _sync(device)
    t0 = time.perf_counter()
    engine.apply_assumptions_bulk(assumptions_tensor)
    engine.propagate(max_rounds=max_rounds, soft_stop_frac=soft_stop_frac,
                     min_rounds=min_rounds)
    if timed:
        _sync(device)
    wall_ms = (time.perf_counter() - t0) * 1e3 if timed else 0.0
    return engine.real_conflict.tolist(), wall_ms


def _dedup_clauses(clauses: list[list[int]]) -> list[list[int]]:
    """Remove duplicate clauses (order-insensitive)."""
    seen: dict[tuple, list[int]] = {}
    for c in clauses:
        key = tuple(sorted(c))
        if key not in seen:
            seen[key] = c
    return list(seen.values())


def _gpu_mem_bytes(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory * 0.5
    return 8.0e9  # conservative 8 GB for CPU


def main():
    ap = argparse.ArgumentParser(
        description="GPAP conflict-clause learning + kissat augmented solve"
    )
    ap.add_argument("--cnf", required=True, help="Path to DIMACS CNF file")
    ap.add_argument("--mode", default="random", choices=["random", "pairs", "triplets"],
                    help="Assumption generation mode (default: random)")
    ap.add_argument("--K", type=int, default=65536,
                    help="Chunk size: GPAP assumption sets per batch (default: 65536)")
    ap.add_argument("--auto-k", action="store_true",
                    help="Compute K from M and GPU memory (overrides --K)")
    ap.add_argument("--L", type=int, default=2,
                    help="Literals per assumption set for --mode random (default: 2)")
    ap.add_argument("--rounds", type=int, default=1,
                    help="Random rounds for --mode random (default: 1)")
    ap.add_argument("--device", default=None, help="torch device (cpu/cuda)")
    ap.add_argument("--warmup", type=int, default=2, help="GPAP warmup runs (default: 2)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    ap.add_argument("--solver", default="kissat404",
                    help="PySAT solver name for solve steps (default: kissat404)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip the plain-solve baseline")
    ap.add_argument("--no-solve", action="store_true",
                    help="Only learn clauses, skip the final augmented solve")
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="Hard cap on GPAP BCP rounds per chunk")
    ap.add_argument("--soft-stop-frac", type=float, default=None,
                    help="Soft stop: stop GPAP when per-round work drops below "
                         "this fraction of peak work")
    ap.add_argument("--min-rounds", type=int, default=0,
                    help="Minimum GPAP rounds before soft-stop may trigger (default: 0)")
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Load formula ONCE — immutable after this; BCPEngine.init_state() reuses it per chunk
    formula = CnfFormula(device)
    formula.parse_cnf_file(args.cnf)
    N, M = formula.num_vars, formula.num_clauses

    # K sizing
    if args.auto_k:
        K = compute_k_safe(M, N, _gpu_mem_bytes(device), cap=65536)
    else:
        K = args.K

    print(f"\nFormula : {os.path.basename(args.cnf)}  ({N:,} vars, {M:,} clauses)")
    print(f"Device  : {device}  solver={args.solver}  mode={args.mode}  K={K}")
    if args.mode == "random":
        print(f"          L={args.L}  rounds={args.rounds}")
    cutoff = ""
    if args.max_rounds is not None:
        cutoff += f"  max_rounds={args.max_rounds}"
    if args.soft_stop_frac is not None:
        cutoff += f"  soft_stop_frac={args.soft_stop_frac}  min_rounds={args.min_rounds}"
    if cutoff:
        print(f"GPAP    :{cutoff}")

    # Build the chunk generator and compute total_chunks for progress display
    if args.mode == "random":
        def _random_gen():
            for i in range(args.rounds):
                yield generate_random_assumptions(N, K, args.L, args.seed + i)
        chunk_gen = _random_gen()
        total_chunks = args.rounds
    elif args.mode == "pairs":
        chunk_gen = generate_all_pairs_chunks(N, K)
        total_chunks = math.ceil(count_all_pairs(N) / K)
        print(f"          total pairs: {count_all_pairs(N):,}  → {total_chunks} chunk(s)")
    else:  # triplets
        chunk_gen = generate_all_triplets_chunks(N, K)
        total_chunks = math.ceil(count_all_triplets(N) / K)
        print(f"          total triplets: {count_all_triplets(N):,}  → {total_chunks} chunk(s)")
    print()

    # --- Step 1: plain baseline solve ---
    baseline_ms: float | None = None
    result_plain: str | None = None
    if not args.no_baseline:
        print(f"[1/3] Plain solve ({args.solver}) ...", end=" ", flush=True)
        cnf_plain = CNF(from_file=args.cnf)
        result_plain, baseline_ms = _solve(cnf_plain, args.solver)
        print(f"→ {result_plain}  {baseline_ms:.1f} ms")
    else:
        print("[1/3] Plain solve skipped  (--no-baseline)")

    # --- Step 2: GPAP clause learning ---
    print(f"[2/3] GPAP clause learning ...")
    total_learned: list[list[int]] = []
    total_gpap_ms = 0.0

    # Single engine, reused across all chunks via reset_state() (no re-allocation)
    engine = BCPEngine(formula)
    warmed_up = False

    for chunk_idx, chunk in enumerate(chunk_gen, 1):
        if not warmed_up:
            for _ in range(args.warmup):
                _run_gpap_chunk(engine, chunk, device,
                                max_rounds=args.max_rounds,
                                soft_stop_frac=args.soft_stop_frac,
                                min_rounds=args.min_rounds,
                                timed=False)
            warmed_up = True

        conflicts, ms = _run_gpap_chunk(
            engine, chunk, device,
            max_rounds=args.max_rounds, soft_stop_frac=args.soft_stop_frac,
            min_rounds=args.min_rounds,
        )
        raw = extract_conflict_clauses(chunk, conflicts)
        total_learned.extend(raw)
        total_gpap_ms += ms
        conflict_rate = sum(conflicts) / len(conflicts) if conflicts else 0.0
        print(f"  [{chunk_idx:4d}/{total_chunks}] {ms:>7.0f} ms  "
              f"conflict={conflict_rate:.1%}  +{len(raw)} clauses  "
              f"(total: {len(total_learned):,})")

    learned = _dedup_clauses(total_learned)
    dup_dropped = len(total_learned) - len(learned)
    print(f"\n  Total learned: {len(total_learned):,}  →  {len(learned):,} unique"
          + (f"  ({dup_dropped:,} duplicates dropped)" if dup_dropped else ""))
    print(f"  GPAP time: {total_gpap_ms:.1f} ms total  ({total_gpap_ms/total_chunks:.0f} ms/chunk avg)")

    if not learned:
        print("\nNo non-trivial conflict clauses learned.")
        return

    if args.no_solve:
        print("\n(--no-solve: skipping augmented solve)")
        return

    # --- Step 3: augmented solve ---
    print(f"\n[3/3] Augmented solve ({args.solver} + {len(learned):,} clauses) ...",
          end=" ", flush=True)
    cnf_aug = CNF(from_file=args.cnf)
    for clause in learned:
        cnf_aug.append(clause)
    result_aug, aug_ms = _solve(cnf_aug, args.solver)
    print(f"→ {result_aug}  {aug_ms:.1f} ms")

    # --- Summary ---
    print()
    print("=" * 58)
    if baseline_ms is not None:
        print(f"  Plain solve      : {baseline_ms:>9.1f} ms  [{result_plain}]")
    print(f"  GPAP learning    : {total_gpap_ms:>9.1f} ms  (+{len(learned):,} clauses, "
          f"{M:,} → {M + len(learned):,})")
    print(f"  Augmented solve  : {aug_ms:>9.1f} ms  [{result_aug}]")
    if baseline_ms is not None:
        total_ms = total_gpap_ms + aug_ms
        solve_speedup = baseline_ms / aug_ms if aug_ms > 0 else float("inf")
        e2e_speedup = baseline_ms / total_ms if total_ms > 0 else float("inf")
        print(f"  {'─' * 48}")
        print(f"  Solve speedup    : {solve_speedup:>8.2f}x  (plain / augmented-only)")
        print(f"  End-to-end       : {e2e_speedup:>8.2f}x  (plain / GPAP+augmented)")
    print("=" * 58)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Use GPAP BCP to learn conflict clauses and solve the augmented formula.

For each conflicting assumption set {a1, ..., aL}, the clause (¬a1 ∨ ... ∨ ¬aL)
is entailed by the original formula (BCP proves {a1,...,aL} → conflict).
Only non-trivial BCP-derived conflicts are learned (round_count > 1, via
engine.real_conflict), so learned clauses are never subsumed by an existing clause.

Modes:
  random   — K random assumption sets per round (default), generated on-GPU
  pairs    — exhaustive enumeration of all (±v1, ±v2) pairs, v1 < v2 (on-GPU)
  triplets — exhaustive enumeration of all (±v1, ±v2, ±v3) triplets, v1<v2<v3

End-to-end timing: assumption generation, engine reset, and apply+propagate are
timed separately, plus a true wall-clock for the whole learning loop — so the gap
between GPU kernel time and real wall time is explicit. --compare-propagate adds a
PySAT baseline over the same assumption sets.

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
from gpap.clsdata.occ_list import lit_index_to_ext_lit
from gpap.solver.bcp_engine import BCPEngine
from gpap.solver.clause_learner import extract_conflict_clauses_tensor
from gpap.utils.gpu_budget import compute_k_safe, gpu_mem_available
from scripts.profile_engine import (
    count_all_pairs,
    count_all_triplets,
    generate_all_pairs_chunks_gpu,
    generate_all_triplets_chunks_gpu,
    generate_random_assumptions_gpu,
)
from gpap.solver.external_solver import ExternalSolver

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


def _tensor_to_dimacs(t: torch.Tensor) -> list[list[int]]:
    """Convert a (K, L) lit-index tensor to DIMACS assumption sets (−1 = padding).

    These are the *assumed* literals (not negated) — for feeding PySAT propagate.
    """
    return [[lit_index_to_ext_lit(i) for i in row if i >= 0] for row in t.cpu().tolist()]


def _dedup_clauses(clauses: list[list[int]]) -> list[list[int]]:
    """Remove duplicate clauses (order-insensitive)."""
    seen: dict[tuple, list[int]] = {}
    for c in clauses:
        key = tuple(sorted(c))
        if key not in seen:
            seen[key] = c
    return list(seen.values())




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
    ap.add_argument("--compare-propagate", action="store_true",
                    help="Run a PySAT propagate baseline over the SAME assumption sets "
                         "and report sets/sec for GPAP vs PySAT (slow for large totals)")
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

    # Load formula ONCE — immutable after this; BCPEngine reuses it per chunk
    formula = CnfFormula(device)
    formula.parse_cnf_file(args.cnf)
    N, M = formula.num_vars, formula.num_clauses

    # K sizing: no cap when auto-computing — use full memory-safe K
    if args.auto_k:
        max_cls    = int(formula.clause_sizes.max().item())
        total_lits = int(formula.clause_sizes.sum().item())
        K = compute_k_safe(M, N, gpu_mem_available(device), cap=None,
                           max_clause_size=max_cls, total_lits=total_lits,
                           deduct_formula=False)
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

    # Build the chunk generator (yields (K, L) GPU lit-index tensors) and total_chunks
    if args.mode == "random":
        def _random_gen():
            for i in range(args.rounds):
                yield generate_random_assumptions_gpu(N, K, args.L, device, args.seed + i)
        chunk_gen = _random_gen()
        total_chunks = args.rounds
    elif args.mode == "pairs":
        chunk_gen = generate_all_pairs_chunks_gpu(N, K, device)
        total_chunks = math.ceil(count_all_pairs(N) / K)
        print(f"          total pairs: {count_all_pairs(N):,}  → {total_chunks} chunk(s)")
    else:  # triplets
        chunk_gen = generate_all_triplets_chunks_gpu(N, K, device)
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

    # --- Step 2: GPAP clause learning (per-phase timing + true wall-clock) ---
    print(f"[2/3] GPAP clause learning ...")
    total_learned: list[list[int]] = []
    t_gen = t_reset = t_prop = t_extract = 0.0
    total_sets = 0
    pysat_dimacs: list[list[int]] = []  # only populated with --compare-propagate

    # Single engine, reused across all chunks via reset_state() (no re-allocation)
    engine = BCPEngine(formula)
    warmed_up = False

    gen_iter = iter(chunk_gen)
    chunk_idx = 0
    while True:
        # --- phase: generation (next chunk → GPU tensor) ---
        _sync(device); g0 = time.perf_counter()
        try:
            chunk_t = next(gen_iter)
        except StopIteration:
            break
        _sync(device); t_gen += (time.perf_counter() - g0) * 1e3
        chunk_idx += 1
        K_chunk = chunk_t.size(0)
        total_sets += K_chunk

        # Warmup on the first chunk only (kernel caching) — excluded from every timer
        if not warmed_up and args.warmup > 0:
            for _ in range(args.warmup):
                engine.reset_state(K_chunk)
                engine.apply_assumptions_bulk(chunk_t)
                engine.propagate(max_rounds=args.max_rounds,
                                 soft_stop_frac=args.soft_stop_frac,
                                 min_rounds=args.min_rounds)
            _sync(device)
        warmed_up = True

        # --- phase: reset state (in-place when K unchanged) ---
        _sync(device); r0 = time.perf_counter()
        engine.reset_state(K_chunk)
        _sync(device); t_reset += (time.perf_counter() - r0) * 1e3

        # --- phase: apply assumptions + propagate ---
        _sync(device); p0 = time.perf_counter()
        engine.apply_assumptions_bulk(chunk_t)
        engine.propagate(max_rounds=args.max_rounds, soft_stop_frac=args.soft_stop_frac,
                         min_rounds=args.min_rounds)
        _sync(device); ms = (time.perf_counter() - p0) * 1e3
        t_prop += ms

        # --- phase: extract learned clauses (includes the device→host transfer) ---
        _sync(device); e0 = time.perf_counter()
        raw = extract_conflict_clauses_tensor(chunk_t, engine.real_conflict)
        _sync(device); t_extract += (time.perf_counter() - e0) * 1e3
        total_learned.extend(raw)
        if args.compare_propagate:
            pysat_dimacs.extend(_tensor_to_dimacs(chunk_t))

        # len(raw) == #real_conflict rows (each conflicting row yields one clause),
        # so this is the conflict rate without an extra GPU sync.
        conflict_rate = len(raw) / K_chunk
        print(f"  [{chunk_idx:4d}/{total_chunks}] {ms:>7.0f} ms  "
              f"conflict={conflict_rate:.1%}  +{len(raw)} clauses  "
              f"(total: {len(total_learned):,})")

    # Wall = sum of the explicitly-timed phases — warmup is never added to these,
    # so this is the honest production cost (no benchmarking artifacts, no terminal I/O).
    wall_total_ms = t_gen + t_reset + t_prop + t_extract

    learned = _dedup_clauses(total_learned)
    dup_dropped = len(total_learned) - len(learned)
    print(f"\n  Total learned: {len(total_learned):,}  →  {len(learned):,} unique"
          + (f"  ({dup_dropped:,} duplicates dropped)" if dup_dropped else ""))
    print(f"  Phase timing ({total_sets:,} assumption sets over {chunk_idx} chunk(s)):")
    print(f"    generation : {t_gen:>9.1f} ms")
    print(f"    reset      : {t_reset:>9.1f} ms")
    print(f"    propagate  : {t_prop:>9.1f} ms  (GPU kernel)")
    print(f"    extract    : {t_extract:>9.1f} ms  (device→host)")
    print(f"    {'─' * 28}")
    print(f"    GPAP wall  : {wall_total_ms:>9.1f} ms  "
          f"({total_sets / (wall_total_ms / 1e3):,.0f} sets/s)")

    # print the clauses if there aren't too many
    print("\nLearned clauses:"
          + ("  (omitted, too many)" if len(learned) > 20 else ""))
    for c in learned[:20]:
        print("  " + " ".join(str(lit) for lit in c))

    # --- Optional: PySAT propagate baseline over the SAME assumption sets ---
    if args.compare_propagate:
        print(f"\nPySAT propagate baseline over the same {len(pysat_dimacs):,} sets ...",
              end=" ", flush=True)
        ext = ExternalSolver(args.cnf)
        _, _, pysat_s = ext.propagate_many(pysat_dimacs)
        pysat_ms = pysat_s * 1e3
        print(f"→ {pysat_ms:.1f} ms")
        gpap_wall_sps = total_sets / (wall_total_ms / 1e3) if wall_total_ms > 0 else float("inf")
        pysat_sps = len(pysat_dimacs) / pysat_s if pysat_s > 0 else float("inf")
        speedup = pysat_ms / wall_total_ms if wall_total_ms > 0 else float("inf")
        print(f"  GPAP  : {gpap_wall_sps:>12,.0f} sets/s  (wall)")
        print(f"  PySAT : {pysat_sps:>12,.0f} sets/s")
        print(f"  GPAP wall speedup vs PySAT: {speedup:.1f}x")

    if not learned:
        print("\nNo non-trivial conflict clauses learned.")
        return

    # validate the learned clauses with the external solver (sanity check)
    print(f"\nValidating {len(learned):,} learned clauses (PySAT propagate) ...",
          end=" ", flush=True)
    external_solver = ExternalSolver(args.cnf)
    for clause in learned:
        # check that the clause is entailed by the original formula
        # (i.e., that the negation of the clause is unsat under propagation)
        assumptions = [-lit for lit in clause]  # assume the negation of the learned clause
        res, _, _ = external_solver.propagate_assumptions(assumptions)
        if not res:
            print(f"\n  Clause {clause} failed validation! Expected conflict but got {res}")
            return
    print("OK (all entailed)")

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
    print(f"  GPAP learning    : {wall_total_ms:>9.1f} ms wall  "
          f"({t_prop:.1f} ms kernel)  (+{len(learned):,} clauses, "
          f"{M:,} → {M + len(learned):,})")
    print(f"  Augmented solve  : {aug_ms:>9.1f} ms  [{result_aug}]")
    if baseline_ms is not None:
        total_ms = wall_total_ms + aug_ms
        solve_speedup = baseline_ms / aug_ms if aug_ms > 0 else float("inf")
        e2e_speedup = baseline_ms / total_ms if total_ms > 0 else float("inf")
        print(f"  {'─' * 48}")
        print(f"  Solve speedup    : {solve_speedup:>8.2f}x  (plain / augmented-only)")
        print(f"  End-to-end       : {e2e_speedup:>8.2f}x  (plain / GPAP-wall + augmented)")
    print("=" * 58)


if __name__ == "__main__":
    main()

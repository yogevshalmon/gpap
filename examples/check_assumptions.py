#!/usr/bin/env python3
"""Checking many assumption sets at once — the engine's core primitive.

    python examples/check_assumptions.py [formula.cnf]

Every set is propagated to a BCP fixpoint independently of the others. That is the operation
GPAP parallelizes: not one big propagation, but K of them at once. Pass as many sets as you
like — they are split into GPU-sized chunks internally and come back in input order.
"""
import sys
import time

import numpy as np

from pathlib import Path

from gpap.engine import AssumptionChecker

# Anchored on this file, not on the working directory: `python examples/…` from anywhere but
# the repo root used to be a FileNotFoundError. Same spelling as examples/mine_clauses.py.
DEFAULT = Path(__file__).resolve().parent.parent / "benchmarks" / "sampling" / "10.sk_1_46.cnf"


def tutorial() -> None:
    """A three-variable formula, so every step can be checked by hand:
        (x1 v x2) ^ (~x1 v x3) ^ (~x2 v ~x3)
    """
    checker = AssumptionChecker()
    checker.formula.init_formula(3, [[1, 2], [-1, 3], [-2, -3]])

    sets = [[1], [-1, -2], [2, 3], []]
    labels = ["x1=T", "x1=F,x2=F", "x2=T,x3=T", "(no assumptions)"]

    # `assignments=True` also returns the (n, 2N) state. It is the expensive part of a large
    # call, so it is opt-in; 'conflict' and 'num_propagated' always come back.
    r = checker.check_assumptions(sets, assignments=True)
    conflict = r["conflict"].cpu().numpy()
    implied = r["num_propagated"].cpu().numpy()
    assign = r["assignment"].cpu().numpy()

    print(f"{len(sets)} assumption sets, {int(conflict.sum())} conflicted:\n")
    for i, label in enumerate(labels):
        if conflict[i]:
            print(f"  {label:16s} -> CONFLICT")
            continue
        # Literal l of variable v: TRUE at index 2(v-1), FALSE at 2(v-1)+1. Neither set = free.
        state = []
        for v in range(1, 4):
            t, f = assign[i][2 * (v - 1)], assign[i][2 * (v - 1) + 1]
            state.append(f"x{v}={'T' if t else 'F' if f else 'free'}")
        print(f"  {label:16s} -> ok ({implied[i]} implied):  " + "  ".join(state))
    print("\nRow 0 is the point: assuming x1 alone forces x3 (via ~x1 v x3) and then ~x2\n"
          "(via ~x2 v ~x3) — two implications from one assumption, on the GPU.")


def survey(path: str) -> None:
    """The shape of most real uses: a large population of candidate assumptions, filtered in
    bulk. 200,000 sets is more than one launch holds, which the caller never has to know."""
    checker = AssumptionChecker(path)
    nv = checker.formula.num_vars
    n_sets, n_lits = 200_000, 8

    rng = np.random.default_rng(12345)
    picks = rng.integers(1, nv + 1, size=(n_sets, n_lits))
    signs = rng.integers(0, 2, size=(n_sets, n_lits)) * 2 - 1
    sets = (picks * signs).tolist()

    t0 = time.perf_counter()
    r = checker.check_assumptions(sets)          # flags only: ~5 bytes per set
    secs = time.perf_counter() - t0

    conflicted = int(r["conflict"].sum())
    print(f"{path}: {nv:,} vars, {checker.formula.num_clauses:,} clauses")
    print(f"{n_sets:,} random {n_lits}-literal assumption sets, "
          f"{checker.chunk_size(n_sets, n_lits):,} per launch")
    print(f"  {n_sets - conflicted:,} survived BCP, {conflicted:,} conflicted  "
          f"({secs * 1e3:.1f} ms, {n_sets / secs:.2e} sets/s)")
    print(f"  {float(r['num_propagated'].double().mean()):.1f} implications per set on average")


def main(argv: list) -> int:
    if len(argv) > 1:
        survey(argv[1])
    else:
        tutorial()
        print()
        survey(DEFAULT)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

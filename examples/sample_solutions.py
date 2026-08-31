#!/usr/bin/env python3
"""Sampling a CNF's solutions with GPAP — the three-line version, then what it gives you.

    python examples/sample_solutions.py [formula.cnf]

There is no batch size to pick and no tuning to do: GPAP sizes the GPU batch to the card and the
formula, picks the propagate kernel, orders decisions so propagation does the work, and verifies
on the device that every row it returns satisfies every clause.
"""
import sys

import numpy as np

import gpap

DEFAULT = "benchmarks/sampling/10.sk_1_46.cnf"


def main(path: str) -> int:
    # ---- the whole API, for the common case ------------------------------------------------
    samples, info = gpap.sample(path, n=50_000, seed=5)

    # `samples` is a numpy (rows, S) uint8 bit matrix: one row per solution, one column per
    # SUPPORT variable. It is projected on the GPU, so only these bits cross the bus.
    print(f"{path}: {info['nv']:,} vars, {info['n_clauses']:,} clauses")
    print(f"  {samples.shape[0]:,} solutions x {samples.shape[1]} variables  ({samples.dtype})")
    print(f"  {info['samples_per_s']:,.0f} solutions/s, {info['success_rate']:.0%} of "
          f"trajectories succeeded, sound-check={info['sound_check']}")

    # `info['cols']` names the columns: column j is the value of DIMACS variable cols[j].
    cols = info["cols"]
    print(f"  columns are variables {cols[:6]}{' ...' if len(cols) > 6 else ''}")
    for i in range(min(3, len(samples))):
        lits = [v if samples[i, j] else -v for j, v in enumerate(cols)]
        print(f"    solution {i}: {' '.join(map(str, lits[:12]))}"
              f"{' ...' if len(lits) > 12 else ''}")

    # ---- the sampled object is a DISTRIBUTION, so ask distribution questions ----------------
    distinct = len(np.unique(samples, axis=0))
    print(f"  {distinct:,} distinct of {len(samples):,} drawn "
          f"({distinct / max(len(samples), 1):.1%})")
    if len(samples):
        ones = samples.mean(axis=0)
        print(f"  per-variable P(true): min {ones.min():.2f}, max {ones.max():.2f} "
              f"(0.5 everywhere = no variable is pinned by the formula)")

    # ---- other things worth knowing about --------------------------------------------------
    #   gpap.sample(path, n=..., full_models=True)   -> full DIMACS models instead of bits
    #   gpap.sample(path, n=..., max_samples=1000)   -> cap what is kept on the host
    #   gpap.sample(path, n=..., uniformity="exact") -> add a uniformity readout (extra pass)
    #   gpap.sample_clauses(nv, clauses, n=...)      -> an in-memory clause list
    #   gpap.sample_circuit("b10.bench", cycles=25)  -> unroll a circuit and sample its inputs
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))

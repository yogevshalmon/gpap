#!/usr/bin/env python3
"""Prove clauses a formula already entails — the probe primitive, and the mining built on it.

    python examples/mine_clauses.py [formula.cnf]

Assume a set of literals, propagate, and see whether BCP hits a conflict. If it does, the formula
entails the NEGATION of that set — a clause. Sweep every k-subset of a literal pool and you have
every entailed clause of width k that propagation alone can see:

    k = 1   assuming l conflicts       =>  the unit (~l)        the BCP-visible backbone
    k = 2   assuming (a, b) conflicts  =>  the clause (~a | ~b)

Every clause found is entailed, so adding it leaves the solution set EXACTLY unchanged while
making later propagation stronger. That is what makes this safe to run before anything else.

Two layers are shown, in the order you would reach for them:
  1. `LiteralProbe` — the primitive: one pool, one width, one budget.
  2. `mine_entailed` — the strategy: several widths, a fixpoint at width 1, subsumption filtering.

The command line front for layer 2 is `gpap mine` (see `gpap mine --help`).
"""
import sys
from pathlib import Path

import torch
from pysat.formula import CNF

from gpap.engine.probe import LiteralProbe, ProbeBudget
from gpap.mining import mine_entailed

DEFAULT = Path(__file__).resolve().parent.parent / "benchmarks" / "sampling" / "10.sk_1_46.cnf"


def main(path: str) -> int:
    if not torch.cuda.is_available():
        print("this example needs a CUDA GPU (there is no CPU execution path)")
        return 1
    device = torch.device("cuda")
    cnf = CNF(from_file=path)
    print(f"{Path(path).name}: {cnf.nv:,} variables, {len(cnf.clauses):,} clauses\n")

    # ---------------------------------------------------------------- 1. the primitive
    probe = LiteralProbe(cnf.nv, cnf.clauses, device)
    free = probe.free_literals()
    print(f"free literals (what is worth probing at all): {len(free):,}")
    print(f"  the other {2 * cnf.nv - len(free):,} were already settled by the formula's own "
          f"unit propagation, so probing them would prove nothing.\n")

    # ALWAYS ask what a width costs before running it. The count is exact and needs no GPU work,
    # and it is the difference between a two-second sweep and one that cannot be run at all.
    for k in (1, 2, 3):
        print(f"  width {k}: {probe.count(size=k):,} candidate tuples")
    print()

    failed = probe.probe(size=1)                       # pool=None => every free literal
    print(f"width 1 swept {failed.probed:,} literals in {failed.seconds:.2f}s "
          f"-> {len(failed):,} failed literal(s)")
    if failed.exhausted:
        # This is the property worth understanding: an EXHAUSTED sweep that finds nothing is a
        # proof that nothing of that width exists (that BCP can see). A sweep cut short by a
        # budget proves nothing at all — same empty result, completely different meaning.
        print("  the sweep was exhaustive, so this list is COMPLETE: no other literal fails.")
    for lit in failed.conflicting[:5].reshape(-1):
        print(f"  assuming {int(lit):+d} conflicts  =>  ({-int(lit):+d}) is entailed")
    print()

    # A narrower, deeper question: pairs over just the first few free variables, under a budget.
    pool = free[:40]
    pairs = probe.probe(pool, size=2, budget=ProbeBudget(seconds=5.0, max_tuples=100_000))
    print(f"width 2 over a {len(pool)}-literal pool: {pairs.probed:,} of {pairs.total:,} probed "
          f"-> {len(pairs):,} binary clause(s)")
    for row in pairs.conflicting[:3]:
        a, b = int(row[0]), int(row[1])
        print(f"  assuming ({a:+d}, {b:+d}) conflicts  =>  ({-a:+d} | {-b:+d}) is entailed")
    del probe
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 2. the strategy
    print("\nmine_entailed(): several widths, a fixpoint at width 1, subsumption filtering")
    res = mine_entailed(cnf.nv, cnf.clauses, device, widths=(1, 2), seconds=10.0)
    print(f"  {res.summary()}")
    for w, n in sorted(res.by_width.items()):
        why = "exhaustive" if res.exhausted.get(w) else res.skipped.get(w, "stopped early")
        print(f"    width {w}: {n:,} clause(s)  [{why}]")
    print(f"\n  stronger formula = {len(cnf.clauses):,} original + {len(res.clauses):,} mined "
          f"clauses, with EXACTLY the same solutions.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT)))

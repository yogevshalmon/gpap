#!/usr/bin/env python3
"""What each literal implies — one sweep, three classic preprocessing results.

    python examples/literal_implications.py [formula.cnf]

Probe a literal and propagate. Either it CONFLICTS (the literal is failed, so its negation is a
unit) or you get its CLOSURE: everything unit propagation derives from it. That single object is
what three well-known techniques read:

    failed literals       the conflict bit                       -> units, the BCP backbone
    hyper-binary res.     m in closure(l) IS the clause (~l | m)
    equivalent literals   l == m when each is in the other's closure  -> SAT sweeping / ELS

On a CPU each literal is its own propagation, which is why solvers probe only roots of the binary
implication graph and accept the coverage they lose. Here every literal is one ROW of one launch,
so the exhaustive answer costs about what a partial one would — that is the whole point of GPAP.
"""
import sys
from pathlib import Path

import torch
from pysat.formula import CNF

from gpap.implications import (
    closure_bytes,
    equivalence_classes,
    hyper_binary,
    implications,
)

DEFAULT = Path(__file__).resolve().parent.parent / "benchmarks" / "sampling" / "10.sk_1_46.cnf"


def main(path: str) -> int:
    if not torch.cuda.is_available():
        print("this example needs a CUDA GPU (there is no CPU execution path)")
        return 1
    cnf = CNF(from_file=path)
    print(f"{Path(path).name}: {cnf.nv:,} variables, {len(cnf.clauses):,} clauses\n")

    # ALWAYS ask the cost first. The closure is P x 2N and dense, so on a big unroll it is
    # gigabytes — `implications` refuses rather than reserving, and the fix is a narrower pool.
    print(f"closure for every free literal would need "
          f"~{closure_bytes(cnf.nv, 2 * cnf.nv) / 2**20:.0f} MB (upper bound)\n")

    imp = implications(cnf.nv, cnf.clauses, torch.device("cuda"))
    print(f"probed {len(imp):,} free literals in ONE batched sweep\n")

    # ---------------------------------------------------------------- 1. failed literals
    print(f"failed literals: {imp.failed.size:,}  ->  that many proven units")
    for u in imp.units[:5]:
        print(f"  assuming {-u[0]:+d} conflicts  =>  ({u[0]:+d}) is entailed")

    # ---------------------------------------------------------------- 2. equivalent literals
    classes = equivalence_classes(imp)
    removable = sum(len(c) - 1 for c in classes) // 2      # classes come in +/- mirror pairs
    print(f"\nequivalence classes: {len(classes):,}  ->  {removable:,} variables removable")
    for cl in classes[:3]:
        print(f"  {' == '.join(f'{x:+d}' for x in cl)}")
    print("  (each is mutual implication: every model agrees on them, so one name suffices —")
    print("   feed `equivalence_edges(imp)` to preprocess.equivalent_literal_substitution)")

    # ---------------------------------------------------------------- 3. hyper-binary resolvents
    hb = hyper_binary(imp)
    print(f"\nhyper-binary resolvents: {len(hb):,} binary clauses the formula entails "
          f"({len(hb) / max(1, len(cnf.clauses)):.0f}x the formula)")
    for c in hb[:3]:
        print(f"  ({c[0]:+d} | {c[1]:+d})")
    print("  NOTE these are entailed but LOGICALLY REDUNDANT by construction — propagation")
    print("  already derives them, which is how they were found. There is one per closure EDGE,")
    print("  so the set is normally much larger than the formula and is UNFILTERED: this is an")
    print("  analysis product, not something to concatenate onto a CNF. Making it useful needs")
    print("  transitive reduction (see the module's FUTURE WORK). For clauses that are actually")
    print("  worth adding, use `gpap mine`.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT)))

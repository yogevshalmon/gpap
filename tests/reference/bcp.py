"""Pure-Python reference BCP, the oracle for the engine's kernels.

BCP fixpoints are confluent: for a formula and a decision set, the fixpoint and whether it
conflicts do not depend on propagation order. So this simple worklist implementation and the
engine must agree exactly on the conflict flag and on the assigned literals of every row that did
not conflict. A conflicted row's partial state depends on order and is not compared.
"""
from collections import defaultdict


def lit_idx(ext: int) -> int:
    return 2 * (abs(ext) - 1) + (0 if ext > 0 else 1)


def ref_bcp(clauses: list, decisions: list[int]) -> tuple[bool, frozenset]:
    """Unit-propagate `decisions` (literal indices, applied in order; duplicates are skipped,
    l and ~l together conflict) over `clauses` (DIMACS literals), including the formula's own
    unit clauses. Returns (conflict, set of assigned literal indices)."""
    idx_clauses = [[lit_idx(x) for x in c] for c in clauses]
    contains: dict[int, list[int]] = defaultdict(list)
    for ci, lits in enumerate(idx_clauses):
        for l in lits:
            contains[l].append(ci)

    asg: set[int] = set()
    newly: list[int] = []
    conflict = False

    def assign(lit: int) -> None:
        nonlocal conflict
        if lit in asg:
            return
        if (lit ^ 1) in asg:
            conflict = True
            return
        asg.add(lit)
        newly.append(lit)

    for lit in decisions:
        if conflict:
            break
        assign(lit)

    work = list(range(len(idx_clauses)))      # initial pass catches the formula's own units
    while not conflict and (work or newly):
        while newly and not conflict:
            lit = newly.pop()
            work.extend(contains[lit ^ 1])    # clauses that just lost a literal
        if conflict or not work:
            break
        lits = idx_clauses[work.pop()]
        if any(l in asg for l in lits):       # satisfied: never forces
            continue
        free = [l for l in lits if (l ^ 1) not in asg]
        if not free:                          # every literal falsified
            conflict = True
        elif len(free) == 1:                  # unit: force the survivor
            assign(free[0])
    return conflict, frozenset(asg)


def assigned_set(assignment_row) -> frozenset:
    """Engine (2N,) int8 assignment row -> set of TRUE lit indices (torch or numpy)."""
    return frozenset(i for i, v in enumerate(assignment_row.tolist()) if v == 1)

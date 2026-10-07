"""Unit tests for the CPU preprocessing primitives: unit_closure and
equivalent_literal_substitution (ELS).

End-to-end ELS soundness is in tests/sampling/test_sampler.py (test_els_*). Binary edges are
built the way simplify_cnf_els builds them: (l1 ∨ l2) gives ¬l1→l2 and ¬l2→l1.
"""
from gpap.sampling.preprocess import equivalent_literal_substitution, unit_closure
from gpap.engine.formula import ext_lit_to_lit_index


def _norm(clauses):
    return sorted(tuple(sorted(set(c))) for c in clauses)


def _binary_edges(clauses):
    """Implication edges from the binary clauses (the sampler's ELS edge source)."""
    edges = []
    for c in clauses:
        if len(c) == 2:
            i1, i2 = ext_lit_to_lit_index(c[0]), ext_lit_to_lit_index(c[1])
            edges.append((i1 ^ 1, i2))
            edges.append((i2 ^ 1, i1))
    return edges


# ---------------------------------------------------------------- unit closure
def test_propagates_and_keeps_units():
    # unit (1) ⇒ in (¬1 ∨ 2) strip ¬1 → (2) ⇒ assign 2; (2 ∨ 3) satisfied.
    clauses = [[-1, 2], [2, 3]]
    out, assigned, status = unit_closure(clauses, units=[[1]])
    assert status == "ok"
    assert assigned == {1: True, 2: True}
    norm = _norm(out)
    assert (1,) in norm and (2,) in norm          # asserted units retained (equivalence)
    assert (2, 3) not in norm                       # satisfied clause dropped


def test_detects_unsat():
    # units (1) and (¬1) contradict.
    out, _, status = unit_closure([[5, 6]], units=[[1], [-1]])
    assert status == "unsat"


def test_existing_units_picked_up():
    # unit clauses already in the formula cascade the same way
    clauses = [[1], [-1, 2], [-2, 3], [3, 4, 5]]
    out, assigned, status = unit_closure(clauses, units=[])
    assert status == "ok"
    assert assigned == {1: True, 2: True, 3: True}


# ---------------------------------------------------------------- ELS
def _els(clauses, nv):
    return equivalent_literal_substitution(clauses, _binary_edges(clauses), nv)


def test_merges_equivalent_pair():
    # (¬1∨2)∧(¬2∨1) ⇒ 1≡2; the extra clause (1∨3) keeps 1 alive. Var 2 is removed.
    clauses = [[-1, 2], [-2, 1], [1, 3]]
    red, var_map, stats = _els(clauses, 3)
    assert stats["status"] == "ok"
    assert stats["vars_removed"] == 1 and var_map == {2: 1}
    assert 2 not in {abs(l) for c in red for l in c}     # var 2 substituted away


def test_chain_collapses_to_one_var():
    # 1≡2≡3 via a cycle of implications; two of the three vars vanish.
    clauses = [[-1, 2], [-2, 1], [-2, 3], [-3, 2], [1, 4]]
    red, var_map, stats = _els(clauses, 4)
    assert stats["vars_removed"] == 2
    assert {abs(l) for c in red for l in c} <= {1, 4}


def test_equivalence_to_negation_is_unsat():
    # A class holding both a literal and its negation ⇒ UNSAT. Edges make 1≡2 and 1≡¬2
    # (lit indices: 1→0, ¬1→1, 2→2, ¬2→3), so union closes 0,1,2,3 into one class.
    edges = [[0, 2], [2, 0], [0, 3], [3, 0]]
    red, var_map, stats = equivalent_literal_substitution([[1, 2], [3]], edges, 3)
    assert stats["status"] == "unsat"


def test_no_equivalences_is_noop():
    # All clauses ≥3 literals → no binary edges → nothing merges.
    clauses = [[1, 2, 3], [2, 3, 4], [-1, 4, 5]]
    red, var_map, stats = _els(clauses, 5)
    assert stats["vars_removed"] == 0 and var_map == {}
    assert _norm(red) == _norm(clauses)

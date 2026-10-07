"""The Python face of entry point 1 (`BCPEngine`): small hand-checked scenarios, the flush cache,
counters, `check_sets` and edge cases. The fixpoint against the reference propagator, across
kernels and splits, is tests/engine/test_propagation.py.
"""
import os

import numpy as np
import pytest
import torch

from gpap.engine.bcp_engine import BCPEngine, make_assumption_tensor
from gpap.engine.formula import CnfFormula, ext_lit_to_lit_index

from tests.conftest import TEST_DATA_DIR
from tests.support import DEV

pytestmark = pytest.mark.gpu


def _engine(clauses, num_vars, K):
    formula = CnfFormula(DEV)
    formula.init_formula(num_vars, clauses)
    engine = BCPEngine(formula)
    engine.reset(K)
    return engine


def _assum(list_of_lists):
    return make_assumption_tensor(list_of_lists, DEV)


# ---------------------------------------------------------------- scenarios

def test_reset_state():
    engine = _engine([[1, 2], [-1, 3]], num_vars=3, K=2)
    sat = engine.sat_rows(torch.arange(2, device=DEV))
    assert sat.shape == (2, 2)
    assert sat[0].tolist() == [2, 2]
    assert engine.assignment.shape == (2, 6)
    assert (engine.assignment == 0).all()
    assert engine.active.all() and not engine.conflict.any()


def test_apply_assumption_no_conflict():
    # (x1 OR x2), (~x1 OR x3): x1=T falsifies ~x1 in clause 1 -> unit -> x3 forced
    engine = _engine([[1, 2], [-1, 3]], num_vars=3, K=1)
    engine.apply_and_propagate(_assum([[ext_lit_to_lit_index(1)]]))
    sat = engine.sat_rows(torch.tensor([0], device=DEV))[0]
    assert sat[0].item() == 2      # clause 0 lost nothing
    assert sat[1].item() == 1      # clause 1 lost ~x1; its survivor x3 was forced TRUE
    assert not engine.conflict[0].item()
    assert engine.assignment[0, ext_lit_to_lit_index(3)].item() == 1


def test_unit_clauses_flushed_at_reset():
    # the formula's own unit (x3) is asserted by the construction-time flush cache
    engine = _engine([[3], [-3, 2], [1, -2]], num_vars=3, K=2)
    assert engine.assignment[0, ext_lit_to_lit_index(3)].item() == 1
    assert engine.assignment[0, ext_lit_to_lit_index(2)].item() == 1   # implied by (-3, 2)
    assert engine.assignment[1, ext_lit_to_lit_index(1)].item() == 1   # implied by (1, -2)
    assert not engine.conflict.any()


def test_contradicting_units_conflict_at_construction():
    engine = _engine([[1], [-1]], num_vars=1, K=4)
    assert engine.flush_conflict
    assert engine.conflict.all() and not engine.active.any()


def test_empty_clause_is_unsat():
    engine = _engine([[1, 2], []], num_vars=2, K=2)
    assert engine.flush_conflict and engine.conflict.all()


def test_bcp_unit_propagation():
    # (~x1 OR x2), (~x2 OR x3): chain x1=T -> x2=T -> x3=T
    engine = _engine([[-1, 2], [-2, 3]], num_vars=3, K=1)
    engine.apply_and_propagate(_assum([[ext_lit_to_lit_index(1)]]))
    assert not engine.conflict[0].item()
    assert engine.assignment[0, 2].item() == 1     # x2 TRUE
    assert engine.assignment[0, 4].item() == 1     # x3 TRUE
    assert engine.num_propagated[0].item() == 2    # implications only (not the decision)


def test_bcp_conflict_detection():
    # (x1 OR x2), (~x1), (~x2): units force x1=F,x2=F -> clause 0 empties at the flush
    engine = _engine([[1, 2], [-1], [-2]], num_vars=2, K=1)
    assert engine.flush_conflict and engine.conflict[0].item()


def test_multiple_batches():
    # (x1 OR x2), (~x1 OR x2): batch 0 assumes x2=T (fine), batch 1 x2=F (conflict)
    engine = _engine([[1, 2], [-1, 2]], num_vars=2, K=2)
    engine.apply_and_propagate(_assum([
        [ext_lit_to_lit_index(2)],
        [ext_lit_to_lit_index(-2)],
    ]))
    assert not engine.conflict[0].item()
    assert engine.conflict[1].item()


def test_parse_and_propagate():
    formula = CnfFormula(DEV)
    formula.parse_cnf_file(os.path.join(TEST_DATA_DIR, "unit_prop_chain.cnf"))
    engine = BCPEngine(formula)
    engine.reset(1)
    # (x1 OR x2), (~x1 OR x2), (~x2 OR x3), (~x2 OR x4): x1=F forces x2 -> x3, x4
    engine.apply_and_propagate(_assum([[ext_lit_to_lit_index(-1)]]))
    assert not engine.conflict[0].item()
    assert engine.num_propagated[0].item() == 3
    for v in (2, 3, 4):
        assert engine.assignment[0, ext_lit_to_lit_index(v)].item() == 1


def test_already_satisfied_unit():
    # sat_count==1 with the survivor already TRUE must be skipped, not re-propagated
    engine = _engine([[1, 2], [-1, 2]], num_vars=2, K=1)
    engine.apply_and_propagate(
        _assum([[ext_lit_to_lit_index(1), ext_lit_to_lit_index(2)]]))
    assert not engine.conflict[0].item()
    assert engine.num_propagated[0].item() == 0


def test_reassume_implied_literal_is_noop():
    """Assuming a literal the units flush already implied must not count its falsifications
    twice; that would corrupt sat counts and produce false conflicts."""
    clauses = [[1], [-1, 2], [-2, 3, 4], [-3, 4, 5]]
    engine = _engine(clauses, num_vars=5, K=4)
    assert engine.assignment[0, 0].item() == 1 and engine.assignment[0, 2].item() == 1
    dec = torch.tensor([[0], [2], [0], [6]], dtype=torch.long, device=DEV)  # x1,x2,x1,x4
    engine.apply_and_propagate(dec)
    assert not engine.conflict.any(), "re-assuming an implied literal produced a false conflict"
    # sat counts must equal ground truth (non-falsified literals under the assignment)
    sat = engine.sat_rows(torch.arange(4, device=DEV))
    for c_i, cl in enumerate(clauses):
        for k in range(4):
            truth = sum(1 for l in cl
                        if engine.assignment[k, ext_lit_to_lit_index(l) ^ 1] != 1)
            assert int(sat[k, c_i]) == truth, (k, c_i)


def test_recache_flush_incremental_units():
    """The miner flow: fold entailed units into the flush cache, reset(K) restores them."""
    clauses = [[-1, 2], [-2, 3], [1, 4]]
    engine = _engine(clauses, num_vars=4, K=1)
    engine.apply_and_propagate(_assum([[ext_lit_to_lit_index(1)]]))   # x1 -> x2 -> x3
    engine.recache_flush()
    engine.reset(3)
    for v in (1, 2, 3):
        assert (engine.assignment[:, ext_lit_to_lit_index(v)] == 1).all()


def test_m0_formula():
    engine = _engine([], num_vars=4, K=2)
    assert not engine.conflict.any()
    engine.apply_and_propagate(_assum([[0], [3]]))
    assert not engine.conflict.any()
    assert engine.assignment[0, 0].item() == 1 and engine.assignment[1, 3].item() == 1


def test_counters_accumulate():
    engine = _engine([[-1, 2], [-2, 3]], num_vars=3, K=2)
    assert engine.props_total() == 0 and engine.launches == 0    # flush not counted
    engine.apply_and_propagate(_assum([[0], [0]]))               # x1=T per row
    assert engine.launches == 1
    assert engine.decs_total() == 2                              # one decision per row
    assert engine.props_total() == 6                             # +2 implications per row
    assert engine.ops_total() > 0


def test_envelope_rejection():
    f = CnfFormula(DEV)
    f.init_formula(40_000, [list(range(1, 300))])                # 2N>65535 and len>255
    with pytest.raises(ValueError):
        BCPEngine(f)


def test_make_assumption_tensor_shapes():
    t = make_assumption_tensor([[0, 2], [5]], DEV)
    assert t.shape == (2, 2) and t.dtype == torch.int64
    assert t[1].tolist() == [5, -1]
    assert make_assumption_tensor([[], []], DEV).shape == (2, 0)


# ---------------------------------------------------------------- check_sets

def _six_var_engine():
    f = CnfFormula(DEV)
    f.init_formula(6, [[1, 2], [-1, 3], [-2, -3], [4, 5], [-4, 6], [-5, -6]])
    return BCPEngine(f)


def test_check_sets_flat_form():
    """`check_sets` over flattened DIMACS literals and offsets. An empty set is legal (it
    yields the flush state), and a tiny chunk gives the same answers."""
    eng = _six_var_engine()
    lits = np.array([1, 2, -1, -2, -3], dtype=np.int32)    # [1,2] | [] | [-1,-2,-3]
    off = np.array([0, 2, 2, 5], dtype=np.int64)
    conflict, prop, assign, sat = eng.check_sets(lits, off, assignments=True)
    assert conflict.dtype == bool and conflict.shape == (3,)
    assert not conflict[1] and prop[1] == 0                 # the empty set: flush state
    assert assign.shape == (3, 12) and sat is None
    c1, p1, _, _ = eng.check_sets(lits, off, chunk=1)       # 3 launches, same answers
    np.testing.assert_array_equal(c1, conflict)
    np.testing.assert_array_equal(p1[~c1], prop[~conflict])


def test_check_sets_rejects_offsets_past_the_literal_array():
    """The binding checks offsets against len(lits); the core cannot, since a C pointer has
    no length, and would read past the array."""
    eng = _six_var_engine()
    with pytest.raises(ValueError, match="exceeds len"):
        eng.check_sets(np.array([1, 2], dtype=np.int32),
                       np.array([0, 2, 4], dtype=np.int64))

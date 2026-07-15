"""Unit tests for the packed kernel engine (BCPEngine): fixpoint semantics, the flush cache,
counters and edge cases. Semantic ground truth = the pure-Python reference (reference_bcp.py);
scenario-level expectations are spelled out per test. CUDA-only (the engine has no CPU path).
"""
import os

import pytest
import torch

from gpap.engine.formula import CnfFormula, ext_lit_to_lit_index
from gpap.engine.bcp_engine import BCPEngine, make_assumption_tensor
from gpap.engine.kernels import kernel_available

from tests.reference_bcp import ref_bcp, assigned_set

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
AVAIL = torch.cuda.is_available() and kernel_available()
pytestmark = pytest.mark.skipif(not AVAIL, reason="requires CUDA + a working triton")
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _engine(clauses, num_vars, K):
    formula = CnfFormula(DEV)
    formula.init_formula(num_vars, clauses)
    engine = BCPEngine(formula)
    engine.reset(K)
    return engine


def _assum(list_of_lists):
    return make_assumption_tensor(list_of_lists, DEV)


class TestBCPEngine:
    def test_reset_state(self):
        engine = _engine([[1, 2], [-1, 3]], num_vars=3, K=2)
        sat = engine.sat_rows(torch.arange(2, device=DEV))
        assert sat.shape == (2, 2)
        assert sat[0].tolist() == [2, 2]
        assert engine.assignment.shape == (2, 6)
        assert (engine.assignment == 0).all()
        assert engine.active.all() and not engine.conflict.any()

    def test_apply_assumption_no_conflict(self):
        # (x1 OR x2), (~x1 OR x3): x1=T falsifies ~x1 in clause 1 -> unit -> x3 forced
        engine = _engine([[1, 2], [-1, 3]], num_vars=3, K=1)
        engine.apply_and_propagate(_assum([[ext_lit_to_lit_index(1)]]))
        sat = engine.sat_rows(torch.tensor([0], device=DEV))[0]
        assert sat[0].item() == 2      # clause 0 untouched (no ~x1/~x3 in it... no falsification)
        assert sat[1].item() == 1      # clause 1 lost ~x1; survivor x3 was forced TRUE
        assert not engine.conflict[0].item()
        assert engine.assignment[0, ext_lit_to_lit_index(3)].item() == 1

    def test_unit_clauses_flushed_at_reset(self):
        # the formula's own unit (x3) is asserted by the construction-time flush cache
        engine = _engine([[3], [-3, 2], [1, -2]], num_vars=3, K=2)
        assert engine.assignment[0, ext_lit_to_lit_index(3)].item() == 1
        assert engine.assignment[0, ext_lit_to_lit_index(2)].item() == 1   # implied by (-3, 2)
        assert engine.assignment[1, ext_lit_to_lit_index(1)].item() == 1   # implied by (1, -2)
        assert not engine.conflict.any()

    def test_contradicting_units_conflict_at_construction(self):
        engine = _engine([[1], [-1]], num_vars=1, K=4)
        assert engine.flush_conflict
        assert engine.conflict.all() and not engine.active.any()

    def test_empty_clause_is_unsat(self):
        engine = _engine([[1, 2], []], num_vars=2, K=2)
        assert engine.flush_conflict and engine.conflict.all()

    def test_bcp_unit_propagation(self):
        # (~x1 OR x2), (~x2 OR x3) — chain: x1=T -> x2=T -> x3=T
        engine = _engine([[-1, 2], [-2, 3]], num_vars=3, K=1)
        engine.apply_and_propagate(_assum([[ext_lit_to_lit_index(1)]]))
        assert not engine.conflict[0].item()
        assert engine.assignment[0, 2].item() == 1     # x2 TRUE
        assert engine.assignment[0, 4].item() == 1     # x3 TRUE
        assert engine.num_propagated[0].item() == 2    # implications only (not the decision)

    def test_bcp_conflict_detection(self):
        # (x1 OR x2), (~x1), (~x2): units force x1=F,x2=F -> clause 0 empties at the FLUSH
        engine = _engine([[1, 2], [-1], [-2]], num_vars=2, K=1)
        assert engine.flush_conflict and engine.conflict[0].item()

    def test_multiple_batches(self):
        # (x1 OR x2), (~x1 OR x2): batch 0 assumes x2=T (fine), batch 1 x2=F (conflict)
        engine = _engine([[1, 2], [-1, 2]], num_vars=2, K=2)
        engine.apply_and_propagate(_assum([
            [ext_lit_to_lit_index(2)],
            [ext_lit_to_lit_index(-2)],
        ]))
        assert not engine.conflict[0].item()
        assert engine.conflict[1].item()

    def test_parse_and_propagate(self):
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

    def test_already_satisfied_unit(self):
        # sat_count==1 with the survivor already TRUE must be skipped, not re-propagated
        engine = _engine([[1, 2], [-1, 2]], num_vars=2, K=1)
        engine.apply_and_propagate(
            _assum([[ext_lit_to_lit_index(1), ext_lit_to_lit_index(2)]]))
        assert not engine.conflict[0].item()
        assert engine.num_propagated[0].item() == 0

    def test_reassume_implied_literal_is_noop(self):
        """Regression (2026-07-03): assuming a literal the units flush ALREADY implied must not
        double-count its falsifications (the old bulk path double-decremented sat counts ->
        garbage forced literals -> FALSE conflicts)."""
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

    @pytest.mark.parametrize("mode", ["scalar", "wave"])
    def test_matches_reference_random(self, mode):
        """Kernel fixpoints == the pure-Python reference on random CNFs (confluence), for BOTH
        propagate variants (serial trail and the warp-wide wavefront)."""
        import random
        rng = random.Random(11)
        for _ in range(25):
            nv = rng.randint(3, 30)
            clauses = []
            for _ in range(rng.randint(nv, 4 * nv)):
                vs = rng.sample(range(1, nv + 1), rng.randint(1, min(4, nv)))
                clauses.append([v if rng.random() < 0.5 else -v for v in vs])
            K = 16
            decs = [[2 * (v - 1) + rng.randint(0, 1)
                     for v in rng.sample(range(1, nv + 1), rng.randint(0, min(4, nv)))]
                    for _ in range(K)]
            f = CnfFormula(DEV)
            f.init_formula(nv, clauses)
            eng = BCPEngine(f, mode=mode)
            eng.reset(K)
            eng.apply_and_propagate(_assum(decs))
            for k in range(K):
                conf, asg = ref_bcp(clauses, decs[k])
                assert bool(eng.conflict[k].item()) == conf, (clauses, decs[k])
                if not conf:
                    assert assigned_set(eng.assignment[k].cpu()) == asg, (clauses, decs[k])

    def test_recache_flush_incremental_units(self):
        """The miner flow: fold entailed units into the flush cache, reset(K) restores them."""
        clauses = [[-1, 2], [-2, 3], [1, 4]]
        engine = _engine(clauses, num_vars=4, K=1)
        engine.apply_and_propagate(_assum([[ext_lit_to_lit_index(1)]]))   # x1 -> x2 -> x3
        engine.recache_flush()
        engine.reset(3)
        for v in (1, 2, 3):
            assert (engine.assignment[:, ext_lit_to_lit_index(v)] == 1).all()

    def test_m0_formula(self):
        engine = _engine([], num_vars=4, K=2)
        assert not engine.conflict.any()
        engine.apply_and_propagate(_assum([[0], [3]]))
        assert not engine.conflict.any()
        assert engine.assignment[0, 0].item() == 1 and engine.assignment[1, 3].item() == 1

    def test_counters_accumulate(self):
        engine = _engine([[-1, 2], [-2, 3]], num_vars=3, K=2)
        assert engine.props_total() == 0 and engine.launches == 0    # flush not counted
        engine.apply_and_propagate(_assum([[0], [0]]))               # x1=T per row
        assert engine.launches == 1
        assert engine.decs_total() == 2                              # one decision per row
        assert engine.props_total() == 6                             # +2 implications per row

    def test_envelope_rejection(self):
        f = CnfFormula(DEV)
        f.init_formula(40_000, [list(range(1, 300))])                # 2N>65535 AND len>255
        with pytest.raises(ValueError):
            BCPEngine(f)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

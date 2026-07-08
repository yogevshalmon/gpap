import os
import torch
import pytest

from gpap.engine.formula import CnfFormula
from gpap.engine.formula import ext_lit_to_lit_index
from gpap.engine.bcp_engine import BCPEngine, make_assumption_tensor

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


class TestBCPEngine:
    def _make_engine(self, clauses, num_vars, num_batches):
        device = torch.device("cpu")
        formula = CnfFormula(device)
        formula.init_formula(num_vars, clauses)
        engine = BCPEngine(formula)
        engine.init_state(num_batches=num_batches)
        return engine

    def _assum(self, list_of_lists):
        return make_assumption_tensor(list_of_lists, torch.device("cpu"))

    def test_init_state(self):
        engine = self._make_engine([[1, 2], [-1, 3]], num_vars=3, num_batches=2)
        assert engine.sat_count.shape == (2, 2)
        assert engine.sat_count[0].tolist() == [2, 2]
        assert engine.assignment.shape == (2, 6)
        assert (engine.assignment == 0).all()

    def test_apply_assumption_no_conflict(self):
        # (x1 OR x2), (~x1 OR x3)
        engine = self._make_engine([[1, 2], [-1, 3]], num_vars=3, num_batches=1)

        # assume x1=TRUE (lit_idx=0)
        engine.apply_assumptions_bulk(self._assum([[ext_lit_to_lit_index(1)]]))

        # x1 TRUE -> ~x1 falsified -> clause 1 (~x1 OR x3) loses one lit
        # sat_count[0,0] should still be 2 (x1 is in clause 0 but occ[0] = clauses with ~x1 = clause 1)
        # Actually: occ[0] (lit idx 0, pos x1) = clauses containing ~x1 = clause 1
        # so sat_count[0,1] -= 1 => was 2, now 1
        assert engine.sat_count[0, 0].item() == 2  # clause 0 not affected
        assert engine.sat_count[0, 1].item() == 1  # clause 1 lost ~x1
        assert engine.conflict[0].item() == False

    def test_apply_assumption_conflict(self):
        # (x1), (~x1)
        engine = self._make_engine([[1], [-1]], num_vars=1, num_batches=1)

        # assume x1=TRUE: clause 1 (~x1) becomes conflict (sat_count -> 0)
        engine.apply_assumptions_bulk(self._assum([[ext_lit_to_lit_index(1)]]))

        # after assumption: sat_count[0,1] should be 0 (conflict)
        assert engine.sat_count[0, 1].item() == 0

    def test_bcp_unit_propagation(self):
        # (~x1 OR x2), (~x2 OR x3) — chain: x1=T -> x2=T -> x3=T
        engine = self._make_engine([[-1, 2], [-2, 3]], num_vars=3, num_batches=1)

        # assume x1=TRUE
        engine.apply_assumptions_bulk(self._assum([[ext_lit_to_lit_index(1)]]))

        # after assumption: clause 0 has ~x1 falsified -> sat_count=1, lit_sum=x2(idx=2)
        assert engine.sat_count[0, 0].item() == 1

        conflict = engine.propagate()
        assert conflict[0].item() == False

        # x2 should be propagated TRUE (lit idx 2)
        assert engine.assignment[0, 2].item() == 1
        # x3 should be propagated TRUE (lit idx 4)
        assert engine.assignment[0, 4].item() == 1
        assert engine.num_propagated[0].item() == 2

    def test_bcp_conflict_detection(self):
        # (x1 OR x2), (~x1), (~x2)
        # assume x1=F -> clause 0 loses x1, clause 1 satisfied
        # then x2 is forced TRUE by clause 0 -> but clause 2 (~x2) conflicts
        engine = self._make_engine([[1, 2], [-1], [-2]], num_vars=2, num_batches=1)

        # assume x1=FALSE (lit_idx=1, ~x1 TRUE)
        engine.apply_assumptions_bulk(self._assum([[ext_lit_to_lit_index(-1)]]))

        conflict = engine.propagate()
        assert conflict[0].item() == True

    def test_multiple_batches(self):
        # (x1 OR x2), (~x1 OR x2)
        engine = self._make_engine([[1, 2], [-1, 2]], num_vars=2, num_batches=2)

        # batch 0: x2=TRUE, batch 1: x2=FALSE
        engine.apply_assumptions_bulk(self._assum([
            [ext_lit_to_lit_index(2)],   # x2=TRUE
            [ext_lit_to_lit_index(-2)],  # x2=FALSE
        ]))

        conflict = engine.propagate()

        # batch 0: x2=TRUE satisfies no clause directly (occ doesn't track satisfaction),
        # but neither clause conflicts. Actually:
        # occ[2] (pos x2) = clauses with ~x2 = none (no clause has ~x2)
        # Hmm, let me reconsider. clauses: (x1 OR x2), (~x1 OR x2)
        # ~x2 doesn't appear in any clause, so occ[lit_idx(x2)] = occ[2] = clauses with ~x2 = empty
        # So setting x2=TRUE doesn't change any sat_count. Both clauses still have sat_count=2.
        # No units, no conflicts. That's correct - clauses are still open.
        assert conflict[0].item() == False

        # batch 1: x2=FALSE (~x2 TRUE, lit_idx=3)
        # occ[3] (neg x2) = clauses with x2 = clauses 0, 1
        # sat_count[1, 0] -= 1 => 1 (unit: remaining = x1, idx 0)
        # sat_count[1, 1] -= 1 => 1 (unit: remaining = ~x1, idx 1)
        # BCP: force x1=T from clause 0, then ~x1 from clause 1 -> contradiction
        assert conflict[1].item() == True

    def test_parse_and_propagate(self):
        """Test with CNF file."""
        device = torch.device("cpu")
        formula = CnfFormula(device)
        formula.parse_cnf_file(os.path.join(TEST_DATA_DIR, "unit_prop_chain.cnf"))

        engine = BCPEngine(formula)
        engine.init_state(num_batches=1)

        # (x1 OR x2), (~x1 OR x2), (~x2 OR x3), (~x2 OR x4)
        # assume x1=FALSE: clause 0 loses x1 -> unit: x2 forced
        # then x2=TRUE: occ[2] = clauses with ~x2 = clauses 2,3 lose ~x2
        # clause 2 becomes unit: x3 forced. clause 3 becomes unit: x4 forced.
        engine.apply_assumptions_bulk(self._assum([[ext_lit_to_lit_index(-1)]]))
        conflict = engine.propagate()

        assert conflict[0].item() == False
        # x2, x3, x4 should be propagated
        assert engine.num_propagated[0].item() == 3
        # x2=TRUE (idx 2), x3=TRUE (idx 4), x4=TRUE (idx 6)
        assert engine.assignment[0, 2].item() == 1
        assert engine.assignment[0, 4].item() == 1
        assert engine.assignment[0, 6].item() == 1

    def test_already_satisfied_unit(self):
        """When sat_count==1 but remaining literal is already TRUE, skip it."""
        # (x1 OR x2), (~x1 OR x2)
        # If we assume both x1=TRUE and x2=TRUE:
        # occ[0] = clauses with ~x1 = clause 1 -> sat_count[1] -= 1 = 1
        # occ[2] = clauses with ~x2 = none
        # After assumptions, clause 1 has sat_count=1, lit_sum = x2's idx
        # But x2 is already TRUE -> should skip, not propagate
        engine = self._make_engine([[1, 2], [-1, 2]], num_vars=2, num_batches=1)
        engine.apply_assumptions_bulk(self._assum([[ext_lit_to_lit_index(1), ext_lit_to_lit_index(2)]]))
        conflict = engine.propagate()

        assert conflict[0].item() == False
        assert engine.num_propagated[0].item() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

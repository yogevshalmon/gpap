import os
import torch
import pytest

from gpap.solver.assumption_checker import AssumptionChecker

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


class TestAssumptionChecker:
    def test_basic_check(self):
        checker = AssumptionChecker(device=torch.device("cpu"))
        # (x1 OR x2), (~x1 OR x3), (~x2 OR ~x3)
        checker.formula.init_formula(3, [[1, 2], [-1, 3], [-2, -3]])

        results = checker.check_assumptions([
            [1, 2, 3],     # x1=T, x2=T, x3=T
            [1, -2, -3],   # x1=T, x2=F, x3=F
        ])

        # batch 0: x1=T,x2=T,x3=T
        #   clause 2 (~x2 OR ~x3): both x2,x3 TRUE -> ~x2,~x3 falsified -> sat_count=0 -> conflict
        assert results['conflict'][0].item() == True

        # batch 1: x1=T,x2=F,x3=F
        #   clause 0 (x1 OR x2): x2=F -> occ[ext(-2)]=occ[3]=clauses with x2=clause 0, sat_count-=1
        #   x1=T -> occ[0]=clauses with ~x1=clause 1, sat_count[1]-=1
        #   clause 1: sat_count=1, unit: lit_sum=x3(idx=4)... but x3=F(lit_idx=5 is TRUE)
        #   So x3's negation idx=5 is TRUE. lit_sum for clause 1 should be x3(4).
        #   assignment[5]=1 (x3=FALSE means ~x3 TRUE, idx=5=1).
        #   occ[5] = clauses with x3 = clause 1: sat_count[1] -= 1 => 0 => conflict? No wait...
        #   Let me trace more carefully.
        # Actually this depends on order of assumption application. Let's just check no crash.
        # The formula is satisfiable under x1=T,x2=F,x3=F? clause 0: x1=T, sat. clause 1: ~x1=F, x3=F, sat_count could be 0.
        # clause 1 = (~x1 OR x3): ~x1=F, x3=F -> both falsified -> conflict!
        assert results['conflict'][1].item() == True

    def test_no_conflict(self):
        checker = AssumptionChecker(device=torch.device("cpu"))
        # (x1 OR x2), (~x1 OR x2)
        checker.formula.init_formula(2, [[1, 2], [-1, 2]])

        results = checker.check_assumptions([
            [2],  # x2=TRUE -> no clause has ~x2, so no sat_count changes
        ])
        assert results['conflict'][0].item() == False

    def test_from_cnf_file(self):
        checker = AssumptionChecker(
            cnf_file=os.path.join(TEST_DATA_DIR, "simple_3v_4c.cnf"),
            device=torch.device("cpu"),
        )
        assert checker.formula.num_vars == 3
        assert checker.formula.num_clauses == 4

    def test_tensor_input(self):
        checker = AssumptionChecker(device=torch.device("cpu"))
        # (x1 OR x2), (~x2 OR x3)
        checker.formula.init_formula(3, [[1, 2], [-2, 3]])

        # K=2, N=3 tensor
        assumptions = torch.tensor([
            [1, -1, 0],   # x1=T, x2=F, x3=unassigned
            [0, 1, 0],    # x2=T
        ], dtype=torch.float32)

        results = checker.check_assumptions_tensor(assumptions)
        # batch 0: x2=F -> clause 0 (x1 OR x2) loses x2, sat_count=1
        #   x1=T -> clause 1 (~x2 OR x3) not affected by x1
        #   After assumptions, BCP: clause 0 unit, remaining=x1(idx=0), already TRUE -> skip
        #   clause 1: ~x2 falsified? No, x2=F means ~x2 is TRUE -> occ[3] = clauses with x2 = clause 0
        #   Wait: x2=F means lit_idx=3 (~x2) is TRUE. occ[3] = clauses with x2 (clause 0).
        #   sat_count[0] -= 1 (from x2 being falsified in clause 0)
        #   Also x1=T means lit_idx=0 is TRUE. occ[0] = clauses with ~x1 = none (no clause has ~x1)
        # So after assumptions: clause 0 has sat_count=1 (x1 remaining), clause 1 has sat_count=2
        # BCP: clause 0 unit, lit_sum = x1(0), already TRUE -> skip. No more units.
        assert results['conflict'][0].item() == False

    def test_propagation_count(self):
        checker = AssumptionChecker(device=torch.device("cpu"))
        # chain: (~x1 OR x2), (~x2 OR x3)
        checker.formula.init_formula(3, [[-1, 2], [-2, 3]])

        results = checker.check_assumptions([[1]])  # x1=TRUE
        # x1=T -> clause 0 unit (x2 forced) -> x2=T -> clause 1 unit (x3 forced)
        assert results['conflict'][0].item() == False
        assert results['num_propagated'][0].item() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

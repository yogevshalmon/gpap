import os
import torch
import pytest

from gpap.clsdata.cnf_formula import CnfFormula
from gpap.clsdata.occ_list import OccurrenceList, ext_lit_to_lit_index, lit_index_negation

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


class TestLiteralIndexing:
    def test_positive_literal(self):
        assert ext_lit_to_lit_index(1) == 0
        assert ext_lit_to_lit_index(2) == 2
        assert ext_lit_to_lit_index(3) == 4

    def test_negative_literal(self):
        assert ext_lit_to_lit_index(-1) == 1
        assert ext_lit_to_lit_index(-2) == 3
        assert ext_lit_to_lit_index(-3) == 5

    def test_negation(self):
        assert lit_index_negation(0) == 1
        assert lit_index_negation(1) == 0
        assert lit_index_negation(4) == 5
        assert lit_index_negation(5) == 4


class TestOccurrenceList:
    def test_basic_construction(self):
        device = torch.device("cpu")
        # (~x1 OR x2), (x1 OR ~x2 OR x3), (~x1 OR ~x2), (~x3)
        clauses = [[-1, 2], [1, -2, 3], [-1, -2], [-3]]
        occ = OccurrenceList.from_clauses(3, clauses, device)

        assert occ.num_lits == 6

        # occ[l] = clauses where ~l appears
        # occ[0] (pos x1) -> clauses with ~x1: clauses 0, 2 (which have lit -1)
        assert occ.get_clause_indices(0).tolist() == [0, 2]

        # occ[1] (neg x1) -> clauses with x1: clause 1
        assert occ.get_clause_indices(1).tolist() == [1]

        # occ[2] (pos x2) -> clauses with ~x2: clauses 1, 2
        assert occ.get_clause_indices(2).tolist() == [1, 2]

        # occ[3] (neg x2) -> clauses with x2: clause 0
        assert occ.get_clause_indices(3).tolist() == [0]

        # occ[4] (pos x3) -> clauses with ~x3: clause 3
        assert occ.get_clause_indices(4).tolist() == [3]

        # occ[5] (neg x3) -> clauses with x3: clause 1
        assert occ.get_clause_indices(5).tolist() == [1]


class TestCnfFormula:
    def test_init_formula(self):
        device = torch.device("cpu")
        formula = CnfFormula(device)
        clauses = [[-1, 2], [1, -2, 3], [-1, -2], [-3]]
        formula.init_formula(3, clauses)

        assert formula.num_vars == 3
        assert formula.num_clauses == 4
        assert formula.clause_sizes.tolist() == [2, 3, 2, 1]

        # lit_sum for each clause:
        # clause 0: ~x1(1) + x2(2) = 3
        # clause 1: x1(0) + ~x2(3) + x3(4) = 7
        # clause 2: ~x1(1) + ~x2(3) = 4
        # clause 3: ~x3(5) = 5
        assert formula.init_lit_sum.tolist() == [3, 7, 4, 5]

    def test_parse_cnf_file(self):
        device = torch.device("cpu")
        formula = CnfFormula(device)
        formula.parse_cnf_file(os.path.join(TEST_DATA_DIR, "simple_3v_4c.cnf"))

        assert formula.num_vars == 3
        assert formula.num_clauses == 4
        assert formula.clause_sizes.tolist() == [2, 3, 2, 1]
        assert formula.init_lit_sum.tolist() == [3, 7, 4, 5]

    def test_single_literal_clauses(self):
        device = torch.device("cpu")
        formula = CnfFormula(device)
        # (x1), (~x1)
        clauses = [[1], [-1]]
        formula.init_formula(1, clauses)

        assert formula.num_vars == 1
        assert formula.num_clauses == 2
        assert formula.clause_sizes.tolist() == [1, 1]
        # clause 0: x1(0) = 0
        # clause 1: ~x1(1) = 1
        assert formula.init_lit_sum.tolist() == [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

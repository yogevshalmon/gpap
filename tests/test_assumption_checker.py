import os

import pytest
import torch

from gpap.engine.checker import AssumptionChecker
from gpap.engine import core_available

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
AVAIL = torch.cuda.is_available() and core_available()
pytestmark = pytest.mark.skipif(not AVAIL, reason="requires CUDA + a working triton")


class TestAssumptionChecker:
    def test_basic_check(self):
        checker = AssumptionChecker()
        # (x1 OR x2), (~x1 OR x3), (~x2 OR ~x3)
        checker.formula.init_formula(3, [[1, 2], [-1, 3], [-2, -3]])

        results = checker.check_assumptions([
            [1, 2, 3],     # clause 2 (~x2 OR ~x3) fully falsified -> conflict
            [1, -2, -3],   # clause 1 (~x1 OR x3) fully falsified -> conflict
        ])
        assert results['conflict'][0].item() is True or bool(results['conflict'][0])
        assert bool(results['conflict'][1])

    def test_no_conflict(self):
        checker = AssumptionChecker()
        checker.formula.init_formula(2, [[1, 2], [-1, 2]])
        results = checker.check_assumptions([[2]])   # x2=TRUE satisfies both clauses
        assert not bool(results['conflict'][0])

    def test_from_cnf_file(self):
        checker = AssumptionChecker(
            cnf_file=os.path.join(TEST_DATA_DIR, "simple_3v_4c.cnf"))
        assert checker.formula.num_vars == 3
        assert checker.formula.num_clauses == 4

    def test_propagation_count(self):
        checker = AssumptionChecker()
        # chain: (~x1 OR x2), (~x2 OR x3)
        checker.formula.init_formula(3, [[-1, 2], [-2, 3]])
        results = checker.check_assumptions([[1]])   # x1=TRUE -> x2 -> x3
        assert not bool(results['conflict'][0])
        assert results['num_propagated'][0].item() == 2

    def test_sat_count_shape(self):
        checker = AssumptionChecker()
        checker.formula.init_formula(3, [[1, 2], [-1, 3], [-2, -3]])
        results = checker.check_assumptions([[1], [2]])
        assert results['sat_count'].shape == (2, 3)
        assert results['assignment'].shape == (2, 6)


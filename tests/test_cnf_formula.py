"""Literal indexing, the reference formula build, and the shipped CnfFormula facade.

The reference build (tests/reference_formula.py) runs on any host, so this file keeps DIMACS
parsing and structure construction covered on machines with no libgpap; tests/test_core.py is
where the core is compared against the reference entry for entry. The facade tests at the bottom
gate on the core being present, since the shipped CnfFormula has no other builder.
"""
import os

import pytest

from gpap.engine import _core
from gpap.engine.formula import CnfFormula, ext_lit_to_lit_index, lit_index_negation
from tests.reference_formula import ReferenceFormula

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
SIMPLE = [[-1, 2], [1, -2, 3], [-1, -2], [-3]]     # (~1|2), (1|~2|3), (~1|~2), (~3)


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


class TestReferenceOccurrenceLists:
    def test_basic_construction(self):
        # SIMPLE segregates as: clause 0 (~1|2) BIN pair (1,2); clause 1 (1|~2|3) LONG id 0;
        # clause 2 (~1|~2) BIN pair (1,3); clause 3 (~3) UNIT lit 5 -> NO occ entry.
        f = ReferenceFormula.from_clauses(3, SIMPLE)
        assert (f.num_bin, f.num_long) == (2, 1)
        assert f.bin_pairs == [(1, 2), (1, 3)]
        assert f.unit_lits == [5]

        # binary prefix: payload = the OTHER literal setting l TRUE forces
        assert f.occ_bin(0).tolist() == [2, 3]  # x1 TRUE: clause 0 forces x2, clause 2 forces ~x2
        assert f.occ_bin(3).tolist() == [1]     # x2 FALSE: clause 0 forces ~x1
        assert f.occ_bin(2).tolist() == [1]     # x2 TRUE: clause 2 forces ~x1
        # long suffix: payload = the long clause's re-index
        assert f.occ_long(1).tolist() == [0]    # x1 FALSE falsifies +1 in long clause 0
        assert f.occ_long(2).tolist() == [0]    # x2 TRUE falsifies -2 in long clause 0
        assert f.occ_long(5).tolist() == [0]    # x3 FALSE falsifies +3 in long clause 0
        # the unit clause (~3) contributes NO entry: occ[4] is empty
        assert f.occ(4).tolist() == []
        # RAW storage carries the binary tag (the sign bit) so a kernel can classify an entry
        # from the value it already loaded; occ_bin() above is the masked view.
        from tests.reference_formula import BIN_TAG
        assert f.occ(0).tolist() == [2 | BIN_TAG, 3 | BIN_TAG]
        assert all(x < 0 for x in f.occ(0).tolist())      # binaries tagged
        assert all(x >= 0 for x in f.occ_long(1).tolist())  # long clause ids are not


class TestReferenceFormula:
    def test_from_clauses(self):
        f = ReferenceFormula.from_clauses(3, SIMPLE)
        assert (f.num_vars, f.num_clauses) == (3, 4)
        assert f.clause_sizes.tolist() == [2, 3, 2, 1]
        # init_lit_sum (sum of lit-indices) per clause:
        #   clause 0: ~x1(1) + x2(2) = 3;  clause 1: x1(0) + ~x2(3) + x3(4) = 7
        #   clause 2: ~x1(1) + ~x2(3) = 4; clause 3: ~x3(5) = 5
        assert f.init_lit_sum.tolist() == [3, 7, 4, 5]

    def test_from_file(self):
        f = ReferenceFormula.from_file(os.path.join(TEST_DATA_DIR, "simple_3v_4c.cnf"))
        assert (f.num_vars, f.num_clauses) == (3, 4)
        assert f.clause_sizes.tolist() == [2, 3, 2, 1]
        assert f.init_lit_sum.tolist() == [3, 7, 4, 5]

    def test_single_literal_clauses(self):
        f = ReferenceFormula.from_clauses(1, [[1], [-1]])
        assert (f.num_vars, f.num_clauses) == (1, 2)
        assert f.clause_sizes.tolist() == [1, 1]
        assert f.init_lit_sum.tolist() == [0, 1]     # x1 -> 0, ~x1 -> 1

    def test_empty_clause_is_dropped_and_flagged(self):
        f = ReferenceFormula.from_clauses(2, [[1, 2], []])
        assert f.has_empty_clause
        assert f.num_clauses == 1


@pytest.mark.skipif(not _core.available(), reason="libgpap not built (see cpp/README.md)")
class TestCnfFormulaFacade:
    """The shipped CnfFormula is a facade over the core's builder — no CUDA needed to BUILD
    (formula construction is host-side C++), so these run wherever the library exists."""

    def test_init_formula_adopts_the_core_build(self):
        import torch
        f = CnfFormula(torch.device("cpu"))
        f.init_formula(3, SIMPLE)
        assert (f.num_vars, f.num_clauses) == (3, 4)
        assert f.max_clause_len == 3
        assert f.total_lits == 8
        # on-demand materialization from the core's clause arrays
        assert f.clause_sizes.tolist() == [2, 3, 2, 1]
        assert f.init_lit_sum.tolist() == [3, 7, 4, 5]

    def test_parse_cnf_file(self):
        import torch
        f = CnfFormula(torch.device("cpu"))
        f.parse_cnf_file(os.path.join(TEST_DATA_DIR, "simple_3v_4c.cnf"))
        assert (f.num_vars, f.num_clauses) == (3, 4)
        assert f.clause_sizes.tolist() == [2, 3, 2, 1]
        assert f.init_lit_sum.tolist() == [3, 7, 4, 5]


@pytest.mark.skipif(not _core.available(), reason="libgpap not built (see cpp/README.md)")
class TestReadDimacs:
    """`read_dimacs` is THE list-of-lists entry point (the core's parser + clause read-back) —
    it is what replaced the third-party parser, so its fidelity is what licenses that removal."""

    def test_matches_the_reference_parser_as_clause_sets(self, tmp_path):
        from gpap.engine import read_dimacs
        p = tmp_path / "f.cnf"
        p.write_text("c comment\np cnf 4 3\n1 -2 0 2 3\n-4 0\n-1 -3 4 0\n")
        nv, clauses, sup = read_dimacs(str(p))
        ref = ReferenceFormula.from_file(str(p))
        assert nv == ref.num_vars and len(clauses) == ref.num_clauses
        # within-clause order is ascending literal index (rebuilt from the CSR), so compare sets
        assert [sorted(c) for c in clauses] == \
               [sorted(c) for c in [[1, -2], [2, 3, -4], [-1, -3, 4]]]
        assert sup is None

    def test_duplicate_literals_survive_the_round_trip(self):
        f = _core.Formula.from_clauses(2, [[1, 1, -2], [2]])
        assert [sorted(c) for c in f.clauses()] == [[-2, 1, 1], [2]]

    def test_empty_clause_is_reappended(self, tmp_path):
        from gpap.engine import read_dimacs
        p = tmp_path / "unsat.cnf"
        p.write_text("p cnf 2 2\n1 2 0\n0\n")
        nv, clauses, _ = read_dimacs(str(p))
        assert [] in clauses and len(clauses) == 2

    def test_header_n_wins_over_max_var_used(self, tmp_path):
        """A header that declares MORE variables than the clauses use is honoured — the
        parser's rule, so `c ind`-only support variables keep their slots. Pinned because the
        third-party parser this replaced recounted from the clauses (nv would read 10 here),
        which silently drops declared-but-unconstrained sampling variables."""
        from gpap.engine import read_dimacs
        p = tmp_path / "h.cnf"
        p.write_text("p cnf 20 2\n1 10 0\n-10 2 0\n")
        nv, clauses, _ = read_dimacs(str(p))
        assert nv == 20 and len(clauses) == 2

    def test_support_lines_c_ind_and_cr(self, tmp_path):
        """Both header conventions, and the CORE parse must agree with the pure-Python scanner
        (`parse_independent_support`) — the gate that keeps the two parsers one."""
        from gpap.engine import read_dimacs
        from gpap.sampling import parse_independent_support
        p = tmp_path / "s.cnf"
        p.write_text("c ind 2 5 0\ncr 3\np cnf 5 2\n1 2 0\n-3 4 5 0\n")
        _, _, sup = read_dimacs(str(p))
        assert sup == [1, 2, 4]                     # 0-based, ascending, deduped
        assert parse_independent_support(str(p)) == sup

"""Literal indexing, the reference formula build, and the shipped CnfFormula facade.

The reference build (tests/reference_formula.py) runs on any host, so this file keeps DIMACS
parsing and structure construction covered on machines with no libgpap; tests/test_core.py is
where the core is compared against the reference entry for entry. The facade tests at the bottom
gate on the core being present, since the shipped CnfFormula has no other builder.
"""
import ctypes
import os

import numpy as np
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

    def test_a_directory_is_an_error_not_a_process_abort(self, tmp_path):
        """fopen("rb") SUCCEEDS on a directory on Linux and ftell then reports LONG_MAX. Sizing
        the read buffer from it threw std::length_error out through the extern "C" boundary,
        where there is no handler — so `gpap sample <a directory>` called std::terminate() and
        core-dumped the whole process, the host's included. The ABI promises a gpap_status."""
        with pytest.raises(_core.GpapError) as e:
            _core.Formula.from_file(str(tmp_path))
        assert e.value.status == 4                                  # GPAP_ERR_IO

    def test_a_non_seekable_stream_parses_its_actual_bytes(self, tmp_path):
        """A FIFO (also /dev/stdin and a <(...) process substitution) has no ftell size: it
        returns -1, the sized read was skipped, and the caller got the EMPTY formula back with
        GPAP_OK — zero variables, trivially satisfiable, no warning. Same silent-different-
        formula failure the tokenizer is strict about, one layer down."""
        import os
        import threading
        text = "p cnf 3 3\n1 2 0\n-1 3 0\n2 -3 0\n"
        fifo = tmp_path / "f.cnf"
        os.mkfifo(fifo)
        threading.Thread(target=lambda: fifo.write_text(text), daemon=True).start()
        info = _core.Formula.from_file(str(fifo)).info
        plain = tmp_path / "plain.cnf"
        plain.write_text(text)
        ref = _core.Formula.from_file(str(plain)).info
        assert (info.num_vars, info.num_clauses) == (ref.num_vars, ref.num_clauses) == (3, 3)

    def test_a_file_ftell_sizes_as_empty_is_still_read(self, tmp_path):
        """procfs/sysfs report st_size == 0 while having content, so ftell says 0 and the old
        reader parsed nothing. Not a CNF, but it must be REJECTED rather than accepted as the
        empty formula — the distinction the fix is about."""
        with pytest.raises(_core.GpapError) as e:
            _core.Formula.from_file("/proc/self/status")
        assert e.value.status == 4                                  # read, then refused as non-DIMACS

    def test_an_empty_file_is_still_the_empty_formula(self, tmp_path):
        """The other side of the same fix: a genuinely empty file is legal, not an error."""
        p = tmp_path / "empty.cnf"
        p.write_text("")
        info = _core.Formula.from_file(str(p)).info
        assert (info.num_vars, info.num_clauses) == (0, 0)


@pytest.mark.skipif(not _core.available(), reason="libgpap not built (see cpp/README.md)")
class TestOccTagAndSplitAgree:
    """The binary/long split is read TWO different ways, and they must never disagree.

    `_wave_propagate` classifies an occ entry from the SIGN BIT of the payload it has already
    loaded (`pay_v < 0`), because gathering that segment's `occ_mid` again would be a third
    shuffle gather in the engine's hottest loop. `_clean_packed` and the scalar `_propagate` use
    `occ_mid[l]` instead. A binary entry that landed at or above `occ_mid` would be a long clause
    to the restore (which would index `init_ptr` with a NEGATIVE payload) and a binary to wave —
    an out-of-bounds device write that no readout could attribute. Nothing else states the
    coupling, so it is pinned here rather than left implied by the array comparison."""

    @staticmethod
    def _check(f):
        import numpy as np
        offsets, mid, occ = f.occ()
        two_n, num_long = 2 * f.info.num_vars, f.info.num_long
        for lit in range(len(mid)):
            start, split, end = int(offsets[lit]), int(mid[lit]), int(offsets[lit + 1])
            assert start <= split <= end, (lit, start, split, end)
            binaries, longs = occ[start:split], occ[split:end]
            assert np.all(binaries < 0), f"untagged entry below occ_mid[{lit}]"
            assert np.all(longs >= 0), f"tagged entry at/above occ_mid[{lit}]"
            assert np.all((binaries & 0x7FFFFFFF) < two_n), f"binary payload past 2N at {lit}"
            assert np.all(longs < num_long), f"long payload past num_long at {lit}"

    #: Shapes where one of the formula's vectors is EMPTY. Reading them back used to be
    #: undefined behaviour: `std::vector::data()` on an empty vector may be nullptr and
    #: `memcpy(dst, nullptr, 0)` is UB at any length — UBSan flags it, and two of the three
    #: copies in gpap_formula_occ carried an `!empty()` guard while the third did not.
    DEGENERATE = [
        (4, [[1, 2], [-1, 3, 4], [2], [-2, -3], [1, 2, 3, 4]]),
        (3, [[1, 1], [1, -1], [2, -2, 3], [3, 3, 3]]),       # dup / tautological
        (5, [[1, 2], [3, 4]]),                                # no long clauses at all
        (5, [[1, 2, 3], [-1, -2, -3]]),                       # no binaries at all
        (2, [[1], [2]]),                                      # units only: occ_clauses empty
        (3, []),                                              # clause_sizes / lit_sum empty
        (0, []),                                              # occ_offsets / occ_mid empty too
    ]

    def test_on_mixed_and_degenerate_formulas(self):
        for nv, clauses in self.DEGENERATE:
            self._check(_core.Formula.from_clauses(nv, clauses))

    def test_every_read_back_accessor_survives_an_empty_vector(self):
        """Not just occ(): the same null-source memcpy sat in gpap_formula_clause_arrays, and
        gpap_formula_support had the guard. Touch them ALL on the shapes that empty them — the
        failure is invisible without a sanitizer, so this exists to be run under one."""
        for nv, clauses in self.DEGENERATE:
            f = _core.Formula.from_clauses(nv, clauses)
            offsets, mid, occ = f.occ()
            sizes, sums = f.clause_arrays()
            back = f.clauses()
            # NOT `... or nv == 0`: that disjunct short-circuited the assertion to True on
            # the (0, []) row, i.e. on the one degenerate shape this table exists for.
            assert len(offsets) == 2 * nv + 1
            assert len(sizes) == len(sums) == f.info.num_clauses == len(back)
            assert [sorted(c) for c in back] == [sorted(c) for c in clauses if c]

    def test_on_a_real_benchmark(self):
        import pathlib
        cnf = sorted(pathlib.Path("benchmarks/sampling").glob("*.cnf"))
        if not cnf:
            pytest.skip("no benchmark CNFs in the checkout")
        f = _core.Formula.from_file(str(cnf[0]))
        assert f.info.num_bin > 0 and f.info.num_long > 0, "fixture must hold both kinds"
        self._check(f)


def read_support(path):
    """The 0-based independent support the core parsed out of `c ind` / `cr`.

    `Formula.support` is a method on the facade; this goes at the ABI so the parser is what is
    under test. Shared by both support-line test classes — there is one reader, not one per
    class, and so the "needs the library" guard is one IMPERATIVE skip here rather than a
    decorator per class. A `skipif` MARK on this function is what CI caught: pytest only honours
    marks on collected items, so on a helper it silently guards nothing and every caller raised
    GpapError on a tree with no libgpap."""
    if not _core.available():
        pytest.skip("libgpap not built (see cpp/README.md)")
    f = _core.Formula.from_file(str(path))
    n = ctypes.c_int32(0)
    _core.load().gpap_formula_support(f._h, None, ctypes.byref(n))
    if not n.value:
        return []
    out = np.zeros(n.value, dtype=np.int32)
    _core.load().gpap_formula_support(
        f._h, out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.byref(n))
    return out[:n.value].tolist()


class TestSupportLineIsBoundedByItsLine:
    """REGRESSION. The `c ind` scanner stopped at `line_end` by testing the CURSOR, but strtol
    skips leading whitespace including \\r and \\n — so any cursor left short of the newline
    (one \\r on a CRLF file, or one trailing space) parsed straight past it and pulled the NEXT
    line's first literal into the independent support.

    Silent, and it enlarges the projection set of every CRLF-formatted benchmark by one variable
    per `c ind` line. Nothing under benchmarks/ triggers it (410 `c ind` files, all LF), which is
    exactly why it survived: it is only reachable from an externally supplied CNF."""

    @staticmethod
    def _support(tmp_path, raw):
        p = tmp_path / "s.cnf"
        p.write_bytes(raw)
        return read_support(p)

    @pytest.mark.parametrize(
        "name,raw",
        [
            ("lf", b"p cnf 5 2\nc ind 2 3 0\n5 2 0\n-3 4 0\n"),
            ("crlf", b"p cnf 5 2\r\nc ind 2 3 0\r\n5 2 0\r\n-3 4 0\r\n"),
            ("trailing space", b"p cnf 5 2\nc ind 2 3 0 \n5 2 0\n-3 4 0\n"),
            ("trailing tab", b"p cnf 5 2\nc ind 2 3 0\t\n5 2 0\n-3 4 0\n"),
        ],
    )
    def test_line_endings_do_not_leak_the_next_clause(self, tmp_path, name, raw):
        # variable 5 opens the next line; 0-based support must stay {1, 2} in every encoding
        assert self._support(tmp_path, raw) == [1, 2], name

    def test_a_cr_line_is_bounded_too(self, tmp_path):
        # the `cr <v...>` variant takes the same path and has no terminating 0 at all
        assert self._support(tmp_path, b"p cnf 5 2\r\ncr 2 3\r\n5 2 0\r\n-3 4 0\r\n") == [1, 2]


class TestSupportLineRanges:
    """`c ind` / `cr` values get the SAME range discipline as literals.

    They did not, and the parser is otherwise strict about a file silently becoming a different
    formula. `c ind 4294967297` wrapped through an int32 cast into variable 1; a value past
    INT64 range came back as support index **-2**, and LONG_MIN reached `-v` — undefined
    behaviour, the same trap gpap_internal.hpp documents for literals one width down (LONG_MIN
    is exactly representable, so strtol does not flag it). A negative index is not merely wrong:
    this ABI documents its output as 0-based variable indices, and only one downstream consumer
    happened to filter them.

    Out-of-range values are DROPPED, not refused, matching the parser's existing policy for a
    support variable the header never declared — `c ind` is a comment line and tools write junk
    there."""

    @staticmethod
    def _support(tmp_path, line):
        p = tmp_path / "s.cnf"
        p.write_text(f"p cnf 5 2\n{line}\n1 2 0\n-3 4 5 0\n")
        return read_support(p)

    @pytest.mark.parametrize("line", [
        "c ind 4294967297 0",              # wraps through int32 -> used to be variable 1
        "c ind 9223372036854775808 0",     # past int64 -> used to be support index -2
        "c ind -9223372036854775808 0",    # LONG_MIN: -v is UB, and strtol does not flag it
        "c ind -2147483648 0",             # INT32_MIN, the same trap one width down
        "c ind 99 0",                      # in range for int32, past the header's N
    ])
    def test_unrepresentable_values_are_dropped(self, tmp_path, line):
        assert self._support(tmp_path, line) == []

    def test_valid_values_survive_either_polarity_and_either_keyword(self, tmp_path):
        assert self._support(tmp_path, "c ind 2 3 0") == [1, 2]
        assert self._support(tmp_path, "c ind -2 -3 0") == [1, 2]     # sign is ignored
        assert self._support(tmp_path, "cr 1 4") == [0, 3]
        assert self._support(tmp_path, "c ind 2 99 3 0") == [1, 2]    # junk dropped, rest kept

    def test_every_returned_index_is_a_valid_variable(self, tmp_path):
        for line in ("c ind 4294967297 2 -9223372036854775808 3 0", "c ind 0 0", "c ind x y 0"):
            for v in self._support(tmp_path, line):
                assert 0 <= v < 5, (line, v)

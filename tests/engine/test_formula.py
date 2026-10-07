"""Literal indexing, the reference formula build, the core's DIMACS parser and builder, and the
CnfFormula facade.

The reference tests (tests/reference/formula.py) run without libgpap; the rest need the library
(not a GPU) and compare it against the reference.
"""
import ctypes
import os
import random

import numpy as np
import pytest

from gpap.engine import _core
from gpap.engine.formula import CnfFormula, ext_lit_to_lit_index, lit_index_negation
from tests.reference.formula import ReferenceFormula
from tests.conftest import REPO_ROOT, TEST_DATA_DIR
from tests.support import random_cnf

SIMPLE = [[-1, 2], [1, -2, 3], [-1, -2], [-3]]     # (~1|2), (1|~2|3), (~1|~2), (~3)


# ---------------------------------------------------------------- literal indexing
def test_positive_literal():
    assert ext_lit_to_lit_index(1) == 0
    assert ext_lit_to_lit_index(2) == 2
    assert ext_lit_to_lit_index(3) == 4


def test_negative_literal():
    assert ext_lit_to_lit_index(-1) == 1
    assert ext_lit_to_lit_index(-2) == 3
    assert ext_lit_to_lit_index(-3) == 5


def test_negation():
    assert lit_index_negation(0) == 1
    assert lit_index_negation(1) == 0
    assert lit_index_negation(4) == 5
    assert lit_index_negation(5) == 4


# ---------------------------------------------------------------- reference occurrence lists
def test_basic_construction():
    # SIMPLE splits as: clause 0 (~1|2) binary pair (1,2); clause 1 (1|~2|3) long id 0;
    # clause 2 (~1|~2) binary pair (1,3); clause 3 (~3) unit lit 5, with no occ entry.
    f = ReferenceFormula.from_clauses(3, SIMPLE)
    assert (f.num_bin, f.num_long) == (2, 1)
    assert f.bin_pairs == [(1, 2), (1, 3)]
    assert f.unit_lits == [5]

    # binary prefix: payload = the other literal, which setting l TRUE forces
    assert f.occ_bin(0).tolist() == [2, 3]  # x1 TRUE: clause 0 forces x2, clause 2 forces ~x2
    assert f.occ_bin(3).tolist() == [1]     # x2 FALSE: clause 0 forces ~x1
    assert f.occ_bin(2).tolist() == [1]     # x2 TRUE: clause 2 forces ~x1
    # long suffix: payload = the long clause's re-index
    assert f.occ_long(1).tolist() == [0]    # x1 FALSE falsifies +1 in long clause 0
    assert f.occ_long(2).tolist() == [0]    # x2 TRUE falsifies -2 in long clause 0
    assert f.occ_long(5).tolist() == [0]    # x3 FALSE falsifies +3 in long clause 0
    # the unit clause (~3) adds no entry: occ[4] is empty
    assert f.occ(4).tolist() == []
    # raw storage carries the binary tag (the sign bit); occ_bin() above is the masked view
    from tests.reference.formula import BIN_TAG
    assert f.occ(0).tolist() == [2 | BIN_TAG, 3 | BIN_TAG]
    assert all(x < 0 for x in f.occ(0).tolist())      # binaries tagged
    assert all(x >= 0 for x in f.occ_long(1).tolist())  # long clause ids are not


# ---------------------------------------------------------------- reference formula
def test_from_clauses():
    f = ReferenceFormula.from_clauses(3, SIMPLE)
    assert (f.num_vars, f.num_clauses) == (3, 4)
    assert f.clause_sizes.tolist() == [2, 3, 2, 1]
    # init_lit_sum (sum of lit-indices) per clause:
    #   clause 0: ~x1(1) + x2(2) = 3;  clause 1: x1(0) + ~x2(3) + x3(4) = 7
    #   clause 2: ~x1(1) + ~x2(3) = 4; clause 3: ~x3(5) = 5
    assert f.init_lit_sum.tolist() == [3, 7, 4, 5]


def test_from_file():
    f = ReferenceFormula.from_file(os.path.join(TEST_DATA_DIR, "simple_3v_4c.cnf"))
    assert (f.num_vars, f.num_clauses) == (3, 4)
    assert f.clause_sizes.tolist() == [2, 3, 2, 1]
    assert f.init_lit_sum.tolist() == [3, 7, 4, 5]


def test_single_literal_clauses():
    f = ReferenceFormula.from_clauses(1, [[1], [-1]])
    assert (f.num_vars, f.num_clauses) == (1, 2)
    assert f.clause_sizes.tolist() == [1, 1]
    assert f.init_lit_sum.tolist() == [0, 1]     # x1 -> 0, ~x1 -> 1


def test_empty_clause_is_dropped_and_flagged():
    f = ReferenceFormula.from_clauses(2, [[1, 2], []])
    assert f.has_empty_clause
    assert f.num_clauses == 1


# ---------------------------------------------------------------- cnf formula facade
# CnfFormula is a facade over the core's builder. Construction is host-side C++, so these build
# on a CPU device.

@pytest.mark.core
def test_init_formula_adopts_the_core_build():
    import torch
    f = CnfFormula(torch.device("cpu"))
    f.init_formula(3, SIMPLE)
    assert (f.num_vars, f.num_clauses) == (3, 4)
    assert f.max_clause_len == 3
    assert f.total_lits == 8
    # on-demand materialization from the core's clause arrays
    assert f.clause_sizes.tolist() == [2, 3, 2, 1]
    assert f.init_lit_sum.tolist() == [3, 7, 4, 5]


@pytest.mark.core
def test_parse_cnf_file():
    import torch
    f = CnfFormula(torch.device("cpu"))
    f.parse_cnf_file(os.path.join(TEST_DATA_DIR, "simple_3v_4c.cnf"))
    assert (f.num_vars, f.num_clauses) == (3, 4)
    assert f.clause_sizes.tolist() == [2, 3, 2, 1]
    assert f.init_lit_sum.tolist() == [3, 7, 4, 5]


# ---------------------------------------------------------------- read dimacs
# `read_dimacs` returns clauses as lists of literals, via the core's parser and clause read-back.

@pytest.mark.core
def test_matches_the_reference_parser_as_clause_sets(tmp_path):
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


@pytest.mark.core
def test_duplicate_literals_survive_the_round_trip():
    f = _core.Formula.from_clauses(2, [[1, 1, -2], [2]])
    assert [sorted(c) for c in f.clauses()] == [[-2, 1, 1], [2]]


@pytest.mark.core
def test_empty_clause_is_reappended(tmp_path):
    from gpap.engine import read_dimacs
    p = tmp_path / "unsat.cnf"
    p.write_text("p cnf 2 2\n1 2 0\n0\n")
    nv, clauses, _ = read_dimacs(str(p))
    assert [] in clauses and len(clauses) == 2


@pytest.mark.core
def test_header_n_wins_over_max_var_used(tmp_path):
    """A header declaring more variables than the clauses use is honoured, so support
    variables that appear in no clause keep their slots."""
    from gpap.engine import read_dimacs
    p = tmp_path / "h.cnf"
    p.write_text("p cnf 20 2\n1 10 0\n-10 2 0\n")
    nv, clauses, _ = read_dimacs(str(p))
    assert nv == 20 and len(clauses) == 2


@pytest.mark.core
def test_support_lines_c_ind_and_cr(tmp_path):
    """Both support-line conventions; the core's parse agrees with the Python scanner
    `parse_independent_support`."""
    from gpap.engine import read_dimacs
    from gpap.sampling import parse_independent_support
    p = tmp_path / "s.cnf"
    p.write_text("c ind 2 5 0\ncr 3\np cnf 5 2\n1 2 0\n-3 4 5 0\n")
    _, _, sup = read_dimacs(str(p))
    assert sup == [1, 2, 4]                     # 0-based, ascending, deduped
    assert parse_independent_support(str(p)) == sup


@pytest.mark.core
def test_a_directory_is_an_error_not_a_process_abort(tmp_path):
    """On Linux fopen succeeds on a directory and ftell reports LONG_MAX. The core must return
    a gpap_status, not throw through the C boundary and abort the process."""
    with pytest.raises(_core.GpapError) as e:
        _core.Formula.from_file(str(tmp_path))
    assert e.value.status == 4                                  # GPAP_ERR_IO


@pytest.mark.core
def test_a_non_seekable_stream_parses_its_actual_bytes(tmp_path):
    """A FIFO (like /dev/stdin or <(...)) has no ftell size. Its bytes must be parsed, not
    replaced by the empty formula."""
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


@pytest.mark.core
def test_a_file_ftell_sizes_as_empty_is_still_read(tmp_path):
    """procfs files report size 0 but have content. This one is not a CNF, so it must be
    rejected, not read as the empty formula."""
    with pytest.raises(_core.GpapError) as e:
        _core.Formula.from_file("/proc/self/status")
    assert e.value.status == 4                                  # read, then refused as non-DIMACS


@pytest.mark.core
def test_an_empty_file_is_still_the_empty_formula(tmp_path):
    """A truly empty file is the empty formula, not an error."""
    p = tmp_path / "empty.cnf"
    p.write_text("")
    info = _core.Formula.from_file(str(p)).info
    assert (info.num_vars, info.num_clauses) == (0, 0)


# ---------------------------------------------------------------- occ tag and split agree
# The binary/long split is read two ways that must agree. `_wave_propagate` classifies an occ
# entry by the sign bit of its payload (saving a gather in the hot loop); `_clean_packed` and the
# scalar `_propagate` use `occ_mid[l]`. A binary entry at or above `occ_mid` would make the
# restore index `init_ptr` with a negative payload, an out-of-bounds device write.

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


#: Shapes where one of the formula's vectors is empty. Reading those back must not memcpy from
#: `data()` of an empty vector, which may be nullptr (UB at any length).
_DEGENERATE = [
    (4, [[1, 2], [-1, 3, 4], [2], [-2, -3], [1, 2, 3, 4]]),
    (3, [[1, 1], [1, -1], [2, -2, 3], [3, 3, 3]]),       # dup / tautological
    (5, [[1, 2], [3, 4]]),                                # no long clauses at all
    (5, [[1, 2, 3], [-1, -2, -3]]),                       # no binaries at all
    (2, [[1], [2]]),                                      # units only: occ_clauses empty
    (3, []),                                              # clause_sizes / lit_sum empty
    (0, []),                                              # occ_offsets / occ_mid empty too
]


@pytest.mark.core
def test_on_mixed_and_degenerate_formulas():
    for nv, clauses in _DEGENERATE:
        _check(_core.Formula.from_clauses(nv, clauses))


@pytest.mark.core
def test_every_read_back_accessor_survives_an_empty_vector():
    """Call every read-back accessor on the shapes that empty its vectors. The null-source
    memcpy is only visible under a sanitizer, so this test is meant to run under one."""
    for nv, clauses in _DEGENERATE:
        f = _core.Formula.from_clauses(nv, clauses)
        offsets, mid, occ = f.occ()
        sizes, sums = f.clause_arrays()
        back = f.clauses()
        # No `or nv == 0` escape: the (0, []) row must be checked too.
        assert len(offsets) == 2 * nv + 1
        assert len(sizes) == len(sums) == f.info.num_clauses == len(back)
        assert [sorted(c) for c in back] == [sorted(c) for c in clauses if c]


@pytest.mark.core
def test_on_a_real_benchmark():
    import pathlib
    cnf = sorted((pathlib.Path(REPO_ROOT) / "benchmarks" / "sampling").glob("*.cnf"))
    if not cnf:
        pytest.skip("no benchmark CNFs in the checkout")
    f = _core.Formula.from_file(str(cnf[0]))
    assert f.info.num_bin > 0 and f.info.num_long > 0, "fixture must hold both kinds"
    _check(f)


def read_support(path):
    """The 0-based independent support the core parsed from `c ind` / `cr`, read through the
    ABI so the parser itself is under test."""
    f = _core.Formula.from_file(str(path))
    n = ctypes.c_int32(0)
    _core.load().gpap_formula_support(f._h, None, ctypes.byref(n))
    if not n.value:
        return []
    out = np.zeros(n.value, dtype=np.int32)
    _core.load().gpap_formula_support(
        f._h, out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.byref(n))
    return out[:n.value].tolist()


# ------------------------------------------------- support line is bounded by its line
# strtol skips leading whitespace, including \r and \n. A `c ind` scan that stops short of the
# newline (a \r on a CRLF file, a trailing space) must not read the next line's first literal
# into the support. The bundled benchmarks are all LF, so only these tests reach it.

def _support_of_bytes(tmp_path, raw):
    p = tmp_path / "s.cnf"
    p.write_bytes(raw)
    return read_support(p)


@pytest.mark.core
@pytest.mark.parametrize(
    "name,raw",
    [
        ("lf", b"p cnf 5 2\nc ind 2 3 0\n5 2 0\n-3 4 0\n"),
        ("crlf", b"p cnf 5 2\r\nc ind 2 3 0\r\n5 2 0\r\n-3 4 0\r\n"),
        ("trailing space", b"p cnf 5 2\nc ind 2 3 0 \n5 2 0\n-3 4 0\n"),
        ("trailing tab", b"p cnf 5 2\nc ind 2 3 0\t\n5 2 0\n-3 4 0\n"),
    ],
)
def test_line_endings_do_not_leak_the_next_clause(tmp_path, name, raw):
    # variable 5 opens the next line; 0-based support must stay {1, 2} in every encoding
    assert _support_of_bytes(tmp_path, raw) == [1, 2], name


@pytest.mark.core
def test_a_cr_line_is_bounded_too(tmp_path):
    # the `cr <v...>` variant takes the same path and has no terminating 0 at all
    assert _support_of_bytes(tmp_path, b"p cnf 5 2\r\ncr 2 3\r\n5 2 0\r\n-3 4 0\r\n") == [1, 2]


# ---------------------------------------------------------------- support line ranges
# `c ind` / `cr` values get the same range checks as literals: no int32 wrap, no negative index,
# no `-v` on LONG_MIN (see gpap_internal.hpp). Out-of-range values are dropped, not refused, as
# for a support variable past the header's N: `c ind` is a comment line and tools write junk
# there.

def _support_of_line(tmp_path, line):
    p = tmp_path / "s.cnf"
    p.write_text(f"p cnf 5 2\n{line}\n1 2 0\n-3 4 5 0\n")
    return read_support(p)


@pytest.mark.core
@pytest.mark.parametrize("line", [
    "c ind 4294967297 0",              # would wrap through int32 to variable 1
    "c ind 9223372036854775808 0",     # past int64
    "c ind -9223372036854775808 0",    # LONG_MIN: -v is UB, and strtol does not flag it
    "c ind -2147483648 0",             # INT32_MIN, the same trap one width down
    "c ind 99 0",                      # in range for int32, past the header's N
])
def test_unrepresentable_values_are_dropped(tmp_path, line):
    assert _support_of_line(tmp_path, line) == []


@pytest.mark.core
def test_valid_values_survive_either_polarity_and_either_keyword(tmp_path):
    assert _support_of_line(tmp_path, "c ind 2 3 0") == [1, 2]
    assert _support_of_line(tmp_path, "c ind -2 -3 0") == [1, 2]     # sign is ignored
    assert _support_of_line(tmp_path, "cr 1 4") == [0, 3]
    assert _support_of_line(tmp_path, "c ind 2 99 3 0") == [1, 2]    # junk dropped, rest kept


@pytest.mark.core
def test_every_returned_index_is_a_valid_variable(tmp_path):
    for line in ("c ind 4294967297 2 -9223372036854775808 3 0", "c ind 0 0", "c ind x y 0"):
        for v in _support_of_line(tmp_path, line):
            assert 0 <= v < 5, (line, v)


# ------------------------------------------------- the core's builder vs the reference

@pytest.mark.core
def test_formula_structures_match_the_reference_builder():
    rng = random.Random(7)
    for nv, nc in [(6, 10), (40, 120), (200, 900)]:
        clauses = random_cnf(rng, nv, nc)
        f = ReferenceFormula.from_clauses(nv, clauses)
        cf = _core.Formula.from_clauses(nv, clauses)

        assert cf.info.num_vars == f.num_vars
        assert cf.info.num_clauses == f.num_clauses
        assert cf.info.num_bin == f.num_bin
        assert cf.info.num_long == f.num_long
        off, mid, cls_ = cf.occ()
        np.testing.assert_array_equal(off, f.occ_offsets)
        np.testing.assert_array_equal(mid, f.occ_mid)
        np.testing.assert_array_equal(cls_, f.occ_clauses)
        sizes, lsum = cf.clause_arrays()
        np.testing.assert_array_equal(sizes, f.clause_sizes)
        np.testing.assert_array_equal(lsum, f.init_lit_sum)


@pytest.mark.core
def test_dimacs_parse_matches_the_reference_parser(tmp_path):
    rng = random.Random(11)
    clauses = random_cnf(rng, 30, 80)
    p = tmp_path / "f.cnf"
    with open(p, "w") as fh:
        fh.write("c a comment\np cnf 30 %d\n" % len(clauses))
        for c in clauses:                       # two clauses per line: the token stream matters
            fh.write(" ".join(map(str, c)) + " 0 ")
    f = ReferenceFormula.from_file(str(p))
    cf = _core.Formula.from_file(str(p))
    assert (cf.info.num_vars, cf.info.num_clauses) == (f.num_vars, f.num_clauses)
    np.testing.assert_array_equal(cf.clause_arrays()[0], f.clause_sizes)


@pytest.mark.core
def test_envelope_matches():
    """The packed-split envelope in cpp/src/runtime.cpp and its Python mirror (used to
    pre-check formulas without CUDA) agree."""
    from gpap.engine.bcp_engine import packed_envelope
    for nv, ml in [(10, 3), (32767, 5), (33000, 5), (40000, 300), (9_000_000, 3)]:
        assert _core.packed_envelope(nv, ml)[0] == packed_envelope(nv, ml)[0]

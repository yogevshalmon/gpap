"""User-facing input validation, found by exercising the public surface as a user would.

Every case here was a real observed failure before its guard existed:

  * `apply_and_propagate` with a CUDA decisions tensor bypassed the host path's row-count
    check entirely — a 2-row tensor against K=4 was an out-of-bounds DEVICE read whose garbage
    became literal indices ("succeeded" silently). The host path validated; the device path
    did not. Memory safety, not politeness.
  * `--cycles 0` reached the unroller and died as `KeyError: 'LINE2'` on one circuit and
    `RecursionError` on another — the wrong error for a plainly invalid input, found only
    after the (possibly expensive) parse.
  * a truncated `.aag` died as a bare `IndexError: list index out of range`.
  * `gpap mine --seconds 0` help said "0 = unbounded" while the implementation treated it as
    a deadline of now — every width reported `[budget]` with 0 probed. The two meanings of
    zero produced opposite behaviors in neighboring flags.
  * top-level `sample_clauses(support=...)` raised a bare IndexError on an out-of-range var —
    and silently mis-projected when a 1-BASED list happened to stay in range.
  * `LiteralProbe.probe(size=0)` went all the way to the device for its "bad argument".
"""
import pytest
import torch

from gpap.circuits.aig2cnf import build_cnf_aig
from gpap.circuits.aiger import read_aiger
from gpap.circuits.bench import build_cnf, parse_bench
from gpap.cli import _k_cap_arg
from gpap.engine import BCPEngine, CnfFormula, LiteralProbe, core_available

AVAIL = torch.cuda.is_available() and core_available()
needs_gpu = pytest.mark.skipif(not AVAIL, reason="requires CUDA + a built libgpap")
# The DIMACS parser lives in the C++ core, so its tests need the LIBRARY but no GPU device —
# except that loading libgpap needs libcuda, so on a driverless CI runner these skip too.
needs_core = pytest.mark.skipif(not core_available(),
                                reason="requires a built libgpap (the parser is in the core)")

TINY_BENCH = """\
INPUT(a)
OUTPUT(y)
r = DFF(a)
y = AND(a, r)
"""

TINY_AAG = """\
aag 3 2 0 1 1
2
4
6
6 2 4
"""


# --------------------------------------------------------------------- host-side (no GPU)

@pytest.mark.parametrize("cycles", [0, -3])
def test_bench_unroll_rejects_bad_cycles(tmp_path, cycles):
    p = tmp_path / "t.bench"
    p.write_text(TINY_BENCH)
    with pytest.raises(ValueError, match="cycles must be >= 1"):
        build_cnf(parse_bench(str(p)), cycles, seed=1)


@pytest.mark.parametrize("cycles", [0, -3])
def test_aig_unroll_rejects_bad_cycles(tmp_path, cycles):
    p = tmp_path / "t.aag"
    p.write_text(TINY_AAG)
    with pytest.raises(ValueError, match="cycles must be >= 1"):
        build_cnf_aig(read_aiger(str(p)), cycles, seed=1)


def test_truncated_aag_is_a_clear_error(tmp_path):
    p = tmp_path / "bad.aag"
    p.write_text("aag 3 2 0 1 1\n2\n")          # header declares 4 definition lines, has 1
    with pytest.raises(ValueError, match="truncated"):
        read_aiger(str(p))


def test_k_cap_arg_accepts_auto_and_positive_ints():
    assert _k_cap_arg("auto") == "auto"
    assert _k_cap_arg("4096") == 4096


@pytest.mark.parametrize("bad", ["0", "-5", "frog"])
def test_k_cap_arg_rejects_nonpositive_and_junk(bad):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _k_cap_arg(bad)


def test_sample_clauses_support_is_validated_as_zero_based():
    import gpap
    with pytest.raises(ValueError, match="0-BASED"):
        gpap.sample_clauses(4, [[1, 2]], n=10, support=[4])


# --------------------------------------------------------------------- engine (GPU)

@needs_gpu
def test_device_path_apply_validates_row_count():
    """The CUDA-tensor fast path must reject a row mismatch exactly like the host path does:
    apply_dev hands the core a raw pointer it reads K rows from, so a short tensor is an
    out-of-bounds device read. Before the guard this call 'succeeded'."""
    dev = torch.device("cuda")
    f = CnfFormula(dev)
    f.init_formula(6, [[1, 2], [-1, 3]])
    eng = BCPEngine(f)
    eng.reset(4)
    short = torch.full((2, 1), -1, dtype=torch.int64, device=dev)
    with pytest.raises(ValueError, match="2 rows but the engine is sized for K=4"):
        eng.apply_and_propagate(short)
    # and the matching shape still goes through
    eng.apply_and_propagate(torch.full((4, 1), -1, dtype=torch.int64, device=dev))


@needs_gpu
def test_probe_size_validated_on_host():
    pr = LiteralProbe(6, [[1, 2], [-1, 3]], torch.device("cuda"))
    with pytest.raises(ValueError, match="probe size must be >= 1"):
        pr.probe(None, 0)


@needs_gpu
def test_probe_tuples_ragged_input_is_a_clear_error():
    pr = LiteralProbe(6, [[1, 2], [-1, 3]], torch.device("cuda"))
    with pytest.raises(ValueError, match="rectangular"):
        pr.probe_tuples([[1], [2, 3]])


# --------------------------------------------------------------------- strict DIMACS parser

# The tokenizer used to DROP anything it did not recognize ("stray character: ignore"), so a
# corrupted file silently became a DIFFERENT formula — `1 2 frog 0` parsed as (1 2), and a
# weighted-CNF `w 1 0.9` line became clauses (0.9 tokenizes as literal 0 = a terminator!).
# Strict on corruption, lenient on structure; both halves are gated here.

PARSER_REJECT = {
    "garbage-token":     "p cnf 3 2\n1 2 frog 0\n-1 3 0\n",
    "glued-token":       "p cnf 3 1\n1a 2 0\n",
    "decimal-literal":   "p cnf 3 1\n1.5 2 0\n",
    "negative-zero":     "p cnf 2 1\n1 -0 2 0\n",
    "stray-dash":        "p cnf 2 1\n1 - 2 0\n",
    "mid-line-comment":  "p cnf 2 1\n1 2 0 c trailing\n",
    "wcnf-header":       "p wcnf 3 2\n1 2 0\n",
    "header-no-counts":  "p cnf\n1 2 0\n",       # sscanf must not read counts off the next line
    "conflicting-hdrs":  "p cnf 3 1\np cnf 4 1\n1 2 0\n",
    "weight-lines":      "p cnf 2 0\nw 1 0.9\nw 2 0.5\n",
    "overflow-literal":  "p cnf 3 1\n99999999999 2 0\n",
}

PARSER_ACCEPT = {
    "no-header":         "1 2 0\n-1 3 0\n",
    "header-undercount": "p cnf 3 1\n1 2 0\n-1 3 0\n2 3 0\n",
    "var-above-header":  "p cnf 3 2\n1 9 0\n-1 3 0\n",
    "no-final-zero-eof": "p cnf 3 2\n1 2 0\n-1 3",
    "empty-clause":      "p cnf 3 2\n1 2 0\n0\n",
    "taut-and-dup":      "p cnf 3 2\n1 -1 0\n2 2 3 0\n",
    "satlib-trailer":    "p cnf 3 2\n1 2 0\n-1 3 0\n%\n0\n\n",
    "c-ind-support":     "c ind 1 2 0\np cnf 3 1\n1 2 3 0\n",
    "crlf-endings":      "p cnf 3 1\r\n1 2 0\r\n",
    "multi-clause-line": "p cnf 3 2\n1 2 0 -1 3 0\n",
    "clause-spans-lines": "p cnf 3 1\n1\n2\n3 0\n",
    # the QuickSampler/HTS distribution repeats the IDENTICAL header — harmless, accepted
    "identical-dup-hdr": "p cnf 3 1\np cnf 3 1\n1 2 0\n",
}


@needs_core
@pytest.mark.parametrize("text", PARSER_REJECT.values(), ids=list(PARSER_REJECT))
def test_dimacs_rejects_corruption(tmp_path, text):
    from gpap.engine import read_dimacs
    p = tmp_path / "z.cnf"
    p.write_text(text)
    with pytest.raises(Exception, match="line \\d"):
        read_dimacs(str(p))


@needs_core
@pytest.mark.parametrize("text", PARSER_ACCEPT.values(), ids=list(PARSER_ACCEPT))
def test_dimacs_keeps_structural_leniency(tmp_path, text):
    from gpap.engine import read_dimacs
    p = tmp_path / "z.cnf"
    p.write_text(text)
    read_dimacs(str(p))                          # must not raise


@needs_gpu
def test_mine_entailed_seconds_none_is_unbounded():
    """`seconds=None` (what the CLI now maps `--seconds 0` to) must actually sweep — the CLI
    help promised "0 = unbounded" while the old passthrough made it a deadline of now."""
    from gpap.mining import mine_entailed
    res = mine_entailed(6, [[1, 2], [-1, 3], [-2, 3]], torch.device("cuda"),
                        widths=(1,), seconds=None)
    assert res.probed.get(1, 0) > 0
    assert not res.skipped

"""Input validation on the public surface: cycle counts, truncated AIGER, `--k-cap`, 0-based
`support`, the row count on the CUDA apply path, probe sizes, the strict DIMACS parser and
`--seconds 0`. Each invalid input gets a clear error instead of a crash or a wrong result.
"""
import pytest
import torch

from gpap.circuits.aig2cnf import build_cnf_aig
from gpap.circuits.aiger import read_aiger
from gpap.circuits.bench import build_cnf, parse_bench
from gpap.cli import _k_cap_arg
from gpap.engine import BCPEngine, CnfFormula, LiteralProbe


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

@pytest.mark.gpu
def test_device_path_apply_validates_row_count():
    """The CUDA-tensor path rejects a row-count mismatch like the host path: apply_dev reads K
    rows from a raw pointer, so a short tensor would be an out-of-bounds device read."""
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


@pytest.mark.gpu
def test_probe_size_validated_on_host():
    pr = LiteralProbe(6, [[1, 2], [-1, 3]], torch.device("cuda"))
    with pytest.raises(ValueError, match="probe size must be >= 1"):
        pr.probe(None, 0)


@pytest.mark.gpu
def test_probe_tuples_ragged_input_is_a_clear_error():
    pr = LiteralProbe(6, [[1, 2], [-1, 3]], torch.device("cuda"))
    with pytest.raises(ValueError, match="rectangular"):
        pr.probe_tuples([[1], [2, 3]])


# --------------------------------------------------------------------- strict DIMACS parser

# Strict on corruption, lenient on structure. Skipping unknown tokens would silently parse a
# different formula: `1 2 frog 0` as (1 2), and `w 1 0.9` as clauses (0.9 reads as literal 0).

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
    # the QuickSampler/HTS benchmark files repeat the identical header
    "identical-dup-hdr": "p cnf 3 1\np cnf 3 1\n1 2 0\n",
}


@pytest.mark.gpu
@pytest.mark.parametrize("text", PARSER_REJECT.values(), ids=list(PARSER_REJECT))
def test_dimacs_rejects_corruption(tmp_path, text):
    from gpap.engine import read_dimacs
    p = tmp_path / "z.cnf"
    p.write_text(text)
    with pytest.raises(Exception, match="line \\d"):
        read_dimacs(str(p))


@pytest.mark.gpu
@pytest.mark.parametrize("text", PARSER_ACCEPT.values(), ids=list(PARSER_ACCEPT))
def test_dimacs_keeps_structural_leniency(tmp_path, text):
    from gpap.engine import read_dimacs
    p = tmp_path / "z.cnf"
    p.write_text(text)
    read_dimacs(str(p))                          # must not raise


@pytest.mark.gpu
def test_mine_entailed_seconds_none_is_unbounded():
    """`seconds=None` (what `--seconds 0` maps to) sweeps with no deadline."""
    from gpap.mining import mine_entailed
    res = mine_entailed(6, [[1, 2], [-1, 3], [-2, 3]], torch.device("cuda"),
                        widths=(1,), seconds=None)
    assert res.probed.get(1, 0) > 0
    assert not res.skipped

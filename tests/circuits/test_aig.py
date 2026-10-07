"""Native AIGER (.aig/.aag) input support.

Load-bearing claims:
  * the CNF that `build_cnf_aig` emits describes EXACTLY the input sequences whose simulation
    reproduces the pinned target (checked by brute force over all 2^(PI*T) sequences);
  * `simulate_aig` agrees with `bench.simulate` on the same circuit across formats;
  * the ABC route on an AIG input yields the same projected solution set as the native route;
  * AIGER reset values are honoured (0/1) or safely rejected (`x`, and reset-1 under ABC).

Handwritten-`aag` tests run everywhere; cross-format tests need an `abc` binary and skip
without one (scripts/build_abc.sh).
"""
import itertools
import os

import pytest

from gpap.circuits.abc_backend import abc_available, find_abc
from gpap.circuits.aig2cnf import build_cnf_aig, convert_aig_one, simulate_aig
from gpap.circuits.aiger import read_aiger

from tests.conftest import SEQ_DIR as BENCH
from tests.support import enumerate_projections

# 2 inputs, 1 latch (reset 0), 1 output = ~latch; latch chases and(and(i0,i1), latch).
AAG_SEQ = """aag 5 2 1 1 2
2
4
6 10
7
8 2 4
10 8 6
i0 a
i1 b
o0 y
"""

# Same circuit with the latch RESET to 1.
AAG_SEQ_R1 = AAG_SEQ.replace("6 10\n", "6 10 1\n")

# Combinational: y = a AND b (no latches — unrolling must clamp to T=1).
AAG_COMB = """aag 3 2 0 1 1
2
4
6
6 2 4
"""


def _brute_force(aig, T, target):
    """All input sequences (as support-ordered bit tuples, input-major) whose simulation hits
    `target`. Support order must match build_cnf_aig: [(k, t) for k in inputs for t in range(T)]."""
    n = aig.n_inputs
    out = set()
    for bits in itertools.product((0, 1), repeat=n * T):
        seq = [[bool(bits[k * T + t]) for k in range(n)] for t in range(T)]
        if simulate_aig(aig, seq, T) == target:
            out.add(bits)
    return out


def _write(tmp_path, name, text):
    p = str(tmp_path / name)
    with open(p, "w") as f:
        f.write(text)
    return p


# ------------------------------------------------------------------ native path (no abc needed)

@pytest.mark.parametrize("text,reset", [(AAG_SEQ, 0), (AAG_SEQ_R1, 1)])
def test_native_aag_matches_brute_force(tmp_path, text, reset):
    aig = read_aiger(_write(tmp_path, "t.aag", text))
    assert aig.latches[0][2] == reset
    T = 3
    b, support, target, labels = build_cnf_aig(aig, T, seed=7)
    got = enumerate_projections(b.clauses, support)
    want = _brute_force(aig, T, target)
    assert want, "pinned target unreachable — bad test circuit"
    assert got == want
    assert labels == [f"{n}@{t}" for n in ("a", "b") for t in range(T)]


def test_native_comb_clamps_to_one_cycle(tmp_path):
    aig = read_aiger(_write(tmp_path, "c.aag", AAG_COMB))
    b, support, target, labels = build_cnf_aig(aig, 25, seed=1)   # T forced to 1
    assert len(support) == 2 and labels == ["i0@0", "i1@0"]
    got = enumerate_projections(b.clauses, support)
    want = {bits for bits in itertools.product((0, 1), repeat=2)
            if (bool(bits[0]) and bool(bits[1])) == target[0]}
    assert got == want


def test_uninitialized_latch_rejected(tmp_path):
    p = _write(tmp_path, "x.aag", AAG_SEQ.replace("6 10\n", "6 10 6\n"))   # reset = own literal
    with pytest.raises(ValueError, match="UNINITIALIZED"):
        read_aiger(p)


def test_binary_writer_roundtrip(tmp_path):
    """write_aiger_binary -> read_aiger must preserve circuit BEHAVIOUR (simulation equality on
    every input sequence) and the I/O symbol names, across renumbering/topo-sorting."""
    from gpap.circuits.aiger import write_aiger_binary
    src = read_aiger(_write(tmp_path, "t.aag", AAG_SEQ))
    p = str(tmp_path / "t.aig")
    write_aiger_binary(src, p)
    back = read_aiger(p)
    assert (back.n_inputs, back.n_latches, back.n_outputs) == (2, 1, 1)
    assert back.input_names == src.input_names and back.output_names == src.output_names
    T = 4
    for bits in itertools.product((0, 1), repeat=2 * T):
        seq = [[bool(bits[k * T + t]) for k in range(2)] for t in range(T)]
        assert simulate_aig(src, seq, T) == simulate_aig(back, seq, T)


def test_provenance_comment(tmp_path):
    from gpap.circuits.bench import parse_encoder_comment
    p = _write(tmp_path, "t.aag", AAG_SEQ)
    out = str(tmp_path / "t.cnf")
    convert_aig_one(p, out, 3, 1, quiet=True)                     # native
    assert parse_encoder_comment(out) == "tseitin"
    header = open(out).read().splitlines()[:8]
    assert any(ln.startswith("c ind") for ln in header)


# ------------------------------------------------------------------ cross-format (needs abc)

needs_abc = pytest.mark.skipif(not abc_available(), reason="no `abc` binary (scripts/build_abc.sh)")


def _bench_to_aig(tmp_path, name):
    """b<name>.bench -> binary .aig via ABC (zero pins the don't-care init to our reset-0)."""
    import subprocess
    src = os.path.abspath(os.path.join(BENCH, f"{name}.bench"))
    dst = str(tmp_path / f"{name}.aig")
    subprocess.run([find_abc(), "-q", f'read_bench "{src}"; strash; zero; write_aiger -s "{dst}"'],
                   check=True, capture_output=True, cwd=str(tmp_path))
    return dst


@needs_abc
def test_simulate_aig_matches_bench_simulate(tmp_path):
    """Binary parse + latch semantics: simulate the ABC-written .aig of b06 and the original
    .bench side by side (I/O matched BY NAME — ABC may reorder)."""
    import random
    from gpap.circuits.bench import parse_bench, simulate
    aig = read_aiger(_bench_to_aig(tmp_path, "b06"))
    circ = parse_bench(os.path.join(BENCH, "b06.bench"))
    inputs, outputs, _d, _g = circ
    assert sorted(aig.input_names.values()) == sorted(inputs)
    rng, T = random.Random(3), 5
    for _ in range(20):
        seq_named = [{i: rng.random() < 0.5 for i in inputs} for _ in range(T)]
        seq_aig = [[seq_named[t][aig.input_names[k]] for k in range(aig.n_inputs)]
                   for t in range(T)]
        ours = simulate(circ, seq_named, T)
        theirs = simulate_aig(aig, seq_aig, T)
        assert all(theirs[j] == ours[aig.output_names[j]] for j in range(aig.n_outputs))


@needs_abc
def test_native_aig_unroll_matches_brute_force_b06(tmp_path):
    aig = read_aiger(_bench_to_aig(tmp_path, "b06"))
    T = 3
    b, support, target, _lab = build_cnf_aig(aig, T, seed=4)
    assert enumerate_projections(b.clauses, support) == _brute_force(aig, T, target)


@needs_abc
def test_abc_route_on_aig_input_preserves_projections(tmp_path):
    from gpap.circuits.abc_backend import build_cnf_abc_aig
    path = _bench_to_aig(tmp_path, "b06")
    aig = read_aiger(path)
    T, seed = 3, 4
    bn, sn, tn, ln = build_cnf_aig(aig, T, seed)
    ba, sa, ta, la = build_cnf_abc_aig(aig, path, T, seed)
    assert (tn, ln) == (ta, la)                    # same simulated target, same label order
    assert enumerate_projections(bn.clauses, sn) == enumerate_projections(ba.clauses, sa)


@needs_abc
def test_abc_route_on_ascii_aag_input(tmp_path):
    """ABC's read_aiger does NOT accept ASCII `aag`; the abc route must still work on .aag input
    (it rewrites the parsed circuit as canonical binary first) and preserve the projection set."""
    from gpap.circuits.abc_backend import build_cnf_abc_aig
    p = _write(tmp_path, "t.aag", AAG_SEQ)
    aig = read_aiger(p)
    T, seed = 3, 5
    bn, sn, tn, ln = build_cnf_aig(aig, T, seed)
    ba, sa, ta, la = build_cnf_abc_aig(aig, p, T, seed)
    assert (tn, ln) == (ta, la)
    assert enumerate_projections(bn.clauses, sn) == enumerate_projections(ba.clauses, sa)


@needs_abc
def test_abc_route_rejects_reset1_aig(tmp_path):
    p = _write(tmp_path, "r1.aag", AAG_SEQ_R1)
    from gpap.circuits.abc_backend import build_cnf_abc_aig
    with pytest.raises(RuntimeError, match="reset=1"):
        build_cnf_abc_aig(read_aiger(p), p, 3, 1)

"""Bit-parallel simulation (gpap/circuits/simulate.py): the rejection baseline, rare-target
selection and the sample check. Everything rests on `_run_bench`/`_run_aig` agreeing bit for bit
with the scalar reference simulators that define the pinned-target semantics.
"""
import itertools

import numpy as np
import pytest

from gpap.circuits.simulate import (_run_aig, _run_bench, _topo_ands, _topo_gates, rare_target,
                                    sim_baseline, simulate_check)


# A small sequential circuit exercising every supported gate op + a DFF.
BENCH = """\
INPUT(a)
INPUT(b)
INPUT(c)
OUTPUT(y)
OUTPUT(z)
s = DFF(n1)
n1 = XOR(a, s)
n2 = NAND(b, c)
n3 = NOR(a, n2)
n4 = NOT(n3)
y = AND(n1, n4)
z = OR(n2, s)
"""

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


@pytest.fixture
def bench_circ(tmp_path):
    p = tmp_path / "t.bench"
    p.write_text(BENCH)
    from gpap.circuits.bench import parse_bench
    return str(p), parse_bench(str(p))


@pytest.fixture
def aig_circ(tmp_path):
    p = tmp_path / "t.aag"
    p.write_text(AAG_SEQ)
    from gpap.circuits.aiger import read_aiger
    return str(p), read_aiger(str(p))


def test_bitparallel_bench_matches_reference(bench_circ):
    from gpap.circuits.bench import simulate
    _, circ = bench_circ
    T, n_words = 3, 2
    rng = np.random.default_rng(7)
    given = {(inp, t): rng.integers(0, np.uint64(2**64 - 1), n_words, dtype=np.uint64,
                                    endpoint=True)
             for inp in circ[0] for t in range(T)}
    last = _run_bench(circ, _topo_gates(circ), T, n_words, given=given)
    for i in range(n_words * 64):
        w, bit = divmod(i, 64)
        in_seq = [{inp: bool((int(given[(inp, t)][w]) >> bit) & 1) for inp in circ[0]}
                  for t in range(T)]
        ref = simulate(circ, in_seq, T)
        got = {o: bool((int(last[o][w]) >> bit) & 1) for o in circ[1]}
        assert got == ref


def test_bitparallel_aig_matches_reference_exhaustive(aig_circ):
    from gpap.circuits.aig2cnf import simulate_aig
    _, aig = aig_circ
    T = 4
    seqs = list(itertools.product((0, 1), repeat=2 * T))      # every input sequence
    given = {}
    for k in range(2):
        for t in range(T):
            bits = np.array([s[k * T + t] for s in seqs], dtype=np.uint8)
            given[(k, t)] = np.packbits(bits, bitorder="little").view(np.uint64)
    last = _run_aig(aig, _topo_ands(aig), T, len(seqs) // 64, given=given)
    for i, s in enumerate(seqs):
        in_seq = [[bool(s[t]), bool(s[T + t])] for t in range(T)]
        ref = simulate_aig(aig, in_seq, T)
        got = [bool((int(last[j][i // 64]) >> (i % 64)) & 1) for j in range(aig.n_outputs)]
        assert got == ref


def test_p_accept_exact_on_combinational_and(tmp_path):
    # y = AND(a, b): p_accept is 3/4 (target 0) or 1/4 (target 1); the estimate must land
    # within binomial noise of the one the seed picked.
    p = tmp_path / "and.bench"
    p.write_text("INPUT(a)\nINPUT(b)\nOUTPUT(y)\ny = AND(a, b)\n")
    from gpap.circuits.bench import parse_bench, random_target
    tgt = random_target(parse_bench(str(p)), 1, seed=5)
    want = 0.25 if tgt["y"] else 0.75
    r = sim_baseline(str(p), 1, seed=5, n_min=1 << 18, budget_s=0.2, hard_cap_s=2.0)
    assert r["p_accept"] == pytest.approx(want, abs=0.01)
    assert r["accepted_per_s"] > 0 and r["patterns_per_s"] > 0


def test_rare_target_is_rarer_and_consistent(bench_circ):
    path, circ = bench_circ
    T = 3
    tgt, est = rare_target("bench", circ, T, seed=5)
    r_rare = sim_baseline(path, T, seed=5, target_mode="rare",
                          n_min=1 << 18, budget_s=0.2, hard_cap_s=2.0)
    r_rand = sim_baseline(path, T, seed=5, n_min=1 << 18, budget_s=0.2, hard_cap_s=2.0)
    assert r_rare["p_accept"] <= r_rand["p_accept"] + 1e-9    # never easier than default
    assert r_rare["p_accept"] == pytest.approx(est, abs=0.05)  # measured ~ selection estimate
    # determinism: same seed -> same target (the converters re-derive it independently)
    assert rare_target("bench", circ, T, seed=5)[0] == tgt


def test_rare_target_cnf_is_sat_with_matching_projections(tmp_path):
    # AND of 3 inputs: the rare pattern is y=1 (p=1/8), which has exactly one solution.
    p = tmp_path / "and3.bench"
    p.write_text("INPUT(a)\nINPUT(b)\nINPUT(c)\nOUTPUT(y)\ny = AND(a, b, c)\n")
    from itertools import product
    from gpap.circuits.bench import convert_one, parse_support_labels
    from pysat.formula import CNF
    from pysat.solvers import Solver
    cnf_p = str(tmp_path / "and3.cnf")
    convert_one(str(p), cnf_p, 1, seed=5, target_mode="rare", quiet=True)
    cnf = CNF(from_file=cnf_p)
    sup = sorted(parse_support_labels(cnf_p))
    with Solver(name="cadical195", bootstrap_with=cnf.clauses) as s:
        projs = {bits for bits in product((0, 1), repeat=len(sup))
                 if s.solve(assumptions=[v if b else -v for v, b in zip(sup, bits)])}
    assert projs == {(1, 1, 1)}


def test_simulate_check_accepts_real_solutions_and_flags_corruption(bench_circ):
    path, circ = bench_circ
    from gpap.circuits.bench import build_cnf
    from pysat.formula import CNF
    from pysat.solvers import Solver
    T = 3
    b, support, _target, labels = build_cnf(circ, T, seed=5)
    s = Solver(name="cadical195", bootstrap_with=CNF(from_clauses=b.clauses))
    rows = []
    while s.solve() and len(rows) < 32:
        m = set(s.get_model())
        rows.append([1 if v in m else 0 for v in support])
        s.add_clause([(-v if v in m else v) for v in support])
    s.delete()
    assert rows
    bits = np.array(rows, dtype=np.uint8)
    assert simulate_check(path, T, seed=5, bits=bits, col_labels=labels) == 0
    bad = bits.copy()
    bad[0] ^= 1                                    # flip a whole row: no longer a solution
    assert simulate_check(path, T, seed=5, bits=bad, col_labels=labels) >= 1


@pytest.mark.gpu
def test_simulate_check_through_pipeline_bits(tmp_path, bench_circ):
    # End-to-end: GPAP-emitted sample rows must drive the circuit to the pinned outputs.
    import torch
    path, circ = bench_circ
    from gpap.circuits.bench import convert_one
    from gpap.sampling.pipeline import sample_file
    cnf_p = str(tmp_path / "t.cnf")
    convert_one(path, cnf_p, 3, seed=5, quiet=True)
    info = sample_file(cnf_p, torch.device("cuda"), n_time=500, n_cov=0, out_cap=200,
                       configs=None, time_limit=20.0, mine=False)
    mat, cols, wm = info["samples"], info["cols"], info["wire_map"]
    assert mat is not None and len(mat) > 0
    labels = [wm[v] for v in cols]
    assert simulate_check(path, 3, seed=5, bits=mat, col_labels=labels) == 0


def test_vacuous_pin_reads_p_accept_one(tmp_path):
    # constant output: every pattern is accepted
    p = tmp_path / "const.bench"
    p.write_text("INPUT(a)\nOUTPUT(y)\nn1 = AND(a)\nn2 = NOT(a)\nn3 = OR(n1, n2)\ny = BUFF(n3)\n")
    r = sim_baseline(str(p), 1, seed=5, n_min=1 << 16, budget_s=0.1, hard_cap_s=1.0)
    assert r["p_accept"] == 1.0

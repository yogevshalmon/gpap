"""Tests for the compact sample-output path (gpap.sampling.sampler.sample_bits + output writers).

The load-bearing claim: the GPU-side bit projection is SOUND — every emitted row is a valid
support projection (extends to a full model), forced vars are pinned, don't-cares vary, and
ELS-equivalent columns stay equal. Sampling needs CUDA (packed-kernel-only engine); the
writer round-trips are host-only.
"""
import numpy as np
import pytest
import torch
from pysat.formula import CNF
from pysat.solvers import Solver

from gpap.sampling.sampler import GPUSampler
from gpap.sampling.output import write_samples

CUDA = torch.cuda.is_available()
try:
    from gpap.engine.kernels import kernel_available
    KERNEL = CUDA and kernel_available()
except Exception:  # noqa: BLE001
    KERNEL = False
needs_gpu = pytest.mark.skipif(not KERNEL, reason="engine is CUDA+triton-only")
DEV = torch.device("cuda") if CUDA else torch.device("cpu")


def _enum_projs(clauses, support):
    s = Solver(name="cadical195", bootstrap_with=CNF(from_clauses=clauses))
    out = set()
    while s.solve():
        m = set(s.get_model())
        out.add(tuple((v + 1) if (v + 1) in m else -(v + 1) for v in support))
        s.add_clause([-((v + 1) if (v + 1) in m else -(v + 1)) for v in support])
    s.delete()
    return out


def _bits_to_projs(M, cols, support):
    ci = {v: i for i, v in enumerate(cols)}
    return {tuple((v + 1) if row[ci[v + 1]] else -(v + 1) for v in support) for row in M}


# --------------------------------------------------------------------------- projection soundness

@needs_gpu
def test_bits_plain_projection_matches_exact():
    clauses, sup = [[1, 2], [-1, 3], [-2, -3]], [0, 1, 2]
    exact = _enum_projs(clauses, sup)
    smp = GPUSampler.from_clauses(3, clauses, DEV, k_cap=256, seed=1, support=sup)
    M, cols = smp.sample_bits(4000, config=(True, 1, False), max_chunks=60)
    got = _bits_to_projs(M, cols, sup)
    assert set(np.unique(M).tolist()) <= {0, 1}
    assert got <= exact and got == exact                 # sound AND complete


@needs_gpu
def test_bits_simplify_forced_and_dontcare():
    import random
    rng = random.Random(3)
    clauses = [[1]]                                       # force x1
    sup = list(range(1, 7))                               # support 1..6 (0-based 0..5)
    core = sup[1:-1]                                      # last support var in no clause -> don't-care
    for _ in range(4):
        vs = rng.sample(core, 3)
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    sup0 = [v - 1 for v in sup]
    exact = _enum_projs(clauses, sup0)
    smp = GPUSampler.from_clauses(6, clauses, DEV, k_cap=512, seed=2, support=sup0,
                                  simplify=True)
    M, cols = smp.sample_bits(60_000, config=(True, 1, False), max_chunks=120)
    got = _bits_to_projs(M, cols, sup0)
    assert got == exact
    ci = {v: i for i, v in enumerate(cols)}
    assert (M[:, ci[1]] == 1).all()                       # x1 forced TRUE in every sample
    assert set(np.unique(M[:, ci[6]]).tolist()) == {0, 1}  # don't-care var takes both polarities


@needs_gpu
def test_bits_els_equivalent_columns_equal():
    import random
    rng = random.Random(5)
    S, var, clauses = 6, 7, []
    for _ in range(4):
        vs = rng.sample(range(1, S + 1), 3)
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    clauses += [[-2, 3], [-3, 2], [-4, 5], [-5, 4]]       # s2==s3, s4==s5
    g = var; var += 1
    clauses += [[-g, 1], [-1, g], [g, 6]]
    nv, sup0 = var - 1, list(range(S))
    exact = _enum_projs(clauses, sup0)
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=512, seed=2, support=sup0,
                                  els=True)
    M, cols = smp.sample_bits(80_000, config=(True, 1, False), max_chunks=160)
    assert _bits_to_projs(M, cols, sup0) == exact
    ci = {v: i for i, v in enumerate(cols)}
    assert (M[:, ci[2]] == M[:, ci[3]]).all()             # equivalent vars stay EQUAL
    assert (M[:, ci[4]] == M[:, ci[5]]).all()


@needs_gpu
def test_bits_respects_cap():
    clauses, sup = [[1, 2], [-1, 3], [-2, -3]], [0, 1, 2]
    smp = GPUSampler.from_clauses(3, clauses, DEV, k_cap=256, seed=1, support=sup)
    M, _ = smp.sample_bits(100, config=(True, 1, False), max_chunks=60)
    assert M.shape[0] <= 100


# --------------------------------------------------------------------------- writers round-trip

def test_writers_roundtrip(tmp_path):
    M = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    cols = [5, 8, 12]
    # npy round-trips exactly
    p = tmp_path / "s.npy"
    write_samples(str(p), M, cols, fmt="npy")
    assert np.array_equal(np.load(str(p)), M)
    # bits: one 0/1 line per sample after the `c` header; wire labels appear in the legend
    p = tmp_path / "s.bits"
    write_samples(str(p), M, cols, fmt="bits", wire_map={5: "a@0", 8: "b@0", 12: "c@1"})
    lines = [ln for ln in open(str(p)).read().splitlines() if not ln.startswith("c")]
    assert lines == ["101", "011", "110"]
    assert "a@0" in open(str(p)).read()
    # dimacs: signed vars per column bit, 0-terminated
    p = tmp_path / "s.dimacs"
    write_samples(str(p), M, cols, fmt="dimacs")
    assert open(str(p)).readline().split() == ["5", "-8", "12", "0"]


# --------------------------------------------------------------------------- wire-label plumbing

def test_bench_wire_labels_roundtrip(tmp_path):
    from gpap.circuits.bench import convert_one, parse_support_labels
    cnf = str(tmp_path / "s386.cnf")
    convert_one("benchmarks/circuits/bench/seq/s386.bench", cnf, 2, 1, pin_outputs=None)
    wm = parse_support_labels(cnf)
    assert wm and all("@" in lbl for lbl in wm.values())  # every support var -> wire@cycle

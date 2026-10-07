"""Entry point 1's answer: the BCP fixpoint each kernel reaches, against the pure-Python reference.

A fixpoint is unique, so on a row that does not conflict the assigned-literal set must equal
`tests/reference/bcp.py` exactly, for both propagate kernels and both packed splits. A conflicted
row's partial state is unspecified; only its conflict flag is compared.
"""
import random

import numpy as np
import pytest

from gpap.engine import _core
from gpap.engine.bcp_engine import BCPEngine
from gpap.engine.formula import CnfFormula

from tests.reference.bcp import assigned_set, lit_idx, ref_bcp
from tests.support import DEV, decision_matrix, random_assumptions, random_cnf

pytestmark = pytest.mark.gpu


def _mixed_cnf(rng, nv):
    """Every clause kind the engine stores differently: units, binaries and long clauses."""
    widths = [1, 2, 2, 2, 3, 3, 3, 4, 4]
    return [[v if rng.random() < 0.5 else -v
             for v in rng.sample(range(1, nv + 1), min(rng.choice(widths), nv))]
            for _ in range(rng.randint(nv, 3 * nv))]


@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("shift", [16, 8])
def test_fixpoint_matches_the_reference(mode, shift):
    """Clause-length segregation decides which literal a clause forces (binaries from the occ
    payload, longs from the packed sum field), so an error there shows in the assignment first
    and in the conflict flag only sometimes. Both are compared."""
    rng = random.Random(2026)
    compared = 0
    for _ in range(20):
        nv = rng.randint(4, 26)
        clauses = _mixed_cnf(rng, nv)
        f = _core.Formula.from_clauses(nv, clauses)
        if f.info.has_empty_clause:
            continue
        eng = _core.Engine(f, mode=mode, force_shift=shift)
        if eng.flush_conflict:
            continue
        K = 24
        sets = random_assumptions(rng, nv, K, wmax=4)
        eng.reset(K)
        eng.apply(decision_matrix(sets))
        active, conflict = eng.flags()
        np.testing.assert_array_equal(active, ~conflict)      # a row is active iff not conflicted
        asg = eng.rows_assignment(np.arange(K, dtype=np.int32))
        for i, s in enumerate(sets):
            want_conflict, want_asg = ref_bcp(clauses, [lit_idx(x) for x in s])
            assert bool(conflict[i]) == want_conflict, (mode, shift, nv, s)
            if not want_conflict:
                assert assigned_set(asg[i]) == want_asg, (mode, shift, nv, s)
                compared += 1
    assert compared > 100, f"only {compared} surviving rows compared"


def _free_decisions(rng, engine, K, B, contradict_rows=()):
    """(K, B) literal indices over distinct free variables per row, as the sampler decides.
    Rows in `contradict_rows` get an l/~l pair instead, to take the conflict path."""
    asg = engine.assignment.cpu()
    dec = np.full((K, B), -1, dtype=np.int64)
    for k in range(K):
        free = [v for v in range(asg.size(1) // 2) if asg[k, 2 * v] == 0 and asg[k, 2 * v + 1] == 0]
        if not free:
            continue
        if k in contradict_rows and B >= 2:
            v = rng.choice(free)
            dec[k, 0], dec[k, 1] = 2 * v, 2 * v + 1
            continue
        for j, v in enumerate(rng.sample(free, min(B, len(free)))):
            dec[k, j] = 2 * v + rng.randint(0, 1)
    return dec


def _python_engine(nv, clauses, mode=None):
    f = CnfFormula(DEV)
    f.init_formula(nv, clauses)
    return BCPEngine(f, mode=mode)


@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_decisions_applied_in_steps_reach_the_reference_fixpoint(mode):
    """Three rounds of decisions on top of each other, one row given a contradiction. The
    reference propagates the cumulative decisions from scratch, which reaches the same fixpoint."""
    import torch
    rng = random.Random(11)
    for _ in range(15):
        nv = rng.randint(4, 40)
        clauses = random_cnf(rng, nv, rng.randint(nv, 4 * nv))
        K, B = 32, rng.choice([1, 2, 4])
        eng = _python_engine(nv, clauses, mode)
        eng.reset(K)
        cum = [[] for _ in range(K)]
        for step in range(3):
            contradict = {rng.randrange(K)} if (step == 1 and B >= 2) else ()
            dec = _free_decisions(rng, eng, K, B, contradict_rows=contradict)
            for k in range(K):
                cum[k].extend(int(x) for x in dec[k] if x >= 0)
            eng.apply_and_propagate(torch.from_numpy(dec).to(DEV))
            for k in range(K):
                conf, asg = ref_bcp(clauses, cum[k])
                assert bool(eng.conflict[k].item()) == conf
                if not conf:
                    assert assigned_set(eng.assignment[k].cpu()) == asg


def test_fixpoint_matches_the_reference_on_a_circuit(circuit_cnfs):
    """An unrolled circuit: deep implication cascades rather than random clauses."""
    import torch
    inst = circuit_cnfs["s386-c5"]
    nv, clauses = inst["nv"], [list(c) for c in inst["clauses"]]
    rng = random.Random(3)
    K = 24
    eng = _python_engine(nv, clauses)
    eng.reset(K)
    dec = _free_decisions(rng, eng, K, 8)
    eng.apply_and_propagate(torch.from_numpy(dec).to(DEV))
    for k in range(K):
        conf, asg = ref_bcp(clauses, [int(x) for x in dec[k] if x >= 0])
        assert bool(eng.conflict[k].item()) == conf
        if not conf:
            assert assigned_set(eng.assignment[k].cpu()) == asg


def test_scalar_and_wave_reach_the_same_fixpoint():
    """The two kernels differ only in propagation order, so they must agree exactly. This is
    what licenses dispatching between them at run time."""
    rng = random.Random(19)
    nv, nc, K = 20, 55, 12
    clauses = random_cnf(rng, nv, nc, kmin=2)
    dec = np.array([[rng.randrange(2 * nv)] for _ in range(K)], dtype=np.int64)
    rows = np.arange(K, dtype=np.int32)
    out = {}
    for m in ("scalar", "wave"):
        core = _core.Engine(_core.Formula.from_clauses(nv, clauses), mode=m)
        core.reset(K)
        core.apply(dec)
        out[m] = (core.flags()[1], core.rows_assignment(rows), core.num_propagated())
    for a, b in zip(out["scalar"], out["wave"]):
        np.testing.assert_array_equal(a, b)


def test_the_wide_split_engages_past_the_16_bit_envelope():
    """2N > 65535 must select the 8|24 split by itself, and literal indices far above 65535 must
    still come back exact from the 24-bit sum field. An implication ladder x_i -> x_{i+1} over
    40k variables makes every decision propagate a long prefix or suffix."""
    nv = 40_000
    clauses = [[-i, i + 1] for i in range(1, nv)]
    eng = _core.Engine(_core.Formula.from_clauses(nv, clauses))
    assert eng.shift == 8
    mid = 20_000
    dec = np.array([[lit_idx(mid)], [lit_idx(-mid)]], dtype=np.int64)
    eng.reset(2)
    eng.apply(dec)
    _, conflict = eng.flags()
    asg = eng.rows_assignment(np.arange(2, dtype=np.int32))
    for k in range(2):
        conf, want = ref_bcp(clauses, [int(dec[k, 0])])
        assert bool(conflict[k]) == conf
        if not conf:
            assert assigned_set(asg[k]) == want

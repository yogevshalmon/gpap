"""The C++ core's own contract: formula build, both entry points, readout, degenerate shapes.

Two oracles keep this honest, and NEITHER is a second GPU engine:

  * the formula build is checked against `tests/reference_formula.py` — a pure-numpy
    re-derivation of the structures, which runs on any host;
  * propagation is checked against `tests/reference_bcp.py` — a pure-Python BCP.

Anything the core emits is additionally verifiable from the outside: a model must satisfy every
clause, and these tests check that directly rather than trusting either implementation.

(This file once also A/B'd the core against a retired PyTorch/Triton-JIT engine —
the port gate. That engine is gone; the checks that survived it are the ones below.)
"""
import random

import numpy as np
import pytest
import torch

from gpap.engine import _core

from tests.reference_bcp import assigned_set, ref_bcp
from tests.reference_formula import ReferenceFormula

CUDA = torch.cuda.is_available()
HAVE_CORE = _core.available()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not CUDA, reason="requires CUDA"),
    pytest.mark.skipif(not HAVE_CORE, reason="libgpap not built (see cpp/README.md)"),
]
DEV = torch.device("cuda") if CUDA else torch.device("cpu")


def _random_cnf(rng, nv, nc, kmax=4, kmin=1):
    """`kmin` is load-bearing, not decoration. At kmin=1 a formula of this size almost always
    contains a contradictory pair of UNIT clauses, so it is UNSAT before a single decision is
    made: every row is conflicted at construction and the trajectory kernel does nothing. Two
    tests in the retired equivalence suite were passing that way — comparing zero models to zero
    models. Propagation tests use kmin=2 (real implication chains, no trivial UNSAT); tests that
    need SOLUTIONS to exist use kmin=3 at a low clause ratio."""
    out = []
    for _ in range(nc):
        vs = rng.sample(range(1, nv + 1), rng.randint(min(kmin, nv), min(kmax, nv)))
        out.append([v if rng.random() < 0.5 else -v for v in vs])
    return out


# ---------------------------------------------------------------- formula structures

def test_formula_structures_match_the_reference_builder():
    rng = random.Random(7)
    for nv, nc in [(6, 10), (40, 120), (200, 900)]:
        clauses = _random_cnf(rng, nv, nc)
        f = ReferenceFormula.from_clauses(nv, clauses)
        cf = _core.Formula.from_clauses(nv, clauses)

        assert cf.info.num_vars == f.num_vars
        assert cf.info.num_clauses == f.num_clauses
        off, cls_ = cf.occ()
        np.testing.assert_array_equal(off, f.occ_offsets)
        np.testing.assert_array_equal(cls_, f.occ_clauses)
        sizes, lsum = cf.clause_arrays()
        np.testing.assert_array_equal(sizes, f.clause_sizes)
        np.testing.assert_array_equal(lsum, f.init_lit_sum)


def test_dimacs_parse_matches_the_reference_parser(tmp_path):
    rng = random.Random(11)
    clauses = _random_cnf(rng, 30, 80)
    p = tmp_path / "f.cnf"
    with open(p, "w") as fh:
        fh.write("c a comment\np cnf 30 %d\n" % len(clauses))
        for c in clauses:                       # two clauses per line: the token stream matters
            fh.write(" ".join(map(str, c)) + " 0 ")
    f = ReferenceFormula.from_file(str(p))
    cf = _core.Formula.from_file(str(p))
    assert (cf.info.num_vars, cf.info.num_clauses) == (f.num_vars, f.num_clauses)
    np.testing.assert_array_equal(cf.clause_arrays()[0], f.clause_sizes)


def test_k_calibration_matches():
    """The sizing model has ONE implementation (cpp/src/calibrate.cpp); Python binds it. This
    asserts the binding actually reaches it rather than reintroducing a second copy — which is
    what this test used to be gating, back when the arithmetic was written out twice."""
    from gpap.engine.calibrate import compute_k_safe as py_k
    for M, N in [(4407, 1701), (109432, 42051), (201437, 77151)]:
        for budget in (1 << 30, 6 << 30):
            for cap in (0, 16384):
                assert _core.compute_k_safe(M, N, budget, cap or None, 8) == \
                    py_k(M, N, budget, cap=cap or None, decisions_per_round=8)


def test_envelope_matches():
    """Same two-copies story as the K model: the packed splits are declared in
    cpp/src/runtime.cpp and mirrored in Python so a suite driver can pre-check without CUDA."""
    from gpap.engine.bcp_engine import packed_envelope
    for nv, ml in [(10, 3), (32767, 5), (33000, 5), (40000, 300), (9_000_000, 3)]:
        assert _core.packed_envelope(nv, ml)[0] == packed_envelope(nv, ml)[0]


# ---------------------------------------------------------------- entry point 1

@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_apply_matches_the_reference_bcp(mode):
    """Conflict flags AND the fixpoint itself, row by row, against the pure-Python oracle."""
    rng = random.Random(3)
    for trial in range(8):
        nv, nc = 25, 70
        clauses = _random_cnf(rng, nv, nc, kmin=2)
        K, L = 16, 3
        dec = [[rng.randrange(2 * nv) for _ in range(L)] for _ in range(K)]

        core = _core.Engine(_core.Formula.from_clauses(nv, clauses), mode=mode)
        core.reset(K)
        core.apply(np.array(dec, dtype=np.int64))

        act, conf = core.flags()
        np.testing.assert_array_equal(act, ~conf)          # a row is active iff not conflicted
        asg = core.rows_assignment(np.arange(K, dtype=np.int32))
        for k in range(K):
            ref_conflict, ref_asg = ref_bcp(clauses, dec[k])
            assert bool(conf[k]) == ref_conflict, f"row {k} trial {trial}"
            if not ref_conflict:
                assert assigned_set(asg[k]) == ref_asg


def test_scalar_and_wave_reach_the_same_fixpoint():
    """BCP fixpoints are confluent, so the two propagate variants must agree EXACTLY — they
    differ only in propagation order. This is what licenses the runtime auto-dispatch."""
    rng = random.Random(19)
    nv, nc, K = 20, 55, 12
    clauses = _random_cnf(rng, nv, nc, kmin=2)
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


def test_flush_cache_and_units():
    """The formula's own unit clauses are asserted ONCE at construction and broadcast by reset."""
    clauses = [[1], [-1, 2], [-2, 3], [4, 5]]
    core = _core.Engine(_core.Formula.from_clauses(5, clauses))
    assert not core.flush_conflict
    # the unit x1 implies x2 then x3; x4/x5 stay free. (lit index of +v is 2(v-1))
    assert assigned_set(core.flushed_assignment()) == {0, 2, 4}


def test_unsat_units_conflict_every_row():
    core = _core.Engine(_core.Formula.from_clauses(2, [[1], [-1]]))
    assert core.flush_conflict
    core.reset(4)
    assert core.flags()[1].all()


# ---------------------------------------------------------------- entry point 2

@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("B", [1, 8])
def test_trajectory_emits_only_real_models(mode, B):
    """THE soundness gate: every row the selection reports must satisfy every clause, checked
    outside the engine. Also pins the on-device check (`check_sound`) against that verdict."""
    rng = random.Random(5)
    nv, nc, K, seed = 60, 150, 512, 1234567
    clauses = _random_cnf(rng, nv, nc, kmin=3)
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses), mode=mode)

    order = torch.randperm(nv, generator=torch.Generator(device=DEV).manual_seed(9),
                           device=DEV).to(torch.int32).contiguous()
    core.reset(K)
    core.trajectory_dev(order.data_ptr(), nv, nv, B, seed)
    n = core.select_complete(core.pair_mask(None), 0)
    assert n > 0, "this formula should yield models at both B values"
    assert core.check_sound(0) == 0

    bits = core.var_bits(0, np.arange(nv, dtype=np.int32), n)
    for row in bits[: min(n, 64)]:
        for c in clauses:
            assert any((row[abs(lit) - 1] == 1) == (lit > 0) for lit in c)
    # the selection is exactly the rows that are active and fully assigned
    rows = core.selection_rows(0, n)
    assert np.all(np.diff(rows) > 0)                       # ascending, no duplicates
    assert core.flags()[0][rows].all()


@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_trajectory_counters_agree_with_the_assignment(mode):
    """decs + implications == literals assigned; the counters are what every reported
    propagation rate is derived from, so they must not be free-floating."""
    rng = random.Random(23)
    nv, nc, K = 40, 110, 256
    clauses = _random_cnf(rng, nv, nc, kmin=3)
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses), mode=mode)
    order = torch.arange(nv, device=DEV, dtype=torch.int32).contiguous()
    core.reset(K)
    core.trajectory_dev(order.data_ptr(), nv, nv, 4, 999)
    c = core.counters()
    assert c.launches == 1
    assert c.props > c.decs > 0                            # some literals came from propagation
    assert c.ops > 0
    # implications, per row, are props - decs; the ABI already subtracts
    assert int(core.num_propagated().sum()) == c.props - c.decs


@pytest.mark.slow
@pytest.mark.parametrize("name", ["b03-c25", "b06-c10", "s386-c10"])
def test_circuit_models_are_sound(circuit_cnfs, name):
    """The same gate at production scale: an unrolled circuit CNF, support-first decision order,
    production-shaped K. Every emitted row must satisfy every clause of the real instance."""
    if name not in circuit_cnfs:
        pytest.skip(f"{name} not in the generated matrix")
    inst = circuit_cnfs[name]
    nv, clauses, support = inst["nv"], inst["clauses"], inst["support"]
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses))

    gen = torch.Generator(device=DEV).manual_seed(5)
    rank = torch.argsort(torch.randperm(nv, generator=gen, device=DEV))
    smask = torch.zeros(nv, dtype=torch.bool, device=DEV)
    smask[torch.tensor(support, dtype=torch.long, device=DEV)] = True
    in_sup = smask[rank]
    order = torch.cat([rank[in_sup], rank[~in_sup]]).to(torch.int32).contiguous()

    core.reset(2048)
    core.trajectory_dev(order.data_ptr(), order.numel(), int(in_sup.sum().item()), 8, 20260728)
    n = core.select_complete(core.pair_mask(None), 0)
    assert core.check_sound(0) == 0
    if n:
        bits = core.var_bits(0, np.arange(nv, dtype=np.int32), n)
        for row in bits[: min(n, 8)]:
            for c in clauses:
                assert any((row[abs(lit) - 1] == 1) == (lit > 0) for lit in c)


# ---------------------------------------------------------------- degenerate shapes
# Each of these was a real bug found while porting: the flush cache used a null packed pointer
# as its "not built yet" sentinel (which M == 0 also produces), the core rejected num_vars == 0,
# and it rejected an empty decision order. All three are reachable through simplification.

def test_m0_formula_resets_every_row():
    """A formula with no clauses: reset must still restore ALL K rows, not just row 0."""
    core = _core.Engine(_core.Formula.from_clauses(4, []))
    core.reset(2)
    core.apply(np.array([[0], [3]], dtype=np.int64))       # lit indices: +x1 and -x2
    act, conf = core.flags()
    assert act.all() and not conf.any()
    asg = core.rows_assignment(np.array([0, 1], dtype=np.int32))
    assert assigned_set(asg[0]) == {0} and assigned_set(asg[1]) == {3}


def test_zero_variable_formula():
    """Unit propagation / ELS can determine EVERY variable, leaving an empty formula."""
    cf = _core.Formula.from_clauses(0, [])
    assert cf.info.num_vars == 0 and cf.info.num_clauses == 0
    core = _core.Engine(cf)
    core.reset(8)
    assert core.trajectory(np.zeros(0, dtype=np.int32), 0, 8, 42) == 0   # empty order = no-op
    # nothing is free, so every active row is a (vacuous) complete model
    assert core.select_complete(core.pair_mask(None), 0) == 8


def test_zero_column_readout_is_a_noop():
    """Gathering ZERO variables is legal: the sampler's collect path asks for all N variables,
    and after ELS/unit-propagation N can be 0."""
    core = _core.Engine(_core.Formula.from_clauses(0, []))
    core.reset(4)
    n = core.select_complete(core.pair_mask(None), 0)
    assert core.var_bits(0, np.zeros(0, dtype=np.int32), n).shape == (n, 0)
    assert core.assignment_rows(0, n).shape == (n, 0)


def test_readout_never_writes_past_the_caller_buffer():
    """The three gathers take an explicit row CAPACITY and write min(capacity, selection).

    Without it their output size was only *implied* by the selection count, so a caller whose
    count was stale by one chunk silently overflowed the heap. Asking for fewer rows than are
    selected must truncate — and truncate to the FIRST rows, not to arbitrary ones."""
    rng = random.Random(77)
    nv = 12
    clauses = _random_cnf(rng, nv, 18, kmin=3)
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses))
    core.reset(64)
    core.trajectory(np.arange(nv, dtype=np.int32), nv, 4, 2024)
    n = core.select_complete(core.pair_mask(None), 0)
    assert n > 2, "need several complete rows to have something to truncate"

    cols = np.arange(nv, dtype=np.int32)
    full_bits = core.var_bits(0, cols, n)
    full_asg = core.assignment_rows(0, n)
    full_rows = core.selection_rows(0, n)
    half = n // 2
    np.testing.assert_array_equal(core.var_bits(0, cols, half), full_bits[:half])
    np.testing.assert_array_equal(core.assignment_rows(0, half), full_asg[:half])
    np.testing.assert_array_equal(core.selection_rows(0, half), full_rows[:half])
    # Asking for MORE than exists is also safe: the extra rows are simply not written.
    assert core.var_bits(0, cols, n + 5).shape == (n + 5, nv)


def test_empty_decision_order_is_a_noop():
    rng = random.Random(41)
    clauses = _random_cnf(rng, 8, 12)
    core = _core.Engine(_core.Formula.from_clauses(8, clauses))
    core.reset(4)
    before = core.flags()[0].copy()
    assert core.trajectory(np.zeros(0, dtype=np.int32), 0, 4, 7) == 0
    np.testing.assert_array_equal(core.flags()[0], before)


def test_out_of_envelope_formula_builds_but_engine_rejects():
    """The packed split is the ENGINE's choice: shape queries still work on a formula that no
    split fits, and suite drivers catch a ValueError from the engine to skip the instance."""
    cf = _core.Formula.from_clauses(40_000, [list(range(1, 300))])   # 2N>65535 AND len>255
    assert cf.info.shift == 0 and cf.info.num_vars == 40_000
    with pytest.raises(ValueError):
        _core.Engine(cf)


def test_wide_split_is_selected_and_sound():
    """8|24 'wide' split: 2N > 65535 forces the second envelope, and it must still be sound."""
    rng = random.Random(31)
    nv = 40000
    clauses = _random_cnf(rng, nv, 60000, kmax=4, kmin=3)
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses))
    assert core.shift == 8
    order = torch.randperm(nv, generator=torch.Generator(device=DEV).manual_seed(2),
                           device=DEV).to(torch.int32).contiguous()
    core.reset(64)
    core.trajectory_dev(order.data_ptr(), nv, nv, 8, 4242)
    core.select_complete(core.pair_mask(None), 0)
    assert core.check_sound(0) == 0

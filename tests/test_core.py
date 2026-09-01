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
import ctypes
import random

import numpy as np
import pytest
import torch

from gpap.engine import _core
from gpap.engine._core import check

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
        assert cf.info.num_bin == f.num_bin
        assert cf.info.num_long == f.num_long
        off, mid, cls_ = cf.occ()
        np.testing.assert_array_equal(off, f.occ_offsets)
        np.testing.assert_array_equal(mid, f.occ_mid)
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
                assert _core.compute_k_safe(M, N, budget, cap or None) == \
                    py_k(M, N, budget, cap=cap or None)


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
def test_check_sound_agrees_with_an_independent_recount(mode):
    """The soundness check has TEETH, gated against a SECOND computation of the same fact.

    `check_sound() == 0` is evidence only if the kernel can say otherwise, and since the check is
    opt-in now (`verify_sound`) that evidence is the only reason to pay for it. So instead of
    trusting a 0, recompute the verdict on the host from `sat_rows` — which reports every clause
    kind in the ORIGINAL order, long words and bitmap-synthesized binaries alike — over a mixed
    bag of rows (some conflicted, some not) and require the two to agree row for row.

    Note what does NOT prove this: pointing the check at conflicted rows and expecting them all
    to be unsound. A row can conflict by a CONTRADICTORY PUSH (some literal forced to both
    polarities) having emptied nothing, and on the scalar kernel that is how nearly every
    conflict surfaces — a falsified long clause is normally caught one decrement earlier, when
    it goes unit and forces a literal whose negation is already set."""
    rng = random.Random(11)
    nv, nc, K = 40, 90, 256
    clauses = _random_cnf(rng, nv, nc, kmin=2)          # kmin=2 -> exercise the binary pair path
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses), mode=mode)
    core.reset(K)
    # Wide random assumptions: plenty of rows conflict, a few survive.
    dec = np.array([[2 * v + rng.randint(0, 1) for v in rng.sample(range(nv), 12)]
                    for _ in range(K)], dtype=np.int64)
    core.apply(dec)

    rows = np.arange(K, dtype=np.int32)
    sat = core.sat_rows(rows)                            # (K, M) non-falsified count per clause
    want = int((sat.min(axis=1) == 0).sum())             # rows with SOME emptied clause
    core.selection_set(core.SCRATCH_SLOT, rows)
    assert core.check_sound(core.SCRATCH_SLOT) == want
    assert want > 0, "this fixture must actually contain unsound rows, or it proves nothing"


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


@pytest.mark.parametrize("entry", ["trajectory", "apply"])
def test_a_zero_width_launch_does_not_report_the_previous_one(entry):
    """The per-launch counters describe THE LAST CALL, and reset() does not clear them — so the
    degenerate launch (an empty decision order, a zero-width apply) has to clear them itself or
    it publishes the PREVIOUS launch's propagation as its own. Both entry points grew the same
    early return for the same degenerate formula (one whose variables unit propagation already
    determined) and only apply zeroed; the trajectory reported stale counters until 2026-09-01.

    It fails against the unfixed library, not merely passes against this one: the first launch
    is asserted to have produced non-zero work, so `all zero` afterwards is a real transition."""
    nv, K = 12, 8
    clauses = [[1, 2, 3], [-1, 4], [-2, -4], [3, -4, 5], [-5, 6], [-6, 7, 8]]
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses))
    order = np.arange(nv, dtype=np.int32)

    core.reset(K)
    core.trajectory(order, nv, 1, 12345)
    worked = core.num_propagated()
    assert worked.any(), "the priming launch must do work, or the check below is vacuous"

    core.reset(K)
    if entry == "trajectory":
        rounds = core.trajectory(np.empty(0, dtype=np.int32), 0, 1, 999)
        assert rounds == 0
    else:
        core.apply(np.empty((K, 0), dtype=np.int64))
    assert not core.num_propagated().any(), \
        f"a zero-width {entry} reported the previous launch's implications"


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


# ---------------------------------------------------------------- ABI preconditions

# NOTE — there is deliberately NO test here that feeds gpap_engine_apply an out-of-range literal
# and watches it corrupt a later row. Demonstrating that hazard REQUIRES an index past 2N (any
# smaller one stays inside its own row's bitmap words), and the propagate loop dereferences the
# popped literal as `occ_off[lit]`, `occ_mid[lit]`, `occ_off[lit+1]` — arrays of exactly 2N and
# 2N+1 entries. So the demonstration is an unbounded out-of-bounds DEVICE read whose garbage
# bounds then drive a store; it passes here only because the driver hands back zeroed padding,
# and a CUDA_ERROR_ILLEGAL_ADDRESS is sticky for the context, i.e. it would cascade-fail every
# later GPU test in the process rather than failing alone.
#
# The guarantee that actually matters is that the PACKAGED faces validate before any launch, and
# that is gated twice, safely: tests/test_assumption_checker.py (test_out_of_range_literal_raises
# over five bad literals, plus test_bad_literal_cannot_corrupt_another_row) and
# tests/test_mining.py::test_rejects_out_of_range_literals. The raw ABI's precondition is stated
# in gpap.h.


def test_n_support_past_the_order_length_is_refused():
    """n_support is the end of decision phase 1 and the kernel walks `order` up to it, so
    n_support > P was an out-of-bounds read of the caller's buffer that still returned OK."""
    nv = 8
    core = _core.Engine(_core.Formula.from_clauses(nv, [[1, 2, 3], [-1, 4], [5, 6, 7], [-5, 8]]))
    core.reset(4)
    order = np.arange(nv, dtype=np.int32)
    with pytest.raises(_core.GpapError):
        core.trajectory(order, n_support=nv * 10, B=4, seed=7)
    with pytest.raises(_core.GpapError):
        core.trajectory(order, n_support=-1, B=4, seed=7)
    core.trajectory(order, n_support=nv, B=4, seed=7)          # the boundary value is legal


def test_an_order_entry_past_num_vars_is_refused_on_the_host_face():
    """Same cross-row write as the apply hazard, via `ab_row + (v >> 4)` — and here the check is
    free, because gpap_engine_trajectory is already walking the array to stage it."""
    nv = 8
    core = _core.Engine(_core.Formula.from_clauses(nv, [[1, 2, 3], [-1, 4], [5, 6, 7], [-5, 8]]))
    core.reset(4)
    bad = np.arange(nv, dtype=np.int32)
    bad[0] = 10_000
    with pytest.raises(_core.GpapError):
        core.trajectory(bad, n_support=nv, B=4, seed=7)


@pytest.mark.slow          # ~1.2 GB of VRAM: the slice loop only engages past 256 MB
def test_readout_slicing_matches_an_unsliced_read():
    """`var_bits` / `assignment_rows` SLICE the readout so the (rows, cols) host transient stays
    under `kReadoutSliceBytes` (256 MB, cpp/src/engine.cpp). The loop only engages when
    rows > 256MB/cols — a wide formula with many rows — which no other test reaches, so the
    row-pointer offsetting and the per-slice copy-back were unexercised.

    A wide, nearly-free formula gets there cheaply: 2N past the 16|16 envelope forces the 8|24
    split as a bonus."""
    nv = 100_000                                  # 2N = 200,000 -> wide split, small slice step
    clauses = [[v, -(v + 1)] for v in range(1, 400)]
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses))
    step = max(32, (256 << 20) // nv) // 32 * 32
    # ONE slice plus a short tail is all the boundary arithmetic needs, and the buffers here are
    # (K x nv) and (K x 2N) BYTES on the host: at 3 slices this test allocated 2.4 GB, which is
    # not a cost a unit test should impose (and it is what stalled the suite under ASan).
    K = step + 96
    core.reset(K)
    # A TRAJECTORY first, so the rows actually DIFFER. Straight after reset every row holds the
    # same broadcast baseline, and then a slicing loop that dropped its row-pointer offset would
    # compare equal to itself and this test would prove nothing.
    core.trajectory(np.arange(nv, dtype=np.int32), nv, 64, 20260831)
    rows = np.arange(K, dtype=np.int32)
    core.selection_set(core.SCRATCH_SLOT, rows)
    cols = np.arange(nv, dtype=np.int32)

    sliced = core.var_bits(core.SCRATCH_SLOT, cols, K)
    assert sliced.shape == (K, nv)
    assert K > step, f"precondition: {K} rows must exceed the {step}-row slice"
    # NON-VACUITY: rows on either side of a slice boundary must be distinguishable, or a loop
    # that ignored the offset would pass by comparing identical rows.
    assert not np.array_equal(sliced[0], sliced[step]), "rows must differ across the boundary"

    piece = max(1, step // 4)                     # each request below the slice threshold
    for r0 in range(0, K, piece):
        core.selection_set(1, rows[r0:r0 + piece])
        ref = core.var_bits(1, cols, min(piece, K - r0))
        assert np.array_equal(ref, sliced[r0:r0 + len(ref)]), f"slice boundary at row {r0}"

    # assignment_rows runs the same loop at 2N columns, so its step is half — K already crosses
    # it twice. Read a bounded prefix: the point is the boundary, not the volume.
    step2 = max(32, (256 << 20) // (2 * nv)) // 32 * 32
    rows2 = min(K, step2 + 96)
    assert rows2 > step2, "precondition: assignment_rows must cross a slice too"
    asg = core.assignment_rows(core.SCRATCH_SLOT, rows2)
    assert asg.shape == (rows2, 2 * nv)
    assert np.array_equal(sliced[:rows2], asg[:, 0::2]), \
        "var_bits must equal the even literal columns, across both slice boundaries"


@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("shift", [16, 8])
def test_a_row_never_writes_outside_itself(mode, shift):
    """Device-side spill detector. `compute-sanitizer` ships with the CUDA toolkit, which this
    project deliberately does not require, so there is no device memcheck to lean on — and every
    kernel does raw per-row pointer arithmetic (`st_row + c`, `ab_row + (lit >> 5)`,
    `trail_row + tail`).

    A row whose decisions are ALL padding does no work: the seed loop assigns nothing, the trail
    stays empty, and neither its packed words nor its bitmap are touched. Interleaving such
    CANARY rows among rows doing deep propagation turns any cross-row write into a visible
    change. The detector is known to fire: aiming an out-of-range literal index at a canary's
    bitmap words trips exactly that canary (see
    test_an_out_of_range_apply_literal_corrupts_a_DIFFERENT_row, which is the same mechanism)."""
    from tests.reference_bcp import lit_idx

    nv = 300
    rng = random.Random(31337)
    clauses = ([[-v, v + 1] for v in range(1, nv)]                       # deep implication chains
               + [[-v, -(v + 1), -(v + 2), nv] for v in range(1, nv - 2)])
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses), mode=mode, force_shift=shift)
    assert not core.flush_conflict

    K = 257                                     # odd, so canaries land at every alignment
    core.reset(K)
    rows = np.arange(K, dtype=np.int32)
    before_packed = core.packed_rows(rows).copy()
    before_bits = core.rows_assignment(rows).copy()

    canaries = [i for i in range(K) if i % 3 == 1]
    L = 6
    dec = np.full((K, L), -1, dtype=np.int64)
    for i in range(K):
        if i in canaries:
            continue                            # all padding: this row must not move
        for j, v in enumerate(rng.sample(range(1, nv + 1), L)):
            dec[i, j] = lit_idx(v if rng.random() < 0.5 else -v)
    core.apply(dec)

    after_packed = core.packed_rows(rows)
    after_bits = core.rows_assignment(rows)
    for i in canaries:
        assert np.array_equal(before_packed[i], after_packed[i]), f"packed row {i} was written"
        assert np.array_equal(before_bits[i], after_bits[i]), f"bitmap row {i} was written"
    worked = sum(1 for i in range(K)
                 if i not in canaries and not np.array_equal(before_bits[i], after_bits[i]))
    assert worked > K // 3, f"only {worked} rows did any work — the fixture proves little"


@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_probe_returns_exactly_the_conflicting_tuples(mode):
    """The probe's SET, not just its counts. Existing coverage checks the candidate arithmetic
    (`gpap_probe_count` vs a binomial), the ceiling, and that width-1 agrees with the Python
    miner — but nothing compares the returned tuples against an independent decision procedure.
    Brute-force every k-subset through the reference BCP and require set equality, at both
    `same_var` settings and three widths.

    An `exhausted` sweep is the only one this can be asked of: it is what licenses "these are ALL
    of them", which is exactly the claim under test."""
    import itertools

    from tests.reference_bcp import lit_idx, ref_bcp

    rng = random.Random(4242)
    compared = conflicts_seen = 0
    for _ in range(6):
        nv = rng.randint(5, 12)
        clauses = []
        for _ in range(rng.randint(nv, 3 * nv)):
            w = min(rng.choice([1, 2, 2, 3, 3, 4]), nv)
            clauses.append([v if rng.random() < 0.5 else -v
                            for v in rng.sample(range(1, nv + 1), w)])
        f = _core.Formula.from_clauses(nv, clauses)
        if f.info.has_empty_clause:
            continue
        core = _core.Engine(f, mode=mode)
        if core.flush_conflict:
            continue
        pool = sorted(core.free_literals(), key=lambda x: (abs(x), x < 0))
        if len(pool) < 3:
            continue
        for k, same_var in ((1, False), (2, False), (2, True), (3, False)):
            sweep = core.probe_combinations(pool, k, distinct_vars=not same_var)
            if not sweep.stats.exhausted:
                continue
            got = {tuple(row) for row in sweep.conflicts()}
            want = set()
            for combo in itertools.combinations(pool, k):
                if not same_var and len({abs(x) for x in combo}) != k:
                    continue
                if ref_bcp(clauses, [lit_idx(x) for x in combo])[0]:
                    want.add(combo)
            assert got == want, (mode, k, same_var,
                                 sorted(got - want)[:3], sorted(want - got)[:3])
            assert sweep.stats.conflicts == len(want), "stats.conflicts must match the set"
            compared += 1
            conflicts_seen += len(want)
    assert compared >= 8 and conflicts_seen > 50, \
        f"fixture proves little: {compared} sweeps, {conflicts_seen} conflicts"


def test_a_slot_with_no_selection_yields_nothing_not_the_uninitialized_mask():
    """`gpap_engine_reset` sets a slot's count to 0 to say "nothing selected this chunk", and
    `ensure_rows` used to overwrite that by COMPACTING THE MASK — device memory that
    `DevBuf::reserve` never zeroes and only `gpap_select_complete` fills. A slot nobody had
    selected into therefore produced arbitrary row indices, and `gpap_check_sound` on it
    pronounced a verdict over rows the caller never chose."""
    nv = 40
    clauses = [[1, 2, 3], [-1, 4], [5, 6, 7], [-5, 8]] + [[i, -(i + 1)] for i in range(9, nv)]
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses))
    order = np.arange(nv, dtype=np.int32)
    mask = core.pair_mask(None)

    core.reset(64)
    core.trajectory(order, nv, 8, 7)
    n1 = core.select_complete(mask, 0)
    assert n1 > 0, "fixture must select something, or the reset below proves nothing"
    assert len(core.selection_rows(0, n1)) == n1

    core.reset(64)                       # new chunk: slot 0 has no selection yet
    buf = np.full(64, -7, dtype=np.int32)
    check(_core.load().gpap_selection_rows(core._h, 0, _core._p(buf, ctypes.c_int32), 64), "rows")
    assert np.all(buf == -7), f"the library wrote into an unselected slot: {buf[:8].tolist()}"
    unsound = ctypes.c_int32(-1)
    check(_core.load().gpap_check_sound(core._h, 0, 0, ctypes.byref(unsound)), "sound")
    assert unsound.value == 0            # over ZERO rows, not over compacted garbage
    assert core.select_complete(mask, 0) == 0     # and the real answer is 0 before a trajectory


@pytest.mark.parametrize("B", [1, 8, 64])
def test_the_warp_wide_decision_scan_picks_identically(B):
    """`bcp.py` claims the warp-wide decision scan makes "bit-identical picks" — the same first B
    still-free variables in `order`, the same per-(chunk, row, var) polarity hash — and that claim
    is the whole reason VSCAN is ON by default in the wave kernel. It was checked only for
    LAUNCHABILITY (tests/test_kernel_table.py runs every vscan value and asserts the cubin
    exists); nothing compared results.

    Compare the model SETS, not the counts: identical counts from different picks would pass."""
    rng = random.Random(808)
    compared = models_seen = 0
    for trial in range(4):
        nv = rng.randint(30, 120)
        # Clause/var ~0.7: deciding 64 variables in ONE round conflicts almost every
        # trajectory on a tightly constrained formula, and a cell with no models compares two
        # empty sets. The non-vacuity assertion at the end catches that, so keep the fixture
        # loose enough for every B to deliver.
        clauses = [[rng.choice([1, -1]) * v
                    for v in rng.sample(range(1, nv + 1), rng.choice([2, 3, 3, 4]))]
                   for _ in range(max(2, (7 * nv) // 10))]
        f = _core.Formula.from_clauses(nv, clauses)
        if f.info.has_empty_clause:
            continue
        order = np.random.default_rng(trial).permutation(nv).astype(np.int32)

        def run(mode, vscan, seed):
            core = _core.Engine(f, mode=mode, vscan=vscan)
            core.reset(512)
            core.trajectory(order, nv, B, seed)
            n = core.select_complete(core.pair_mask(None), 0)
            if not n:
                return 0, b"", b""
            assert core.check_sound(0) == 0
            return (n, core.selection_rows(0, n).tobytes(),
                    core.var_bits(0, np.arange(nv, dtype=np.int32), n).tobytes())

        for seed in (7, 20260901):
            on, off, scalar = run("wave", 1, seed), run("wave", -1, seed), run("scalar", 0, seed)
            assert on == off, f"wave vscan on/off diverged (B={B}, nv={nv}, seed={seed})"
            assert on[0] == scalar[0], f"wave and scalar disagree on the count (B={B}, nv={nv})"
            compared += 1
            models_seen += on[0]
    assert compared >= 6 and models_seen > 0, \
        f"fixture proves little: {compared} cells, {models_seen} models"

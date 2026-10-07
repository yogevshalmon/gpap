"""The core's C ABI: the flush cache, the trajectory kernel, readout, degenerate shapes and
preconditions. Every emitted model is checked against the clauses outside the engine.

Fixpoint correctness is tests/engine/test_propagation.py; the formula build is
tests/engine/test_formula.py.
"""
import ctypes
import random

import numpy as np
import pytest
import torch

from gpap.engine import _core
from gpap.engine._core import check

from tests.reference.bcp import assigned_set
from tests.support import DEV, random_cnf

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------- what Python binds

def test_k_calibration_matches():
    """Python's compute_k_safe is the binding of cpp/src/calibrate.cpp, not a second copy."""
    from gpap.engine.calibrate import compute_k_safe as py_k
    for M, N in [(4407, 1701), (109432, 42051), (201437, 77151)]:
        for budget in (1 << 30, 6 << 30):
            for cap in (0, 16384):
                assert _core.compute_k_safe(M, N, budget, cap or None) == \
                    py_k(M, N, budget, cap=cap or None)


def test_envelope_matches():
    """The packed-split envelope in cpp/src/runtime.cpp and its Python mirror (used to
    pre-check formulas without CUDA) agree."""
    from gpap.engine.bcp_engine import packed_envelope
    for nv, ml in [(10, 3), (32767, 5), (33000, 5), (40000, 300), (9_000_000, 3)]:
        assert _core.packed_envelope(nv, ml)[0] == packed_envelope(nv, ml)[0]


# ---------------------------------------------------------------- the flush cache

def test_flush_cache_and_units():
    """The formula's unit clauses are asserted once at construction and broadcast by reset."""
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


# ---------------------------------------------------------------- the trajectory kernel

@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_check_sound_agrees_with_an_independent_recount(mode):
    """`check_sound` counts the same unsound rows as a host recount from `sat_rows`, over a mix
    of conflicted and surviving rows.

    Expecting every conflicted row to be unsound would be wrong: a row can conflict by forcing
    both polarities of a literal without emptying any clause, which is how most conflicts show
    up on the scalar kernel."""
    rng = random.Random(11)
    nv, nc, K = 40, 90, 256
    clauses = random_cnf(rng, nv, nc, kmin=2)          # kmin=2 -> exercise the binary pair path
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses), mode=mode)
    core.reset(K)
    # Wide random assumptions: plenty of rows conflict, a few survive.
    dec = np.array([[2 * v + rng.randint(0, 1) for v in rng.sample(range(nv), 12)]
                    for _ in range(K)], dtype=np.int64)
    core.apply(dec)

    rows = np.arange(K, dtype=np.int32)
    sat = core.sat_rows(rows)                            # (K, M) non-falsified count per clause
    want = int((sat.min(axis=1) == 0).sum())             # rows with some emptied clause
    core.selection_set(core.SCRATCH_SLOT, rows)
    assert core.check_sound(core.SCRATCH_SLOT) == want
    assert want > 0, "this fixture must actually contain unsound rows, or it proves nothing"


@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("B", [1, 8])
def test_trajectory_emits_only_real_models(mode, B):
    """Every selected row satisfies every clause, checked outside the engine, and
    `check_sound` agrees."""
    rng = random.Random(5)
    nv, nc, K, seed = 60, 150, 512, 1234567
    clauses = random_cnf(rng, nv, nc, kmin=3)
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
    """The counters agree with the assignment (decisions + implications); every reported
    propagation rate is derived from them."""
    rng = random.Random(23)
    nv, nc, K = 40, 110, 256
    clauses = random_cnf(rng, nv, nc, kmin=3)
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
    """Per-launch counters describe the last call and reset() does not clear them, so an empty
    launch (empty order, zero-width apply) must clear them itself. The priming launch must do
    work, so that the zeros afterwards mean something."""
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
    """Emitted rows satisfy every clause on unrolled circuit CNFs, with a support-first order
    and a realistic K."""
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
# No clauses, no variables, an empty decision order: all reachable after simplification.

def test_m0_formula_resets_every_row():
    """With no clauses, reset still restores all K rows, not just row 0."""
    core = _core.Engine(_core.Formula.from_clauses(4, []))
    core.reset(2)
    core.apply(np.array([[0], [3]], dtype=np.int64))       # lit indices: +x1 and -x2
    act, conf = core.flags()
    assert act.all() and not conf.any()
    asg = core.rows_assignment(np.array([0, 1], dtype=np.int32))
    assert assigned_set(asg[0]) == {0} and assigned_set(asg[1]) == {3}


def test_zero_variable_formula():
    """Unit propagation or ELS can fix every variable, leaving an empty formula."""
    cf = _core.Formula.from_clauses(0, [])
    assert cf.info.num_vars == 0 and cf.info.num_clauses == 0
    core = _core.Engine(cf)
    core.reset(8)
    assert core.trajectory(np.zeros(0, dtype=np.int32), 0, 8, 42) == 0   # empty order = no-op
    # nothing is free, so every active row is a (vacuous) complete model
    assert core.select_complete(core.pair_mask(None), 0) == 8


def test_zero_column_readout_is_a_noop():
    """Gathering zero variables is legal: the sampler asks for all N variables, and N can be 0
    after simplification."""
    core = _core.Engine(_core.Formula.from_clauses(0, []))
    core.reset(4)
    n = core.select_complete(core.pair_mask(None), 0)
    assert core.var_bits(0, np.zeros(0, dtype=np.int32), n).shape == (n, 0)
    assert core.assignment_rows(0, n).shape == (n, 0)


def test_readout_never_writes_past_the_caller_buffer():
    """The three gathers take an explicit row capacity and write min(capacity, selection), so a
    stale count cannot overflow the caller's buffer. Fewer rows than selected truncates to the
    first rows."""
    rng = random.Random(77)
    nv = 12
    clauses = random_cnf(rng, nv, 18, kmin=3)
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
    # Asking for more than exists is safe too: the extra rows are not written.
    assert core.var_bits(0, cols, n + 5).shape == (n + 5, nv)


def test_empty_decision_order_is_a_noop():
    rng = random.Random(41)
    clauses = random_cnf(rng, 8, 12)
    core = _core.Engine(_core.Formula.from_clauses(8, clauses))
    core.reset(4)
    before = core.flags()[0].copy()
    assert core.trajectory(np.zeros(0, dtype=np.int32), 0, 4, 7) == 0
    np.testing.assert_array_equal(core.flags()[0], before)


def test_out_of_envelope_formula_builds_but_engine_rejects():
    """The packed split is the engine's choice: shape queries work on a formula no split fits,
    and the engine raises ValueError (suite drivers skip on it)."""
    cf = _core.Formula.from_clauses(40_000, [list(range(1, 300))])   # 2N>65535 and len>255
    assert cf.info.shift == 0 and cf.info.num_vars == 40_000
    with pytest.raises(ValueError):
        _core.Engine(cf)


def test_small_formulas_take_the_16_bit_split():
    core = _core.Engine(_core.Formula.from_clauses(4, [[1, 2], [-1, 2], [1, -2], [3, 4]]))
    assert core.shift == 16


def test_wide_split_is_selected_and_sound():
    """8|24 'wide' split: 2N > 65535 forces the second envelope, and it must still be sound."""
    rng = random.Random(31)
    nv = 40000
    clauses = random_cnf(rng, nv, 60000, kmax=4, kmin=3)
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses))
    assert core.shift == 8
    order = torch.randperm(nv, generator=torch.Generator(device=DEV).manual_seed(2),
                           device=DEV).to(torch.int32).contiguous()
    core.reset(64)
    core.trajectory_dev(order.data_ptr(), nv, nv, 8, 4242)
    core.select_complete(core.pair_mask(None), 0)
    assert core.check_sound(0) == 0


# ---------------------------------------------------------------- ABI preconditions

# There is deliberately no test feeding gpap_engine_apply an out-of-range literal. Showing the
# hazard needs an index past 2N, which makes the kernel read the occ arrays out of bounds; a
# CUDA_ERROR_ILLEGAL_ADDRESS is sticky for the context and would fail every later GPU test. The
# packaged faces validate literals before any launch, gated in
# tests/engine/test_assumption_checker.py and
# tests/mining/test_mining.py::test_rejects_out_of_range_literals. gpap.h states the raw ABI's
# precondition.


def test_n_support_past_the_order_length_is_refused():
    """The kernel walks `order` up to n_support, so n_support past the order length (or
    negative) must be refused."""
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
    """An order entry past num_vars would write into another row (`ab_row + (v >> 4)`). The
    host face checks it while staging the array."""
    nv = 8
    core = _core.Engine(_core.Formula.from_clauses(nv, [[1, 2, 3], [-1, 4], [5, 6, 7], [-5, 8]]))
    core.reset(4)
    bad = np.arange(nv, dtype=np.int32)
    bad[0] = 10_000
    with pytest.raises(_core.GpapError):
        core.trajectory(bad, n_support=nv, B=4, seed=7)


@pytest.mark.slow          # large: the slice loop only engages past 256 MB
def test_readout_slicing_matches_an_unsliced_read():
    """`var_bits` and `assignment_rows` read in slices so the host buffer stays under
    `kReadoutSliceBytes` (256 MB, cpp/src/engine.cpp). Slicing only engages when
    rows > 256MB/cols, so this uses a wide, nearly free formula (which also takes the 8|24
    split)."""
    nv = 100_000                                  # 2N = 200,000 -> wide split, small slice step
    clauses = [[v, -(v + 1)] for v in range(1, 400)]
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses))
    step = max(32, (256 << 20) // nv) // 32 * 32
    # One slice plus a short tail is enough to cross a boundary; the host buffers are
    # (K x nv) and (K x 2N) bytes, so keep K small.
    K = step + 96
    core.reset(K)
    # Run a trajectory first so rows differ. After a reset every row is the same baseline, and
    # a slice loop that dropped its row offset would still compare equal.
    core.trajectory(np.arange(nv, dtype=np.int32), nv, 64, 20260831)
    rows = np.arange(K, dtype=np.int32)
    core.selection_set(core.SCRATCH_SLOT, rows)
    cols = np.arange(nv, dtype=np.int32)

    sliced = core.var_bits(core.SCRATCH_SLOT, cols, K)
    assert sliced.shape == (K, nv)
    assert K > step, f"precondition: {K} rows must exceed the {step}-row slice"
    # Rows on either side of the boundary must differ, or an ignored offset would go unseen.
    assert not np.array_equal(sliced[0], sliced[step]), "rows must differ across the boundary"

    piece = max(1, step // 4)                     # each request below the slice threshold
    for r0 in range(0, K, piece):
        core.selection_set(1, rows[r0:r0 + piece])
        ref = core.var_bits(1, cols, min(piece, K - r0))
        assert np.array_equal(ref, sliced[r0:r0 + len(ref)]), f"slice boundary at row {r0}"

    # assignment_rows has 2N columns, so its step is half; read just past one boundary.
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
    """A device-side check for cross-row writes, since compute-sanitizer (CUDA toolkit) is not
    a dependency and the kernels do raw per-row pointer arithmetic.

    A row whose decisions are all padding touches neither its packed words nor its bitmap.
    Interleaving such canary rows with rows doing deep propagation makes any write into another
    row visible."""
    from tests.reference.bcp import lit_idx

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
    """The probe returns exactly the conflicting tuples: brute-force every k-subset through the
    reference BCP and compare sets, for both `same_var` settings and three widths. Only an
    exhausted sweep claims to have found all of them, so only those are compared."""
    import itertools

    from tests.reference.bcp import lit_idx, ref_bcp

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
    """After reset a slot selects nothing, so reading it must not compact the uninitialized
    mask into arbitrary row indices, and `gpap_check_sound` on it covers zero rows."""
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
    assert unsound.value == 0            # over zero rows
    assert core.select_complete(mask, 0) == 0     # and the real answer is 0 before a trajectory


@pytest.mark.parametrize("B", [1, 8, 64])
def test_the_warp_wide_decision_scan_picks_identically(B):
    """The warp-wide decision scan (VSCAN, on by default in the wave kernel) picks the same
    variables and polarities as the scalar scan, so wave with and without it and the scalar
    kernel draw the same models. Compare the model sets, not counts."""
    rng = random.Random(808)
    compared = models_seen = 0
    for trial in range(4):
        nv = rng.randint(30, 120)
        # Clause/var ~0.7 keeps the formula loose enough that B=64 still yields models; a cell
        # with none would compare two empty sets.
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
            # Decisions depend only on `order`, the free set (a BCP fixpoint) and the polarity
            # hash, so the two kernels complete the same rows with the same bits.
            assert on == scalar, f"wave and scalar diverged on the models (B={B}, nv={nv})"
            compared += 1
            models_seen += on[0]
    assert compared >= 6 and models_seen > 0, \
        f"fixture proves little: {compared} cells, {models_seen} models"


@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("soft", [False, True])
def test_both_packed_splits_draw_identical_trajectories(mode, soft):
    """The 8|24 and 16|16 splits differ only in how the packed word is divided, so at one seed
    they complete the same rows with the same bits. The formulas are loose so each yields
    models; two empty sets would compare equal."""
    rng = random.Random(3)
    formulas = [(6, [[3], [1, 2, -3], [-1, 4], [-2, -4], [3, -4, 5], [-5, 1], [2, 5, 6],
                     [-6, -2]])]
    for _ in range(5):
        nv = rng.randint(8, 16)
        formulas.append((nv, random_cnf(rng, nv, nv, kmin=2)))
    for nv, clauses in formulas:
        f = _core.Formula.from_clauses(nv, clauses)
        order = np.arange(nv, dtype=np.int32)
        got = {}
        for shift in (16, 8):
            core = _core.Engine(f, mode=mode, force_shift=shift)
            assert core.shift == shift
            core.reset(128)
            core.trajectory(order, nv, 2, 17, soft=soft)
            n = core.select_complete(core.pair_mask(None), 0)
            assert n > 0, f"no models on nv={nv}: the comparison would be vacuous"
            assert core.check_sound(0) == 0
            got[shift] = (core.selection_rows(0, n).tobytes(),
                          core.var_bits(0, np.arange(nv, dtype=np.int32), n).tobytes())
        assert got[16] == got[8], f"the splits diverged on nv={nv} (mode={mode}, soft={soft})"

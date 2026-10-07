"""CLEAN mode (gpap_engine_apply_clean; docs/findings/CLEAN_MODE_FINDINGS.md).

A clean launch gives the same conflict flags and counters as a plain launch, and leaves every
packed row bit-identical to the post-flush baseline, so the next reset() can skip the broadcast
("reset-lite"). Also: the guards on readouts that would mix baseline and fixpoint state, and the
callers that use clean mode. The readout is `packed_rows` (raw words); `sat_rows` masks the sum
field, where a wrong restore could hide.
"""

import random

import numpy as np
import pytest
import torch

from gpap.engine import _core
from tests.support import DEV, decision_matrix, random_assumptions, random_cnf

pytestmark = pytest.mark.gpu


def _awkward_sets(rng, nv, n):
    """Assumption sets of width 0..4 with duplicate literals and l/~l pairs mixed in."""
    sets = random_assumptions(rng, nv, n, wmax=4)
    for s in sets:
        if s and rng.random() < 0.25:
            s.append(s[0])                       # duplicate literal
        if s and rng.random() < 0.25:
            s.append(-s[0])                      # l / ~l pair -> in-seed contradiction
    return sets


@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("shift", [16, 8])
def test_state_identity_fuzz(mode, shift):
    """After each clean launch every packed row equals the baseline (conflicted rows too),
    flags and counters equal a plain-apply twin, and conflict flags equal the reference BCP.
    The sets include duplicates and l/~l pairs to hit each way a launch can stop early."""
    from tests.reference.bcp import lit_idx, ref_bcp

    rng = random.Random(11)
    lite_seen = 0
    for trial in range(8):
        nv = rng.randint(6, 13)
        clauses = random_cnf(rng, nv, rng.randint(4, 23))   # units included
        f = _core.Formula.from_clauses(nv, clauses)
        if f.info.has_empty_clause:
            continue
        eng = _core.Engine(f, mode=mode, force_shift=shift)
        twin = _core.Engine(f, mode=mode, force_shift=shift)
        if eng.flush_conflict:
            continue
        K = rng.randint(3, 39)
        eng.reset(K)
        baseline = eng.packed_rows(np.arange(K))

        for round_ in range(4):
            sets = _awkward_sets(rng, nv, K)
            d = decision_matrix(sets)
            eng.reset(K)
            eng.apply_clean(d)
            twin.reset(K)
            twin.apply(d)

            ta, tc = twin.flags()
            ca, cc = eng.flags()
            assert (cc == tc).all() and (ca == ta).all(), (mode, shift, trial, round_, "flags")
            assert (eng.num_propagated() == twin.num_propagated()).all(), \
                (mode, shift, trial, round_, "counters")
            post = eng.packed_rows(np.arange(K))
            assert (post == baseline).all(), \
                (mode, shift, trial, round_, "packed rows not restored",
                 np.argwhere(post != baseline)[:5].tolist())
            for i, s in enumerate(sets):                     # independent oracle
                exp_conflict, _ = ref_bcp(clauses, [lit_idx(x) for x in s])
                assert bool(cc[i]) == exp_conflict, (mode, shift, trial, round_, i, s)
        lite_seen += eng.reset_stats()[1]
    assert lite_seen > 0, "no reset ever went lite — CLEAN mode silently disabled"


@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_deep_row_guard_streams_exactly(mode):
    """A probe that assigns every variable pushes the restore past the M/8 threshold into the
    stream-the-row path, which must restore exactly too.

    The chain is ternary (-v | -v | v+1) so that it has packed words: a binary chain would have
    none (test_pure_binary_formula_is_stateless). The duplicate literal still forces v+1 as
    (-v | v+1) would."""
    nv = 400
    clauses = [[-(v), -(v), v + 1] for v in range(1, nv)]    # 1 => 2 => ... => nv, len-3 links
    f = _core.Formula.from_clauses(nv, clauses)
    eng = _core.Engine(f, mode=mode)
    eng.reset(4)
    baseline = eng.packed_rows(np.arange(4))
    d = np.array([[0], [2 * (nv - 1) + 1], [-1], [4]], dtype=np.int64)  # deep, deep, empty, deep
    eng.reset(4)
    eng.apply_clean(d)
    a, c = eng.flags()
    assert not c.any(), "chain probes conflict nothing"
    assert (eng.packed_rows(np.arange(4)) == baseline).all(), "stream-arm restore not exact"
    assert eng.num_propagated()[0] == nv - 1, "deep probe really propagated the whole chain"


def test_k_shrink_grow_bookkeeping():
    """Shrinking K stays lite (rows past the chunk stay clean); growing past the high-water
    mark broadcasts again. The state is exact throughout."""
    nv = 12
    clauses = [[1, 2, 3], [-1, 4], [-2, -4], [3, -4, 5], [-5, 6]]
    f = _core.Formula.from_clauses(nv, clauses)
    eng = _core.Engine(f)
    eng.reset(64)
    baseline64 = eng.packed_rows(np.arange(64))
    d = np.array([[0]] * 64, dtype=np.int64)

    eng.reset(64)                                    # lite (nothing dirtied since baseline read)
    eng.apply_clean(d)
    eng.reset(16)                                    # lite: 64 >= 16
    eng.apply_clean(d[:16])
    eng.reset(64)                                    # lite: rows 16..64 stayed clean
    full0, lite0 = eng.reset_stats()
    assert (eng.packed_rows(np.arange(64)) == baseline64).all()
    assert lite0 == 3, f"expected exactly the three lite resets, got {lite0}"

    eng.apply_clean(d)
    eng.reset(256)                                   # growth past high-water -> full broadcast
    full1, lite1 = eng.reset_stats()
    assert full1 == full0 + 1 and lite1 == lite0
    assert (eng.packed_rows(np.arange(64)) == baseline64).all()


@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_pure_binary_formula_is_stateless(mode):
    """A formula of clauses with at most 2 literals has no packed words. Propagation, flags,
    counters, sat_rows and reset-lite must still behave as on any other formula."""
    from tests.reference.bcp import lit_idx, ref_bcp

    nv = 40
    clauses = [[-(v), v + 1] for v in range(1, nv)] + [[1, 2]]   # binary chain + one extra edge
    f = _core.Formula.from_clauses(nv, clauses)
    assert (f.info.num_bin, f.info.num_long) == (nv, 0)
    eng = _core.Engine(f, mode=mode)
    twin = _core.Engine(f, mode=mode)
    eng.reset(4)
    twin.reset(4)
    assert eng.packed_rows(np.arange(4)).shape == (4, 0)

    d = np.array([[0], [2 * (nv - 1) + 1], [-1], [1]], dtype=np.int64)   # +1, -nv, pad, -1
    eng.apply_clean(d)
    twin.apply(d)
    ea, ec = eng.flags()
    ta, tc = twin.flags()
    assert (ea == ta).all() and (ec == tc).all()
    assert (eng.num_propagated() == twin.num_propagated()).all()
    assert eng.num_propagated()[0] == nv - 1, "the whole chain propagated through the edges"
    for i, dec in enumerate([[1], [-nv], [], [-1]]):
        exp_conflict, _ = ref_bcp(clauses, [lit_idx(x) for x in dec])
        assert bool(ec[i]) == exp_conflict

    # sat_rows synthesizes counts from the bitmap. On the fully propagated row each chain clause
    # has one non-falsified literal, and (1|2) keeps both.
    sat = twin.sat_rows(np.array([0], dtype=np.int32))
    assert sat.shape == (1, nv)
    assert sat[0].tolist() == [1] * (nv - 1) + [2]

    eng.reset(4)
    assert eng.reset_stats()[1] >= 1, "reset-lite must fire with nothing to broadcast"


def test_plain_apply_and_trajectory_invalidate():
    """A plain apply or a trajectory leaves rows dirty, so the next reset is full and restores
    the baseline."""
    nv = 10
    clauses = [[1, 2], [-1, 3], [-2, -3], [4, 5, 6]]
    f = _core.Formula.from_clauses(nv, clauses)
    eng = _core.Engine(f)
    eng.reset(8)
    baseline = eng.packed_rows(np.arange(8))
    d = np.array([[0]] * 8, dtype=np.int64)

    eng.apply(d)                                     # plain apply: rows now dirty
    eng.reset(8)
    assert eng.reset_stats() == (2, 0)               # create-time none; both resets full
    assert (eng.packed_rows(np.arange(8)) == baseline).all()

    eng.apply_clean(d)
    eng.reset(8)                                     # lite
    order = np.arange(nv, dtype=np.int32)
    eng.trajectory(order, n_support=nv, B=1, seed=7)  # trajectory: rows now dirty
    eng.reset(8)
    full, lite = eng.reset_stats()
    assert (full, lite) == (3, 1)
    assert (eng.packed_rows(np.arange(8)) == baseline).all()


def test_recache_moves_the_baseline():
    """recache_flush changes the baseline: the next reset is full, and later clean launches
    restore to the new baseline."""
    nv = 8
    clauses = [[-1, 2], [-2, 3], [3, 4, 6], [-3, -5]]
    f = _core.Formula.from_clauses(nv, clauses)
    eng = _core.Engine(f)
    eng.reset(1)
    eng.apply(np.array([[0]], dtype=np.int64))       # make x1 true -> x2 -> x3, ~x5
    eng.recache_flush()
    eng.reset(6)
    full, lite = eng.reset_stats()
    assert lite == 0, "reset after recache must be a full broadcast"
    new_baseline = eng.packed_rows(np.arange(6))
    # probe ~x6 on the new baseline: falsifies +6 in (3|4|6), the one packed word here
    eng.apply_clean(np.array([[2 * 5 + 1]] * 6, dtype=np.int64))
    assert (eng.packed_rows(np.arange(6)) == new_baseline).all()
    eng.reset(6)
    assert eng.reset_stats()[1] == 1


def test_gpap_clean_env_disables(monkeypatch):
    """GPAP_CLEAN=0 (read at engine creation) makes apply_clean a plain apply: the fixpoint
    state stays and no reset goes lite."""
    monkeypatch.setenv("GPAP_CLEAN", "0")
    nv = 8
    clauses = [[-1, -1, 2], [-2, -2, 3], [1, 4]]     # ternary, so the change lands in packed words
    f = _core.Formula.from_clauses(nv, clauses)
    eng = _core.Engine(f)
    eng.reset(4)
    baseline = eng.packed_rows(np.arange(4))
    eng.reset(4)
    eng.apply_clean(np.array([[0]] * 4, dtype=np.int64))       # x1 -> x2 -> x3: dirties words
    assert (eng.packed_rows(np.arange(4)) != baseline).any(), \
        "with GPAP_CLEAN=0 the fixpoint state must survive the launch"
    eng.reset(4)
    assert eng.reset_stats()[1] == 0


# ---------------------------------------------------------------- clean mode guards
# After a clean launch the packed words are the baseline but the bitmap is the fixpoint, so
# calls whose answer would mix the two refuse.

def _engine_after_clean():
    f = _core.Formula.from_clauses(8, [[-1, 2], [-2, 3], [3, 4], [-3, -5], [1, 6]])
    e = _core.Engine(f)
    e.reset(4)
    e.apply_clean(np.array([[0]] * 4, dtype=np.int64))    # x1 TRUE -> propagates
    return e


def test_recache_flush_refuses_after_clean():
    """Caching the mixed state would make an inconsistent flush cache that every later launch
    propagates from."""
    e = _engine_after_clean()
    with pytest.raises(Exception, match="apply_clean|recache"):
        e.recache_flush()


def test_packed_derived_readouts_refuse_after_clean():
    """sat_rows and check_sound read the packed words, which are the baseline here."""
    e = _engine_after_clean()
    with pytest.raises(Exception, match="apply_clean|baseline"):
        e.sat_rows([0, 1])


def test_raw_packed_readout_stays_open():
    """gpap_packed_rows is the raw readout the identity tests above rely on, so it is not
    guarded."""
    e = _engine_after_clean()
    assert e.packed_rows([0, 1]).shape[0] == 2


def test_plain_apply_leaves_everything_usable():
    """Control: the guards key off a clean launch, not off clean mode being enabled."""
    f = _core.Formula.from_clauses(8, [[-1, 2], [-2, 3], [3, 4], [-3, -5], [1, 6]])
    e = _core.Engine(f)
    e.reset(4)
    e.apply(np.array([[0]] * 4, dtype=np.int64))
    assert e.sat_rows([0]).shape == (1, e.num_clauses)
    e.recache_flush()                                     # must not raise
    assert e.reset_stats()[1] == 0


def test_reset_clears_the_divergence():
    """reset() restores the bitmap, so the state is consistent and recache is allowed."""
    e = _engine_after_clean()
    e.reset(4)
    e.recache_flush()                                     # must not raise
    assert e.sat_rows([0]).shape[0] == 1


@pytest.mark.parametrize("rows", [1, 3, 8])
def test_apply_validates_row_count(rows):
    """Both entry points check the row count. The C side reads K*L int64s, so a short array
    would be read past its end and the garbage used as literal indices."""
    f = _core.Formula.from_clauses(6, [[1, 2], [-1, 3]])
    e = _core.Engine(f)
    e.reset(4)
    d = np.zeros((rows, 1), dtype=np.int64)      # parametrized 1/3/8, never the sized 4
    for fn in (e.apply, e.apply_clean):
        with pytest.raises(ValueError, match="rows but the engine is sized"):
            fn(d)


# ---------------------------------------------------------------- consumers
# The callers that use clean mode, end to end.

def test_probe_sweep_identical_and_lite(monkeypatch):
    """The probe sweep finds the same conflict tuples with clean mode on and off, and with it
    on, resets after the first chunk go lite."""
    nv = 8
    clauses = [[1, 2, 3], [-1, -2], [-2, -3], [-1, -3], [4, 5], [-5, 6], [-4, -6, 7]]
    f = _core.Formula.from_clauses(nv, clauses)

    eng_on = _core.Engine(f)
    with eng_on.probe_combinations(None, 2, batch=13) as sweep:   # many small chunks
        on = sweep.conflicts().tolist()
    full_on, lite_on = eng_on.reset_stats()

    monkeypatch.setenv("GPAP_CLEAN", "0")
    eng_off = _core.Engine(f)
    with eng_off.probe_combinations(None, 2, batch=13) as sweep:
        off = sweep.conflicts().tolist()
    assert on == off, "conflict tuples must not depend on the reset mechanism"
    assert len(on) > 0, "precondition: this pool must yield real conflicts"
    assert eng_off.reset_stats()[1] == 0
    assert lite_on > 0, "multi-chunk sweep with CLEAN on must reset lite"


def test_checker_flags_only_goes_lite(monkeypatch):
    """Flags-only AssumptionChecker calls use clean mode (lite resets, same answers);
    assignments=True uses the plain kernel. The second half uses a fresh checker because the
    first reset after a clean call is legitimately lite."""
    from gpap.engine.checker import AssumptionChecker

    CLAUSES = [[1, 2], [-1, 3], [-2, -3], [4, 5], [-4, 6]]
    sets = [[1], [2, 4], [-3], [], [1, -1], [5]] * 40
    ck = AssumptionChecker.from_clauses(6, CLAUSES)
    one = ck.check_assumptions(sets)
    monkeypatch.setattr(ck, "chunk_size", lambda n, max_lits=1: 17)
    before = ck.engine._core.reset_stats()[1]
    many = ck.check_assumptions(sets)
    gained = ck.engine._core.reset_stats()[1] - before
    assert torch.equal(one["conflict"], many["conflict"])
    # Compare with the chunk count, not > 0: the first reset is lite anyway (left clean by the
    # previous call), so > 0 would pass without any chunking.
    n_chunks = -(-len(sets) // 17)
    assert gained >= n_chunks - 1, (
        f"expected ~{n_chunks} lite resets across {n_chunks} chunks, got {gained}")

    ck2 = AssumptionChecker.from_clauses(6, CLAUSES)
    monkeypatch.setattr(ck2, "chunk_size", lambda n, max_lits=1: 17)
    r = ck2.check_assumptions(sets, assignments=True)
    assert ck2.engine._core.reset_stats()[1] == 0, \
        "state-reading calls must keep the plain kernel (no lite resets)"
    assert int(r["assignment"].sum()) > 0, "assignments must be the fixpoint, not baseline"


def test_mining_identical_under_env(monkeypatch):
    """gpap.mining yields the same clauses with clean mode on and off."""
    from gpap.mining import mine_entailed

    clauses = [[-1, 2], [-2, 3], [-1, -3, 4], [-4, 5], [1, 2, 3], [-5, -1, 3]]
    on = mine_entailed(6, clauses, DEV, widths=(1, 2), drop="none")
    monkeypatch.setenv("GPAP_CLEAN", "0")
    off = mine_entailed(6, clauses, DEV, widths=(1, 2), drop="none")
    assert sorted(map(tuple, on.clauses)) == sorted(map(tuple, off.clauses))

"""CLEAN-mode identity gates (gpap_engine_apply_clean — docs/findings/CLEAN_MODE_FINDINGS.md).

THE invariant: a CLEAN launch answers exactly what the plain launch answers — conflict flags on
EVERY row (confluence makes them exact), counters row for row on the same kernel — and exits with
every packed row bit-identical to the post-flush baseline, so the next reset() skips the K·M
broadcast ("reset-lite"). The fuzz is engineered to hit every partial-stop mode the kernels
have: seed contradictions (the check-then-set path), forced-both-polarities (the wave
OR-then-check stray-bit case), mid-occurrence-list conflicts, deep chains (the in-kernel
deep-row guard that streams a row instead), duplicate literals, empty sets, K shrink/grow.

`packed_rows` (raw words, sum field included) is the readout — `sat_rows` masks the sum field,
which is exactly where a wrong restore could hide.
"""

import numpy as np
import pytest
import torch

from gpap.engine import _core
from tests.support import DEV

pytestmark = pytest.mark.gpu


def _random_cnf(rng, nv):
    """Random clauses INCLUDING units, so the flush assigns literals and seed-vs-baseline
    contradictions are reachable."""
    n_cl = int(rng.integers(4, 24))
    out = []
    for _ in range(n_cl):
        w = int(rng.integers(1, 5))
        vs = rng.choice(nv, size=min(w, nv), replace=False) + 1
        out.append([int(v) * (1 if rng.random() < 0.5 else -1) for v in vs])
    return out


def _random_sets(rng, nv, n):
    """Widths 0..4, with deliberate duplicates and l/~l pairs mixed in."""
    sets = []
    for _ in range(n):
        w = int(rng.integers(0, 5))
        s = [int(v) * (1 if rng.random() < 0.5 else -1)
             for v in rng.choice(nv, size=min(w, nv), replace=False) + 1]
        if s and rng.random() < 0.25:
            s.append(s[0])                       # duplicate literal
        if s and rng.random() < 0.25:
            s.append(-s[0])                      # l / ~l pair -> in-seed contradiction
        sets.append(s)
    return sets


def _to_dec_matrix(sets, nv):
    L = max(1, max((len(s) for s in sets), default=1))
    d = np.full((len(sets), L), -1, dtype=np.int64)
    for i, s in enumerate(sets):
        for j, lit in enumerate(s):
            d[i, j] = 2 * (abs(lit) - 1) + (0 if lit > 0 else 1)
    return d


@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("shift", [16, 8])
def test_state_identity_fuzz(mode, shift):
    """After every CLEAN launch: packed == the broadcast baseline BIT-EXACT on every row
    (conflicted rows included), flags/counters == a plain-apply twin, conflict flags == the
    pure-Python reference BCP. Both kernels, both packed splits."""
    from tests.reference.bcp import lit_idx, ref_bcp

    rng = np.random.default_rng(11)
    lite_seen = 0
    for trial in range(8):
        nv = int(rng.integers(6, 14))
        clauses = _random_cnf(rng, nv)
        f = _core.Formula.from_clauses(nv, clauses)
        if f.info.has_empty_clause:
            continue
        eng = _core.Engine(f, mode=mode, force_shift=shift)
        twin = _core.Engine(f, mode=mode, force_shift=shift)
        if eng.flush_conflict:
            continue
        K = int(rng.integers(3, 40))
        eng.reset(K)
        baseline = eng.packed_rows(np.arange(K))

        for round_ in range(4):
            sets = _random_sets(rng, nv, K)
            d = _to_dec_matrix(sets, nv)
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
    """An implication chain where one probe assigns EVERY variable pushes the clean tail past
    the M/8 threshold into the stream-the-row arm — which must restore just as exactly.

    The chain is TERNARY (-v | -v | v+1): a plain binary chain would segregate into implication
    edges with no packed words at all (that case is test_pure_binary_formula_is_stateless), and
    this test exists to exercise the LONG-clause restore. The duplicate literal makes each
    falsification decrement twice, so the chain still forces v+1 exactly as (-v | v+1) does —
    and each deep probe walks 2(nv-1) long occ entries against M/8 = (nv-1)/8."""
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
    """The clean-rows high-water: shrink stays lite (rows beyond the chunk keep their clean
    state), growth past the high-water broadcasts again — and the state is exact throughout."""
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
    eng.reset(64)                                    # lite: rows 16..64 never lost cleanliness
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
    """A formula of ONLY <=2-literal clauses has num_long == 0: no packed words, no broadcast,
    nothing for CLEAN to restore — yet propagation, flags, counters, the sat_rows synthesis and
    reset-lite must all behave exactly as on any other formula."""
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

    # sat_rows synthesizes every count from the bitmap (there are no words to read): on the
    # fully-propagated chain row every clause has exactly one non-falsified (= TRUE) literal,
    # except (1|2) which keeps both.
    sat = twin.sat_rows(np.array([0], dtype=np.int32))
    assert sat.shape == (1, nv)
    assert sat[0].tolist() == [1] * (nv - 1) + [2]

    eng.reset(4)
    assert eng.reset_stats()[1] >= 1, "reset-lite must fire with nothing to broadcast"


def test_plain_apply_and_trajectory_invalidate():
    """A plain apply and a trajectory both dirty without a restorable record: the next reset
    must be FULL, and the state after it must again be the baseline."""
    nv = 10
    clauses = [[1, 2], [-1, 3], [-2, -3], [4, 5, 6]]
    f = _core.Formula.from_clauses(nv, clauses)
    eng = _core.Engine(f)
    eng.reset(8)
    baseline = eng.packed_rows(np.arange(8))
    d = np.array([[0]] * 8, dtype=np.int64)

    eng.apply(d)                                     # plain apply -> claim dropped
    eng.reset(8)
    assert eng.reset_stats() == (2, 0)               # create-time none; both resets full
    assert (eng.packed_rows(np.arange(8)) == baseline).all()

    eng.apply_clean(d)
    eng.reset(8)                                     # lite
    order = np.arange(nv, dtype=np.int32)
    eng.trajectory(order, n_support=nv, B=1, seed=7)  # trajectory -> claim dropped
    eng.reset(8)
    full, lite = eng.reset_stats()
    assert (full, lite) == (3, 1)
    assert (eng.packed_rows(np.arange(8)) == baseline).all()


def test_recache_moves_the_baseline():
    """assume_units-style recache: the baseline itself changes, so the claim is dropped AND the
    next clean cycle restores to the NEW baseline."""
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
    """GPAP_CLEAN=0 (read at engine create) turns apply_clean into plain apply: state SURVIVES
    (differs from baseline) and no reset ever goes lite — the A/B lever."""
    monkeypatch.setenv("GPAP_CLEAN", "0")
    nv = 8
    clauses = [[-1, -1, 2], [-2, -2, 3], [1, 4]]     # ternary links: the dirt lands in WORDS
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


class TestCleanModeGuards:
    """The state a CLEAN launch leaves behind is asymmetric — packed is the BASELINE, abits are
    the FIXPOINT — so the calls whose answer would silently come from the wrong half REFUSE.
    Found by code review: each of these used to return a plausible wrong number."""

    @staticmethod
    def _engine_after_clean():
        f = _core.Formula.from_clauses(8, [[-1, 2], [-2, 3], [3, 4], [-3, -5], [1, 6]])
        e = _core.Engine(f)
        e.reset(4)
        e.apply_clean(np.array([[0]] * 4, dtype=np.int64))    # x1 TRUE -> propagates
        return e

    def test_recache_flush_refuses_after_clean(self):
        """THE dangerous one: caching (baseline packed, fixpoint abits) would bake a
        self-contradictory flush cache — abits claiming literals TRUE whose clause decrements
        packed never recorded — and every later launch would propagate from it, undetectably."""
        e = self._engine_after_clean()
        with pytest.raises(Exception, match="apply_clean|recache"):
            e.recache_flush()

    def test_packed_derived_readouts_refuse_after_clean(self):
        """sat_rows / check_sound answer FROM the packed words, which are the baseline here."""
        e = self._engine_after_clean()
        with pytest.raises(Exception, match="apply_clean|baseline"):
            e.sat_rows([0, 1])

    def test_raw_packed_readout_stays_open(self):
        """gpap_packed_rows is the documented raw/debug readout and is how the identity gate
        PROVES the restore happened — guarding it would break the gate."""
        e = self._engine_after_clean()
        assert e.packed_rows([0, 1]).shape[0] == 2

    def test_plain_apply_leaves_everything_usable(self):
        """Positive control: the guards key off the CLEAN launch, not off clean mode existing."""
        f = _core.Formula.from_clauses(8, [[-1, 2], [-2, 3], [3, 4], [-3, -5], [1, 6]])
        e = _core.Engine(f)
        e.reset(4)
        e.apply(np.array([[0]] * 4, dtype=np.int64))
        assert e.sat_rows([0]).shape == (1, e.num_clauses)
        e.recache_flush()                                     # must NOT raise
        assert e.reset_stats()[1] == 0

    def test_reset_clears_the_divergence(self):
        """reset() re-broadcasts abits, so the pair is consistent again and recache is legal."""
        e = self._engine_after_clean()
        e.reset(4)
        e.recache_flush()                                     # must NOT raise
        assert e.sat_rows([0]).shape[0] == 1

    @pytest.mark.parametrize("rows", [1, 3, 8])
    def test_apply_validates_row_count(self, rows):
        """MEMORY SAFETY, not politeness: the C side stages K*L int64s, so a short array is read
        past its end and the garbage becomes literal indices — and an index past 2N makes the
        kernel write into a LATER row. Both entry points check."""
        f = _core.Formula.from_clauses(6, [[1, 2], [-1, 3]])
        e = _core.Engine(f)
        e.reset(4)
        d = np.zeros((rows, 1), dtype=np.int64)      # parametrized 1/3/8, never the sized 4
        for fn in (e.apply, e.apply_clean):
            with pytest.raises(ValueError, match="rows but the engine is sized"):
                fn(d)


class TestConsumers:
    """The two call sites that opted in, exercised end to end."""

    def test_probe_sweep_identical_and_lite(self, monkeypatch):
        """The core probe sweep: identical conflict tuples with CLEAN on vs off, and the ON
        engine's resets actually went lite after chunk 1 (forced multi-chunk via batch)."""
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

    def test_checker_flags_only_goes_lite(self, monkeypatch):
        """AssumptionChecker flags-only calls ride the clean kernel (lite resets, same answers);
        assignments=True keeps the plain kernel and readable state. (A single carried-over lite
        at a flags-only -> arrays boundary would be legitimate — the previous clean launch really
        did leave baseline state — so the arrays half uses a FRESH checker for a crisp zero.)"""
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
        # Tied to the CHUNK COUNT, not just > 0: the boundary reset alone (the previous call's
        # last clean apply makes this call's FIRST reset lite) satisfies a bare > 0, so that
        # form would stay green even if chunking stopped happening entirely.
        n_chunks = -(-len(sets) // 17)
        assert gained >= n_chunks - 1, (
            f"expected ~{n_chunks} lite resets across {n_chunks} chunks, got {gained}")

        ck2 = AssumptionChecker.from_clauses(6, CLAUSES)
        monkeypatch.setattr(ck2, "chunk_size", lambda n, max_lits=1: 17)
        r = ck2.check_assumptions(sets, assignments=True)
        assert ck2.engine._core.reset_stats()[1] == 0, \
            "state-reading calls must keep the plain kernel (no lite resets)"
        assert int(r["assignment"].sum()) > 0, "assignments must be the fixpoint, not baseline"

    def test_mining_identical_under_env(self, monkeypatch):
        """gpap.mining end to end: identical clause sets with CLEAN on vs off."""
        from gpap.mining import mine_entailed

        clauses = [[-1, 2], [-2, 3], [-1, -3, 4], [-4, 5], [1, 2, 3], [-5, -1, 3]]
        on = mine_entailed(6, clauses, DEV, widths=(1, 2), drop="none")
        monkeypatch.setenv("GPAP_CLEAN", "0")
        off = mine_entailed(6, clauses, DEV, widths=(1, 2), drop="none")
        assert sorted(map(tuple, on.clauses)) == sorted(map(tuple, off.clauses))

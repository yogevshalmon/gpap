"""The job queue (docs/ENGINE.md, "The queue"): entry point 1 for many sets on few rows.

A queued launch gives every set the conflict flag and the implied count a launch of its own
gives, and leaves every row at the post-flush baseline, packed words and bitmap. The sweeps that
run on it (`gpap_probe_*`, and `gpap_check_sets` when it returns flags only) report what the
chunked sweeps report, in the same order. `GPAP_QUEUE=0` is the chunked sweep.

Each test also asserts that the queue ran (`queue_stats`): two chunked sweeps agree trivially.
"""
import itertools
import random

import numpy as np
import pytest

from gpap.engine import _core
from tests.reference.bcp import lit_idx, ref_bcp
from tests.support import random_cnf

pytestmark = pytest.mark.engine

#: Entails -1 (assuming 1 forces 2 and -2), so a width-1 sweep finds a unit to fold in.
UNIT_CLAUSES = [[-1, 2], [-1, -2], [3, 4, 5], [-3, 6], [-6, -4, 7], [2, 5, 8], [-8, -5, 3]]


def _engine(monkeypatch, f, queue, **kw):
    """An engine with the queue on or off. The switch is read when the engine is created."""
    monkeypatch.setenv("GPAP_QUEUE", "1" if queue else "0")
    return _core.Engine(f, **kw)


def _tuples(rng, nv, n, k):
    """`n` tuples of `k` DIMACS literals, some with a repeated literal or an l / ~l pair, the two
    ways a tuple can stop before it propagates anything."""
    out = []
    for _ in range(n):
        t = [v if rng.random() < 0.5 else -v for v in rng.sample(range(1, nv + 1), k)]
        if k >= 2 and rng.random() < 0.15:
            t[1] = t[0]
        elif k >= 2 and rng.random() < 0.15:
            t[1] = -t[0]
        out.append(t)
    return np.array(out, dtype=np.int32)


def _flat(sets):
    lits = np.array([x for s in sets for x in s], dtype=np.int32)
    offsets = np.cumsum([0] + [len(s) for s in sets]).astype(np.int64)
    return lits, offsets


def _baseline(f, rows, **kw):
    """The post-flush state of `rows` rows, read off a fresh engine: packed words, assignment."""
    e = _core.Engine(f, **kw)
    e.reset(rows)
    return e.packed_rows(np.arange(rows)), e.rows_assignment(np.arange(rows))


def _assert_rows_at_baseline(eng, f, what, **kw):
    rows = eng.K
    packed, assign = _baseline(f, rows, **kw)
    post = eng.packed_rows(np.arange(rows))
    assert (post == packed).all(), (what, "packed rows not restored",
                                    np.argwhere(post != packed)[:5].tolist())
    post = eng.rows_assignment(np.arange(rows))
    assert (post == assign).all(), (what, "bitmap not restored",
                                    np.argwhere(post != assign)[:5].tolist())


@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("shift", [16, 8])
def test_queued_sweep_reports_the_reference_and_restores_every_row(monkeypatch, mode, shift):
    """On few rows and many tuples (so every row runs several), a queued sweep reports exactly
    the tuples the reference BCP conflicts on, in input order, and afterwards every row equals
    the baseline, conflicted jobs included."""
    rng = random.Random(23)
    launches = 0
    for trial in range(8):
        nv = rng.randint(6, 13)
        clauses = random_cnf(rng, nv, rng.randint(4, 23))    # units included
        f = _core.Formula.from_clauses(nv, clauses)
        if f.info.has_empty_clause:
            continue
        kw = dict(mode=mode, force_shift=shift)
        eng = _engine(monkeypatch, f, True, **kw)
        if eng.flush_conflict:
            continue
        rows = rng.randint(2, 7)
        for k in (1, 2, 3):
            t = _tuples(rng, nv, rng.randint(rows + 1, 60), k)
            with eng.probe_tuples(t, batch=rows) as sweep:
                got = [tuple(x) for x in sweep.conflicts().tolist()]
                assert int(sweep.stats.probed) == len(t)
            want = [tuple(x) for x in t.tolist()
                    if ref_bcp(clauses, [lit_idx(v) for v in x])[0]]
            assert got == want, (mode, shift, trial, k)
            assert eng.K == rows, "a pinned batch is the queue's row count"
            _assert_rows_at_baseline(eng, f, (mode, shift, trial, k), **kw)
        launches += eng.queue_stats()[0]
    assert launches > 0, "no sweep ran on the queue"


@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("shift", [16, 8])
def test_flag_checks_on_the_queue_give_the_chunked_answers(monkeypatch, mode, shift):
    """`check_sets` without assignments or sat counts runs on the queue: the same conflict flags
    as the chunked check and, on every set that does not conflict, the same implied count. The
    sets have widths 0 to 5, repeated literals and l / ~l pairs."""
    rng = random.Random(31)
    nv = 24
    clauses = random_cnf(rng, nv, 70, kmin=2, kmax=4)
    sets = []
    for _ in range(300):
        s = [v if rng.random() < 0.5 else -v
             for v in rng.sample(range(1, nv + 1), rng.randint(0, 4))]
        if s and rng.random() < 0.2:
            s.append(s[0])
        elif s and rng.random() < 0.2:
            s.append(-s[0])
        sets.append(s)
    lits, offsets = _flat(sets)
    f = _core.Formula.from_clauses(nv, clauses)
    kw = dict(mode=mode, force_shift=shift)
    queued = _engine(monkeypatch, f, True, **kw)
    if queued.flush_conflict:
        pytest.skip("precondition: this seed's formula must not be UNSAT at the root")
    chunked = _engine(monkeypatch, f, False, **kw)

    cq, pq, _, _ = queued.check_sets(lits, offsets, chunk=5)
    cc, pc, _, _ = chunked.check_sets(lits, offsets, chunk=5)
    np.testing.assert_array_equal(cq, cc)
    ok = ~cc
    assert ok.sum() > 50 and cc.sum() > 5, "precondition: both outcomes must occur"
    np.testing.assert_array_equal(pq[ok], pc[ok])
    assert queued.queue_stats() == (queued.queue_stats()[0], len(sets))
    assert queued.queue_stats()[0] > 0, "the flags-only check did not run on the queue"
    assert chunked.queue_stats() == (0, 0), "GPAP_QUEUE=0 must not launch the queue"
    _assert_rows_at_baseline(queued, f, (mode, shift), **kw)


def test_a_check_that_reads_state_stays_on_chunks(monkeypatch):
    """A queued row is back at the baseline when its job ends, so a check that returns
    assignments or sat counts cannot run on the queue: it keeps its own row per set."""
    f = _core.Formula.from_clauses(8, [[-1, 2], [-2, 3], [3, 4], [-3, -5, 6]])
    eng = _engine(monkeypatch, f, True)
    lits, offsets = _flat([[1], [5], [-3, 1]])
    _, _, assign, _ = eng.check_sets(lits, offsets, assignments=True)
    assert assign[0, lit_idx(3)] == 1, "the assignment of set 0 must hold its implications"
    _, _, _, sat = eng.check_sets(lits, offsets, sat_count=True)
    assert sat is not None and sat.shape == (3, 4)
    assert eng.queue_stats() == (0, 0)


@pytest.mark.parametrize("k,budget", [(1, None), (2, None), (2, 37), (3, 101)])
def test_combination_sweeps_agree_with_and_without_a_budget(monkeypatch, k, budget):
    """Queue and chunks enumerate and report alike: the same conflicting tuples in the same
    order, the same probed count under a work budget, the same `exhausted`."""
    rng = random.Random(7)
    nv = 16
    clauses = random_cnf(rng, nv, 44, kmin=2, kmax=4)
    f = _core.Formula.from_clauses(nv, clauses)
    kw = {} if budget is None else {"max_tuples": budget}
    out = []
    for queue in (True, False):
        eng = _engine(monkeypatch, f, queue)
        if eng.flush_conflict:
            pytest.skip("precondition: this seed's formula must not be UNSAT at the root")
        with eng.probe_combinations(None, k, batch=3, **kw) as sweep:
            st = sweep.stats
            out.append((sweep.conflicts().tolist(), int(st.total), int(st.probed),
                        int(st.exhausted), int(st.stop)))
        assert (eng.queue_stats()[0] > 0) == queue
    assert out[0] == out[1]
    assert out[0][2] == (budget if budget is not None else out[0][1])
    assert k == 1 or budget is not None or len(out[0][0]) > 0, \
        "precondition: the full pair sweep must yield real conflicts"


def test_an_expired_deadline_stops_the_queued_sweep_before_its_first_launch(monkeypatch):
    """`seconds` is checked before each launch, as it is before each chunk."""
    f = _core.Formula.from_clauses(8, UNIT_CLAUSES)
    eng = _engine(monkeypatch, f, True)
    with eng.probe_combinations(None, 2, seconds=1e-9) as sweep:
        assert int(sweep.stats.probed) == 0 and int(sweep.stats.exhausted) == 0
        assert _core.PROBE_STOP[int(sweep.stats.stop)] == "deadline"
    assert eng.queue_stats() == (0, 0)


def test_the_engine_stays_exact_around_queued_sweeps(monkeypatch):
    """What the miner does to one engine: sweeps on a shrinking and a growing row count, a
    proven unit folded into the baseline, and a plain launch in between. Every sweep agrees with
    a chunked twin put through the same steps, and the rows end at the moved baseline."""
    f = _core.Formula.from_clauses(8, UNIT_CLAUSES)
    eng = _engine(monkeypatch, f, True)
    twin = _engine(monkeypatch, f, False)

    def sweep(e, k, batch):
        with e.probe_combinations(None, k, batch=batch) as s:
            return s.conflicts().tolist()

    for k, batch in ((1, 7), (2, 3), (2, 9)):
        assert sweep(eng, k, batch) == sweep(twin, k, batch), (k, batch)
    assert [1] in sweep(eng, 1, 5), "precondition: literal 1 must fail alone"

    d = np.array([[lit_idx(3)], [lit_idx(-4)], [lit_idx(8)]], dtype=np.int64)
    for e in (eng, twin):
        e.reset(3)
        e.apply(d)                                   # a plain launch dirties rows the queue used
    np.testing.assert_array_equal(eng.flags()[1], twin.flags()[1])
    np.testing.assert_array_equal(eng.num_propagated(), twin.num_propagated())
    # A queued sweep straight after the plain launch: it must find those rows dirty and rebuild
    # them, bitmap included, before any job runs on them.
    assert sweep(eng, 2, 3) == sweep(twin, 2, 3), "after a plain launch"
    _assert_rows_at_baseline(eng, f, "after a plain launch")

    for e in (eng, twin):
        e.assume_units(np.array([-1], dtype=np.int32))   # the baseline moves
    for k, batch in ((1, 4), (2, 6)):
        assert sweep(eng, k, batch) == sweep(twin, k, batch), ("after the unit", k, batch)
    rows = eng.K
    twin.reset(rows)
    assert (eng.packed_rows(np.arange(rows)) == twin.packed_rows(np.arange(rows))).all()
    assert (eng.rows_assignment(np.arange(rows)) == twin.rows_assignment(np.arange(rows))).all()
    assert eng.queue_stats()[0] >= 6 and twin.queue_stats() == (0, 0)


#: Two independent blocks of a 640-variable formula (40 bitmap words a row): UNIT_CLAUSES on
#: variables 1 to 8, and block 2 — a chain on 601 to 608, and 800 all-positive (so satisfiable)
#: three-literal clauses on 609 to 638. A sweep over block 1 assigns at most 16 literals a job,
#: fewer than the row's bitmap words, and does less occurrence work than an eighth of its 805
#: packed words, so each job puts back only the words it touched (past either bound it rewrites
#: the whole row): whatever an earlier launch left in block 2's words is still there afterwards.
B2 = 600
TWO_BLOCKS = (UNIT_CLAUSES
              + [[-(B2 + 1), B2 + 2], [-(B2 + 2), B2 + 3], [-(B2 + 3), B2 + 4],
                 [B2 + 5, B2 + 6, B2 + 7], [-(B2 + 5), B2 + 8]]
              + [list(c) for c in itertools.islice(
                  itertools.combinations(range(B2 + 9, B2 + 39), 3), 800)])
TWO_BLOCKS_NV = 640
BLOCK1 = np.array([s * v for v in range(1, 9) for s in (1, -1)], dtype=np.int32)
BLOCK2_BITS = slice(2 * B2, 2 * (B2 + 8))


def _sweep_conflicts(eng, k, batch):
    """A width-k sweep over block 1, `batch` rows."""
    with eng.probe_combinations(BLOCK1, k, batch=batch) as s:
        return s.conflicts().tolist()


@pytest.mark.cuda_only
def test_a_sweep_after_a_trajectory_rebuilds_its_rows(monkeypatch):
    """A trajectory (the sampler's launch) assigns rows and keeps no record to restore them
    from, so it drops the packed claim. A queued sweep on the same engine afterwards must then
    rebuild every row it uses, packed words and bitmap: the conflicts of a chunked twin, and
    every row back at the baseline, also in the word the sweep never writes."""
    f = _core.Formula.from_clauses(TWO_BLOCKS_NV, TWO_BLOCKS)
    eng = _engine(monkeypatch, f, True)
    twin = _engine(monkeypatch, f, False)
    assert _sweep_conflicts(eng, 2, 4) == _sweep_conflicts(twin, 2, 4)   # raises the claim
    launches = eng.queue_stats()[0]

    eng.reset(8)
    clean_bitmap = eng.rows_assignment(np.arange(4))[:, BLOCK2_BITS]
    clean_packed = eng.packed_rows(np.arange(4))
    eng.trajectory(np.arange(TWO_BLOCKS_NV, dtype=np.int32), n_support=TWO_BLOCKS_NV, B=1, seed=7)
    assert (eng.rows_assignment(np.arange(4))[:, BLOCK2_BITS] != clean_bitmap).any(), \
        "precondition: the trajectory must assign block 2 in the rows the sweep takes"
    assert (eng.packed_rows(np.arange(4)) != clean_packed).sum() > 100, \
        "precondition: the trajectory must move packed words, block 2's among them"

    assert _sweep_conflicts(eng, 2, 4) == _sweep_conflicts(twin, 2, 4), "after a trajectory"
    assert eng.queue_stats()[0] > launches, "the sweep after the trajectory did not queue"
    _assert_rows_at_baseline(eng, f, "after a trajectory")


def test_a_sweep_after_a_clean_launch_rebuilds_the_bitmap(monkeypatch):
    """A CLEAN launch puts the packed words back but leaves each row's bitmap at its fixpoint,
    so it keeps the packed claim, which says nothing of the bitmap. A queued sweep straight
    after it copies the bitmap all the same, as every queued call does at its start: the
    conflicts of a chunked twin, and every row back at the baseline, also in the word the sweep
    never writes."""
    f = _core.Formula.from_clauses(TWO_BLOCKS_NV, TWO_BLOCKS)
    eng = _engine(monkeypatch, f, True)
    twin = _engine(monkeypatch, f, False)
    assert _sweep_conflicts(eng, 2, 4) == _sweep_conflicts(twin, 2, 4)
    launches = eng.queue_stats()[0]

    eng.reset(4)
    clean_bitmap = eng.rows_assignment(np.arange(4))[:, BLOCK2_BITS]
    d = np.array([[lit_idx(B2 + 1)], [lit_idx(-(B2 + 4))], [lit_idx(B2 + 5)], [lit_idx(B2 + 3)]],
                 dtype=np.int64)
    eng.apply_clean(d)
    assert (eng.rows_assignment(np.arange(4))[:, BLOCK2_BITS] != clean_bitmap).any(), \
        "precondition: the CLEAN launch must leave block 2 assigned in the rows the sweep takes"

    assert _sweep_conflicts(eng, 2, 4) == _sweep_conflicts(twin, 2, 4), "after a CLEAN launch"
    assert eng.queue_stats()[0] > launches, "the sweep after the CLEAN launch did not queue"
    _assert_rows_at_baseline(eng, f, "after a CLEAN launch")


@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_a_deep_job_takes_the_streaming_restores(monkeypatch, mode):
    """A job that assigns every variable is past both restores' thresholds (more occurrence
    work than M/8, more trail entries than bitmap words), so its row is rewritten whole. That
    arm must leave the baseline too, with shallow jobs before and after it on the same row."""
    nv = 400
    z = nv + 1
    clauses = [[-(v), v + 1, z] for v in range(1, nv)] + [[-z]]   # 1 => 2 => ... => nv
    f = _core.Formula.from_clauses(z, clauses)
    eng = _engine(monkeypatch, f, True, mode=mode)
    sets = [[nv], [1], [nv - 1], [1], [-nv], [1]] * 3
    lits, offsets = _flat(sets)
    conflict, implied, _, _ = eng.check_sets(lits, offsets, chunk=2)
    assert not conflict.any(), "chain probes conflict nothing"
    assert implied.tolist() == [0, nv - 1, 1, nv - 1, nv - 1, nv - 1] * 3
    assert eng.queue_stats()[0] > 0
    _assert_rows_at_baseline(eng, f, mode, mode=mode)


def test_gpap_queue_0_is_the_chunked_sweep(monkeypatch):
    """The switch gives the sweep as it ran before the queue: one tuple per row per launch, so
    a sweep of many chunks resets many times, and the queue never launches."""
    f = _core.Formula.from_clauses(8, UNIT_CLAUSES)
    eng = _engine(monkeypatch, f, False)
    with eng.probe_combinations(None, 2, batch=13) as sweep:
        probed = int(sweep.stats.probed)
    full, lite = eng.reset_stats()
    assert eng.queue_stats() == (0, 0)
    assert full + lite >= -(-probed // 13), "a chunked sweep resets once per chunk"


def test_gpap_clean_0_keeps_the_sweeps_in_chunks(monkeypatch):
    """A queued row restores itself with CLEAN mode's restore, so GPAP_CLEAN=0, the escape hatch
    from that restore, keeps every sweep in chunks on the plain kernel: the queue never launches
    and no reset is lite, which is how a silent fallback to it shows (mining's reset record)."""
    monkeypatch.setenv("GPAP_CLEAN", "0")
    f = _core.Formula.from_clauses(8, UNIT_CLAUSES)
    eng = _engine(monkeypatch, f, True)
    twin = _engine(monkeypatch, f, False)
    with eng.probe_combinations(None, 2, batch=4) as sweep:
        got = sweep.conflicts().tolist()
    with twin.probe_combinations(None, 2, batch=4) as sweep:
        assert got == sweep.conflicts().tolist()
    lits, offsets = _flat([[1], [2, -3], [4], [-5, 6]] * 10)
    eng.check_sets(lits, offsets, chunk=4)
    assert eng.queue_stats() == (0, 0)
    assert eng.reset_stats()[1] == 0


@pytest.mark.parametrize("pin,set_mode,want", [
    (None, None, "wave"), ("scalar", None, "scalar"), (None, "scalar", "scalar")])
def test_the_queue_runs_wave_unless_a_kernel_is_pinned(monkeypatch, pin, set_mode, want):
    """The queue's rows are the warps the device runs at once, and a WAVE row is one warp on
    both backends, so the queue runs WAVE on every formula, also on one so small that a chunk of
    it runs SCALAR. A pin wins: $GPAP_KERNEL, or set_mode(). After the call the engine is as
    after a reset to its rows, so its own kernel is the one the pin or the dispatch rule gives
    for them, whatever the queue ran: WAVE on CUDA, where the queue's rows are at most the
    resident warps, and SCALAR on OpenCL, which cannot count them."""
    rng = random.Random(3)
    nv = 40
    f = _core.Formula.from_clauses(nv, random_cnf(rng, nv, 90, kmin=2, kmax=3))
    assert not _core.dispatch_wave(int(f.info.num_clauses), 1 << 18), \
        "precondition: a full-size chunk of this formula runs SCALAR"
    if pin:
        monkeypatch.setenv("GPAP_KERNEL", pin)
    else:
        monkeypatch.delenv("GPAP_KERNEL", raising=False)
    eng = _engine(monkeypatch, f, True)
    if eng.flush_conflict:
        pytest.skip("precondition: this seed's formula must not be UNSAT at the root")
    if set_mode:
        eng.set_mode(set_mode)
    with eng.probe_combinations(None, 1, batch=4):
        pass
    assert eng.queue_stats()[0] > 0
    assert eng.K == 4
    after = want if (pin or set_mode) else (
        "wave" if _core.dispatch_wave(int(f.info.num_clauses), eng.K) else "scalar")
    assert eng.mode == after, "the engine's kernel after the call: a reset to its rows'"
    assert eng.queue_mode() == want, "the kernel the queued launches ran"


def test_counters_count_the_queued_work(monkeypatch):
    """`counters()` reports the same decisions and the same implications for a queued sweep as
    for the chunked one, on a sweep where nothing conflicts (a conflicted set's counts are not
    specified)."""
    nv = 60
    clauses = [[-v, v + 1, nv + 1] for v in range(1, nv)] + [[-(nv + 1)]]
    f = _core.Formula.from_clauses(nv + 1, clauses)
    pool = np.arange(1, nv + 1, dtype=np.int32)          # positive literals: chains, no conflict
    got = []
    for queue in (True, False):
        eng = _engine(monkeypatch, f, queue)
        c0 = eng.counters()
        with eng.probe_combinations(pool, 1, batch=5) as sweep:
            assert len(sweep.conflicts()) == 0
        c1 = eng.counters()
        assert (eng.queue_stats()[0] > 0) == queue
        got.append((int(c1.decs - c0.decs), int(c1.props - c0.props)))
    assert got[0] == got[1]
    assert got[0][0] == nv and got[0][1] == nv + nv * (nv - 1) // 2


def test_mining_proves_the_same_clauses_on_the_queue(monkeypatch):
    """gpap.mining, the application on top of the sweeps, proves the same clauses whether they
    run on the queue or in chunks; and the probe it is built on does run on the queue."""
    from gpap.engine.probe import LiteralProbe
    from gpap.mining import mine_entailed
    from tests.support import DEV

    clauses = [[-1, 2], [-2, 3], [-1, -3, 4], [-4, 5], [1, 2, 3], [-5, -1, 3]]
    monkeypatch.setenv("GPAP_QUEUE", "1")
    on = mine_entailed(6, clauses, DEV, widths=(1, 2), drop="none")
    probe = LiteralProbe(6, clauses, DEV)
    with probe.engine.probe_combinations(None, 1, batch=3):   # more literals than rows
        pass
    assert probe.engine.queue_stats()[0] > 0, "the miner's engine did not run on the queue"
    monkeypatch.setenv("GPAP_QUEUE", "0")
    off = mine_entailed(6, clauses, DEV, widths=(1, 2), drop="none")
    assert sorted(map(tuple, on.clauses)) == sorted(map(tuple, off.clauses))
    assert len(on.clauses) > 0, "precondition: this formula must entail something"


def test_a_sweep_that_fits_the_rows_stays_a_chunk(monkeypatch):
    """With no more sets than rows, every row would take one set: that is a chunk, which does
    it with less bookkeeping, so the queue does not launch. One row fewer, and it does."""
    f = _core.Formula.from_clauses(8, UNIT_CLAUSES)
    eng = _engine(monkeypatch, f, True)
    t = np.array([[1], [2], [3], [-4], [5]], dtype=np.int32)
    with eng.probe_tuples(t, batch=5) as sweep:
        five = sweep.conflicts().tolist()
    assert eng.queue_stats() == (0, 0), "five sets on five rows must run as one chunk"
    with eng.probe_tuples(t, batch=4) as sweep:
        four = sweep.conflicts().tolist()
    assert eng.queue_stats()[1] == 5, "five sets on four rows must run on the queue"
    assert five == four == [[1]]


def test_a_pinned_first_launch_takes_one_set_per_row(monkeypatch):
    """A queued call's first launch takes the sets a chunk would, so it is never longer than the
    chunk it replaces (a later one grows only while launches stay short). With the rows pinned
    a chunk is one set per row: five sets on four rows are two launches, four sets and then one,
    for a sweep and a check alike."""
    f = _core.Formula.from_clauses(8, UNIT_CLAUSES)
    eng = _engine(monkeypatch, f, True)
    t = np.array([[1], [2], [3], [-4], [5]], dtype=np.int32)
    with eng.probe_tuples(t, batch=4) as sweep:
        assert sweep.conflicts().tolist() == [[1]]
    assert eng.queue_stats() == (2, 5), "a sweep: four sets, then one"
    lits, offsets = _flat(t.tolist())
    conflict, _, _, _ = eng.check_sets(lits, offsets, chunk=4)
    assert conflict.tolist() == [1, 0, 0, 0, 0]
    assert eng.queue_stats() == (4, 10), "a check: four sets, then one"


def test_the_launch_buffers_leave_a_tiny_formula_its_rows(monkeypatch):
    """The queue's launch buffers come out of the memory budget in bytes, so a formula whose
    rows are a few bytes still gets every row the device runs at once (one per resident warp,
    or 4,096 where the library cannot count them), not one."""
    f = _core.Formula.from_clauses(8, UNIT_CLAUSES)
    eng = _engine(monkeypatch, f, True)
    rng = random.Random(5)
    sets = [[rng.choice((1, -1)) * rng.randint(2, 8)] for _ in range(20000)]
    lits, offsets = _flat(sets)
    eng.check_sets(lits, offsets)
    assert eng.queue_stats()[1] == len(sets)
    assert eng.K == (_core.resident_warps() or 4096)


def test_an_unpinned_first_launch_takes_a_whole_chunk(monkeypatch):
    """Unpinned, a chunk is the memory batch, far more sets than the queue has rows on a small
    formula, and so is the first launch: a whole width-2 sweep is one launch, not one set per
    row and then more launches, each with its fixed cost."""
    rng = random.Random(11)
    nv = 90
    f = _core.Formula.from_clauses(nv, random_cnf(rng, nv, 120, kmin=3, kmax=3))
    eng = _engine(monkeypatch, f, True)
    if eng.flush_conflict:
        pytest.skip("precondition: this seed's formula must not be UNSAT at the root")
    with eng.probe_combinations(None, 2) as sweep:
        probed = int(sweep.stats.probed)
    launches, sets = eng.queue_stats()
    assert sets == probed > eng.K, "precondition: more sets than rows, all on the queue"
    assert launches == 1


# Built by the test against the internal header as it stands: no public path reaches the middle
# of a queued call, so the program holds one open and calls the engine from inside it.
_HELD_SRC = r"""
#include "gpap_internal.hpp"
#include <cstdio>
static void report(const char *what, gpap_status st) {
    std::printf("%s|%d|%s\n", what, (int)st, st == GPAP_OK ? "" : gpap_last_error());
}
int main(int, char **argv) {
    gpap_formula *f = nullptr;
    gpap_engine *e = nullptr;
    if (gpap_formula_from_dimacs(argv[1], &f) != GPAP_OK) return 2;
    if (gpap_engine_create(f, nullptr, &e) != GPAP_OK) return 3;
    {
        gpap::QueuedCall q(e);
        report("begin", q.begin(4, 1 << 16, 1, 4));
        std::printf("engaged|%d|\n", q.engaged() ? 1 : 0);
        const int64_t dec[4] = {0, 2, 4, 6};
        const int32_t order[1] = {0};
        int32_t rounds = 0;
        report("reset", gpap_engine_reset(e, 4));
        report("apply", gpap_engine_apply(e, dec, 1));
        report("apply_clean", gpap_engine_apply_clean(e, dec, 1));
        report("trajectory", gpap_engine_trajectory(e, order, 1, 1, 1, 5, 0, &rounds));
        report("recache_flush", gpap_engine_recache_flush(e));
        report("set_mode", gpap_engine_set_mode(e, GPAP_MODE_SCALAR));
        gpap::QueuedCall nested(e);
        report("nested_begin", nested.begin(4, 1 << 16, 1, 4));
        const int32_t jobs[2] = {0, 2};
        int8_t conflict[2];
        int32_t implied[2];
        report("launch", q.launch(jobs, 2, 1, conflict, implied));
    }
    report("reset_after", gpap_engine_reset(e, 4));
    gpap_engine_free(e);
    gpap_formula_free(f);
}
"""

_REFUSED_WHILE_HELD = ["reset", "apply", "apply_clean", "trajectory", "recache_flush",
                       "set_mode", "nested_begin"]


def _run_internal_program(tmp_path, name, src):
    """Compile `src` against the internal header and the library Python loads, run it on a small
    satisfiable formula, and return its `name|status|message` lines as {name: (status, msg)}."""
    import os
    import shlex
    import shutil
    import subprocess
    from pathlib import Path

    cxx = shlex.split(os.environ["CXX"]) if os.environ.get("CXX") else (
        [c] if (c := shutil.which("c++") or shutil.which("g++")) else None)
    if not cxx:
        pytest.skip("no C++ compiler")
    lib = next((c for c in _core._candidates() if c.exists()), None)   # the one Python loads
    if lib is None:
        pytest.skip("libgpap not built (scripts/build_core.sh)")
    root = Path(__file__).parents[2]
    (tmp_path / f"{name}.cpp").write_text(src)
    exe = tmp_path / name
    r = subprocess.run([*cxx, "-std=c++17", "-O1", f"-I{root / 'cpp/include'}",
                        f"-I{root / 'cpp/src'}", str(tmp_path / f"{name}.cpp"), "-o", str(exe),
                        str(lib), f"-Wl,-rpath,{lib.parent}"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    rng = random.Random(3)
    nv = 40
    clauses = random_cnf(rng, nv, 120, kmin=2, kmax=4)
    cnf = tmp_path / "f.cnf"
    cnf.write_text(f"p cnf {nv} {len(clauses)}\n" +
                   "".join(" ".join(map(str, c)) + " 0\n" for c in clauses))
    out = subprocess.run([str(exe), str(cnf)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return {n: (int(st), msg) for n, st, msg in
            (line.split("|", 2) for line in out.stdout.splitlines())}


def test_a_queued_call_refuses_every_other_engine_call_until_it_ends(tmp_path):
    """While a queued call holds the rows, the engine calls that would change them or the kernel
    (reset, both applies, a trajectory, recache_flush, set_mode, a second queued call) are
    refused, each naming itself, instead of silently breaking the baseline its launches
    restore to. The call's own launch still runs, and the engine takes calls again once the
    call ends."""
    got = _run_internal_program(tmp_path, "held", _HELD_SRC)
    assert got["begin"][0] == 0 and got["engaged"][0] == 1, ("precondition: the call engaged", got)
    served = {n: got[n] for n in _REFUSED_WHILE_HELD
              if got[n][0] == 0 or "queued call" not in got[n][1]}
    assert not served, ("served while a queued call held the rows", served)
    assert got["launch"][0] == 0, got["launch"]
    assert got["reset_after"][0] == 0, got["reset_after"]


_GROWTH_SRC = r"""
#include "gpap_internal.hpp"
#include <cstdio>
int main(int, char **argv) {
    gpap_formula *f = nullptr;
    gpap_engine *e = nullptr;
    if (gpap_formula_from_dimacs(argv[1], &f) != GPAP_OK) return 2;
    if (gpap_engine_create(f, nullptr, &e) != GPAP_OK) return 3;
    {
        gpap::QueuedCall q(e);
        const gpap_status st = q.begin(4, 1 << 16, 1, 4);
        std::printf("engaged|%d|\n", st == GPAP_OK && q.engaged() ? 1 : 0);
        const double no_deadline = 1e300;
        std::printf("first|%lld|\n", (long long)q.next_block(no_deadline));
        const int32_t jobs[2] = {0, 2};
        int8_t conflict[2];
        int32_t implied[2];
        std::printf("launch|%d|\n", (int)q.launch(jobs, 2, 1, conflict, implied));
        std::printf("second|%lld|\n", (long long)q.next_block(no_deadline));
    }
    gpap_engine_free(e);
    gpap_formula_free(f);
}
"""


def test_a_block_grows_from_the_launch_that_ran(tmp_path):
    """The growth rule judges the launch that ran: when a caller launches fewer sets than the
    block (cut where its matrix would pass kQueueMaxLits), the next block is eight times what
    ran, not eight times a block that never ran, so a check whose cut blocks reach narrow sets
    does not jump to a launch many times longer than any it timed."""
    got = _run_internal_program(tmp_path, "growth", _GROWTH_SRC)
    assert got["engaged"][0] == 1, ("precondition: the call engaged", got)
    assert got["first"][0] == 4 and got["launch"][0] == 0, got
    assert got["second"][0] == 16, got


def test_the_python_checker_leaves_the_queue_rows_to_the_core(monkeypatch):
    """A flags-only `AssumptionChecker.check_assumptions` pins no rows, as the C++ face's call
    does not, so the core sizes the queue's rows; a call that reads row state still runs the
    chunks the face sized. The face's chunk size is shrunk to 4 here, so a pin would show."""
    from gpap.engine.checker import AssumptionChecker

    monkeypatch.setenv("GPAP_QUEUE", "1")
    rng = random.Random(5)
    nv = 60
    ck = AssumptionChecker.from_clauses(nv, random_cnf(rng, nv, 150, kmin=2, kmax=4))
    monkeypatch.setattr(AssumptionChecker, "chunk_size", lambda self, n, m=1: 4)
    sets = [[v if rng.random() < 0.5 else -v for v in rng.sample(range(1, nv + 1), 3)]
            for _ in range(20000)]
    ck.check_assumptions(sets)
    core = ck.engine._core
    if core.flush_conflict:
        pytest.skip("precondition: this seed's formula must not be UNSAT at the root")
    assert core.queue_stats()[1] == len(sets), "precondition: the call ran on the queue"
    assert core.K == (_core.resident_warps() or 4096)
    ck.check_assumptions(sets[:10], assignments=True)
    assert core.K == 2      # chunks of 4, 4 and 2: the face's chunk size

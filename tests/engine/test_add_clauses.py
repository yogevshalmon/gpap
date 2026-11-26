"""gpap_engine_add_clauses (cpp/src/formula_state.cpp): clauses added to a live engine.

An add builds the grown formula beside the old one and propagates the new clauses on top of the
old baseline, so the engine must answer exactly as a new engine built on all the clauses would,
with every unit assumed before the add still assumed. Every comparison here is against such a
fresh engine, never against the add's own arithmetic.
"""
import gc
import random

import numpy as np
import pytest

from gpap.engine import _core
from gpap.engine.probe import LiteralProbe

pytestmark = pytest.mark.engine


@pytest.fixture(autouse=True)
def _no_limit(monkeypatch):
    """A memory limit and the reserve are process state: no test here may inherit one or leave
    one behind."""
    monkeypatch.delenv("GPAP_MEM_LIMIT", raising=False)
    monkeypatch.delenv("GPAP_MEM_RESERVE", raising=False)
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    yield
    if _core.available():
        _core.set_mem_limit(None, 0)
        _core.set_mem_reserve(None, 0)


def _planted(rng, nv, n, lens, model):
    """`n` random clauses of the given lengths, each satisfied by `model`, so any union of them
    stays satisfiable and no propagation here ends in a conflict (a conflicted fixpoint's state
    depends on the order the kernel met it)."""
    out = []
    while len(out) < n:
        vs = rng.sample(range(1, nv + 1), rng.choice(lens))
        c = [v if rng.random() < 0.5 else -v for v in vs]
        if any((lit > 0) == model[abs(lit)] for lit in c):
            out.append(c)
    return out


def _model(rng, nv):
    return {v: rng.random() < 0.5 for v in range(1, nv + 1)}


def _sets(rng, nv, n):
    return [[v if rng.random() < 0.5 else -v for v in rng.sample(range(1, nv + 1),
                                                                rng.randint(1, 4))]
            for _ in range(n)]


def _flat(sets):
    lits = np.array([x for s in sets for x in s], dtype=np.int32)
    offsets = np.cumsum([0] + [len(s) for s in sets]).astype(np.int64)
    return lits, offsets


def _engine(nv, clauses, **kw):
    return _core.Engine(_core.Formula.from_clauses(nv, clauses), **kw)


def _assert_same(a, b, sets):
    """`a` answers as `b` does: counts, the baseline (assignment and packed words), the free
    literals, and every set's conflict flag. A fixpoint is unique, so a set that does not
    conflict must also get the same propagation count, assignment and per-clause counts; a
    conflicted row's partial state is unspecified (under the OpenCL WAVE kernel not even
    repeatable), so it is not compared. Once reading row state (chunks), once flags only on a
    queue pinned to 16 rows, which restores rows from the baseline the add built."""
    assert (a.num_clauses, a.num_long, bool(a.flush_conflict)) == \
           (b.num_clauses, b.num_long, bool(b.flush_conflict))
    assert np.array_equal(a.flushed_assignment(), b.flushed_assignment())
    assert np.array_equal(a.free_literals(), b.free_literals())
    a.reset(1)
    b.reset(1)
    assert np.array_equal(a.packed_rows([0]), b.packed_rows([0])), "the baselines' words differ"
    lits, off = _flat(sets)
    ca, pa, xa, sa = a.check_sets(lits, off, assignments=True, sat_count=True)
    cb, pb, xb, sb = b.check_sets(lits, off, assignments=True, sat_count=True)
    assert np.array_equal(ca, cb)
    ok = ~ca
    for x, y in ((pa, pb), (xa, xb), (sa, sb)):
        assert np.array_equal(x[ok], y[ok])
    ca, pa, _, _ = a.check_sets(lits, off, chunk=16)
    cb, pb, _, _ = b.check_sets(lits, off, chunk=16)
    assert np.array_equal(ca, cb)
    assert np.array_equal(pa[~ca], pb[~ca])


_KINDS = {"unit": [1], "binary": [2], "long": [3, 4, 5], "mixed": [1, 2, 3, 4]}


@pytest.mark.parametrize("kind", list(_KINDS))
@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("shift", [16, 8])
def test_an_add_answers_as_a_rebuild(kind, mode, shift):
    """Engine(F0), assume U, add C answers as Engine(F0 + C), assume U: the new clauses keep
    the units assumed before them, their words are the ones the kernels would have kept, and
    what they force is propagated."""
    rng = random.Random(f"{kind}-{mode}-{shift}")      # a str seed is stable across runs
    nv = 60
    model = _model(rng, nv)
    f0 = _planted(rng, nv, 150, [1, 2, 2, 3, 3, 4], model)
    add = _planted(rng, nv, 40, _KINDS[kind], model)
    units = [v if model[v] else -v for v in rng.sample(range(1, nv + 1), 3)]
    opts = dict(mode=mode, force_shift=shift)

    a = _engine(nv, f0, **opts)
    a.assume_units(units)
    a.add_clauses(add)
    b = _engine(nv, f0 + add, **opts)
    b.assume_units(units)
    _assert_same(a, b, _sets(rng, nv, 200))


@pytest.mark.parametrize("bad", [[[61]], [[0, 1]], [[1, -62, 3]]],
                         ids=["past_num_vars", "zero_literal", "past_num_vars_in_a_clause"])
def test_a_refused_add_changes_nothing(bad):
    """A literal the formula cannot hold is refused before anything moves: the engine keeps its
    rows and answers as before."""
    rng = random.Random(5)
    nv = 60
    model = _model(rng, nv)
    f0 = _planted(rng, nv, 150, [1, 2, 3, 4], model)
    a = _engine(nv, f0)
    a.reset(4)
    with pytest.raises(_core.GpapError, match="gpap_engine_add_clauses"):
        a.add_clauses(bad)
    assert a.K == 4, "a refused add must not take the rows"
    _assert_same(a, _engine(nv, f0), _sets(rng, nv, 100))


def test_adding_nothing_changes_nothing():
    """n == 0 is a no-op in the core, as gpap_assume_units with no literal is: the rows stay."""
    rng = random.Random(6)
    nv = 40
    f0 = _planted(rng, nv, 100, [2, 3], _model(rng, nv))
    a = _engine(nv, f0)
    a.reset(4)
    _core.check(_core.load().gpap_engine_add_clauses(a._h, None, None, 0), "add nothing")
    a.add_clauses([])
    assert (a.K, a.num_clauses) == (4, len(f0))


@pytest.mark.parametrize("add", [[[1], [-1]], [[]]], ids=["contradictory_units", "empty_clause"])
def test_an_add_that_makes_the_formula_unsat_conflicts_every_row(add):
    """Clauses that make the formula unsat under its baseline leave it conflicted, as a fresh
    engine's flush would: every row conflicts, every set conflicts, and a queued sweep declines
    (it has no baseline to restore to)."""
    rng = random.Random(7)
    nv = 40
    f0 = _planted(rng, nv, 100, [2, 3], _model(rng, nv))
    a = _engine(nv, f0)
    a.add_clauses(add)
    assert a.flush_conflict
    a.reset(8)
    assert a.flags()[1].all(), "every row must start conflicted"
    lits, off = _flat(_sets(rng, nv, 50))
    assert a.check_sets(lits, off)[0].all()
    assert a.check_sets(lits, off, chunk=16)[0].all()     # more sets than rows: the queue's call
    assert a.queue_stats() == (0, 0), "a queued call has no baseline to restore to"
    assert bool(_engine(nv, f0 + add).flush_conflict), "precondition: a rebuild conflicts too"


@pytest.mark.parametrize("lens, ok", [([0], True), ([0, 1], False)], ids=["empty", "not_empty"])
def test_null_literals_are_taken_for_empty_clauses_only(lens, ok):
    """flat_lits may be NULL when every clause is empty (the C++ face passes an empty vector's
    data()): the empty clause makes the formula unsat. With a clause that has a literal, NULL is
    refused before anything moves."""
    import ctypes
    rng = random.Random(17)
    nv = 40
    a = _engine(nv, _planted(rng, nv, 80, [2, 3], _model(rng, nv)))
    a.reset(4)
    arr = np.array(lens, dtype=np.int32)
    st = _core.load().gpap_engine_add_clauses(
        a._h, None, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(lens))
    if ok:
        _core.check(st, "add an empty clause")
        assert a.flush_conflict
    else:
        assert st == 1, "GPAP_ERR_ARG"
        assert (a.K, bool(a.flush_conflict)) == (4, False)


def test_an_add_on_a_formula_that_already_conflicts_stays_conflicted():
    """A formula whose own units conflict stays conflicted whatever is added: its rows conflict
    and a fresh engine on all the clauses agrees."""
    rng = random.Random(16)
    nv = 40
    f0 = [[1], [-1]] + _planted(rng, nv, 80, [2, 3], _model(rng, nv))
    add = _planted(rng, nv, 20, [1, 2, 3], _model(rng, nv))
    a = _engine(nv, f0)
    assert a.flush_conflict, "precondition"
    a.add_clauses(add)
    assert a.flush_conflict and a.num_clauses == len(f0) + len(add)
    a.reset(4)
    assert a.flags()[1].all()
    assert bool(_engine(nv, f0 + add).flush_conflict)


def test_a_clause_the_split_cannot_hold_is_refused():
    """The split never changes with an add (N does not): on an engine pinned to the 8|24 split,
    a clause longer than its 255 literals is GPAP_ERR_ENVELOPE, and nothing changes."""
    rng = random.Random(8)
    nv = 400
    f0 = _planted(rng, nv, 300, [2, 3], _model(rng, nv))
    a = _engine(nv, f0, force_shift=8)
    a.reset(4)
    with pytest.raises(_core.EnvelopeError, match="packed split"):
        a.add_clauses([list(range(1, 301))])
    assert (a.K, a.num_clauses, a.shift) == (4, len(f0), 8)


def test_dispatch_after_an_add_is_a_fresh_engines():
    """AUTO dispatch keys off the clause count, so an add that crosses the WAVE threshold
    changes the kernel exactly as a rebuild's engine would choose it. Below the threshold a
    batch within the device's resident warps runs WAVE too, so K is the first that exceeds
    them."""
    rng = random.Random(9)
    nv = 300
    model = _model(rng, nv)
    f0 = _planted(rng, nv, 200, [2, 3, 4], model)
    add = _planted(rng, nv, 3000, [3, 4], model)
    K = next((k for k in (4096, 16384)
              if _core.dispatch_wave(len(f0), k) != _core.dispatch_wave(len(f0) + len(add), k)),
             None)
    if K is None:
        pytest.skip("on this device the add does not cross the dispatch threshold")
    a = _engine(nv, f0)
    a.add_clauses(add)
    b = _engine(nv, f0 + add)
    for k in (1, 64, K):
        a.reset(k)
        b.reset(k)
        assert a.mode == b.mode, f"K={k}"


def test_an_add_leaves_the_counters_alone():
    """The add's own flush is bookkeeping, as create's is: the counters and the reset counts a
    caller banks (mining's implications) read as they did before it."""
    rng = random.Random(10)
    nv = 60
    model = _model(rng, nv)
    f0 = _planted(rng, nv, 150, [2, 3, 4], model)
    a = _engine(nv, f0)
    lits, off = _flat(_sets(rng, nv, 64))
    a.check_sets(lits, off)
    before = (a.counters().ops, a.counters().props, a.counters().decs,
              a.counters().launches, a.reset_stats())
    a.add_clauses(_planted(rng, nv, 30, [1, 2], model))   # units and binaries: forced literals
    after = (a.counters().ops, a.counters().props, a.counters().decs,
             a.counters().launches, a.reset_stats())
    assert before == after
    assert before[1] > 0, "precondition: the check did work"


def test_an_add_ends_as_create_ends():
    """After an add the engine holds one row of the new width, as create leaves it, and a reset
    takes rows of that width."""
    rng = random.Random(11)
    nv = 60
    model = _model(rng, nv)
    f0 = _planted(rng, nv, 150, [2, 3, 4], model)
    add = _planted(rng, nv, 25, [3, 4], model)
    a = _engine(nv, f0)
    a.reset(16)
    a.add_clauses(add)
    assert a.K == 1
    assert a.num_long == _engine(nv, f0 + add).num_long
    a.reset(8)
    assert a.packed_rows(list(range(8))).shape == (8, a.num_long)


def test_an_add_that_does_not_fit_keeps_the_old_formula():
    """The new formula is built beside the old one, so under a limit that leaves no room for
    both the add fails with GPAP_ERR_MEMORY, and the engine keeps its old formula, holding no
    rows, with nothing of the new one left on the device."""
    rng = random.Random(12)
    nv = 600
    model = _model(rng, nv)
    f0 = _planted(rng, nv, 4000, [2, 3, 4], model)
    add = _planted(rng, nv, 400, [3, 4], model)
    gc.collect()
    b = _engine(nv, f0 + add)
    grown = b.mem_held_formula            # about the size the new formula takes beside the old
    del b
    gc.collect()
    a = _engine(nv, f0)
    a.trim()
    held = _core.mem_held(0)
    _core.set_mem_limit(f"{held + grown // 2}B", 0)
    with pytest.raises(_core.GpapMemoryError):
        a.add_clauses(add)
    assert a.K == 0, "a failed add leaves no rows"
    assert _core.mem_held(0) == held, "nothing of the new formula may stay counted"
    _core.set_mem_limit(None, 0)
    _assert_same(a, _engine(nv, f0), _sets(rng, nv, 100))


def test_an_add_whose_first_row_does_not_fit_puts_the_old_formula_back():
    """Under a limit that holds the new formula beside the old one but not a row of it, the add
    fails after the swap, in its reset to one row: the engine gets its old formula back,
    untouched, holds no rows, and nothing of the new formula stays counted. The limit is the
    trimmed engine plus what a twin holds in formula buffers after the same add: that covers the
    upload with the engine's own kept buffers to spare, and a row of the grown formula is wider
    than those."""
    rng = random.Random(18)
    nv = 600
    model = _model(rng, nv)
    f0 = _planted(rng, nv, 4000, [2, 3, 4], model)
    add = _planted(rng, nv, 400, [3, 4], model)
    gc.collect()
    twin = _engine(nv, f0)
    twin.add_clauses(add)
    grown = twin.mem_held_formula     # the engine's kept buffers plus the new formula
    del twin
    gc.collect()
    a = _engine(nv, f0)
    a.trim()
    held = _core.mem_held(0)
    _core.set_mem_limit(f"{held + grown}B", 0)
    with pytest.raises(_core.GpapMemoryError):
        a.add_clauses(add)
    assert (a.K, a.num_clauses) == (0, len(f0)), "the old formula back, and no rows"
    assert _core.mem_held(0) == held, "nothing of the new formula may stay counted"
    _core.set_mem_limit(None, 0)
    _assert_same(a, _engine(nv, f0), _sets(rng, nv, 100))


def test_a_rolled_back_add_keeps_the_kernel_choice():
    """The add resets to one row on the grown formula, which on AUTO resolves the kernel for it,
    then applies what the new clauses force. Failed there, the rollback puts back the kernel the
    engine had resolved for its old formula too. The limit is one byte below the least this add
    needs, so it fails at its last allocation, the forced literal's apply, after the reset. K is
    the first batch past the resident warps, where the old formula, below the WAVE threshold,
    runs SCALAR and one row WAVE, so the two choices differ."""
    rng = random.Random(19)
    nv = 300
    model = _model(rng, nv)
    f0 = _planted(rng, nv, 200, [2, 3, 4], model)
    add = _planted(rng, nv, 3000, [3, 4], model) + [[1 if model[1] else -1]]   # forces a literal
    K = next((k for k in (4096, 16384) if not _core.dispatch_wave(len(f0), k)), None)
    if K is None or not _core.dispatch_wave(len(f0), 1):
        pytest.skip("on this device one row and a full batch take the same kernel")

    def attempt(extra):
        """(added, the kernel before, the engine): the add under a limit `extra` bytes above
        the trimmed engine."""
        gc.collect()
        a = _engine(nv, f0)
        a.reset(K)
        before = a.mode
        a.trim()
        _core.set_mem_limit(f"{_core.mem_held(0) + extra}B", 0)
        try:
            a.add_clauses(add)
            return True, before, a
        except _core.GpapMemoryError:
            return False, before, a
        finally:
            _core.set_mem_limit(None, 0)

    lo, hi = 0, 1 << 26                  # no room fails; 64 MiB is plenty
    assert attempt(hi)[0] and not attempt(lo)[0]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if attempt(mid)[0]:
            hi = mid
        else:
            lo = mid
    ok, before, a = attempt(lo)
    assert not ok and before == "scalar", "precondition: the full batch runs SCALAR"
    assert (a.num_clauses, a.mode) == (len(f0), before)


def test_a_hundred_adds_leak_nothing():
    """Each add frees the formula it replaced: the device holds the engine's bytes and nothing
    else after every add, and nothing at all once the engine goes. The grown formula is a fresh
    engine's, without the packed initial words only a first flush reads."""
    rng = random.Random(13)
    nv = 80
    model = _model(rng, nv)
    f0 = _planted(rng, nv, 200, [2, 3, 4], model)
    gc.collect()
    base = _core.mem_held(0)
    a = _engine(nv, f0)
    added = []
    for _ in range(100):
        c = _planted(rng, nv, 1, [1, 2, 3, 4], model)
        a.add_clauses(c)
        added += c
        assert _core.mem_held(0) - base == a.mem_held
    fresh = _engine(nv, f0 + added)
    assert a.mem_held_formula == fresh.mem_held_formula - 4 * fresh.num_long
    del a, fresh
    gc.collect()
    assert _core.mem_held(0) == base


# ---------------------------------------------------------------- the faces

def test_the_faces_read_the_engines_counts_after_an_add():
    """Every readout sized from a count reads the engine's, not the formula it was built from:
    after an add, sat counts have a column per clause of the grown formula and the packed rows a
    word per long clause of it, and LiteralProbe's facts follow the add."""
    rng = random.Random(14)
    nv = 60
    model = _model(rng, nv)
    f0 = _planted(rng, nv, 150, [2, 3, 4], model)
    add = _planted(rng, nv, 20, [2, 3, 4], model)
    a = _engine(nv, f0)
    a.add_clauses(add)
    b = _engine(nv, f0 + add)
    lits, off = _flat(_sets(rng, nv, 32))
    conflict, _, _, sat_a = a.check_sets(lits, off, sat_count=True)
    assert sat_a.shape == (32, len(f0) + len(add))
    ok = ~conflict                      # a conflicted row's counts are unspecified
    assert np.array_equal(sat_a[ok], b.check_sets(lits, off, sat_count=True)[3][ok])
    a.reset(4)
    assert a.sat_rows([0, 1, 2, 3]).shape == (4, len(f0) + len(add))
    assert a.packed_rows([0]).shape == (1, b.num_long)

    p = LiteralProbe(nv, f0)
    v = int(p.free_variables()[0])
    u = v if model[v] else -v
    p.add_clauses([[u]])
    assert abs(u) not in p.free_variables().tolist() and not p.unsat
    assert p.num_clauses == len(f0) + 1
    p.add_clauses([[-u]])
    assert p.unsat


@pytest.mark.gpu             # the sampler's trajectory kernel is CUDA's
def test_a_sampler_refuses_after_its_engine_took_an_add():
    """A sampler is set up for the formula it was made on: the kernel it pins for that clause
    count, and a set_k batch for those rows. Once its engine took clauses, it refuses to run
    rather than run with them, and to take a batch, which would pin the engine's kernel
    again for the old formula's sampler."""
    rng = random.Random(15)
    nv = 60
    model = _model(rng, nv)
    a = _engine(nv, _planted(rng, nv, 150, [2, 3, 4], model))
    s = _core.Sampler(a, seed=1)
    s.run(10, [(0, 8)], max_chunks=1)
    a.add_clauses(_planted(rng, nv, 5, [3], model))
    with pytest.raises(_core.GpapError, match="gpap_engine_add_clauses"):
        s.run(10, [(0, 8)], max_chunks=1)
    mode = a.mode
    with pytest.raises(_core.GpapError, match="gpap_sampler_set_k.*gpap_engine_add_clauses"):
        s.set_k(64)
    assert a.mode == mode, "a refused set_k pins nothing"

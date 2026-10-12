"""The core sampling loop (`gpap_sampler_*`, cpp/src/sample.cpp), tested at the ABI through
gpap.engine._core, with no GPUSampler or pipeline.

Rows are verifiable models; a run is a deterministic function of the seed at a fixed batch
(every sampler here pins `k_cap`); delivery and the soundness self-test stay off the sampling
clock and the RNG stream; each stop valve reports which one fired.
"""
import random

import numpy as np
import pytest

from gpap.engine import _core
from tests.support import random_cnf, rows_satisfy

pytestmark = pytest.mark.gpu

B8 = (True, 8, False)                     # the classic single config


def _sampler(clauses, nv, *, k_cap=256, **kw):
    f = _core.Formula.from_clauses(nv, clauses)
    e = _core.Engine(f)
    return _core.Sampler(e, k_cap=k_cap, **kw)


# ---------------------------------------------------------------- soundness

def test_kept_rows_are_models():
    rng = random.Random(7)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    s = _sampler(clauses, 24, seed=3)
    st = s.run(500, [B8], keep_cap=-1)
    assert st.n_models > 0, "fixture produced nothing — the checks below would be vacuous"
    assert _core.SAMPLE_STOP[st.stop] == "target"
    got = s.bits(0)
    assert got.shape == (s.config_stats(0).kept_rows, 24)
    assert got.shape[0] >= min(st.n_models, 500) > 0
    assert rows_satisfy(got, clauses)


def test_unsat_at_flush_is_a_proof_not_a_failure():
    s = _sampler([[1], [-1]], 1)
    st = s.run(100, [B8])
    assert st.n_models == 0 and st.chunks == 0
    assert _core.SAMPLE_STOP[st.stop] == "unsat"


# ---------------------------------------------------------------- determinism

def test_same_seed_same_bits_and_reseed_restarts_the_stream():
    rng = random.Random(11)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    a = _sampler(clauses, 24, seed=9)
    b = _sampler(clauses, 24, seed=9)
    sa, sb = a.run(300, [B8], keep_cap=-1), b.run(300, [B8], keep_cap=-1)
    assert sa.n_models == sb.n_models and sa.chunks == sb.chunks
    assert np.array_equal(a.bits(0), b.bits(0)), "same seed produced different rows"
    first = a.bits(0)
    a.reseed(9)
    sa2 = a.run(300, [B8], keep_cap=-1)
    assert sa2.n_models == sa.n_models and np.array_equal(a.bits(0), first), \
        "reseed must restore the exact stream (the warmup's determinism contract)"
    c = _sampler(clauses, 24, seed=10)
    c.run(300, [B8], keep_cap=-1)
    assert not np.array_equal(c.bits(0), first), "different seeds produced identical rows"


def test_verify_and_delivery_do_not_touch_the_sampling_stream():
    """verify and keep_cap change nothing about which models a seed finds.

    The knobs are varied one at a time, and rows are compared, not only counts: two runs that
    drew different models can still agree on how many."""
    rng = random.Random(13)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    base = _sampler(clauses, 24, seed=5)
    st0 = base.run(400, [B8], keep_cap=-1)
    assert st0.n_models > 0, "fixture produced nothing — the comparisons would be vacuous"
    rows0 = base.bits(0)

    # (a) verify on: same stream, same rows.
    v = _sampler(clauses, 24, seed=5)
    stv = v.run(400, [B8], keep_cap=-1, verify=True)
    assert (stv.n_models, stv.chunks) == (st0.n_models, st0.chunks)
    assert np.array_equal(v.bits(0), rows0), "the soundness self-test moved the sampling stream"

    # (b) delivery off: no gather, same run.
    plain = _sampler(clauses, 24, seed=5).run(400, [B8])
    assert (plain.n_models, plain.chunks) == (st0.n_models, st0.chunks)


# ---------------------------------------------------------------- accounting

def test_verify_covers_every_model_and_bills_apart():
    rng = random.Random(17)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    s = _sampler(clauses, 24, seed=5)
    st = s.run(400, [B8], keep_cap=-1, verify=True)
    assert st.n_models > 0
    assert st.sound_rows_checked == st.n_models, "the verdict must cover EVERY emitted row"
    assert st.unsound_rows == 0
    assert st.verify_s >= 0 and st.deliver_s >= 0
    off = _sampler(clauses, 24, seed=5).run(400, [B8])
    assert off.sound_rows_checked == 0, "verify off must report no evidence, not fake some"


def test_keep_cap_bounds_the_bucket_not_the_count():
    rng = random.Random(19)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    s = _sampler(clauses, 24, seed=5)
    st = s.run(400, [B8], keep_cap=50)
    assert st.n_models > 50, "fixture too small to exercise the cap"
    assert s.config_stats(0).kept_rows == 50
    got = s.bits(0, rows=7)                       # capacity below kept: the first rows, in order
    assert got.shape == (7, 24)
    assert np.array_equal(got, s.bits(0)[:7]), "a short read must truncate, not resample"


def test_keep_cap_zero_never_gathers():
    rng = random.Random(23)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    s = _sampler(clauses, 24, seed=5)             # keep_cap=0: the timed-run spelling
    st = s.run(400, [B8])
    assert st.n_models > 0
    assert s.config_stats(0).kept_rows == 0 and st.deliver_s == 0.0


# ---------------------------------------------------------------- stop valves

def test_stop_valves_report_which_fired():
    rng = random.Random(29)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    st = _sampler(clauses, 24, seed=5).run(10**9, [B8], max_chunks=2)
    assert _core.SAMPLE_STOP[st.stop] == "chunks" and st.chunks == 2
    st = _sampler(clauses, 24, seed=5).run(10**9, [B8], time_limit_s=1e-9)
    assert _core.SAMPLE_STOP[st.stop] == "time" and st.timed_out == 1


def test_the_give_up_rule_is_opt_in_and_the_budget_is_spent_without_it():
    """`grace_chunks` is the OPT-IN early stop, and 0 (its `{0}` default) means run the budget.

    It defaulted to 2, so a dry run ended after the cold start plus two chunks whatever budget it
    was handed -- ~2 s of a 60 s campaign cell, where three of six of the published zero cells
    did have samples. A budget is a promise about how long the call may take, not a ceiling the
    loop may decline to approach."""
    # UNSAT with no unit clauses: the flush cannot prove it, every trajectory conflicts.
    clauses = [[1, 2], [1, -2], [-1, 2], [-1, -2]]
    st = _sampler(clauses, 2, seed=5).run(100, [B8], grace_chunks=2)
    assert st.n_models == 0
    assert _core.SAMPLE_STOP[st.stop] == "zero_yield"
    assert st.chunks == 3, "single config: 1 cold chunk + grace 2"
    assert st.retried == 0
    # ...and by DEFAULT the same dry formula runs until a budget stops it, not before.
    st = _sampler(clauses, 2, seed=5).run(100, [B8], max_chunks=25)
    assert st.n_models == 0
    assert _core.SAMPLE_STOP[st.stop] == "chunks" and st.chunks == 25, \
        "with grace_chunks unset the run must spend the chunk budget, not abandon it at 3"


@pytest.mark.parametrize("soft0", [False, True], ids=["off-to-on", "on-to-off"])
def test_retry_with_lookahead_flips_once_then_stops(soft0):
    """The zero-yield retry flips the soft polarity once rather than turning it on.

    Parametrized because only the on-to-off direction tells a flip from `soft || retry`."""
    clauses = [[1, 2], [1, -2], [-1, 2], [-1, -2]]
    s = _sampler(clauses, 2, seed=5)
    st = s.run(100, [(True, 8, soft0)], retry_with_lookahead=True, grace_chunks=2)
    assert _core.SAMPLE_STOP[st.stop] == "zero_yield"
    assert st.retried == 1
    assert st.lookahead_used == (0 if soft0 else 1), "the retry must run the OTHER polarity"
    assert st.chunks == 6, "one full window under each polarity"
    # THE FLIP IS NOT THE GIVE-UP RULE. It has its own window, so an opt-in a caller DID ask for
    # keeps working now that the give-up rule defaults to off -- otherwise turning one off would
    # have silently disabled the other.
    st = _sampler(clauses, 2, seed=5).run(100, [(True, 8, soft0)],
                                          retry_with_lookahead=True, max_chunks=12)
    assert st.retried == 1, "the polarity flip must still fire with grace_chunks unset"
    assert _core.SAMPLE_STOP[st.stop] == "chunks"


# ---------------------------------------------------------------- portfolio

def test_cold_start_measures_every_config_and_buckets_stay_per_config():
    rng = random.Random(31)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    s = _sampler(clauses, 24, seed=5)
    st = s.run(1, [(True, 8, False), (True, 1, False), (False, 1, False)], keep_cap=100)
    for i in range(3):
        assert s.config_stats(i).chunks >= 1, f"config {i} never measured (cold start hole)"
    assert 0 <= st.best_config < 3 and 0 <= st.dominant_config < 3
    win = s.bits(-1)
    assert np.array_equal(win, s.bits(st.best_config)), "bits(-1) must be the winner's bucket"
    total_kept = sum(s.config_stats(i).kept_rows for i in range(3))
    assert total_kept <= 3 * 100 and win.shape[0] <= 100


def test_min_measure_chunks_extends_past_the_target():
    rng = random.Random(37)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    s = _sampler(clauses, 24, seed=5)
    s.run(1, [(True, 8, False), (True, 1, False)], min_measure_chunks=3)
    assert max(s.config_stats(i).chunks for i in range(2)) >= 3, \
        "the headline rate would rest on a single noisy chunk"


# ---------------------------------------------------------------- support + columns

def test_support_phase_and_columns():
    rng = random.Random(41)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    s = _sampler(clauses, 24, seed=5)
    s.set_support(list(range(8)))                 # phase 1 = vars 0..7
    st = s.run(300, [B8], keep_cap=-1)
    assert st.n_models > 0
    assert st.n_support_complete >= st.n_models, \
        "support completion is implied by full completion, never rarer"
    s.set_columns([5, 0, 7])
    s.reseed(5)                                   # same stream as a fresh seed-5 sampler
    st = s.run(300, [B8], keep_cap=-1)
    got = s.bits(0)
    assert got.shape[1] == 3
    full = _sampler(clauses, 24, seed=5)
    full.set_support(list(range(8)))
    full.run(300, [B8], keep_cap=-1)
    ref = full.bits(0)
    assert np.array_equal(got, ref[:, [5, 0, 7]]), \
        "columns must be a pure re-projection of the same models (same seed, same stream)"


def test_support_only_stops_after_phase_one():
    """`support_only` decides phase 1 and nothing else.

    A row counts only when fully assigned. If the support determines the rest, results are
    unchanged; if not, nothing completes and the loop gives up instead of emitting a
    half-assigned row. Both outcomes are checked.
    """
    # (a) x3 == x1 and x2, x4 == x3 == x5: deciding {x1, x2} propagates the whole tail, so
    # skipping phase 2 costs nothing.
    det = [[-1, -2, 3], [1, -3], [2, -3], [3, -4], [-3, 4], [4, -5], [-4, 5]]
    for so in (False, True):
        s = _sampler(det, 5, seed=5)
        s.set_support([0, 1])
        st = s.run(200, [B8], keep_cap=-1, support_only=so, max_chunks=4)
        assert st.n_models == st.attempts > 0, \
            f"support_only={so}: every trajectory should complete on a determined tail"
        assert rows_satisfy(s.bits(0), det)
    # Not compared row for row: the phase-2 shuffle draws from the same stream, so the two
    # settings reach different trajectory seeds and find different models. (With a one-variable
    # tail they happen to match, which would let an equality check pass for the wrong reason.)

    # (b) a clause the support does not reach: only phase 2 assigns it.
    loose = [[1, 2], [3, 4]]
    s = _sampler(loose, 4, seed=5)
    s.set_support([0, 1])
    st = s.run(200, [B8], support_only=True, max_chunks=4, grace_chunks=2)
    assert st.n_models == 0, "a row with free variables left is not a model and must not count"
    assert st.n_support_complete > 0, "...but the support phase itself did complete"
    assert _core.SAMPLE_STOP[st.stop] == "zero_yield"
    full = _sampler(loose, 4, seed=5)
    full.set_support([0, 1])
    assert full.run(200, [B8], max_chunks=4).n_models > 0, \
        "precondition: with phase 2 the same instance delivers"


def test_zero_k_cap_means_the_library_cap_not_uncapped():
    """`k_cap=0` resolves to the library cap, not to uncapped (the `{0}` rule).

    `gpap_sampler_create` reads 0 as the library cap, >0 verbatim, <0 uncapped. The wrong
    reading fails silently: the run works, at a batch nobody chose.
    """
    rng = random.Random(67)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    f = _core.Formula.from_clauses(24, clauses)
    e = _core.Engine(f)
    zero = _core.Sampler(e, seed=1)                        # the {0} spelling
    named = _core.Sampler(e, seed=1, k_cap=_core.default_k_cap())
    uncapped = _core.Sampler(e, seed=1, k_cap=-1)
    assert zero.k == named.k == _core.default_k_cap()
    assert uncapped.k > zero.k, \
        "fixture: uncapped must differ, or 'zero means the cap' proves nothing here"


def test_set_k_pins_the_batch():
    rng = random.Random(43)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    s = _sampler(clauses, 24, seed=5)
    auto = s.k
    s.set_k(64)
    assert s.k == 64
    st = s.run(200, [B8])
    assert st.attempts % 64 == 0
    s.set_k(0)
    assert s.k == auto


def test_a_guard_reports_its_own_failure_not_the_previous_one():
    """An argument error reports its own message, not the previous call's.

    `gpap_last_error()` is a thread-local that nothing clears, and both faces append it to every
    non-OK status, so a guard returning a bare code would surface a stale message, even one from
    another sampler.
    """
    rng = random.Random(53)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    stale = _sampler(clauses, 24, seed=5)
    with pytest.raises(Exception) as first:                 # noqa: PT011 -- seeds the thread-local
        stale.set_columns([0, 999])
    assert "set_columns" in str(first.value)

    s = _sampler(clauses, 24, seed=5)
    with pytest.raises(Exception) as e:                     # noqa: PT011
        s.config_stats(0)                                   # no run yet
    assert "config_stats" in str(e.value) and "999" not in str(e.value), str(e.value)

    s.run(200, [B8], keep_cap=5)
    with pytest.raises(Exception) as e2:                    # noqa: PT011
        s.bits(9, rows=1)                                   # only one config exists
    assert "out of range" in str(e2.value) and "999" not in str(e2.value), str(e2.value)


def test_freeing_the_sampler_does_not_touch_a_dead_engine():
    """`gpap_sampler_free` does not dereference the engine it borrows.

    The cycle collector and interpreter shutdown can finalize the engine before the sampler.
    Freeing the engine handle first reproduces that without depending on gc order; under a
    sanitizer build that is the whole test."""
    rng = random.Random(59)
    clauses = random_cnf(rng, 24, 36, kmin=3)
    f = _core.Formula.from_clauses(24, clauses)
    e = _core.Engine(f)
    s = _core.Sampler(e, k_cap=256, seed=5)
    s.run(200, [B8])                       # a run resolves AUTO, so the sampler holds a pin
    lib = _core.load()
    lib.gpap_engine_free(e._h)             # free the engine first, as the collector can
    e._h = None                            # so its own __del__ is a no-op
    del s                                  # must free cleanly, reading nothing through e
    import gc
    gc.collect()


def test_the_sampler_never_overrides_an_explicit_kernel_pin():
    """The sampler pins the kernel to its own batch, but never over a caller's pin.

    After the sampler pins, `gpap_engine_auto` is false for its own pin too, so the sampler
    remembers which kernel it set and stands down once the engine holds a different one.
    """
    rng = random.Random(61)
    clauses = random_cnf(rng, 24, 36, kmin=3)                    # tiny M: scalar at a big K
    f = _core.Formula.from_clauses(24, clauses)
    e = _core.Engine(f)
    assert e.auto_dispatch, "fixture: the engine must start on AUTO"
    s = _core.Sampler(e, seed=1)               # default cap => big batch => scalar for this M
    assert e.mode == "scalar" and not e.auto_dispatch, e.mode

    e.set_mode("wave")                         # the caller re-pins after the sampler
    s.set_k(1024)
    assert e.mode == "wave", "set_k re-dispatched over a caller's pin"
    s.run(100, [B8])
    assert e.mode == "wave", "a run re-dispatched over a caller's pin"

    f2 = _core.Formula.from_clauses(24, clauses)   # pinned before the sampler exists
    e2 = _core.Engine(f2)
    e2.set_mode("wave")
    s2 = _core.Sampler(e2, seed=1)
    s2.run(100, [B8])
    assert e2.mode == "wave", "a pre-existing pin was overridden"

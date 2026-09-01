"""The core sampling loop's own contract (`gpap_sampler_*`, cpp/src/sample.cpp), at the ABI
level through gpap.engine._core — no GPUSampler, no pipeline.

The loop this exercises used to be written twice (gpap::Sampler and Python's GPUSampler); it now
has ONE home, and these tests pin the properties both faces rely on: rows are externally
verifiable models, a run is a deterministic function of the seed, delivery and the soundness
self-test stay OFF the sampling clock (and off the RNG stream), and each stop valve reports
which valve fired.
"""
import random

import numpy as np
import pytest
import torch

from gpap.engine import _core

CUDA = torch.cuda.is_available()
HAVE_CORE = _core.available()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not CUDA, reason="requires CUDA"),
    pytest.mark.skipif(not HAVE_CORE, reason="libgpap not built (see cpp/README.md)"),
]

B8 = (True, 8, False)                     # the classic single config


def _sat_cnf(rng, nv=24, nc=36, kmin=3, kmax=4):
    """Random low-ratio CNF with clauses of >= 3 literals: almost surely satisfiable, decent
    success rate, and never UNSAT-at-flush (no units) — see test_core._random_cnf on kmin."""
    out = []
    for _ in range(nc):
        vs = rng.sample(range(1, nv + 1), rng.randint(kmin, kmax))
        out.append([v if rng.random() < 0.5 else -v for v in vs])
    return out


def _sampler(clauses, nv, *, k_cap=256, **kw):
    f = _core.Formula.from_clauses(nv, clauses)
    e = _core.Engine(f)
    return _core.Sampler(e, k_cap=k_cap, **kw)


def _rows_satisfy(bits, clauses):
    """Every row (columns = all vars, bit j == var j+1 TRUE) must satisfy every clause."""
    for r in range(bits.shape[0]):
        row = bits[r]
        for cl in clauses:
            if not any((row[abs(l) - 1] == 1) == (l > 0) for l in cl):
                return False
    return True


# ---------------------------------------------------------------- soundness

def test_kept_rows_are_models():
    rng = random.Random(7)
    clauses = _sat_cnf(rng)
    s = _sampler(clauses, 24, seed=3)
    st = s.run(500, [B8], keep_cap=-1)
    assert st.n_models > 0, "fixture produced nothing — the checks below would be vacuous"
    assert _core.SAMPLE_STOP[st.stop] == "target"
    got = s.bits(0)
    assert got.shape == (s.config_stats(0).kept_rows, 24)
    assert got.shape[0] >= min(st.n_models, 500) > 0
    assert _rows_satisfy(got, clauses)


def test_unsat_at_flush_is_a_proof_not_a_failure():
    s = _sampler([[1], [-1]], 1)
    st = s.run(100, [B8])
    assert st.n_models == 0 and st.chunks == 0
    assert _core.SAMPLE_STOP[st.stop] == "unsat"


# ---------------------------------------------------------------- determinism

def test_same_seed_same_bits_and_reseed_restarts_the_stream():
    rng = random.Random(11)
    clauses = _sat_cnf(rng)
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
    """verify=1 and keep_cap>0 must change NOTHING about which models a seed finds — the
    delivery-shares-the-RNG bug class (`--out` silently changing the run) pinned at the core."""
    rng = random.Random(13)
    clauses = _sat_cnf(rng)
    plain = _sampler(clauses, 24, seed=5).run(400, [B8])
    loaded = _sampler(clauses, 24, seed=5).run(400, [B8], keep_cap=-1, verify=True)
    assert (plain.n_models, plain.chunks) == (loaded.n_models, loaded.chunks)


# ---------------------------------------------------------------- accounting

def test_verify_covers_every_model_and_bills_apart():
    rng = random.Random(17)
    clauses = _sat_cnf(rng)
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
    clauses = _sat_cnf(rng)
    s = _sampler(clauses, 24, seed=5)
    st = s.run(400, [B8], keep_cap=50)
    assert st.n_models > 50, "fixture too small to exercise the cap"
    assert s.config_stats(0).kept_rows == 50
    got = s.bits(0, rows=7)                       # capacity below kept: prefix, tail untouched
    assert got.shape == (7, 24)


def test_keep_cap_zero_never_gathers():
    rng = random.Random(23)
    clauses = _sat_cnf(rng)
    s = _sampler(clauses, 24, seed=5)             # keep_cap=0: the timed-run spelling
    st = s.run(400, [B8])
    assert st.n_models > 0
    assert s.config_stats(0).kept_rows == 0 and st.deliver_s == 0.0


# ---------------------------------------------------------------- stop valves

def test_stop_valves_report_which_fired():
    rng = random.Random(29)
    clauses = _sat_cnf(rng)
    st = _sampler(clauses, 24, seed=5).run(10**9, [B8], max_chunks=2)
    assert _core.SAMPLE_STOP[st.stop] == "chunks" and st.chunks == 2
    st = _sampler(clauses, 24, seed=5).run(10**9, [B8], time_limit_s=1e-9)
    assert _core.SAMPLE_STOP[st.stop] == "time" and st.timed_out == 1


def test_zero_yield_gives_up_after_cold_start_plus_grace():
    # UNSAT with no unit clauses: the flush cannot prove it, every trajectory conflicts.
    clauses = [[1, 2], [1, -2], [-1, 2], [-1, -2]]
    s = _sampler(clauses, 2, seed=5)
    st = s.run(100, [B8])
    assert st.n_models == 0
    assert _core.SAMPLE_STOP[st.stop] == "zero_yield"
    assert st.chunks == 3, "single config: 1 cold chunk + grace 2, the sample_bits rule"
    assert st.retried == 0


@pytest.mark.parametrize("soft0", [False, True], ids=["off-to-on", "on-to-off"])
def test_retry_with_lookahead_flips_once_then_stops(soft0):
    """The retry FLIPS the risk polarity — it does not turn it on.

    Both headers say "flipped", but the loop computed `cfg.soft || retry_soft`, so a config
    already running soft retried with the IDENTICAL polarity: a second full window of the same
    trajectory, reported as though the other one had been tried. Parametrized because only the
    on->off direction can tell the two implementations apart."""
    clauses = [[1, 2], [1, -2], [-1, 2], [-1, -2]]
    s = _sampler(clauses, 2, seed=5)
    st = s.run(100, [(True, 8, soft0)], retry_with_lookahead=True)
    assert _core.SAMPLE_STOP[st.stop] == "zero_yield"
    assert st.retried == 1
    assert st.lookahead_used == (0 if soft0 else 1), "the retry must run the OTHER polarity"
    assert st.chunks == 6, "one full window under each polarity"


# ---------------------------------------------------------------- portfolio

def test_cold_start_measures_every_config_and_buckets_stay_per_config():
    rng = random.Random(31)
    clauses = _sat_cnf(rng)
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
    clauses = _sat_cnf(rng)
    s = _sampler(clauses, 24, seed=5)
    s.run(1, [(True, 8, False), (True, 1, False)], min_measure_chunks=3)
    assert max(s.config_stats(i).chunks for i in range(2)) >= 3, \
        "the headline rate would rest on a single noisy chunk"


# ---------------------------------------------------------------- support + columns

def test_support_phase_and_columns():
    rng = random.Random(41)
    clauses = _sat_cnf(rng)
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


def test_set_k_pins_the_batch():
    rng = random.Random(43)
    clauses = _sat_cnf(rng)
    s = _sampler(clauses, 24, seed=5)
    auto = s.k
    s.set_k(64)
    assert s.k == 64
    st = s.run(200, [B8])
    assert st.attempts % 64 == 0
    s.set_k(0)
    assert s.k == auto

"""The coverage run's host-RAM bound (`pipeline.cov_batch`).

Collected models are rebuilt as Python ints at `COV_BYTES_PER_CELL` per (row x variable), so the
bound is on rows x nv, not rows alone. Kept rows accumulate over the run's `cov_chunks` chunks,
so the budget is divided by them. The cap must not change `distinct_frac`, the number the
coverage run exists to measure; the GPU test at the bottom checks that.
"""
import os

import pytest

from gpap.sampling.pipeline import COV_BYTES_PER_CELL, COV_HOST_BUDGET, cov_batch

GIB = 1 << 30


def test_a_row_count_alone_is_not_a_memory_bound():
    """Same row ceiling, two formulas: the larger one gets a smaller batch, and both fit."""
    small, big = 12_926, 413_451                    # b13 and b14, Tseitin at T=25
    assert cov_batch(small, 2048) > cov_batch(big, 2048), \
        "the batch must SHRINK as nv grows — that is the entire fix"
    for nv in (small, big):
        peak = cov_batch(nv, 2048) * 4 * nv * COV_BYTES_PER_CELL
        assert peak <= COV_HOST_BUDGET, f"nv={nv:,} would peak at {peak / 1e9:.1f} GB"


@pytest.mark.parametrize("nv", [1, 1_000, 12_926, 413_451, 4_749_526, 50_000_000])
def test_the_budget_holds_at_every_size_and_the_batch_is_never_zero(nv):
    """The batch is never 0: a formula too large for the budget still collects a small sample,
    because `no-coverage-sample` and `not-requested` are different results consumers branch on."""
    n = cov_batch(nv, 2048)
    assert n >= 1
    assert n <= 2048, "cov_k_cap stays an upper bound"
    assert n * 4 * nv * COV_BYTES_PER_CELL <= COV_HOST_BUDGET or n == 1


def test_the_chunk_multiplier_is_applied():
    """Kept rows accumulate over `cov_chunks` chunks, so the budget is divided by them."""
    nv = 100_000
    # exact up to the floor division (budget // (per_row * chunks)), so allow one row of slack
    assert abs(cov_batch(nv, 2048, cov_chunks=8) * 2 - cov_batch(nv, 2048, cov_chunks=4)) <= 2
    assert abs(cov_batch(nv, 2048, cov_chunks=1) - 4 * cov_batch(nv, 2048, cov_chunks=4)) <= 4
    # not a no-op: 4 chunks give a materially smaller batch than 1
    assert cov_batch(nv, 2048, cov_chunks=4) < cov_batch(nv, 2048, cov_chunks=1) / 3


def test_a_small_formula_is_untouched():
    """The cap does not bind on small formulas, so their coverage samples keep their size."""
    assert cov_batch(1_000, 2048) == 2048
    assert cov_batch(3_000, 2048) == 2048


def test_the_budget_is_overridable_for_a_box_that_has_the_ram():
    assert cov_batch(413_451, 2048, budget=64 * GIB) > cov_batch(413_451, 2048, budget=2 * GIB)


@pytest.mark.gpu
def test_the_cap_does_not_move_the_diagnostic_it_protects(monkeypatch):
    """The cap does not change distinct_frac.

    Where the cap binds, the solution space dwarfs the sample, so the capped and uncapped runs
    must both report all rows distinct."""
    import gpap.sampling.pipeline as P
    from gpap.circuits.bench import build_cnf, parse_bench

    from tests.conftest import SEQ_DIR
    from tests.support import DEV
    src = os.path.join(SEQ_DIR, "b13.bench")
    b, sup, _tgt, _lab = build_cnf(parse_bench(src), 25, seed=5)
    nv, clauses = b.nv, [list(c) for c in b.clauses]
    assert cov_batch(nv) < 2048, \
        f"b13 has nv={nv:,}; the cap must BIND here or this test proves nothing"

    out = {}
    for label, budget in (("capped", 1 << 31), ("uncapped", 64 * GIB)):
        monkeypatch.setattr(P, "COV_HOST_BUDGET", budget)
        out[label] = P.sample_clauses(nv, clauses, sup, DEV,
                                      n_time=5_000, n_cov=1000, time_limit=15.0, seed=5)

    assert out["capped"]["n_collected"] < out["uncapped"]["n_collected"], \
        "the capped run must actually have collected fewer rows — else nothing was exercised"
    for label, r in out.items():
        assert r["n_collected"] > 0, f"{label} collected nothing"
        assert r["n_distinct"] == r["n_collected"], \
            f"{label}: distinct_frac moved off 1.0 ({r['n_distinct']}/{r['n_collected']})"

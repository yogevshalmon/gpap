"""The coverage run's host-RAM bound.

`sample_clauses(n_cov=...)` collects models and rebuilds them in Python: `sample()` turns the
kept (rows, N) bit matrix into a list of `nv` ints per row, and `_reconstruct` builds a second
one to map it back to original variables. MEASURED cost: **72 bytes per (row x variable)**.

`cov_k_cap` bounded ROWS, which is the same thing only while `nv` is small — and it was
calibrated on formulas of a few thousand variables. On a large one it is not a memory bound at
all: 2048 rows x 413,451 vars (b14 through the Tseitin encoding the rival path requires; the
abc:fast encoding the CMSGen table uses is 41,938) x 72 B
is 61 GB. The process was OOM-killed with no exception to catch, no row written, and every
instance after it in that sweep silently unmeasured — b14, b17, b18, b20, b21, b22, s35932 all
went missing from the Recurrent table arm this way.

So the bound is on the PRODUCT, and it is divided by `cov_chunks` because the kept rows
ACCUMULATE across the run (the coverage call asks the core for an unlimited keep_cap and
reconstructs every bucket at the end). Budgeting a single chunk left b14 at 5.2 GiB against a
2 GiB budget — right mechanism, wrong multiplier.

What the cap must NOT do is change the diagnostic it protects. `distinct_frac` is the whole
point of the coverage run, and it is 1.0 on every instance large enough for the cap to bind
(the solution space is astronomically larger than the sample), so collecting fewer rows costs
nothing. The GPU test at the bottom is that claim, measured rather than asserted.
"""
import os

import pytest

from gpap.sampling.pipeline import COV_BYTES_PER_CELL, COV_HOST_BUDGET, cov_batch

GIB = 1 << 30


def test_a_row_count_alone_is_not_a_memory_bound():
    """The regression in one line: the same row ceiling, two formulas, 32x the memory."""
    small, big = 12_926, 413_451                    # b13 and b14, Tseitin at T=25
    assert cov_batch(small, 2048) > cov_batch(big, 2048), \
        "the batch must SHRINK as nv grows — that is the entire fix"
    for nv in (small, big):
        peak = cov_batch(nv, 2048) * 4 * nv * COV_BYTES_PER_CELL
        assert peak <= COV_HOST_BUDGET, f"nv={nv:,} would peak at {peak / 1e9:.1f} GB"


@pytest.mark.parametrize("nv", [1, 1_000, 12_926, 413_451, 4_749_526, 50_000_000])
def test_the_budget_holds_at_every_size_and_the_batch_is_never_zero(nv):
    """Never 0: a formula too large for the budget must still collect SOMETHING, so the run
    reports a small coverage sample rather than none — `no-coverage-sample` is a different
    fact from `not-requested` and consumers branch on it."""
    n = cov_batch(nv, 2048)
    assert n >= 1
    assert n <= 2048, "cov_k_cap stays an upper bound"
    assert n * 4 * nv * COV_BYTES_PER_CELL <= COV_HOST_BUDGET or n == 1


def test_the_chunk_multiplier_is_applied():
    """Kept rows accumulate over `cov_chunks` chunks, so the divisor has to be there. Sabotage
    check: with the divisor dropped, this is the assertion that fires."""
    nv = 100_000
    # exact up to the floor division (budget // (per_row * chunks)), so allow one row of slack
    assert abs(cov_batch(nv, 2048, cov_chunks=8) * 2 - cov_batch(nv, 2048, cov_chunks=4)) <= 2
    assert abs(cov_batch(nv, 2048, cov_chunks=1) - 4 * cov_batch(nv, 2048, cov_chunks=4)) <= 4
    # and it is a real divisor, not a no-op: 4 chunks must buy a materially smaller batch
    assert cov_batch(nv, 2048, cov_chunks=4) < cov_batch(nv, 2048, cov_chunks=1) / 3


def test_a_small_formula_is_untouched():
    """The cap must not bind where it was never the problem, or every recorded small-instance
    coverage sample changes size for no reason."""
    assert cov_batch(1_000, 2048) == 2048
    assert cov_batch(3_000, 2048) == 2048


def test_the_budget_is_overridable_for_a_box_that_has_the_ram():
    assert cov_batch(413_451, 2048, budget=64 * GIB) > cov_batch(413_451, 2048, budget=2 * GIB)


@pytest.mark.gpu
def test_the_cap_does_not_move_the_diagnostic_it_protects():
    """distinct_frac is what the coverage run exists to measure. On a formula big enough for the
    cap to bind, the space is astronomically larger than the sample, so a smaller sample must
    give the SAME fraction — otherwise the cap is buying memory with accuracy."""
    import gpap.sampling.pipeline as P
    from gpap.circuits.bench import build_cnf, parse_bench

    from tests.conftest import SEQ_DIR
    from tests.support import DEV
    src = os.path.join(SEQ_DIR, "b13.bench")
    b, sup, _tgt, _lab = build_cnf(parse_bench(src), 25, seed=5)
    nv, clauses = b.nv, [list(c) for c in b.clauses]
    assert cov_batch(nv) < 2048, \
        f"b13 has nv={nv:,}; the cap must BIND here or this test proves nothing"

    saved, out = P.COV_HOST_BUDGET, {}
    try:
        for label, budget in (("capped", 1 << 31), ("uncapped", 64 * GIB)):
            P.COV_HOST_BUDGET = budget
            out[label] = P.sample_clauses(nv, clauses, sup, DEV,
                                          n_time=5_000, n_cov=1000, time_limit=15.0, seed=5)
    finally:
        P.COV_HOST_BUDGET = saved

    assert out["capped"]["n_collected"] < out["uncapped"]["n_collected"], \
        "the capped run must actually have collected fewer rows — else nothing was exercised"
    for label, r in out.items():
        assert r["n_collected"] > 0, f"{label} collected nothing"
        assert r["n_distinct"] == r["n_collected"], \
            f"{label}: distinct_frac moved off 1.0 ({r['n_distinct']}/{r['n_collected']})"

"""The uniformity statistics the evaluation layer runs on (`eval/uniformity/metrics.py`).

They are NOT part of the library: a uniformity number is a CLAIM about a distribution, and the
honest ones need either the whole solution set (enumeration) or conditional sampling (Barbarik).
What is pinned here is the machinery those claims rest on:

  chi2_sf          — against table values (no scipy dependency to compare to at runtime).
  uniformity_stats exact mode — accepts genuinely uniform multinomial counts, rejects a
                   planted skew (deterministic RNG; thresholds far from the noise floor).
  collision mode   — l2_ratio flags a heavy repeat pattern with |S| unknown; collision-free
                   samples report the birthday floor.
  enumerate_projections — the exact-mode input: complete vs capped, and it must say which.

The campaign that uses them end to end is tests/test_uniformity_fuzz.py (and, at full size,
eval/uniformity/fuzz_uniformity.py).
"""
import numpy as np
import pytest

from eval.uniformity.metrics import chi2_sf, enumerate_projections, uniformity_stats


# ------------------------------------------------------------------------------ chi2 survival
def test_chi2_sf_table_values():
    # canonical 5% critical values: chi2_k(x) = 0.05
    for k, x in [(1, 3.841), (2, 5.991), (5, 11.070), (10, 18.307), (50, 67.505)]:
        assert chi2_sf(x, k) == pytest.approx(0.05, abs=2e-3)
    assert chi2_sf(0.0, 3) == 1.0
    assert chi2_sf(1000.0, 3) < 1e-100


# ------------------------------------------------------------------------------ exact mode
def test_exact_mode_accepts_uniform_and_rejects_skew():
    rng = np.random.default_rng(0)
    S, n = 32, 32 * 200
    uni = rng.multinomial(n, [1.0 / S] * S)
    u = uniformity_stats(uni.tolist(), n, S=S)
    assert u["mode"] == "exact" and u["p"] > 0.01 and u["tv"] < 0.05

    w = np.array([3.0] * (S // 2) + [1.0] * (S // 2))     # 3:1 skew — far past the noise floor
    skew = rng.multinomial(n, w / w.sum())
    u2 = uniformity_stats(skew.tolist(), n, S=S)
    assert u2["p"] < 1e-6 and u2["tv"] > 0.2


def test_exact_mode_small_n_reports_tv_only():
    u = uniformity_stats([2, 1, 1], 4, S=64)       # n << 5*S
    assert u["mode"] == "exact" and u["p"] is None and u["tv"] > 0
    assert "chi2 p unreliable" in u["note"]


def test_single_solution_space():
    u = uniformity_stats([7], 7, S=1)
    assert u["mode"] == "exact" and u["p"] == 1.0 and u["tv"] == 0.0


def test_empty_sample_is_its_own_mode():
    """"nothing collected" is not "uniform" — a caller must be able to tell them apart."""
    u = uniformity_stats([], 0)
    assert u["mode"] == "empty" and u["n"] == 0


# ------------------------------------------------------------------------------ collision mode
def test_collision_mode_flags_heavy_skew_without_S():
    # 1000 samples: one value repeated 500x + 500 singletons -> l2_ratio >> 1 for ANY |S|
    counts = [500] + [1] * 500
    u = uniformity_stats(counts, 1000)
    assert u["mode"] == "collision" and u["l2_ratio"] > 100
    assert "NON-UNIFORM" in u["note"]


def test_collision_mode_collision_free_reports_birthday_floor():
    u = uniformity_stats([1] * 2000, 2000)
    assert u["collisions"] == 0 and u["birthday_floor"] > 600_000


def test_collision_mode_mild_repeats_consistent():
    # 2000 samples over ~1900 distinct with doubletons only — small l2_ratio, no flag
    counts = [2] * 100 + [1] * 1800
    u = uniformity_stats(counts, 2000)
    assert u["l2_ratio"] < 1.5 and "consistent" in u["note"]


# ------------------------------------------------------------------------------ enumeration
def test_enumerate_projections_complete():
    clauses, sup = [[1, 2], [-1, 3], [-2, -3]], [1, 2, 3]
    projs, status = enumerate_projections(clauses, sup)
    assert status == "complete"
    # ground truth by hand: (x1|x2)(~x1|x3)(~x2|~x3) has exactly the models 010 and 101
    assert projs == {(0, 1, 0), (1, 0, 1)}


def test_enumerate_projections_cap():
    # free 2^10 hypercube, cap 100 -> capped, not complete
    projs, status = enumerate_projections([[1, -1]], list(range(1, 11)), cap=100)
    assert status == "capped" and len(projs) == 100

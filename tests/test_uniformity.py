"""Uniformity readout (metrics.uniformity_stats + the pipeline wiring).

Statistical layers tested separately:
  chi2_sf        — against table values (no scipy dependency to compare to at runtime).
  uniformity_stats exact mode — accepts genuinely uniform multinomial counts, rejects a
                   planted skew (deterministic RNG; thresholds far from the noise floor).
  collision mode — l2_ratio flags a heavy repeat pattern with |S| unknown; collision-free
                   samples report the birthday floor.
  pipeline       — sample_clauses returns an exact-mode dict on a tiny CNF (needs CUDA — the
                   engine is packed-kernel-only) whose soundness cross-check passes; the
                   readout is OPT-IN (default off).
"""
import numpy as np
import pytest

from gpap.sampling.metrics import (chi2_sf, enumerate_projections, format_uniformity,
                                   uniformity_stats)

try:
    import torch as _torch
    from gpap.engine import core_available as _ka
    _KERNEL = _torch.cuda.is_available() and _ka()
except Exception:  # noqa: BLE001
    _KERNEL = False
needs_gpu = pytest.mark.skipif(not _KERNEL, reason="engine is CUDA+triton-only")


# ------------------------------------------------------------------------------ chi2 survival
def test_chi2_sf_table_values():
    # canonical 5% critical values: chi2_k(x) = 0.05
    for k, x in [(1, 3.841), (5, 11.070), (10, 18.307), (63, 82.529)]:
        assert chi2_sf(x, k) == pytest.approx(0.05, abs=2e-3)
    assert chi2_sf(0.0, 7) == 1.0
    assert chi2_sf(1000.0, 3) < 1e-12
    # median: chi2_k at ~k-2/3 gives ~0.5
    assert 0.4 < chi2_sf(9.34, 10) < 0.6


# ------------------------------------------------------------------------------ exact mode
def test_exact_mode_accepts_uniform_and_rejects_skew():
    rng = np.random.default_rng(11)
    S, n = 64, 6400
    uni = rng.multinomial(n, [1.0 / S] * S)
    u = uniformity_stats(uni.tolist(), n, S=S)
    assert u["mode"] == "exact" and u["p"] is not None
    assert u["p"] > 0.01 and u["tv"] < 0.08

    # planted skew: half the cells get 3x the mass of the other half
    w = np.array([3.0] * (S // 2) + [1.0] * (S // 2))
    skew = rng.multinomial(n, w / w.sum())
    u2 = uniformity_stats(skew.tolist(), n, S=S)
    assert u2["p"] < 1e-6 and u2["tv"] > 0.2
    assert "p=" in format_uniformity(u2)


def test_exact_mode_small_n_reports_tv_only():
    u = uniformity_stats([2, 1, 1], 4, S=64)       # n << 5*S
    assert u["mode"] == "exact" and u["p"] is None and u["tv"] > 0
    assert "TV=" in format_uniformity(u)


def test_single_solution_space():
    u = uniformity_stats([7], 7, S=1)
    assert u["mode"] == "exact" and u["p"] == 1.0 and u["tv"] == 0.0


# ------------------------------------------------------------------------------ collision mode
def test_collision_mode_flags_heavy_skew_without_S():
    # 1000 samples: one value repeated 500x + 500 singletons -> l2_ratio >> 1 for ANY |S|
    counts = [500] + [1] * 500
    u = uniformity_stats(counts, 1000)
    assert u["mode"] == "collision" and u["l2_ratio"] > 100
    assert "NON-UNIFORM" in u["note"]
    assert format_uniformity(u).startswith("SKEW")
    assert "SKEWED" in format_uniformity(u, verbose=True)


def test_collision_mode_collision_free_reports_birthday_floor():
    u = uniformity_stats([1] * 2000, 2000)
    assert u["collisions"] == 0 and u["birthday_floor"] > 600_000
    assert format_uniformity(u) == "OK"
    txt = format_uniformity(u, verbose=True)
    # the verbose line carries the scale number, not just a verdict
    assert "no repeated solution" in txt and f"{u['birthday_floor']:,}" in txt


def test_collision_hint_depends_on_why_S_is_unknown():
    """fast mode (exact never attempted) must point at --uniformity exact; exact mode that was
    defeated (cap hit, or formula over the M gate) must NOT (only barbarik can go further)."""
    u = uniformity_stats([1] * 500, 500)
    u["enum_status"] = "not-attempted"                 # fast mode
    assert "--uniformity exact" in format_uniformity(u, verbose=True)
    u["enum_status"] = "capped"                        # exact mode, space defeated enumeration
    txt = format_uniformity(u, verbose=True)
    assert "too large to enumerate" in txt and "--uniformity exact" not in txt
    u["enum_status"] = "skipped-large"                 # exact mode, formula over the 200k-M gate
    txt = format_uniformity(u, verbose=True)
    assert "M > 200k" in txt and "--uniformity exact" not in txt


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


# ------------------------------------------------------------------------------ pipeline wiring
@needs_gpu
def test_pipeline_exact_uniformity_on_tiny_cnf():
    import torch
    from gpap.sampling.pipeline import sample_clauses
    clauses, sup0 = [[1, 2], [-1, 3], [-2, -3]], [0, 1, 2]
    info = sample_clauses(3, clauses, sup0, torch.device("cuda"), n_time=400, n_cov=0,
                          uniformity="exact", configs=None,
                          time_limit=20.0, mine=False)
    u = info["uniformity"]
    assert u is not None and u["mode"] == "exact" and u["S"] == 2
    assert not u.get("unsound_projections")
    assert u["n"] >= 20 * u["S"]                   # 5*S floor comfortably met
    assert u["tv"] < 0.35                          # sane distribution over the 2 solutions
    assert info["phases"]["uniformity"] < 10.0


@needs_gpu
def test_pipeline_default_is_off():
    import torch
    from gpap.sampling.pipeline import sample_clauses
    info = sample_clauses(3, [[1, 2]], [0, 1], torch.device("cuda"), n_time=100, n_cov=0,
                          configs=None, time_limit=10.0, mine=False)
    # the readout is OPT-IN (it costs an extra sampling pass): no uniformity kwarg -> no readout
    assert info["uniformity"] is None
    assert info["phases"]["uniformity"] == 0.0


@needs_gpu
def test_pipeline_fast_mode_is_solver_free():
    import torch
    from gpap.sampling.pipeline import sample_clauses
    clauses, sup0 = [[1, 2], [-1, 3], [-2, -3]], [0, 1, 2]
    info = sample_clauses(3, clauses, sup0, torch.device("cuda"), n_time=400, n_cov=0,
                          uniformity="fast", configs=None, time_limit=20.0,
                          mine=False)
    u = info["uniformity"]
    # fast: no enumeration attempted, collision/repeat readout only
    assert u is not None and u["mode"] == "collision"
    assert u["enum_status"] == "not-attempted"
    assert u["l2_ratio"] < 1.5                     # uniform-ish sampler on a 2-solution space


@needs_gpu
def test_pipeline_uniformity_disabled():
    import torch
    from gpap.sampling.pipeline import sample_clauses
    info = sample_clauses(3, [[1, 2]], [0, 1], torch.device("cuda"), n_time=100, n_cov=0,
                          uniformity="off", configs=None, time_limit=10.0,
                          mine=False)
    assert info["uniformity"] is None


# ------------------------------------------------------------------------------ 2-wise (vectorized)
def test_two_wise_vectorized_matches_bruteforce():
    import itertools
    import random
    from gpap.sampling.metrics import two_wise_coverage
    rng = random.Random(9)
    support = [2, 5, 7, 9]                          # 1-based vars, non-contiguous
    models = [[v if rng.random() < 0.5 else -v for v in support] for _ in range(40)]
    got = two_wise_coverage(models, support)
    # brute-force reference: a pair is covered iff all 4 joint polarities appear
    covered = total = 0
    for a, b in itertools.combinations(range(len(support)), 2):
        total += 1
        seen = {(support[a] in set(m), support[b] in set(m)) for m in models}
        covered += len(seen) == 4
    assert got == pytest.approx(100.0 * covered / total)
    assert two_wise_coverage([], support) == 0.0
    assert two_wise_coverage(models, [3]) == 0.0    # S < 2

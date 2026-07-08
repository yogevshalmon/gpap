"""A/B tests for the MONO (1-byte count-only) trajectory kernel (gpap/solver/bcp_triton_wide.py).

Mono must be trajectory-identical to every other variant: the decision order, hash polarity and
propagation order are shared, and the unit-time clause scan recovers exactly the literal the XOR
trick would. Same seed => the SAME model set as the 16|16 packed, 8|24 wide and unpacked kernels.
"""
import random

import pytest
import torch

from gpap.solver.bcp_triton import TritonBCP
from gpap.solver.sampler import GPUSampler

CUDA = torch.cuda.is_available()
AVAIL = CUDA and TritonBCP.available()
pytestmark = pytest.mark.skipif(not AVAIL, reason="requires CUDA + a working triton")
DEV = torch.device("cuda") if CUDA else torch.device("cpu")


def _sample(clauses, nv, cfg, *, kernel, force=False, seed=17, chunks=6, k_cap=128):
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=k_cap, seed=seed, use_kernel=kernel)
    kern = smp.engine._fused_kernel
    if force and kernel == "mono":
        kern._force_mono = True
    if force and kernel == "wide":
        kern._force_wide = True
    out = smp.sample(10 ** 9, configs=[cfg], max_chunks=chunks, collect=True,
                     verify_clauses=clauses)
    assert out["all_valid"] and out["gpu_sound"]
    mode = ("mono" if getattr(kern, "_mono_mode", False)
            else "wide" if getattr(kern, "_wide_mode", False)
            else "packed" if smp._last_chunk_packed else "unpacked")
    return sorted(tuple(m) for m in out["models"]), out, mode


def test_mono_matches_all_variants_small():
    """Forced mono vs forced wide vs base 16|16: identical model sets on a small mixed-length
    CNF with formula units (non-trivial cached flush), fused and fsoft configs."""
    clauses = [[3], [1, 2, -3], [-1, 4], [-2, -4], [3, -4, 5], [-5, 1], [2, 5, 6], [-6, -2]]
    for cfg in [(True, 1, "fused"), (True, 2, "fused"), (True, 2, "fsoft")]:
        m_mono, om, mode_m = _sample(clauses, 6, cfg, kernel="mono", force=True)
        assert mode_m == "mono", "forced mono mode did not engage"
        m_wide, _, mode_w = _sample(clauses, 6, cfg, kernel="wide", force=True)
        assert mode_w == "wide"
        m_base, ob, _ = _sample(clauses, 6, cfg, kernel=True)
        assert m_mono == m_base, f"mono kernel changed the model set for {cfg}"
        assert m_mono == m_wide
        assert om["n_models"] == ob["n_models"]


def test_mono_matches_base_random():
    """Random mixed-length CNFs: forced-mono model set == base packed model set."""
    rng = random.Random(11)
    done = 0
    while done < 5:
        nv = rng.randint(8, 16)
        clauses = []
        for _ in range(rng.randint(2 * nv, 4 * nv)):
            vs = rng.sample(range(1, nv + 1), rng.randint(1, min(4, nv)))
            clauses.append([v if rng.random() < 0.5 else -v for v in vs])
        m_mono, om, mode = _sample(clauses, nv, (True, 2, "fused"), kernel="mono",
                                   force=True, seed=200 + done)
        if mode != "mono":       # UNSAT input: flush conflicts -> mono declined; skip draw
            continue
        m_base, ob, _ = _sample(clauses, nv, (True, 2, "fused"), kernel=True, seed=200 + done)
        assert m_mono == m_base
        assert om["n_models"] == ob["n_models"]
        done += 1


def _sample_ladder(nv, clauses, cfg, kernel):
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=32, seed=23, use_kernel=kernel)
    smp.K_fused = smp.K
    out = smp.sample(10 ** 9, configs=[cfg], max_chunks=1, collect=True, verify_clauses=clauses)
    assert out["all_valid"] and out["gpu_sound"]
    mono_ran = bool(getattr(smp.engine._fused_kernel, "_mono_mode", False))
    return sorted(tuple(m) for m in out["models"]), out, mono_ran


def test_mono_engages_past_16bit_boundary():
    """2N > 65535: mono engages WITHOUT forcing and matches the unpacked baseline exactly.
    The implication ladder forces every unit through the clause-scan recovery on literal
    indices far above 65535."""
    nv = 40_000
    clauses = [[-i, i + 1] for i in range(1, nv)]
    for cfg in [(True, 4, "fused"), (True, 4, "fsoft")]:
        m_mono, om, mono_ran = _sample_ladder(nv, clauses, cfg, "mono")
        assert mono_ran, "mono mode should engage naturally at 2N=80,000"
        m_base, ob, base_mono = _sample_ladder(nv, clauses, cfg, True)
        assert not base_mono
        assert om["n_models"] > 0
        assert m_mono == m_base, f"mono kernel diverged from unpacked baseline for {cfg}"
        assert om["n_models"] == ob["n_models"]


def test_mono_defers_small_and_unsat():
    """Small formulas take the base 16|16 path; UNSAT under use_kernel='mono' -> zero models."""
    clauses = [[1, 2], [-1, 2], [1, -2], [-1, -2], [3, 4]]
    smp = GPUSampler.from_clauses(4, clauses, DEV, k_cap=32, seed=5, use_kernel="mono")
    out = smp.sample(1000, configs=[(True, 2, "fused")], max_chunks=3, collect=True)
    assert out["n_models"] == 0
    sat = [[1, 2], [-1, 3], [2, -3]]
    smp2 = GPUSampler.from_clauses(3, sat, DEV, k_cap=32, seed=5, use_kernel="mono")
    out2 = smp2.sample(50, configs=[(True, 1, "fused")], max_chunks=2, collect=True,
                       verify_clauses=sat)
    assert getattr(smp2.engine._fused_kernel, "_mono_mode", None) is False
    assert smp2._last_chunk_packed and out2["n_models"] > 0 and out2["all_valid"]

"""A/B tests for the WIDE (8|24) packed field split of the merged trajectory kernel
(gpap/engine/kernels.py::_bcp_trajectory_packed_kernel, SHIFT=8).

The 8|24 split must be trajectory-identical to the proven paths: same seed => the SAME model set
as both the 16|16 split and the unpacked kernel (order walk, hash polarity and propagation
semantics are shared; only the packed word's bit split differs). `_force_wide` selects 8|24 on
formulas the 16|16 word could handle, so the equality tests run on small CNFs; the boundary test
builds a real 2N > 65535 formula where wide mode engages naturally.
"""
import random

import pytest
import torch

from gpap.engine.kernels import TritonBCP
from gpap.sampling.sampler import GPUSampler

CUDA = torch.cuda.is_available()
AVAIL = CUDA and TritonBCP.available()
pytestmark = pytest.mark.skipif(not AVAIL, reason="requires CUDA + a working triton")
DEV = torch.device("cuda") if CUDA else torch.device("cpu")


def _sample(clauses, nv, cfg, *, kernel, force_wide=False, seed=17, chunks=6, k_cap=128):
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=k_cap, seed=seed, use_kernel=kernel)
    if force_wide:
        smp.engine._fused_kernel._force_wide = True
    out = smp.sample(10 ** 9, configs=[cfg], max_chunks=chunks, collect=True,
                     verify_clauses=clauses)
    wide_ran = bool(getattr(smp.engine._fused_kernel, "_wide_mode", False))
    assert out["all_valid"] and out["gpu_sound"]
    return sorted(tuple(m) for m in out["models"]), out, wide_ran


def test_wide_matches_base_small():
    """Forced 8|24 vs 16|16 vs unpacked on a small mixed-length CNF (with formula units so the
    cached-flush restore is non-trivial): identical model sets for fused and fsoft configs."""
    clauses = [[3], [1, 2, -3], [-1, 4], [-2, -4], [3, -4, 5], [-5, 1], [2, 5, 6], [-6, -2]]
    for cfg in [(True, 1, "fused"), (True, 2, "fused"), (True, 2, "fsoft")]:
        m_wide, ow, wide_ran = _sample(clauses, 6, cfg, kernel="wide", force_wide=True)
        assert wide_ran, "forced wide mode did not engage"
        m_base, ob, _ = _sample(clauses, 6, cfg, kernel=True)
        assert m_wide == m_base, f"wide kernel changed the model set for {cfg}"
        assert ow["n_models"] == ob["n_models"]


def test_wide_matches_base_random():
    """Random mixed-length CNFs: forced-wide model set == base packed model set."""
    rng = random.Random(3)
    done = 0
    while done < 5:
        nv = rng.randint(8, 16)
        clauses = []
        for _ in range(rng.randint(2 * nv, 4 * nv)):
            vs = rng.sample(range(1, nv + 1), rng.randint(1, min(4, nv)))
            clauses.append([v if rng.random() < 0.5 else -v for v in vs])
        m_wide, ow, wide_ran = _sample(clauses, nv, (True, 2, "fused"), kernel="wide",
                                       force_wide=True, seed=100 + done)
        if not wide_ran:        # UNSAT formula -> flush conflicts -> packing declined; skip
            continue
        m_base, ob, _ = _sample(clauses, nv, (True, 2, "fused"), kernel=True, seed=100 + done)
        assert m_wide == m_base
        assert ow["n_models"] == ob["n_models"]
        done += 1


def _sample_ladder(nv, clauses, cfg, *, force_unpacked=False):
    """Boundary-formula run at a SMALL fixed K (K_fused pinned to K so the test stays fast and
    both variants launch identical grids). `force_unpacked` disables the packed path so the run
    uses the UNPACKED kernel on the engine's own tensors — the A/B baseline."""
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=32, seed=23, use_kernel="auto")
    smp.K_fused = smp.K
    if force_unpacked:
        smp.engine._fused_kernel._packed_mode = False   # reset_fused declines -> unpacked kernel
    out = smp.sample(10 ** 9, configs=[cfg], max_chunks=1, collect=True, verify_clauses=clauses)
    wide_ran = bool(getattr(smp.engine._fused_kernel, "_wide_mode", False))
    packed_ran = bool(getattr(smp, "_last_chunk_packed", False))
    assert out["all_valid"] and out["gpu_sound"]
    return sorted(tuple(m) for m in out["models"]), out, wide_ran, packed_ran


def test_wide_engages_past_16bit_boundary():
    """A real 2N > 65535 formula: auto must select the 8|24 WIDE split (no forcing), produce valid
    models, and match the UNPACKED fallback's model set (same seed). Formula: an implication ladder
    x_i -> x_{i+1} over 40k vars — binary clauses, solutions = monotone suffixes, and every forced
    literal exercises the 24-bit XOR recovery on indices far above 65535."""
    nv = 40_000
    clauses = [[-i, i + 1] for i in range(1, nv)]
    for cfg in [(True, 4, "fused"), (True, 4, "fsoft")]:
        m_wide, ow, wide_ran, packed_ran = _sample_ladder(nv, clauses, cfg)
        assert wide_ran and packed_ran, "auto should select the 8|24 wide split at 2N=80,000"
        m_base, ob, base_wide, base_packed = _sample_ladder(nv, clauses, cfg, force_unpacked=True)
        assert not base_packed                     # baseline runs the UNPACKED fallback
        assert ow["n_models"] > 0
        assert m_wide == m_base, f"wide kernel diverged from unpacked baseline for {cfg}"
        assert ow["n_models"] == ob["n_models"]


def test_wide_defers_small_and_unsat():
    """Small formulas must take the base 16|16 path unchanged (wide never engages), and an
    UNSAT formula under use_kernel='wide' must yield zero models."""
    clauses = [[1, 2], [-1, 2], [1, -2], [-1, -2], [3, 4]]
    smp = GPUSampler.from_clauses(4, clauses, DEV, k_cap=32, seed=5, use_kernel="wide")
    out = smp.sample(1000, configs=[(True, 2, "fused")], max_chunks=3, collect=True)
    assert out["n_models"] == 0
    sat = [[1, 2], [-1, 3], [2, -3]]
    smp2 = GPUSampler.from_clauses(3, sat, DEV, k_cap=32, seed=5, use_kernel="wide")
    out2 = smp2.sample(50, configs=[(True, 1, "fused")], max_chunks=2, collect=True,
                       verify_clauses=sat)
    assert getattr(smp2.engine._fused_kernel, "_wide_mode", None) is False
    assert smp2._last_chunk_packed and out2["n_models"] > 0 and out2["all_valid"]

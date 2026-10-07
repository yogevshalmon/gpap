"""A/B tests for the WIDE (8|24) packed field split (kernels.py, SHIFT=8).

The 8|24 split must be trajectory-identical to the default 16|16 split: same seed => the SAME
model set (order walk, hash polarity and propagation semantics are shared; only the packed
word's bit split differs). `force_wide=True` selects 8|24 on formulas the 16|16 word could
handle, so the equality tests run on small CNFs; the boundary test builds a real 2N > 65535
formula where wide mode engages naturally and checks it against the pure-Python reference.
"""
import random

import pytest
import torch

from gpap.engine.formula import CnfFormula
from gpap.engine.bcp_engine import BCPEngine
from gpap.engine import core_available
from gpap.sampling.sampler import GPUSampler

from tests.reference.bcp import ref_bcp, assigned_set

CUDA = torch.cuda.is_available()
AVAIL = CUDA and core_available()
pytestmark = pytest.mark.skipif(not AVAIL, reason="requires CUDA + a built libgpap")
DEV = torch.device("cuda") if CUDA else torch.device("cpu")


def _force_wide(smp, seed, k_cap):
    """Put `smp` on the 8|24 split — BOTH halves of it.

    `GPUSampler` holds two things that name an engine: the `engine` attribute the tests read,
    and the core sampler handle (`_smp`), which is bound at construction and is what executes
    every chunk. Rebinding one without the other is silent: nothing raises, the run simply
    happens somewhere else. The pair is rebuilt here, and `_sample` asserts afterwards that the
    engine it hands back is the one that propagated.
    """
    from gpap.engine import _core as _gcore
    smp.engine = BCPEngine(smp.formula, force_wide=True)
    smp._smp = _gcore.Sampler(smp.engine._core, seed=seed,
                              k_cap=(-1 if k_cap is None else int(k_cap)))
    smp._sup_pushed = "unset"          # the new handle has been told nothing about the support


def _sample(clauses, nv, cfg, *, force_wide=False, seed=17, chunks=6, k_cap=128):
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=k_cap, seed=seed)
    if force_wide:
        _force_wide(smp, seed, k_cap)
    props0 = smp.engine.props_total()
    out = smp.sample(10 ** 9, configs=[cfg], max_chunks=chunks, collect=True,
                     verify_sound=True,
                     verify_clauses=clauses)
    # NON-VACUITY, and the whole reason this helper exists: the engine under test must be the
    # one that actually propagated. Assigning `smp.engine` alone does NOT redirect the run — the
    # CORE sampler handle is created in GPUSampler.__init__ over the engine that existed then —
    # so the swapped-in 8|24 engine sat idle at zero propagations while the original 16|16 one
    # ran every chunk, and both A/Bs below compared the base split against ITSELF.
    assert smp.engine.props_total() > props0, \
        "the engine under test did no work — the run went to a different engine"
    assert out["all_valid"]
    # "ok" iff rows were actually checked. `sound_check` is "off" when the run completed NO row,
    # because a verdict with no evidence is not a verdict — asserting "ok" here used to pass on a
    # run that emitted nothing, which is how test_wide_matches_base_random came to compare two
    # empty model sets for five instances.
    assert out["sound_check"] == ("ok" if out["sound_rows_checked"] else "off")
    return sorted(tuple(m) for m in out["models"]), out, smp.engine.shift


def test_wide_matches_base_small():
    """Forced 8|24 vs 16|16 on a small mixed-length CNF (with formula units so the cached-flush
    restore is non-trivial): identical model sets for plain and soft configs."""
    clauses = [[3], [1, 2, -3], [-1, 4], [-2, -4], [3, -4, 5], [-5, 1], [2, 5, 6], [-6, -2]]
    for cfg in [(True, 1, False), (True, 2, False), (True, 2, "soft")]:
        m_wide, ow, shift_w = _sample(clauses, 6, cfg, force_wide=True)
        assert shift_w == 8, "forced wide mode did not engage"
        m_base, ob, shift_b = _sample(clauses, 6, cfg)
        assert shift_b == 16
        # NON-VACUITY first: two empty model sets are equal, and that is exactly how
        # test_wide_matches_base_random passed on five instances before it grew this line.
        assert ow["n_models"] > 0, f"{cfg} produced nothing — the comparison would be vacuous"
        assert m_wide == m_base, f"wide kernel changed the model set for {cfg}"
        assert ow["n_models"] == ob["n_models"]


def test_wide_matches_base_random():
    """Random mixed-length CNFs: forced-wide model set == base packed model set.

    The instances are deliberately UNDER-constrained (clause/var ~1, width 2-4, no units). The
    test used to draw 2-4x as many clauses including units, and every one of its five instances
    yielded ZERO models — so it asserted that two empty sets are equal, five times, and would
    have passed against a wide kernel that emitted nothing at all. Each instance now delivers
    hundreds of models, which is asserted so the comparison cannot silently go vacuous again."""
    rng = random.Random(3)
    for i in range(5):
        nv = rng.randint(8, 16)
        clauses = [[v if rng.random() < 0.5 else -v
                    for v in rng.sample(range(1, nv + 1), rng.randint(2, min(4, nv)))]
                   for _ in range(nv)]
        m_wide, ow, _ = _sample(clauses, nv, (True, 2, False), force_wide=True, seed=100 + i)
        m_base, ob, _ = _sample(clauses, nv, (True, 2, False), seed=100 + i)
        assert ow["n_models"] > 0, "vacuous: this instance yields no models to compare"
        assert m_wide == m_base
        assert ow["n_models"] == ob["n_models"]


def test_wide_engages_past_16bit_boundary():
    """A real 2N > 65535 formula: the engine must select the 8|24 WIDE split automatically,
    produce valid models, and its fixpoints must match the pure-Python reference (the 24-bit
    XOR recovery is exercised on literal indices far above 65535). Formula: an implication
    ladder x_i -> x_{i+1} over 40k vars — every decision propagates a long suffix/prefix."""
    nv = 40_000
    clauses = [[-i, i + 1] for i in range(1, nv)]
    # sampler path: valid models, wide selected
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=32, seed=23)
    smp.pin_batch(smp.K)
    assert smp.engine.shift == 8, "auto should select the 8|24 wide split at 2N=80,000"
    out = smp.sample(10 ** 9, configs=[(True, 1, False)], max_chunks=1, collect=True,
                     verify_sound=True,
                     verify_clauses=clauses)
    assert out["sound_check"] == "ok" and out["all_valid"] and out["n_models"] > 0
    # fixpoint path vs reference: decide a middle var each way
    f = CnfFormula(DEV)
    f.init_formula(nv, clauses)
    eng = BCPEngine(f)
    eng.reset(2)
    mid = 20_000
    dec = torch.tensor([[2 * (mid - 1)], [2 * (mid - 1) + 1]], dtype=torch.int64, device=DEV)
    eng.apply_and_propagate(dec)
    for k, lits in enumerate([[2 * (mid - 1)], [2 * (mid - 1) + 1]]):
        conf, asg = ref_bcp(clauses, lits)
        assert bool(eng.conflict[k].item()) == conf
        if not conf:
            assert assigned_set(eng.assignment[k].cpu()) == asg


def test_wide_defers_small_and_unsat():
    """Small formulas take the base 16|16 split, and an UNSAT formula yields zero models."""
    clauses = [[1, 2], [-1, 2], [1, -2], [-1, -2], [3, 4]]
    smp = GPUSampler.from_clauses(4, clauses, DEV, k_cap=32, seed=5)
    assert smp.engine.shift == 16
    out = smp.sample(1000, configs=[(True, 2, False)], max_chunks=3, collect=True)
    assert out["n_models"] == 0
    sat = [[1, 2], [-1, 3], [2, -3]]
    smp2 = GPUSampler.from_clauses(3, sat, DEV, k_cap=32, seed=5)
    out2 = smp2.sample(50, configs=[(True, 1, False)], max_chunks=2, collect=True,
                       verify_clauses=sat)
    assert smp2.engine.shift == 16
    assert out2["n_models"] > 0 and out2["all_valid"]


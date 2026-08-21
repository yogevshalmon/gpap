"""Uniformity regression gate, over generated instances whose solution set is known EXACTLY.

The full campaign lives in `eval/uniformity/fuzz_uniformity.py` (see docs/findings/UNIFORMITY_FINDINGS.md);
this file gates the parts that must never regress silently:

  * the generators' own oracle is self-consistent (a broken oracle would hide every other bug);
  * the chi-square machinery does not manufacture skew from an exactly-uniform input;
  * every emitted row is a real solution (SOUNDNESS — the one property that is not a trade-off);
  * at B >= |support| the sampler is statistically indistinguishable from exactly uniform. That
    is the structural claim: one decision round assigns the whole support a uniform random
    polarity, and because GPAP DISCARDS conflicted trajectories instead of backtracking out of
    them, keeping the survivors is exact rejection sampling over the projection.

Deliberately NOT gated: that small-B configs are uniform. They are not, measurably so, and that is
a documented property of this algorithm class (CMSGen included) rather than a defect.
"""
import numpy as np
import pytest

from eval.uniformity.fuzz_uniformity import (
    CONFIGS,
    all_assignments,
    brute_force_models,
    generate,
    sample_control,
    sample_gpap,
)
from gpap.sampling.metrics import uniformity_stats

# One fixed seed: a regression gate must fail for a code reason, never for a draw reason.
FUZZ_SEED = 20260730
N_INSTANCES = 6
S_WINDOW = (8, 200)

# A true uniform sampler trips this only ~1e-6 of the time per instance, so a failure here is a
# real distributional change and not an unlucky campaign.
P_FLOOR = 1e-6


@pytest.fixture(scope="module")
def instances():
    insts = generate(np.random.default_rng(FUZZ_SEED), N_INSTANCES, ["cnf", "circuit"], S_WINDOW)
    assert len(insts) == N_INSTANCES, "generator could not fill the |S| window"
    return insts


def test_oracle_is_self_consistent(instances):
    """Every enumerated solution must satisfy the formula, and the count must be reproducible.

    For the circuit kind `gen_circuit` already proves encoding == truth table over all 2^k inputs
    (it raises otherwise); this re-checks the `cnf` kind against its own clauses, so the oracle is
    never taken on trust."""
    for inst in instances:
        assert inst.S >= S_WINDOW[0] and inst.S <= S_WINDOW[1]
        assert inst.sols.shape[1] == len(inst.support)
        assert len({tuple(r) for r in inst.sols}) == inst.S, "duplicate rows in the solution set"
        if inst.kind != "cnf":
            continue
        for row in inst.sols:
            asg = {(i + 1) if b else -(i + 1) for i, b in enumerate(row)}
            assert all(any(lit in asg for lit in cl) for cl in inst.clauses), \
                f"{inst.name}: enumerated 'solution' does not satisfy the formula"
        assert brute_force_models(inst.nv, inst.clauses).shape[0] == inst.S


def test_brute_force_agrees_with_a_solver():
    """The oracle is solver-free by design; cross-check it against pysat once so a silent bug in
    the exhaustive evaluator cannot pass unnoticed."""
    pysat = pytest.importorskip("pysat.solvers")
    clauses = [[1, 2, -3], [-1, 3], [2, 3], [-2, -3, 4], [-4, 1]]
    mine = {tuple(int(b) for b in r) for r in brute_force_models(4, clauses)}
    theirs = set()
    with pysat.Solver(name="minisat22", bootstrap_with=clauses) as s:
        while s.solve():
            m = set(s.get_model())
            row = tuple(1 if v in m else 0 for v in range(1, 5))
            theirs.add(row)
            s.add_clause([-(v) if row[v - 1] else v for v in range(1, 5)])
    assert mine == theirs


def test_statistics_do_not_manufacture_skew(instances):
    """The negative control: an exactly-uniform input must not be flagged. If this fails, every
    uniformity number this repo reports is suspect."""
    for inst in instances:
        n = 100 * inst.S
        counts, unsound, got, _ = sample_control(inst, n, seed=3, tv=0.0)
        assert (unsound, got) == (0, n)
        u = uniformity_stats(counts, n, S=inst.S)
        assert u["p"] > P_FLOOR, f"{inst.name}: exact uniform flagged as skewed (p={u['p']})"


def test_positive_control_has_power(instances):
    """...and the test must actually be able to see a real skew, or the negative control above is
    passing for the wrong reason. TV=0.15 is well inside what the campaign measures on real
    samplers, so it must be detected on every instance."""
    for inst in instances:
        n = 100 * inst.S
        counts, _, _, _ = sample_control(inst, n, seed=3, tv=0.15)
        u = uniformity_stats(counts, n, S=inst.S)
        assert u["p"] < P_FLOOR, f"{inst.name}: a TV=0.15 skew went undetected (p={u['p']})"


def test_all_assignments_shape():
    a = all_assignments(4)
    assert a.shape == (16, 4)
    assert len({tuple(r) for r in a}) == 16


@pytest.mark.gpu
@pytest.mark.parametrize("cfg", ["b1", "b8"])
def test_every_emitted_row_is_a_solution(instances, cfg):
    """SOUNDNESS. Unlike uniformity this is not a trade-off knob: a row that is not a solution is
    a bug at any B, under any config."""
    for inst in instances:
        counts, unsound, got, _ = sample_gpap(inst, min(20 * inst.S, 4000), 1, CONFIGS[cfg])
        assert got > 0, f"{inst.name}: sampler emitted nothing at {cfg}"
        assert unsound == 0, f"{inst.name}: {unsound}/{got} emitted rows are NOT solutions ({cfg})"
        assert counts.sum() == got


@pytest.mark.gpu
def test_full_support_batch_is_exactly_uniform(instances):
    """B >= |support| == one decision round over the whole support == exact rejection sampling.

    This is the structural guarantee, so it is gated on EVERY generated instance rather than as an
    aggregate: a single instance drifting off uniform here means the rejection-sampling argument
    has stopped holding."""
    for inst in instances:
        n = min(100 * inst.S, 20_000)
        cfg = (True, len(inst.support), False)
        counts, unsound, got, _ = sample_gpap(inst, n, 1, cfg)
        assert unsound == 0 and got > 0
        u = uniformity_stats(counts, got, S=inst.S)
        # Without enough draws chi2 reports p=None, and "no p" must not read as "uniform" —
        # otherwise a sampler that produced almost nothing would pass this gate vacuously.
        assert u["p"] is not None, \
            f"{inst.name}: only {got} samples for |S|={inst.S}; chi2 needs >= 5|S|"
        assert u["p"] > P_FLOOR, (
            f"{inst.name}: B=|support|={len(inst.support)} is not uniform "
            f"(TV={u['tv']:.4f}, p={u['p']:.2e}) — the rejection-sampling property broke")

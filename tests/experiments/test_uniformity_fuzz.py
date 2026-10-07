"""Uniformity regression gate over generated instances whose solution set is known exactly.

Checks that the oracle is self-consistent, the chi-square test neither invents nor misses skew,
every emitted row is a solution, and B >= |support| is uniform (exact rejection sampling). Small-B
configs are not uniform and are not gated. The full campaign is
`eval/uniformity/fuzz_uniformity.py` (docs/findings/UNIFORMITY_FINDINGS.md).
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
from eval.uniformity.metrics import uniformity_stats

# One fixed seed, so a failure comes from the code and not from the draw.
FUZZ_SEED = 20260730
N_INSTANCES = 6
S_WINDOW = (8, 200)

# A uniform sampler trips this about 1e-6 of the time per instance.
P_FLOOR = 1e-6


@pytest.fixture(scope="module")
def instances():
    insts = generate(np.random.default_rng(FUZZ_SEED), N_INSTANCES, ["cnf", "circuit"], S_WINDOW)
    assert len(insts) == N_INSTANCES, "generator could not fill the |S| window"
    return insts


def test_degeneracy_flag_finds_subcube_solution_sets(instances):
    """A solution set that is a full subcube (every support var pinned or free) is uniform under
    any sampler, so the campaign flags it to screen it out of a median."""
    from eval.uniformity.fuzz_uniformity import _degeneracy

    assert _degeneracy(np.array([[a, b, 1] for a in (0, 1) for b in (0, 1)],
                                dtype=np.uint8)) == {"free_dim": 2, "subcube": True}
    assert _degeneracy(np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]],
                                dtype=np.uint8)) == {"free_dim": 2, "subcube": False}
    for inst in instances:                       # every generated instance carries the verdict
        meta = _degeneracy(inst.sols)
        assert inst.meta["subcube"] == meta["subcube"]
        assert inst.meta["free_dim"] == meta["free_dim"] <= len(inst.support)
        # the flag agrees with the definition
        assert meta["subcube"] == (inst.S == (1 << meta["free_dim"]))


def test_oracle_is_self_consistent(instances):
    """Every enumerated solution satisfies the formula and the count is reproducible. Only the
    `cnf` kind is re-checked here; `gen_circuit` checks its own encoding against the truth table."""
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
    """The solver-free oracle matches pysat's model enumeration."""
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
    """Negative control: an exactly uniform input is not flagged."""
    for inst in instances:
        n = 100 * inst.S
        counts, unsound, got, _ = sample_control(inst, n, seed=3, tv=0.0)
        assert (unsound, got) == (0, n)
        u = uniformity_stats(counts, n, S=inst.S)
        assert u["p"] > P_FLOOR, f"{inst.name}: exact uniform flagged as skewed (p={u['p']})"


def test_positive_control_has_power(instances):
    """Positive control: a TV=0.15 skew is detected on every instance, so the negative control
    is not passing for lack of power."""
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
    """Every emitted row is a solution, at any B and under any config."""
    for inst in instances:
        counts, unsound, got, _ = sample_gpap(inst, min(20 * inst.S, 4000), 1, CONFIGS[cfg])
        assert got > 0, f"{inst.name}: sampler emitted nothing at {cfg}"
        assert unsound == 0, f"{inst.name}: {unsound}/{got} emitted rows are NOT solutions ({cfg})"
        assert counts.sum() == got


@pytest.mark.gpu
def test_full_support_batch_is_exactly_uniform(instances):
    """B >= |support| is one decision round over the whole support, and discarding conflicted
    rows makes it exact rejection sampling. Gated per instance, not as an aggregate."""
    for inst in instances:
        n = min(100 * inst.S, 20_000)
        cfg = (True, len(inst.support), False)
        counts, unsound, got, _ = sample_gpap(inst, n, 1, cfg)
        assert unsound == 0 and got > 0
        u = uniformity_stats(counts, got, S=inst.S)
        # With too few draws chi2 reports p=None; that must fail rather than pass vacuously.
        assert u["p"] is not None, \
            f"{inst.name}: only {got} samples for |S|={inst.S}; chi2 needs >= 5|S|"
        assert u["p"] > P_FLOOR, (
            f"{inst.name}: B=|support|={len(inst.support)} is not uniform "
            f"(TV={u['tv']:.4f}, p={u['p']:.2e}) — the rejection-sampling property broke")

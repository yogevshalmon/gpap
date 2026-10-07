"""The tuple-probe primitive (gpap.engine.probe) and entailed-clause mining (gpap.mining).

Every mined clause must be entailed, so adding it leaves the model set unchanged. That is
checked with an independent SAT solver, not by re-running our own propagation.
"""
import itertools as it
import random

import numpy as np
import pytest

import gpap.engine.probe
from gpap.engine.probe import LiteralProbe, ProbeBudget
from gpap.mining import _derivable_mask, _Subsumption, mine_entailed
from gpap.sampling.pipeline import sample_clauses
from tests.reference.combinations import combinations
from tests.support import DEV, random_cnf

pytestmark = pytest.mark.gpu


def _brute_models(nv, clauses):
    """Every satisfying assignment, as sets of DIMACS literals. Brute force, small `nv`."""
    out = []
    for bits in it.product([False, True], repeat=nv):
        asg = {(i + 1) if b else -(i + 1) for i, b in enumerate(bits)}
        if all(any(l in asg for l in c) for c in clauses):
            out.append(asg)
    return out


def _entailed_binaries(nv, mods):
    """Every binary clause true in every model: the oracle's notion of entailed, no engine."""
    lits = [s * v for v in range(1, nv + 1) for s in (1, -1)]
    out = []
    for i, a in enumerate(lits):
        for b in lits[i + 1:]:
            if abs(a) != abs(b) and all((a in m) or (b in m) for m in mods):
                out.append([a, b])
    return out


# ------------------------------------------------------------------ the enumerator

# ---------------------------------------------------------------- combinations
# The reference enumerator (tests/reference/combinations.py), checked against itertools. It
# pins the core's candidate order the way tests/reference/bcp.py pins the kernels. Literals
# are DIMACS.

def test_pairs_match_itertools():
    import itertools
    pool = [3, -3, 5, -5, 11, -11]
    want = sorted(t for t in itertools.combinations(pool, 2) if abs(t[0]) != abs(t[1]))
    got = sorted(map(tuple, combinations(pool, 2).tolist()))
    assert got == want


def test_triples_match_itertools():
    import itertools
    pool = [3, -3, 5, -5, 11, -11, 16]
    want = sorted(t for t in itertools.combinations(pool, 3) if len({abs(x) for x in t}) == 3)
    got = sorted(map(tuple, combinations(pool, 3).tolist()))
    assert got == want


def test_same_variable_pairs_are_dropped():
    """(l, ~l) conflicts regardless of the formula, so probing it proves only the tautology
    (l | ~l)."""
    got = combinations([1, -1, 2, -2], 2).tolist()
    assert [1, -1] not in got and [2, -2] not in got
    assert len(got) == 4                                   # the four cross-variable pairs
    assert len(combinations([1, -1, 2, -2], 2, distinct_vars=False)) == 6


def test_size_one_is_the_pool():
    assert combinations([7, -8, 9], 1).reshape(-1).tolist() == [7, -8, 9]


def test_pool_smaller_than_size():
    assert combinations([3], 2).shape == (0, 2)


def test_max_tuples_truncates():
    assert combinations(list(range(1, 21)), 2, max_tuples=5).shape == (5, 2)


def test_rejects_zero_size():
    with pytest.raises(ValueError, match="size"):
        combinations([1, 2], 0)


# ------------------------------------------------------------------ the primitive

# ---------------------------------------------------------------- literal probe
#  (x1 v x2) ^ (~x1 v x3) ^ (~x2 v ~x3)
_PROBE_CLAUSES = [[1, 2], [-1, 3], [-2, -3]]


def test_failed_literal():
    """x1 implies x2 and ~x2 => assuming x1 conflicts, so ~x1 is entailed."""
    p = LiteralProbe(2, [[-1, 2], [-1, -2]], DEV)
    r = p.probe(size=1)
    assert r.conflicting.tolist() == [[1]]
    assert r.exhausted


def test_exhausted_is_a_proof_of_absence():
    p = LiteralProbe(3, _PROBE_CLAUSES, DEV)
    r = p.probe(size=1)
    assert len(r) == 0
    assert r.exhausted, "no failed literal exists, and we looked at every one"


def test_pool_defaults_to_every_free_literal():
    """`pool=None` probes every free literal, the same as passing them explicitly."""
    p = LiteralProbe(3, _PROBE_CLAUSES, DEV)
    assert p.probe(size=2).total == p.probe(p.free_literals(), size=2).total


def test_count_matches_the_sweep_without_generating_it():
    """count() equals the number a sweep probes. The ceiling is checked against count() so a
    huge width is refused before any enumeration is materialized."""
    p = LiteralProbe(3, _PROBE_CLAUSES, DEV)
    for k in (1, 2, 3):
        assert p.count(size=k) == p.probe(size=k).total


def test_count_saturates_instead_of_wrapping():
    """A wrapped count would compare below the ceiling and let through the sweep it should
    refuse."""
    p = LiteralProbe(400, [[1, 2], [-1, 3]], DEV)
    assert p.count(size=3) > 10**7


def test_max_tuples_is_a_work_budget():
    """`max_tuples` caps how many candidates are probed; it does not cancel the sweep. Partial
    results are still entailed; only `exhausted` is lost, and the result says so."""
    p = LiteralProbe(3, _PROBE_CLAUSES, DEV)
    r = p.probe(size=2, budget=ProbeBudget(max_tuples=2))
    assert r.probed == 2 and r.total == 12
    assert not r.exhausted and r.stopped == "max_tuples"


def test_refuse_over_max_is_opt_in():
    """`refuse_over_max` probes nothing instead of a prefix, for a caller that needs an
    exhaustive answer."""
    p = LiteralProbe(3, _PROBE_CLAUSES, DEV)
    r = p.probe(size=2, budget=ProbeBudget(max_tuples=2, refuse_over_max=True))
    assert r.probed == 0 and r.total == 12 and r.stopped == "max_tuples"


def test_free_literals_excludes_settled_variables():
    """A formula's own units settle variables at construction; probing them is wasted."""
    p = LiteralProbe(3, _PROBE_CLAUSES + [[1]], DEV)
    free = set(p.free_literals().tolist())
    assert 1 not in free and -1 not in free, "x1 is fixed by the unit clause"
    assert not p.is_free(1)


def test_assume_narrows_without_rebuilding():
    p = LiteralProbe(3, _PROBE_CLAUSES, DEV)
    assert p.is_free(1)
    p.assume([1])                                   # assert x1
    assert not p.is_free(1), "x1 must be settled after being assumed"
    assert set(p.free_literals().tolist()).isdisjoint({1, -1})


def test_unsat_formula_probes_nothing():
    p = LiteralProbe(1, [[1], [-1]], DEV)
    assert p.unsat
    r = p.probe(size=1)
    assert r.probed == 0 and r.stopped == "unsat"


def test_chunked_matches_single_launch():
    """A sweep split across launches gives the same answer as one launch."""
    rng = random.Random(3)
    clauses = random_cnf(rng, 12, 40, kmin=2, kmax=3)
    p = LiteralProbe(12, clauses, DEV)
    tuples = combinations(p.free_literals(), 2)
    one = p.probe_tuples(tuples)
    many = p.engine.probe_tuples(tuples, batch=7)     # force many launches
    with many:
        got = many.conflicts()
    assert np.array_equal(np.sort(one.conflicting, axis=0), np.sort(got, axis=0))


def test_rejects_out_of_range_literals():
    """An index past the row's bitmap would write into a later row's words and corrupt another
    tuple's answer, so it is refused."""
    p = LiteralProbe(3, _PROBE_CLAUSES, DEV)
    with pytest.raises(RuntimeError, match="out of range"):
        p.probe([99], size=1)
    with pytest.raises(RuntimeError, match="out of range"):
        p.probe_tuples([[1, 99]])


# ------------------------------------------------------------------ the application

# ---------------------------------------------------------------- mining
def _models(clauses):
    from pysat.solvers import Minisat22
    with Minisat22(bootstrap_with=clauses) as s:
        return {tuple(m) for m in s.enum_models()}


def _check_entailed(nv, clauses, pool=None, **kw):
    """Every mined clause must be entailed, and the augmented formula must have exactly the
    original models, checked with an independent SAT solver."""
    from pysat.solvers import Minisat22
    res = mine_entailed(nv, clauses, DEV, pool=pool, **kw)
    for cl in res.clauses:
        with Minisat22(bootstrap_with=clauses) as s:
            assert not s.solve(assumptions=[-lit for lit in cl]), \
                f"mined clause {cl} is NOT entailed"
    if res.clauses:
        assert _models(clauses) == _models(clauses + res.clauses)
    return res


def test_planted_backbone_and_pair():
    # (1|2),(1|-2) => backbone unit (1), not flush-implied (there are no unit clauses);
    # (-1,-3,-4),(-1,-3,4) => assuming x3 (with x1 forced) conflicts => unit (-3);
    # (-5|-6) is entailed via x7: (-5,-6,7),(-5,-6,-7)
    clauses = [[1, 2], [1, -2], [-1, -3, -4], [-1, -3, 4],
               [-5, -6, 7], [-5, -6, -7], [5, 6, 8], [2, 8, -5]]
    # `pool` is DIMACS literals, so both polarities of x5 and x6 are named explicitly.
    res = _check_entailed(8, clauses, pool=[5, -5, 6, -6])
    assert res.by_width.get(1, 0) >= 2, res.by_width      # units (1) and (-3) at least
    mined = {tuple(sorted(c)) for c in res.clauses}
    assert (-6, -5) in mined or (-5,) in mined or (-6,) in mined, mined


def test_pool_polarity_is_honoured():
    """Naming one polarity probes only that side. (5, 6) proves (-5 | -6); (-5, -6) asks a
    different question."""
    clauses = [[-5, -6, 7], [-5, -6, -7], [5, 6, 8], [2, 8, -5]]
    both = _check_entailed(8, clauses, pool=[5, -5, 6, -6], widths=(2,))
    one = _check_entailed(8, clauses, pool=[5, 6], widths=(2,))
    assert (-6, -5) in {tuple(sorted(c)) for c in both.clauses}
    assert len(one.clauses) <= len(both.clauses), "one polarity cannot prove more"


def test_random_formulas_stay_model_preserving():
    from pysat.solvers import Minisat22
    rng = random.Random(7)
    mined = 0
    for _ in range(10):
        nv = rng.randint(6, 14)
        clauses = random_cnf(rng, nv, rng.randint(2 * nv, 5 * nv), kmin=2, kmax=3)
        with Minisat22(bootstrap_with=clauses) as s:
            if not s.solve():
                continue                                  # UNSAT input: mining may derive it
        # both polarities of the first few variables, as DIMACS literals (from 1; 0 is not one)
        k = min(nv, 6)
        pool = [sign * v for v in range(1, k + 1) for sign in (1, -1)]
        mined += len(_check_entailed(nv, clauses, pool=pool).clauses)
    # `_check_entailed` compares model sets only when something was mined, so require that.
    assert mined, "nothing was mined on any instance: the entailment check never ran"


def test_widths_are_selectable():
    clauses = [[1, 2], [1, -2], [-1, -3, -4], [-1, -3, 4]]
    only_units = mine_entailed(4, clauses, DEV, widths=(1,))
    assert set(only_units.by_width) == {1}
    assert all(len(c) == 1 for c in only_units.clauses)


def test_max_tuples_skips_rather_than_truncates():
    """A width over its `max_tuples` ceiling is skipped and listed in `skipped`, since a
    truncated width proves nothing about what it did not reach."""
    clauses = [[1, 2], [-1, 3], [-2, -3], [4, 5], [-4, 6], [-5, -6]]
    res = mine_entailed(6, clauses, DEV, widths=(1, 2), max_tuples={2: 1})
    assert 2 in res.skipped and "candidates" in res.skipped[2]
    assert res.exhausted.get(2) is False


def test_wide_widths_gated_on_unit_yield():
    clauses = [[1, 2], [-1, 3], [-2, -3]]              # no failed literals at width 1
    res = mine_entailed(3, clauses, DEV, min_units_for_wide=1)
    assert res.by_width.get(1, 0) == 0
    assert res.skipped.get(2) == "too few units at width 1"


def test_unsat_formula_is_reported_not_mined():
    res = mine_entailed(1, [[1], [-1]], DEV)
    assert res.clauses == []
    assert all("unsat" in v for v in res.skipped.values())


def test_result_summary_is_readable():
    res = mine_entailed(4, [[1, 2], [1, -2]], DEV)
    assert "width 1" in res.summary() and "s" in res.summary()


# ------------------------------------------------------------------ the CLI

# ---------------------------------------------------------------- mine CLI
# `gpap mine`, run as a subprocess: these check the command's contract (flags, exit status,
# files written), not a Python call.

def _run(*argv, cwd=None):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "gpap.cli", "mine", *map(str, argv)],
                       capture_output=True, text=True, timeout=600, cwd=cwd)
    assert r.returncode == 0, f"gpap mine failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def _cnf(tmp_path, nv, clauses, name="f.cnf", ind=None):
    p = tmp_path / name
    head = f"p cnf {nv} {len(clauses)}\n"
    if ind:
        head += "c ind " + " ".join(map(str, ind)) + " 0\n"
    p.write_text(head + "".join(" ".join(map(str, c)) + " 0\n" for c in clauses))
    return p


def test_mines_and_writes_a_strengthened_formula(tmp_path):
    clauses = [[-1, 2], [-1, -2], [3, 4], [6, 7]]
    cnf = _cnf(tmp_path, 7, clauses)
    out = tmp_path / "strong.cnf"
    stats = tmp_path / "mine.json"
    txt = _run(cnf, "--width", "1", "--out", out, "--stats", stats)
    assert "exhaustive" in txt
    assert out.exists() and stats.exists()

    import json
    info = json.loads(stats.read_text())
    assert info["by_width"]["1"] >= 1 and info["clauses"]
    # every proven clause must be entailed, so the strengthened formula has the same models
    from pysat.solvers import Minisat22
    for cl in info["clauses"]:
        with Minisat22(bootstrap_with=clauses) as s:
            assert not s.solve(assumptions=[-lit for lit in cl]), cl


def test_pool_support_uses_the_c_ind_lines(tmp_path):
    cnf = _cnf(tmp_path, 7, [[-1, 2], [-1, -2], [3, 4], [6, 7]], ind=[1, 3, 6])
    txt = _run(cnf, "--width", "2", "--pool", "support")
    assert "pool=support (6 literals)" in txt, txt    # 3 variables, both polarities


def test_pool_rejects_nonsense_instead_of_guessing(tmp_path):
    import subprocess
    import sys
    cnf = _cnf(tmp_path, 7, [[1, 2]])
    for bad in ("99", "abc"):
        r = subprocess.run([sys.executable, "-m", "gpap.cli", "mine", str(cnf),
                            "--pool", bad], capture_output=True, text=True, timeout=300)
        assert r.returncode != 0, f"--pool {bad} should have been refused"


def test_huge_width_probes_its_budget_and_reports_coverage(tmp_path):
    """A huge width probes its budget without materializing the enumeration, and reports its
    coverage so "nothing entailed" is distinguishable from "barely started"."""
    nv = 300
    cnf = _cnf(tmp_path, nv, [[i, -(i + 1)] for i in range(1, nv)])
    txt = _run(cnf, "--width", "3", "--max-tuples", "1000", "--seconds", "20")
    assert "probed 1,000 of" in txt, txt
    assert "NOT searched exhaustively" in txt, txt


# ---------------------------------------------------------------- probe dispatch
# The probe sweep calls the C ABI directly and still gets the kernel dispatch, because
# GPAP_MODE_AUTO resolves inside gpap_engine_reset, which every entry point passes through.
# These tests check where the rule lives, not which kernel wins.

def test_probe_dispatches_by_clause_count(monkeypatch):
    monkeypatch.delenv("GPAP_KERNEL", raising=False)
    # M = 6: too few clauses for a warp to have work, so the small-batch term decides
    small = LiteralProbe(6, [[1, 2], [-1, 3], [-2, -3], [4, 5], [-4, 6], [-5, -6]], DEV)
    small.probe(size=1)
    assert small.engine.mode in ("scalar", "wave")
    # M >= 2048: the rule says wave unconditionally, and the sweep must actually run it
    rng = random.Random(3)
    big = [[rng.choice([1, -1]) * rng.randint(1, 400) for _ in range(3)] for _ in range(2500)]
    probe = LiteralProbe(400, big, DEV)
    probe.probe(size=1)
    assert probe.engine.mode == "wave", "M >= 2048 must dispatch wave inside the sweep"


@pytest.mark.parametrize("pinned", ["scalar", "wave"])
def test_an_explicit_mode_is_honoured(pinned, monkeypatch):
    """A measurement pins the kernel; the per-reset dispatch must not take it back."""
    monkeypatch.delenv("GPAP_KERNEL", raising=False)
    rng = random.Random(4)
    big = [[rng.choice([1, -1]) * rng.randint(1, 400) for _ in range(3)] for _ in range(2500)]
    p = LiteralProbe(400, big, DEV, mode=pinned)
    p.probe(size=1)
    assert p.engine.mode == pinned


def test_env_var_pins(monkeypatch):
    monkeypatch.setenv("GPAP_KERNEL", "scalar")
    rng = random.Random(5)
    big = [[rng.choice([1, -1]) * rng.randint(1, 400) for _ in range(3)] for _ in range(2500)]
    p = LiteralProbe(400, big, DEV)
    p.probe(size=1)
    assert p.engine.mode == "scalar"


# ---------------------------------------------------------------- subsumption
# Mining must not emit a clause that a shorter known clause implies: it is entailed but useless,
# and it enlarges the formula, slowing every later propagation. An exact-duplicate check is not
# enough; every subset is checked.

def test_subsumed_clauses_are_not_emitted():
    # (-1 | -2) is given, so any triple (-1 | -2 | x) is implied by it.
    clauses = [[-1, -2], [1, 3], [2, 3], [-3, 4], [-3, -4], [5, 6], [-5, 6], [5, -6]]
    res = mine_entailed(8, clauses, DEV, widths=(1, 2, 3), seconds=30.0)
    assert res.clauses, "nothing mined — a filter that dropped EVERYTHING would pass this"
    known = {tuple(sorted(c)) for c in clauses}
    for cl in res.clauses:
        t = tuple(sorted(cl))
        for r in range(1, len(t)):
            import itertools as it
            for sub in it.combinations(t, r):
                assert sub not in known, f"{cl} is subsumed by the known clause {list(sub)}"
                assert sub not in {tuple(sorted(c)) for c in res.clauses}, \
                    f"{cl} is subsumed by another MINED clause {list(sub)}"


def test_coverage_is_reported_so_empty_is_readable():
    """`0 clauses` from an exhaustive sweep and from a tiny fraction of one mean opposite
    things, so the result reports coverage."""
    nv = 400
    clauses = [[i, -(i + 1)] for i in range(1, nv)]
    partial = mine_entailed(nv, clauses, DEV, widths=(3,), seconds=5.0, max_tuples=500)
    assert partial.candidates[3] > partial.probed[3] > 0
    assert not partial.exhausted[3]
    assert "probed" in partial.summary()

    full = mine_entailed(4, [[1, 2], [-1, 3]], DEV, widths=(2,), seconds=5.0)
    assert full.exhausted[2] and full.probed[2] == full.candidates[2]


# ---------------------------------------------------------------- subsumption filter
# `mining._Subsumption` is the only redundancy filter. It runs on the host, where the formula is
# known; there is no filter in the engine (docs/findings/MINING_FINDINGS.md).
#
# Its reference is the formula reduced by the engine's baseline, not the input clauses. Mining
# folds proven units into the flush, so an input (a | b | c) whose `c` the baseline falsified
# acts as a binary; a conflict on (~a, ~b) then proves (a | b), which no input clause subsumes.

def test_drops_only_what_the_baseline_makes_redundant():
    """(-1 | -2 | 3) with 3 forced false by the flush acts as a binary during the sweep, so the
    proved (-1 | -2) is subsumed by its residue, though by no input clause."""
    clauses = [[-3], [-1, -2, 3], [4, 5], [-4, 5], [4, -5]]
    p = LiteralProbe(5, clauses, DEV)
    assert -3 in p.assigned_literals().tolist(), "the flush must have settled 3"
    sub = _Subsumption(2)
    sub.rebuild(clauses, 5, p.assigned_literals())
    assert sub.subsumes((-2, -1)), "the residue of (-1 | -2 | 3) subsumes it"
    raw = {tuple(sorted(c)) for c in clauses}
    assert not any(x in raw for r in (1, 2) for x in it.combinations((-2, -1), r)), \
        "...and no INPUT clause does, which is why the reference is the reduced formula"


def test_keeps_a_two_clause_inference():
    """A clause proved by resolution of two clauses is not subsumed and is kept.

    (-1|-2|3) and (-1|-2|-3) both go unit under (1, 2) and force opposite literals, proving
    (-1|-2)."""
    clauses = [[-1, -2, 3], [-1, -2, -3], [4, 5], [-4, 5], [4, -5]]
    res = mine_entailed(5, clauses, DEV, widths=(1, 2), seconds=30.0)
    assert [-2, -1] in [sorted(c) for c in res.clauses]


def test_fuzz_never_emits_a_subsumed_clause():
    """On random formulas, no mined clause is subsumed by the baseline-reduced formula."""
    rng = random.Random(11)
    checked = 0
    for _ in range(40):
        nv = rng.randint(4, 12)
        clauses = [c for c in ([list({rng.choice([1, -1]) * rng.randint(1, nv)
                                      for _ in range(rng.randint(1, 3))})
                                for _ in range(rng.randint(4, 24))]) if c]
        res = mine_entailed(nv, clauses, DEV, widths=(1, 2), seconds=30.0)
        if not res.clauses:
            continue
        p = LiteralProbe(nv, clauses, DEV)
        sub = _Subsumption(2)
        sub.rebuild(clauses, nv, p.assigned_literals())
        for cl in res.clauses:
            t = tuple(sorted(cl))
            if len(t) > 1:                      # width 1 is provably never subsumed
                checked += 1
                assert not sub.subsumes(t), f"emitted {t}, which the formula subsumes"
    # Most of these formulas mine units only, which the loop skips, so require one wide clause.
    assert checked, "the fuzz produced no wide clause: the subsumption check never ran"


def test_mining_is_kernel_invariant():
    """What mining returns does not depend on which propagate kernel ran.

    The implication counter is order-dependent on conflicted rows, so no filter may read it; the
    host filter works on a formula both kernels agree on."""
    clauses = [[-4], [-1, -2, 4], [-1, -3, 4], [5, 6], [-5, 6], [5, -6], [2, 3], [1, 7]]
    out = {}
    for mode in ("scalar", "wave"):
        orig = LiteralProbe.__init__
        try:
            gpap.engine.probe.LiteralProbe.__init__ = (
                lambda self, nv, cl, dev=None, _o=orig, _m=mode, **kw: _o(self, nv, cl, dev,
                                                                          mode=_m))
            res = mine_entailed(7, clauses, DEV, widths=(1, 2), seconds=30.0)
        finally:
            gpap.engine.probe.LiteralProbe.__init__ = orig
        out[mode] = sorted(map(sorted, res.clauses))
    assert out["scalar"] == out["wave"]


# ---------------------------------------------------------------- mining work metrics
# Mining reports what the engine did, not only what it found. Clauses/s and probes/s depend on
# the instance; implications/s is engine work, so it compares across instances, GPUs and engine
# changes.

def test_implications_are_counted_and_rated():
    clauses = [[1, 2], [-1, 3], [-2, -3], [3, 4], [-4, 5], [-4, -5], [6, 7], [-6, 7]]
    res = mine_entailed(8, clauses, DEV, widths=(1, 2), seconds=30.0)
    assert res.implications > 0, "a real sweep derives implications"
    assert res.implications_per_s > 0
    assert "implications/s" in res.summary()


def test_implications_survive_the_probe_rebuild():
    """The probe is rebuilt between widths when a wide tier proves something, which resets the
    engine counters; they must be banked first or the total loses a width."""
    clauses = [[1, 2], [1, -2], [-1, -3, -4], [-1, -3, 4],
               [-5, -6, 7], [-5, -6, -7], [5, 6, 8], [2, 8, -5]]
    one = mine_entailed(8, clauses, DEV, widths=(1,), seconds=30.0)
    three = mine_entailed(8, clauses, DEV, widths=(1, 2, 3), seconds=30.0)
    assert three.implications > one.implications, \
        "mining more widths must not report FEWER implications"


def test_width_one_skips_the_vacuous_subsumption_check():
    """A width-1 unit cannot be subsumed: ~u was probed only because it was free, so u was not
    already a unit. The fast path that skips the check relies on this."""
    clauses = [[-1, 2], [-1, -2], [3, 4], [-4, 5], [-4, -5], [6, 7]]
    res = mine_entailed(7, clauses, DEV, widths=(1,), seconds=30.0)
    formula_units = {c[0] for c in clauses if len(c) == 1}
    mined = {c[0] for c in res.clauses if len(c) == 1}
    assert not (formula_units & mined), "a mined unit can never be one the formula had"
    assert len(mined) == len(res.clauses), "width 1 yields units only"


# ---------------------------------------------------------------- mining pipeline defaults
# The pipeline's mining gates do what the help text says. A gate that wrongly declines to mine
# looks exactly like an instance with nothing to mine.

def test_mine_on_forces_past_the_size_gate():
    """`--mine on` forces mining below MINE_MIN_CLAUSES; `auto` respects the gate."""
    from gpap.sampling import pipeline

    assert pipeline.MINE_MIN_CLAUSES > 0
    # a formula far below the gate: `auto` must decline, `on` must not
    nv = 6
    clauses = [[1, 2], [-1, 3], [2, -3], [4, 5], [-4, 5], [4, -5], [6, -6]]
    got = {}
    for mode in (True, "auto"):
        info = sample_clauses(nv, clauses, None, DEV, n_time=200, time_limit=10.0, mine=mode)
        got[mode] = info.get("mined")
    assert got["auto"] is None, "auto must respect the size gate"
    assert got[True] is not None, "--mine on must force past it"


def test_the_keep_if_it_helps_guard_reports_its_decision(circuit_cnfs):
    """The keep-if-it-helps guard reports `kept`/`gain` consistently, and the run is sound.

    Mined clauses are entailed, so the decision cannot change the model set; this checks the
    bookkeeping, not the direction."""
    c = circuit_cnfs["b03-c25"]
    info = sample_clauses(c["nv"], c["clauses"], c["support"], DEV, n_time=2000,
                          time_limit=30.0, mine=True, verify_sound=True)
    mi = info.get("mined")
    if mi is None:
        pytest.skip("nothing mined on this instance")
    assert "kept" in mi and "gain" in mi
    if mi["gain"] is not None and mi["kept"]:
        assert mi["gain"] > 1.0, "clauses were kept, so the guard must have seen a gain"
    assert info["sound_check"] == "ok", "the run must be sound whether or not the clauses were kept"


# -------------------------------------------- the `derivable` drop level (_derivable_mask)

# ---------------------------------------------------------------- derivable
# `drop="derivable"` drops a clause only when propagation already does every job it does. A
# clause (l1|...|lk) forces lj when the other k-1 literals are false; all k directions must be
# covered. See docs/findings/MINING_FINDINGS.md.

def test_a_transitively_derivable_clause_is_detected():
    """(-1|3) is entailed and not subsumed, but propagation derives it through 1 -> 2 -> 3."""
    cls = [[-1, 2], [-2, 3], [4, 5]]
    mask = _derivable_mask([[-1, 3], [4, 5]], cls, 5, DEV)
    assert bool(mask[0]), "(-1|3) is derivable through the chain 1 -> 2 -> 3"


def test_a_genuine_clause_survives():
    """(1|2) where neither literal propagates the other has to be kept."""
    cls = [[1, 2, 3], [1, 2, -3], [4, 5]]
    mask = _derivable_mask([[1, 2]], cls, 5, DEV)
    assert not bool(mask[0]), "(1|2) is a real two-clause inference, not a shortcut"


def test_units_are_never_marked_derivable():
    """A mined unit u came from probing ~u, which was free, so propagation cannot already
    force u."""
    mask = _derivable_mask([[1], [2]], [[1, 2], [-1, 3], [2, -3]], 3, DEV)
    assert not mask.any()


def test_drop_levels_are_a_ladder():
    """Each level removes a superset of the one before it."""
    cls = [[-4], [-1, -2, 4], [-1, -3, 4], [5, 6], [-5, 6], [5, -6], [2, 3], [1, 7]]
    got = {}
    for lvl in ("none", "subsumed", "derivable"):
        r = mine_entailed(7, cls, DEV, widths=(1, 2), seconds=30.0, drop=lvl)
        got[lvl] = {tuple(sorted(c)) for c in r.clauses}
    assert got["derivable"] <= got["subsumed"] <= got["none"]


def test_what_it_drops_really_is_covered_in_every_direction():
    """Every marked clause is covered in every direction, checked with the reference BCP.

    For each literal, assuming the others false must conflict or force it. The candidates are
    the entailed binaries from a brute-force truth table, so no engine code decides what is
    entailed."""
    from tests.reference.bcp import lit_idx, ref_bcp

    rng = random.Random(7)
    checked = 0
    for _ in range(30):
        nv = rng.randint(3, 6)
        cls = random_cnf(rng, nv, rng.randint(3, 9), kmin=2, kmax=3)
        mods = _brute_models(nv, cls)
        if not mods:
            continue
        cands = _entailed_binaries(nv, mods)
        if not cands:
            continue
        for c, marked in zip(cands, _derivable_mask(cands, cls, nv, DEV)):
            if not marked:
                continue
            checked += 1
            for j, want in enumerate(c):
                conflict, assigned = ref_bcp(
                    cls, [lit_idx(-x) for k, x in enumerate(c) if k != j])
                assert conflict or lit_idx(want) in assigned, \
                    f"{c} was dropped but nothing re-derives its {want:+d} direction"
    assert checked, "the fuzz produced no marked clauses, so it proved nothing"


def test_a_one_way_implication_is_kept():
    """A clause covered in only one direction is kept.

    Probing 1 forces 5 and 6, then (-5|-6|2) forces 2, so (-1|2) is entailed. Assuming -2
    leaves (-5|-6), not a unit, so -1 is never forced; dropping the clause would lose that."""
    base = [[-1, 5], [-1, 6], [-5, -6, 2], [3, 4]]
    assert not bool(_derivable_mask([[-1, 2]], base, 6, DEV)[0])
    # a clause covered in both directions is still dropped
    assert bool(_derivable_mask([[-1, 3]], [[-1, 2], [-2, 3], [4, 5]], 5, DEV)[0])

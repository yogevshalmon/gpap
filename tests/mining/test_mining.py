"""The tuple-probe primitive (gpap.engine.probe) and entailed-clause mining (gpap.mining).

The safety property that matters is ENTAILMENT: every clause mining returns must already follow
from the formula, so adding it leaves the model set exactly unchanged. That is checked here the
only way worth checking it — against an independent SAT solver, on the actual clauses, not by
re-running our own propagation.
"""
import itertools as it
import random

import numpy as np
import pytest
import torch

import gpap.engine.probe
from gpap.engine import core_available
from gpap.engine.probe import LiteralProbe, ProbeBudget
from gpap.mining import _derivable_mask, _Subsumption, mine_entailed
from gpap.sampling.pipeline import sample_clauses
from tests.reference.combinations import combinations

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
AVAIL = torch.cuda.is_available() and core_available()
pytestmark = pytest.mark.skipif(not AVAIL, reason="requires CUDA + a built libgpap")


def _random_cnf(rng, nv, nc, kmax=3):
    return [[rng.choice([-1, 1]) * v for v in rng.sample(range(1, nv + 1),
                                                         rng.randint(2, min(kmax, nv)))]
            for _ in range(nc)]


def _brute_models(nv, clauses):
    """Every satisfying assignment, as sets of DIMACS literals. Brute force, small `nv`."""
    out = []
    for bits in it.product([False, True], repeat=nv):
        asg = {(i + 1) if b else -(i + 1) for i, b in enumerate(bits)}
        if all(any(l in asg for l in c) for c in clauses):
            out.append(asg)
    return out


def _entailed_binaries(nv, mods):
    """Every binary clause true in every model — the ORACLE's notion of entailed, no engine."""
    lits = [s * v for v in range(1, nv + 1) for s in (1, -1)]
    out = []
    for i, a in enumerate(lits):
        for b in lits[i + 1:]:
            if abs(a) != abs(b) and all((a in m) or (b in m) for m in mods):
                out.append([a, b])
    return out


# ------------------------------------------------------------------ the enumerator

class TestCombinations:
    """The REFERENCE enumerator (tests/reference/combinations.py), not shipping code.

    It is here to pin the core's candidate order against an independent implementation — the same
    role tests/reference/bcp.py plays for the kernels. It used to live in `gpap.engine.probe` as a public
    helper "for callers that filter candidates before probing"; that caller was measured at 65s to
    drop 0 of 122M and deleted, leaving a second enumerator in the shipping face with nothing to
    do but drift. Literals are DIMACS here, as everywhere else."""

    def test_pairs_match_itertools(self):
        import itertools
        pool = [3, -3, 5, -5, 11, -11]
        want = sorted(t for t in itertools.combinations(pool, 2) if abs(t[0]) != abs(t[1]))
        got = sorted(map(tuple, combinations(pool, 2).tolist()))
        assert got == want

    def test_triples_match_itertools(self):
        import itertools
        pool = [3, -3, 5, -5, 11, -11, 16]
        want = sorted(t for t in itertools.combinations(pool, 3) if len({abs(x) for x in t}) == 3)
        got = sorted(map(tuple, combinations(pool, 3).tolist()))
        assert got == want

    def test_same_variable_pairs_are_dropped(self):
        """A pool of literals holds both polarities, and (l, ~l) conflicts for reasons that have
        nothing to do with the formula — probing it proves only the tautology (l | ~l)."""
        got = combinations([1, -1, 2, -2], 2).tolist()
        assert [1, -1] not in got and [2, -2] not in got
        assert len(got) == 4                                   # the four cross-variable pairs
        assert len(combinations([1, -1, 2, -2], 2, distinct_vars=False)) == 6

    def test_size_one_is_the_pool(self):
        assert combinations([7, -8, 9], 1).reshape(-1).tolist() == [7, -8, 9]

    def test_pool_smaller_than_size(self):
        assert combinations([3], 2).shape == (0, 2)

    def test_max_tuples_truncates(self):
        assert combinations(list(range(1, 21)), 2, max_tuples=5).shape == (5, 2)

    def test_rejects_zero_size(self):
        with pytest.raises(ValueError, match="size"):
            combinations([1, 2], 0)


# ------------------------------------------------------------------ the primitive

class TestLiteralProbe:
    #  (x1 v x2) ^ (~x1 v x3) ^ (~x2 v ~x3)
    CLAUSES = [[1, 2], [-1, 3], [-2, -3]]

    def test_failed_literal(self):
        """x1 implies x2 and ~x2 => assuming x1 conflicts, so ~x1 is entailed."""
        p = LiteralProbe(2, [[-1, 2], [-1, -2]], DEV)
        r = p.probe(size=1)
        assert r.conflicting.tolist() == [[1]]
        assert r.exhausted

    def test_exhausted_is_a_proof_of_absence(self):
        p = LiteralProbe(3, self.CLAUSES, DEV)
        r = p.probe(size=1)
        assert len(r) == 0
        assert r.exhausted, "no failed literal exists, and we looked at every one"

    def test_pool_defaults_to_every_free_literal(self):
        """`pool=None` is the "all the lits" case, and must equal passing them explicitly."""
        p = LiteralProbe(3, self.CLAUSES, DEV)
        assert p.probe(size=2).total == p.probe(p.free_literals(), size=2).total

    def test_count_matches_the_sweep_without_generating_it(self):
        """The ceiling is checked against count(), so count() must be the number a sweep probes.
        REGRESSION: the ceiling used to be compared against an already-materialized enumeration,
        which cannot protect against the case it exists for — at width 3 over a real circuit's
        free literals that array is ~1e11 tuples and the process was OOM-killed reaching it."""
        p = LiteralProbe(3, self.CLAUSES, DEV)
        for k in (1, 2, 3):
            assert p.count(size=k) == p.probe(size=k).total

    def test_count_saturates_instead_of_wrapping(self):
        """A wrapped count would compare BELOW a ceiling and let through the very sweep the
        ceiling exists to refuse."""
        p = LiteralProbe(400, [[1, 2], [-1, 3]], DEV)
        assert p.count(size=3) > 10**7

    def test_max_tuples_is_a_work_budget(self):
        """`max_tuples` says how many candidates to PROBE — it does not cancel the sweep. Every
        conflict found is a genuine entailed clause whatever went unprobed, so partial results
        beat no results; only `exhausted` is lost, and it says so."""
        p = LiteralProbe(3, self.CLAUSES, DEV)
        r = p.probe(size=2, budget=ProbeBudget(max_tuples=2))
        assert r.probed == 2 and r.total == 12
        assert not r.exhausted and r.stopped == "max_tuples"

    def test_refuse_over_max_is_opt_in(self):
        """...but a caller who only wants an answer it can call a PROOF can ask for nothing
        instead of a prefix."""
        p = LiteralProbe(3, self.CLAUSES, DEV)
        r = p.probe(size=2, budget=ProbeBudget(max_tuples=2, refuse_over_max=True))
        assert r.probed == 0 and r.total == 12 and r.stopped == "max_tuples"

    def test_free_literals_excludes_settled_variables(self):
        """A formula's own units settle variables at construction; probing them is wasted."""
        p = LiteralProbe(3, self.CLAUSES + [[1]], DEV)
        free = set(p.free_literals().tolist())
        assert 1 not in free and -1 not in free, "x1 is fixed by the unit clause"
        assert not p.is_free(1)

    def test_assume_narrows_without_rebuilding(self):
        p = LiteralProbe(3, self.CLAUSES, DEV)
        assert p.is_free(1)
        p.assume([1])                                   # assert x1
        assert not p.is_free(1), "x1 must be settled after being assumed"
        assert set(p.free_literals().tolist()).isdisjoint({1, -1})

    def test_unsat_formula_probes_nothing(self):
        p = LiteralProbe(1, [[1], [-1]], DEV)
        assert p.unsat
        r = p.probe(size=1)
        assert r.probed == 0 and r.stopped == "unsat"

    def test_chunked_matches_single_launch(self):
        """A sweep split across launches must give exactly the same answer. The chunk size is the
        core's business, so it is set explicitly here rather than monkeypatched."""
        rng = random.Random(3)
        clauses = _random_cnf(rng, 12, 40)
        p = LiteralProbe(12, clauses, DEV)
        tuples = combinations(p.free_literals(), 2)
        one = p.probe_tuples(tuples)
        many = p.engine.probe_tuples(tuples, batch=7)     # force many launches
        with many:
            got = many.conflicts()
        assert np.array_equal(np.sort(one.conflicting, axis=0), np.sort(got, axis=0))

    def test_rejects_out_of_range_literals(self):
        """An index past the row's bitmap makes the kernel write into a LATER ROW's words, so one
        bad tuple would silently corrupt a different tuple's answer."""
        p = LiteralProbe(3, self.CLAUSES, DEV)
        with pytest.raises(RuntimeError, match="out of range"):
            p.probe([99], size=1)
        with pytest.raises(RuntimeError, match="out of range"):
            p.probe_tuples([[1, 99]])


# ------------------------------------------------------------------ the application

class TestMining:
    @staticmethod
    def _models(clauses):
        from pysat.solvers import Minisat22
        with Minisat22(bootstrap_with=clauses) as s:
            return {tuple(m) for m in s.enum_models()}

    def _check_entailed(self, nv, clauses, pool=None, **kw):
        """Every mined clause must be entailed, and the augmented formula must have exactly the
        original models — verified with an independent SAT solver."""
        from pysat.solvers import Minisat22
        res = mine_entailed(nv, clauses, DEV, pool=pool, **kw)
        for cl in res.clauses:
            with Minisat22(bootstrap_with=clauses) as s:
                assert not s.solve(assumptions=[-lit for lit in cl]), \
                    f"mined clause {cl} is NOT entailed"
        if res.clauses:
            assert self._models(clauses) == self._models(clauses + res.clauses)
        return res

    def test_planted_backbone_and_pair(self):
        # (1|2),(1|-2) => backbone unit (1), not flush-implied (there are no unit clauses);
        # (-1,-3,-4),(-1,-3,4) => assuming x3 (with x1 forced) conflicts => unit (-3);
        # (-5|-6) is entailed via x7: (-5,-6,7),(-5,-6,-7)
        clauses = [[1, 2], [1, -2], [-1, -3, -4], [-1, -3, 4],
                   [-5, -6, 7], [-5, -6, -7], [5, 6, 8], [2, 8, -5]]
        # `pool` is DIMACS LITERALS, so both polarities of x5 and x6 are named explicitly. That
        # is the point of the literal API: a caller can ask about one side only by naming one.
        res = self._check_entailed(8, clauses, pool=[5, -5, 6, -6])
        assert res.by_width.get(1, 0) >= 2, res.by_width      # units (1) and (-3) at least
        mined = {tuple(sorted(c)) for c in res.clauses}
        assert (-6, -5) in mined or (-5,) in mined or (-6,) in mined, mined

    def test_pool_polarity_is_honoured(self):
        """Naming ONE polarity probes only that side — the reason the pool is literals and not
        variables. (5, 6) proves (-5 | -6); (-5, -6) is a different question entirely."""
        clauses = [[-5, -6, 7], [-5, -6, -7], [5, 6, 8], [2, 8, -5]]
        both = self._check_entailed(8, clauses, pool=[5, -5, 6, -6], widths=(2,))
        one = self._check_entailed(8, clauses, pool=[5, 6], widths=(2,))
        assert (-6, -5) in {tuple(sorted(c)) for c in both.clauses}
        assert len(one.clauses) <= len(both.clauses), "one polarity cannot prove more"

    def test_random_formulas_stay_model_preserving(self):
        from pysat.solvers import Minisat22
        rng = random.Random(7)
        mined = 0
        for _ in range(10):
            nv = rng.randint(6, 14)
            clauses = _random_cnf(rng, nv, rng.randint(2 * nv, 5 * nv))
            with Minisat22(bootstrap_with=clauses) as s:
                if not s.solve():
                    continue                                  # UNSAT input: mining may derive it
            # DIMACS literals, both polarities of the first few variables. It used to be
            # `range(min(nv, 6))`, which starts at 0 — not a literal at all, silently dropped by
            # the free-variable filter, so the pool was one variable short and one-sided.
            k = min(nv, 6)
            pool = [sign * v for v in range(1, k + 1) for sign in (1, -1)]
            mined += len(self._check_entailed(nv, clauses, pool=pool).clauses)
        # `_check_entailed` compares model sets only when something WAS mined, so a miner that
        # returned nothing would satisfy every assertion in it, ten times over.
        assert mined, "nothing was mined on any instance: the entailment check never ran"

    def test_widths_are_selectable(self):
        clauses = [[1, 2], [1, -2], [-1, -3, -4], [-1, -3, 4]]
        only_units = mine_entailed(4, clauses, DEV, widths=(1,))
        assert set(only_units.by_width) == {1}
        assert all(len(c) == 1 for c in only_units.clauses)

    def test_max_tuples_skips_rather_than_truncates(self):
        """A truncated width would prove nothing about what it did not reach, so a width over
        its ceiling is skipped outright and SAYS so."""
        clauses = [[1, 2], [-1, 3], [-2, -3], [4, 5], [-4, 6], [-5, -6]]
        res = mine_entailed(6, clauses, DEV, widths=(1, 2), max_tuples={2: 1})
        assert 2 in res.skipped and "candidates" in res.skipped[2]
        assert res.exhausted.get(2) is False

    def test_wide_widths_gated_on_unit_yield(self):
        clauses = [[1, 2], [-1, 3], [-2, -3]]              # no failed literals at width 1
        res = mine_entailed(3, clauses, DEV, min_units_for_wide=1)
        assert res.by_width.get(1, 0) == 0
        assert res.skipped.get(2) == "too few units at width 1"

    def test_unsat_formula_is_reported_not_mined(self):
        res = mine_entailed(1, [[1], [-1]], DEV)
        assert res.clauses == []
        assert all("unsat" in v for v in res.skipped.values())

    def test_result_summary_is_readable(self):
        res = mine_entailed(4, [[1, 2], [1, -2]], DEV)
        assert "width 1" in res.summary() and "s" in res.summary()


# ------------------------------------------------------------------ the CLI

class TestMineCLI:
    """`gpap mine` — the face a user actually reaches for. Run as a subprocess, because what is
    being checked is the command's contract (flags, exit status, files it writes), not a Python
    call."""

    @staticmethod
    def _run(*argv, cwd=None):
        import subprocess
        import sys
        r = subprocess.run([sys.executable, "-m", "gpap.cli", "mine", *map(str, argv)],
                           capture_output=True, text=True, timeout=600, cwd=cwd)
        assert r.returncode == 0, f"gpap mine failed:\n{r.stdout}\n{r.stderr}"
        return r.stdout

    @staticmethod
    def _cnf(tmp_path, nv, clauses, name="f.cnf", ind=None):
        p = tmp_path / name
        head = f"p cnf {nv} {len(clauses)}\n"
        if ind:
            head += "c ind " + " ".join(map(str, ind)) + " 0\n"
        p.write_text(head + "".join(" ".join(map(str, c)) + " 0\n" for c in clauses))
        return p

    def test_mines_and_writes_a_strengthened_formula(self, tmp_path):
        clauses = [[-1, 2], [-1, -2], [3, 4], [6, 7]]
        cnf = self._cnf(tmp_path, 7, clauses)
        out = tmp_path / "strong.cnf"
        stats = tmp_path / "mine.json"
        txt = self._run(cnf, "--width", "1", "--out", out, "--stats", stats)
        assert "exhaustive" in txt
        assert out.exists() and stats.exists()

        import json
        info = json.loads(stats.read_text())
        assert info["by_width"]["1"] >= 1 and info["clauses"]
        # the strengthened file is the original plus what was proven, and every proven clause is
        # ENTAILED — so the two formulas have the same models. Checked, not asserted.
        from pysat.solvers import Minisat22
        for cl in info["clauses"]:
            with Minisat22(bootstrap_with=clauses) as s:
                assert not s.solve(assumptions=[-lit for lit in cl]), cl

    def test_pool_support_uses_the_c_ind_lines(self, tmp_path):
        cnf = self._cnf(tmp_path, 7, [[-1, 2], [-1, -2], [3, 4], [6, 7]], ind=[1, 3, 6])
        txt = self._run(cnf, "--width", "2", "--pool", "support")
        assert "pool=support (6 literals)" in txt, txt    # 3 variables, both polarities

    def test_pool_rejects_nonsense_instead_of_guessing(self, tmp_path):
        import subprocess
        import sys
        cnf = self._cnf(tmp_path, 7, [[1, 2]])
        for bad in ("99", "abc"):
            r = subprocess.run([sys.executable, "-m", "gpap.cli", "mine", str(cnf),
                                "--pool", bad], capture_output=True, text=True, timeout=300)
            assert r.returncode != 0, f"--pool {bad} should have been refused"

    def test_huge_width_probes_its_budget_and_reports_coverage(self, tmp_path):
        """An enormous width must still DO the work asked for. It reports how much of the space
        it covered, so "nothing entailed here" is distinguishable from "we barely started" —
        and it never materializes the enumeration (which OOM-killed the process)."""
        nv = 300
        cnf = self._cnf(tmp_path, nv, [[i, -(i + 1)] for i in range(1, nv)])
        txt = self._run(cnf, "--width", "3", "--max-tuples", "1000", "--seconds", "20")
        assert "probed 1,000 of" in txt, txt
        assert "NOT searched exhaustively" in txt, txt


class TestProbeDispatch:
    """The probe path must get the kernel dispatch, like every other consumer of the engine.

    REGRESSION. `LiteralProbe` pinned "scalar" when the sweep was ported to the core, because the
    dispatch lived in the two WRAPPERS (BCPEngine.plan_for / gpap::Engine::plan_for) and the sweep
    calls the C ABI directly. That is the same hole the batched-assumption path had until
    for a while, and it cost 1.28x (M=14.5k) to 2.20x (M=219k) on the sweep. The
    fix was to resolve GPAP_MODE_AUTO inside gpap_engine_reset — one place every entry point
    passes through — so this test is about WHERE the rule lives, not about which kernel wins."""

    def test_probe_dispatches_by_clause_count(self, monkeypatch):
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
    def test_an_explicit_mode_is_honoured(self, pinned, monkeypatch):
        """A measurement pins the kernel; the per-reset dispatch must not take it back."""
        monkeypatch.delenv("GPAP_KERNEL", raising=False)
        rng = random.Random(4)
        big = [[rng.choice([1, -1]) * rng.randint(1, 400) for _ in range(3)] for _ in range(2500)]
        p = LiteralProbe(400, big, DEV, mode=pinned)
        p.probe(size=1)
        assert p.engine.mode == pinned

    def test_env_var_pins(self, monkeypatch):
        monkeypatch.setenv("GPAP_KERNEL", "scalar")
        rng = random.Random(5)
        big = [[rng.choice([1, -1]) * rng.randint(1, 400) for _ in range(3)] for _ in range(2500)]
        p = LiteralProbe(400, big, DEV)
        p.probe(size=1)
        assert p.engine.mode == "scalar"


class TestSubsumption:
    """REGRESSION: mining must not emit clauses that a shorter known clause already implies.

    They are sound (still entailed) but useless, and they make the formula bigger — which slows
    every later propagation, the exact opposite of why anyone mines. Checking only for an EXACT
    duplicate let 1,504 of them through on b03-c25 (1,511 ternaries where 7 are useful)."""

    def test_subsumed_clauses_are_not_emitted(self):
        # (-1 | -2) is given. Any triple containing it -- (-1 | -2 | x) -- is implied by it.
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

    def test_coverage_is_reported_so_empty_is_readable(self):
        """`0 clauses` from an exhaustive sweep and `0 clauses` from 0.0002% of one mean opposite
        things. The result must let a caller tell them apart."""
        nv = 400
        clauses = [[i, -(i + 1)] for i in range(1, nv)]
        partial = mine_entailed(nv, clauses, DEV, widths=(3,), seconds=5.0, max_tuples=500)
        assert partial.candidates[3] > partial.probed[3] > 0
        assert not partial.exhausted[3]
        assert "probed" in partial.summary()

        full = mine_entailed(4, [[1, 2], [-1, 3]], DEV, widths=(2,), seconds=5.0)
        assert full.exhausted[2] and full.probed[2] == full.candidates[2]


class TestSubsumptionFilter:
    """`mining._Subsumption`: the ONE redundancy filter, and it runs on the host.

    There used to be a second one inside the engine (`skip_subsumed`: drop a conflict the assumed
    literals alone caused). It was sound, but worth 0.0-0.3% of the result buffer on every instance
    measured, while costing an int32 per row and a branch in the innermost loop of both propagate
    kernels — so it was removed and this is what remains. Redundancy is a property of a CLAUSE
    against a FORMULA, so the host is where it belongs: that is where the formula is known.

    The reference is the formula REDUCED by the engine's baseline, not the input clauses. Mining
    folds proven units into the flush, so an input (a | b | c) whose `c` the baseline falsified
    acts as a binary constraint; a conflict on (~a, ~b) then proves (a | b), which no INPUT clause
    subsumes. Testing against the raw input let 17 such clauses through on b03-c25."""

    def test_drops_only_what_the_baseline_makes_redundant(self):
        """(-1 | -2 | 3) with 3 forced FALSE by the flush is a binary constraint during the sweep,
        so assuming (1, 2) proves (-1 | -2) — entailed, and subsumed by the RESIDUE only."""
        clauses = [[-3], [-1, -2, 3], [4, 5], [-4, 5], [4, -5]]
        p = LiteralProbe(5, clauses, DEV)
        assert -3 in p.assigned_literals().tolist(), "the flush must have settled 3"
        sub = _Subsumption(2)
        sub.rebuild(clauses, 5, p.assigned_literals())
        assert sub.subsumes((-2, -1)), "the residue of (-1 | -2 | 3) subsumes it"
        raw = {tuple(sorted(c)) for c in clauses}
        assert not any(x in raw for r in (1, 2) for x in it.combinations((-2, -1), r)), \
            "...and no INPUT clause does, which is why the reference is the reduced formula"

    def test_keeps_a_two_clause_inference(self):
        """A clause proved by RESOLUTION between two clauses is not subsumed and must survive.

        (-1|-2|3) and (-1|-2|-3) each go unit under (1, 2) and force opposite literals, proving
        (-1|-2). Nothing subsumes it. An in-engine filter that keyed on how EARLY the conflict came
        threw 28 such clauses away on b03-c25 before it was removed."""
        clauses = [[-1, -2, 3], [-1, -2, -3], [4, 5], [-4, 5], [4, -5]]
        res = mine_entailed(5, clauses, DEV, widths=(1, 2), seconds=30.0)
        assert [-2, -1] in [sorted(c) for c in res.clauses]

    def test_fuzz_never_emits_a_subsumed_clause(self):
        """The property mining promises, on random formulas: nothing it returns is subsumed by the
        formula it was mined from (reduced by the baseline), and everything it returns is entailed."""
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
        # Most of these random formulas mine units only, and a unit is skipped above — so
        # without this the loop can run 40 instances and test the filter on nothing.
        assert checked, "the fuzz produced no wide clause: the subsumption check never ran"

    def test_mining_is_kernel_invariant(self):
        """What mining RETURNS must not depend on which propagate kernel ran.

        It did once: the engine-side filter read the implication counter, which is order-dependent
        on conflicted rows, so wave and scalar dropped different subsets and b03-c25 emitted 3,096
        clauses under scalar against 3,113 under wave. With the filter on the host, over a formula
        both kernels agree on, the answer is the same by construction — pinned here anyway."""
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


class TestMiningWorkMetrics:
    """Mining reports what the ENGINE did, not only what it found.

    clauses/s and probes/s both depend on the instance (how many clauses it entails, how quickly a
    tuple conflicts). Implications/s is hardware work, so it is the number that compares across
    instances, GPUs and engine changes — the same reason the sampler reports it."""

    def test_implications_are_counted_and_rated(self):
        clauses = [[1, 2], [-1, 3], [-2, -3], [3, 4], [-4, 5], [-4, -5], [6, 7], [-6, 7]]
        res = mine_entailed(8, clauses, DEV, widths=(1, 2), seconds=30.0)
        assert res.implications > 0, "a real sweep derives implications"
        assert res.implications_per_s > 0
        assert "implications/s" in res.summary()

    def test_implications_survive_the_probe_rebuild(self):
        """The probe is REBUILT between widths when a wide tier proves something, which resets the
        engine's counters — they must be banked first or the total silently loses a width."""
        clauses = [[1, 2], [1, -2], [-1, -3, -4], [-1, -3, 4],
                   [-5, -6, 7], [-5, -6, -7], [5, 6, 8], [2, 8, -5]]
        one = mine_entailed(8, clauses, DEV, widths=(1,), seconds=30.0)
        three = mine_entailed(8, clauses, DEV, widths=(1, 2, 3), seconds=30.0)
        assert three.implications > one.implications, \
            "mining more widths must not report FEWER implications"

    def test_width_one_skips_the_vacuous_subsumption_check(self):
        """A width-1 unit cannot be subsumed: it came from probing ~u, which was probed only
        because it was free, so u was not already a unit. Skipping the check must not change the
        output — this pins the invariant, since the fast path assumes it."""
        clauses = [[-1, 2], [-1, -2], [3, 4], [-4, 5], [-4, -5], [6, 7]]
        res = mine_entailed(7, clauses, DEV, widths=(1,), seconds=30.0)
        formula_units = {c[0] for c in clauses if len(c) == 1}
        mined = {c[0] for c in res.clauses if len(c) == 1}
        assert not (formula_units & mined), "a mined unit can never be one the formula had"
        assert len(mined) == len(res.clauses), "width 1 yields units only"


class TestMiningPipelineDefaults:
    """The pipeline's mining GATES — the two knobs a user is told about, checked to do what the
    help text says. Both were wrong at some point and both were invisible when wrong: a gate that
    silently declines to mine looks exactly like an instance with nothing to mine."""

    def test_mine_on_forces_past_the_size_gate(self):
        """`--mine on` is documented as forcing; `auto` respects MINE_MIN_CLAUSES.

        They used to be identical, because the thing `on` bypassed was the ABC provenance gate —
        since deleted — while the SIZE gate applied to both. So on any formula below the
        threshold `--mine on` quietly did nothing at all."""
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

    def test_the_keep_if_it_helps_guard_reports_its_decision(self, circuit_cnfs):
        """The guard reports what it decided, and the run is SOUND either way.

        Whichever way it goes the model set is unchanged — mined clauses are entailed — so this
        pins the bookkeeping (`kept`/`gain` present and consistent), not the direction."""
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

class TestDerivable:
    """`drop="derivable"`: a clause propagation can already do EVERY job of.

    The quantifier is the whole content of the filter — a clause `(l1|...|lk)` forces `lj` when
    the other k-1 literals are false, and only when propagation covers ALL k of those directions
    does dropping it cost the consumer nothing. See docs/findings/MINING_FINDINGS.md.
    """

    def test_a_transitively_derivable_clause_is_detected(self):
        """(-1|3) is entailed and NOT subsumed, but 1 propagates 3 through (-1|2) and (-2|3) --
        i.e. 1 -> 2 -> 3 -- so unit propagation produces the clause on demand."""
        cls = [[-1, 2], [-2, 3], [4, 5]]
        mask = _derivable_mask([[-1, 3], [4, 5]], cls, 5, DEV)
        assert bool(mask[0]), "(-1|3) is derivable through the chain 1 -> 2 -> 3"

    def test_a_genuine_clause_survives(self):
        """(1|2) where neither literal propagates the other has to be kept."""
        cls = [[1, 2, 3], [1, 2, -3], [4, 5]]
        mask = _derivable_mask([[1, 2]], cls, 5, DEV)
        assert not bool(mask[0]), "(1|2) is a real two-clause inference, not a shortcut"

    def test_units_are_never_marked_derivable(self):
        """A mined unit (u) came from probing ~u, which was probed only because it was FREE — so
        propagation cannot already force u, and the test would be vacuous."""
        mask = _derivable_mask([[1], [2]], [[1, 2], [-1, 3], [2, -3]], 3, DEV)
        assert not mask.any()

    def test_drop_levels_are_a_ladder(self):
        """Each level removes a superset of the one before it."""
        cls = [[-4], [-1, -2, 4], [-1, -3, 4], [5, 6], [-5, 6], [5, -6], [2, 3], [1, 7]]
        got = {}
        for lvl in ("none", "subsumed", "derivable"):
            r = mine_entailed(7, cls, DEV, widths=(1, 2), seconds=30.0, drop=lvl)
            got[lvl] = {tuple(sorted(c)) for c in r.clauses}
        assert got["derivable"] <= got["subsumed"] <= got["none"]

    def test_what_it_drops_really_is_covered_in_every_direction(self):
        """The property the filter promises, against an oracle that shares no code with the engine.

        A marked clause must be REPLACEABLE: for every one of its literals, assuming the others
        false must already conflict or force it. The candidates are every ENTAILED binary clause,
        computed by brute force over the truth table — so the fuzz feeds the filter exactly the
        population it is meant to thin, without asking the engine what is entailed."""
        from tests.reference.bcp import lit_idx, ref_bcp

        rng = random.Random(7)
        checked = 0
        for _ in range(30):
            nv = rng.randint(3, 6)
            cls = _random_cnf(rng, nv, rng.randint(3, 9))
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

    def test_a_one_way_implication_is_kept(self):
        """The case that makes the quantifier matter, small enough to read by hand.

        Probing 1 gives 5 and 6, and only THEN does (-5|-6|2) go unit and force 2 — so (-1|2) is
        entailed. The reverse does not propagate: assuming -2 leaves (-5|-6), not a unit, so -1 is
        never forced. One direction covered, one not, and the clause has to stay — dropping it
        would remove a propagation the consumer had."""
        base = [[-1, 5], [-1, 6], [-5, -6, 2], [3, 4]]
        assert not bool(_derivable_mask([[-1, 2]], base, 6, DEV)[0])
        # ...while a clause whose BOTH directions propagate is still dropped
        assert bool(_derivable_mask([[-1, 3]], [[-1, 2], [-2, 3], [4, 5]], 5, DEV)[0])

"""`gpap.implications`: the closure sweep and the three things read off it.

The properties that matter are all SOUNDNESS properties, and every one of them is checked against
a brute-force reference over the full truth table rather than against another GPAP path:

    l implies m       =>  every model with l true has m true
    l == m            =>  every model agrees on them (so substituting one preserves the model set)
    l failed          =>  no model has l true
"""
import itertools
import random

import pytest
import torch

from gpap.engine import core_available
from gpap.implications import (
    closure_bytes,
    derivable_mask,
    equivalence_classes,
    equivalence_edges,
    hyper_binary,
    implications,
)

pytestmark = pytest.mark.gpu

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def setup_module(_):
    if not torch.cuda.is_available() or not core_available():
        pytest.skip("needs CUDA + libgpap", allow_module_level=True)


def models(num_vars, clauses):
    """Every satisfying assignment, as sets of DIMACS literals. Brute force, small `num_vars`."""
    out = []
    for bits in itertools.product([False, True], repeat=num_vars):
        asg = {(i + 1) if b else -(i + 1) for i, b in enumerate(bits)}
        if all(any(l in asg for l in c) for c in clauses):
            out.append(asg)
    return out


def random_cnf(rng, num_vars, n_clauses, lo=1):
    """Random 1-3 literal clauses over variables `lo..num_vars`.

    `lo` exists so a test can PLANT a structure (an equivalence, a failed literal) on the
    variables below it and know the noise cannot settle it — a planted property that the noise
    can propagate away is a property the test silently stops checking."""
    cls = []
    for _ in range(n_clauses):
        w = rng.randint(1, 3)
        c = list({rng.choice([1, -1]) * rng.randint(lo, num_vars) for _ in range(w)})
        if c:
            cls.append(c)
    return cls


class TestClosure:
    def test_a_hand_checkable_formula(self):
        """1 <-> 2 by two binary clauses; 3 is a failed literal (it forces 4 and -4)."""
        cls = [[-1, 2], [1, -2], [-3, 4], [-3, -4], [5, 6]]
        imp = implications(6, cls, DEV)
        assert imp.failed.tolist() == [3]
        assert imp.units == [[-3]]
        assert set(imp.implies(1).tolist()) == {1, 2}
        assert equivalence_classes(imp) == [[-2, -1], [1, 2]]

    def test_settled_literals_are_not_probed(self):
        """A literal the formula's own units already fixed teaches nothing, and a vacuous row in
        the matrix would make it look equivalent to everything."""
        imp = implications(4, [[1], [-2, 3], [2, -3], [4, -4]], DEV)
        assert 1 not in imp.literals.tolist() and -1 not in imp.literals.tolist()

    def test_refuses_a_closure_it_cannot_hold(self):
        """The cost is knowable up front, so an impossible request is refused, not attempted."""
        with pytest.raises(ValueError, match="Narrow the pool"):
            implications(500, [[1, 2], [-1, 3], [2, -3]], DEV, max_bytes=1000)
        assert closure_bytes(500, 100) == 100 * 1000


class TestSoundness:
    """Against the full truth table — the only reference that cannot share a bug with the engine."""

    @pytest.mark.parametrize("seed", range(25))
    def test_implications_hold_in_every_model(self, seed):
        rng = random.Random(seed)
        nv = rng.randint(3, 7)
        cls = random_cnf(rng, nv, rng.randint(3, 10))
        ms = models(nv, cls)
        if not ms:
            pytest.skip("unsat")
        imp = implications(nv, cls, DEV)
        for l in imp.literals.tolist():
            for m in imp.implies(int(l)).tolist():
                for asg in ms:
                    if l in asg:
                        assert m in asg, f"{l} => {m} fails in {sorted(asg)}"

    @pytest.mark.parametrize("seed", range(25))
    def test_failed_literals_are_in_no_model(self, seed):
        """A PLANTED failed literal: `nv` forces both polarities of `nv-1`, so it is in no model.

        Random CNFs of this size almost never contain one — measured 2 failed literals across 20
        SAT instances, i.e. 18 of them asserted nothing. The plant makes every run test the
        property; the random tail still varies what surrounds it."""
        rng = random.Random(seed + 100)
        nv = rng.randint(5, 9)
        f, w = nv, nv - 1              # planted: f forces w and -w; the noise stays off both
        cls = [[-f, w], [-f, -w]] + random_cnf(rng, nv, rng.randint(3, 10), lo=1)
        cls = [c for c in cls if all(abs(x) < w for x in c)] + [[-f, w], [-f, -w]]
        ms = models(nv, cls)
        if not ms:
            pytest.skip("unsat")
        imp = implications(nv, cls, DEV)
        failed = imp.failed.tolist()
        assert f in failed, f"planted failed literal {f} was not detected"
        for l in failed:
            assert all(int(l) not in asg for asg in ms), f"{l} is failed but appears in a model"

    @pytest.mark.parametrize("seed", range(25))
    def test_equivalent_literals_agree_in_every_model(self, seed):
        """The property that licenses substitution: the model SET is preserved.

        ⚠️ A PLANTED equivalence, not a bare random CNF. Measured: over 19 SAT random formulas of
        this shape the sweep produced ZERO equivalence classes, so the loop body never ran and the
        test executed no assertion at all — it passed with `equivalence_classes` stubbed to return
        `[]`. Planting `2k-1 <-> 2k` guarantees the property is actually exercised, and the
        coverage assertion below keeps it that way if the generator ever drifts."""
        rng = random.Random(seed + 200)
        nv = rng.randint(5, 9)
        a, b = 1, 2                    # planted: 1 <-> 2; the noise lives on 3..nv and cannot
        cls = [[-a, b], [a, -b]] + random_cnf(rng, nv, rng.randint(3, 10), lo=3)
        ms = models(nv, cls)
        if not ms:
            pytest.skip("unsat")
        imp = implications(nv, cls, DEV)
        classes = equivalence_classes(imp)
        assert classes, "no equivalence class found on a formula that plants one — vacuous run"
        checked = 0
        for cl in classes:
            for x, y in itertools.combinations(cl, 2):
                for asg in ms:
                    assert (x in asg) == (y in asg), f"{x} == {y} fails in {sorted(asg)}"
                    checked += 1
        assert checked, "the property was never evaluated"

    @pytest.mark.parametrize("seed", range(15))
    def test_hyper_binary_clauses_are_entailed(self, seed):
        """Adding them must leave the model set EXACTLY unchanged."""
        rng = random.Random(seed + 300)
        nv = rng.randint(3, 6)
        cls = random_cnf(rng, nv, rng.randint(3, 9))
        ms = models(nv, cls)
        if not ms:
            pytest.skip("unsat")
        imp = implications(nv, cls, DEV)
        hb = hyper_binary(imp)
        for c in hb:
            for asg in ms:
                assert any(l in asg for l in c), f"{c} is falsified by a model"
        assert models(nv, cls + hb) == ms, "adding them changed the model set"


class TestEquivalenceEdges:
    def test_edges_are_a_superset_of_the_binary_clause_route(self):
        """The route the sampler ships today builds edges from BINARY CLAUSES only. Two mutual
        binary clauses obviously propagate each other, so everything it finds is here too — plus
        the equivalences that only exist THROUGH longer clauses. Verified on b03-c25 at 0 of 7,856
        missing; this is the same property on a formula small enough to read."""
        # 1<->2 spelled out in binaries; 3<->4 only through TERNARIES, and note they have to go
        # unit for BCP to see them -- (-3|4|5) and (-3|4|-5) would need RESOLUTION, which unit
        # propagation cannot do. The unit 5 is what reduces them.
        cls = [[5], [-1, 2], [1, -2], [-3, -5, 4], [-4, -5, 3], [6, 7]]
        imp = implications(7, cls, DEV)
        got = {tuple(c) for c in equivalence_classes(imp)}
        assert (1, 2) in got and (-2, -1) in got, "the binary-clause equivalence must be found"
        assert (3, 4) in got, "and the one only propagation can see"

    def test_edges_feed_the_shipped_ELS(self):
        """The output format is the one `preprocess.equivalent_literal_substitution` consumes, so
        swapping the sampler's binary-clause edges for these is a one-line change."""
        from gpap.sampling.preprocess import equivalent_literal_substitution

        cls = [[-1, 2], [1, -2], [2, 3], [-2, -3, 4]]
        imp = implications(4, cls, DEV)
        edges = equivalence_edges(imp)
        assert edges.ndim == 2 and edges.shape[1] == 2
        new_clauses, var_map, st = equivalent_literal_substitution(cls, edges, 4)
        assert st["status"] == "ok"
        assert st["vars_removed"] >= 1, "1 and 2 are equivalent, so one variable must go"


class TestDerivable:
    def test_a_transitively_derivable_clause_is_detected(self):
        """(-1|3) is entailed and NOT subsumed, but 1 propagates 3 through (-1|2) and (-2|3) --
        i.e. 1 -> 2 -> 3 -- so unit propagation produces the clause on demand."""
        cls = [[-1, 2], [-2, 3], [4, 5]]
        mask = derivable_mask([[-1, 3], [4, 5]], cls, 5, DEV)
        assert bool(mask[0]), "(-1|3) is derivable through the chain 1 -> 2 -> 3"

    def test_a_genuine_clause_survives(self):
        """(1|2) where neither literal propagates the other has to be kept."""
        cls = [[1, 2, 3], [1, 2, -3], [4, 5]]
        mask = derivable_mask([[1, 2]], cls, 5, DEV)
        assert not bool(mask[0]), "(1|2) is a real two-clause inference, not a shortcut"

    def test_units_are_never_marked_derivable(self):
        """A mined unit (u) came from probing ~u, which was probed only because it was FREE — so
        propagation cannot already force u, and the test would be vacuous."""
        mask = derivable_mask([[1], [2]], [[1, 2], [-1, 3], [2, -3]], 3, DEV)
        assert not mask.any()

    def test_drop_levels_are_a_ladder(self):
        """Each level removes a superset of the one before it."""
        from gpap.mining import mine_entailed

        cls = [[-4], [-1, -2, 4], [-1, -3, 4], [5, 6], [-5, 6], [5, -6], [2, 3], [1, 7]]
        got = {}
        for lvl in ("none", "subsumed", "derivable"):
            r = mine_entailed(7, cls, DEV, widths=(1, 2), seconds=30.0, drop=lvl)
            got[lvl] = {tuple(sorted(c)) for c in r.clauses}
        assert got["derivable"] <= got["subsumed"] <= got["none"]

    def test_what_it_drops_really_is_covered_in_every_direction(self):
        """The property the filter promises, against an oracle that shares no code with the engine.

        A marked clause must be REPLACEABLE: for every one of its literals, assuming the others
        false must already conflict or force it. Fuzzed."""
        from tests.reference_bcp import lit_idx, ref_bcp

        rng = random.Random(7)
        checked = 0
        for _ in range(30):
            nv = rng.randint(3, 6)
            cls = random_cnf(rng, nv, rng.randint(3, 9))
            if not models(nv, cls):
                continue
            imp = implications(nv, cls, DEV)
            hb = hyper_binary(imp)
            if not hb:
                continue
            for c, marked in zip(hb, derivable_mask(hb, cls, nv, DEV)):
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
        assert not bool(derivable_mask([[-1, 2]], base, 6, DEV)[0])
        # ...while a clause whose BOTH directions propagate is still dropped
        assert bool(derivable_mask([[-1, 3]], [[-1, 2], [-2, 3], [4, 5]], 5, DEV)[0])


class TestNumbersInTheDocs:
    """The measured claims the module docstring makes, on the instance it names."""

    def test_closure_bytes_matches_the_quoted_size(self):
        # b03-c25: 7,588 free literals x 2N=11,402 -> 82.5 MiB, one BYTE per pair (the matrix is
        # torch.bool). The docs used to quote 11 MB, which was the bit-packed figure for a
        # representation this module does not use.
        assert closure_bytes(5701, 7588) == 7588 * 11402 == 86_518_376

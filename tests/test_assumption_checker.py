import os

import numpy as np
import pytest
import torch

from gpap.engine.checker import AssumptionChecker
from gpap.engine import core_available

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
AVAIL = torch.cuda.is_available() and core_available()
pytestmark = pytest.mark.skipif(not AVAIL, reason="requires CUDA + a built libgpap")


class TestAssumptionChecker:
    def test_basic_check(self):
        checker = AssumptionChecker()
        # (x1 OR x2), (~x1 OR x3), (~x2 OR ~x3)
        checker.formula.init_formula(3, [[1, 2], [-1, 3], [-2, -3]])

        results = checker.check_assumptions([
            [1, 2, 3],     # clause 2 (~x2 OR ~x3) fully falsified -> conflict
            [1, -2, -3],   # clause 1 (~x1 OR x3) fully falsified -> conflict
        ])
        assert results['conflict'][0].item() is True or bool(results['conflict'][0])
        assert bool(results['conflict'][1])

    def test_no_conflict(self):
        checker = AssumptionChecker()
        checker.formula.init_formula(2, [[1, 2], [-1, 2]])
        results = checker.check_assumptions([[2]])   # x2=TRUE satisfies both clauses
        assert not bool(results['conflict'][0])

    def test_from_cnf_file(self):
        checker = AssumptionChecker(
            cnf_file=os.path.join(TEST_DATA_DIR, "simple_3v_4c.cnf"))
        assert checker.formula.num_vars == 3
        assert checker.formula.num_clauses == 4

    def test_propagation_count(self):
        checker = AssumptionChecker()
        # chain: (~x1 OR x2), (~x2 OR x3)
        checker.formula.init_formula(3, [[-1, 2], [-2, 3]])
        results = checker.check_assumptions([[1]])   # x1=TRUE -> x2 -> x3
        assert not bool(results['conflict'][0])
        assert results['num_propagated'][0].item() == 2

    def test_state_arrays_are_opt_in(self):
        """`assignment` (n, 2N) and `sat_count` (n, M) are what makes a big call expensive —
        237 MB for 20k sets on a 1.5k-variable formula — so they are returned only when asked
        for. `conflict` and `num_propagated` are ~5 bytes/set and always present."""
        checker = AssumptionChecker()
        checker.formula.init_formula(3, [[1, 2], [-1, 3], [-2, -3]])
        assert set(checker.check_assumptions([[1], [2]])) == {"conflict", "num_propagated"}
        r = checker.check_assumptions([[1], [2]], assignments=True, sat_count=True)
        assert r['sat_count'].shape == (2, 3)
        assert r['assignment'].shape == (2, 6)


class TestChunking:
    """Any number of sets, split into GPU-sized chunks internally. The user never names K."""

    @staticmethod
    def _formula(checker):
        checker.formula.init_formula(6, [[1, 2], [-1, 3], [-2, -3], [4, 5], [-4, 6], [-5, -6]])

    def _random_sets(self, n, nv=6, seed=0):
        rng = np.random.default_rng(seed)
        return [[int(v) * (1 if rng.random() < 0.5 else -1)
                 for v in rng.choice(nv, size=int(rng.integers(0, 4)), replace=False) + 1]
                for _ in range(n)]

    def test_chunked_matches_single_chunk_exactly(self, monkeypatch):
        """THE contract: how many launches it took must not be observable. Same sets, one chunk
        vs many, compared element for element and IN INPUT ORDER.

        Scoped to the rows the engine actually specifies, which is what confluence buys: a row that
        survives has a UNIQUE BCP fixpoint, so every field of it is reproducible. A row that
        CONFLICTS stopped as soon as some clause emptied, and which clause that was depends on
        propagation order — so its partial assignment/counters are unspecified, and only `conflict`
        itself is. Chunk size can change the propagate kernel (see calibrate.dispatch_mode, whose
        small-batch term binds on a formula this size), which is exactly how a caller can observe
        that: measured here, 809/3,282 conflicted rows differ in num_propagated between batch 5,000
        and batch 97, and 0/1,718 surviving rows differ in anything."""
        ck = AssumptionChecker()
        self._formula(ck)
        sets = self._random_sets(5000)
        one = ck.check_assumptions(sets, assignments=True, sat_count=True)
        assert ck.chunk_size(len(sets), 3) >= len(sets), "precondition: this fits in one chunk"
        monkeypatch.setattr(ck, "chunk_size", lambda n, max_lits=1: 97)   # 52 chunks, uneven
        many = ck.check_assumptions(sets, assignments=True, sat_count=True)
        assert torch.equal(one["conflict"], many["conflict"]), "conflict is specified for EVERY row"
        live = ~one["conflict"]
        assert int(live.sum()) > 0, "precondition: some sets must survive"
        for key in one:
            assert torch.equal(one[key][live], many[key][live]), f"{key} on surviving rows"

    def test_more_sets_than_any_chunk(self, monkeypatch):
        """A set count far larger than a chunk completes and stays correct — the case that used
        to require the caller to loop (and to know K to loop over)."""
        ck = AssumptionChecker()
        self._formula(ck)
        monkeypatch.setattr(ck, "chunk_size", lambda n, max_lits=1: 64)
        sets = [[1, 2], [-1], []] * 200            # 600 sets, ~10 chunks
        r = ck.check_assumptions(sets)
        assert r["conflict"].shape == (600,)
        # the pattern repeats, so every third result must agree
        c = r["conflict"].tolist()
        assert c[0::3] == [c[0]] * 200 and c[1::3] == [c[1]] * 200 and c[2::3] == [c[2]] * 200

    def test_chunk_size_is_bounded_and_positive(self):
        ck = AssumptionChecker()
        self._formula(ck)
        assert ck.chunk_size(10, 3) == 10, "never larger than the input"
        assert ck.chunk_size(10**9, 3) > 0
        assert ck.chunk_size(10**9, 100_000) >= 1, "a huge L must not drive it to zero"
        assert ck.chunk_size(10**9, 3) >= ck.chunk_size(10**9, 100_000), "L costs 8 bytes/row"

    def test_engine_is_built_once_and_reused(self):
        """Rebuilding per call re-uploaded the occurrence CSR and re-ran the units flush — the
        dominant cost of a small call. Replacing the formula must still be honoured."""
        ck = AssumptionChecker()
        self._formula(ck)
        ck.check_assumptions([[1]])
        eng = ck.engine
        ck.check_assumptions([[2]])
        assert ck.engine is eng
        ck.formula.init_formula(2, [[1, 2]])
        assert ck.engine is not eng

    def test_ragged_and_empty_sets(self):
        ck = AssumptionChecker()
        self._formula(ck)
        r = ck.check_assumptions([[], [1], [1, 2, 3, 4, 5, 6], [2]])
        assert not bool(r["conflict"][0]) and r["num_propagated"][0].item() == 0


class TestInputValidation:
    """Literal validation is memory safety, not politeness — see checker._flatten."""

    @staticmethod
    def _checker():
        ck = AssumptionChecker()
        ck.formula.init_formula(3, [[1, 2], [-1, 3], [-2, -3]])
        return ck

    @pytest.mark.parametrize("lit", [99, -99, 4, -4, 0])
    def test_out_of_range_literal_raises(self, lit):
        with pytest.raises(ValueError, match="out of range"):
            self._checker().check_assumptions([[lit]])

    def test_bad_literal_cannot_corrupt_another_row(self):
        """REGRESSION. Literal 99 on a 3-variable formula is internal index 196, and 196 >> 5 is
        six words past row 0's single-word bitmap — so row 0's bad assumption used to assign a
        literal in ROW 6, silently and undetectably."""
        ck = self._checker()
        with pytest.raises(ValueError):
            ck.check_assumptions([[99]] + [[] for _ in range(7)])
        clean = ck.check_assumptions([[] for _ in range(8)], assignments=True)
        assert int(clean["assignment"].sum()) == 0, "no row may be assigned by an empty set"

    def test_empty_input(self):
        """No sets = no launch, and correctly-shaped empties rather than a driver error."""
        ck = self._checker()
        r = ck.check_assumptions([], assignments=True, sat_count=True)
        assert r["conflict"].shape == (0,) and r["num_propagated"].shape == (0,)
        assert r["assignment"].shape == (0, 6) and r["sat_count"].shape == (0, 3)

    def test_oversized_result_is_refused_not_spilled(self):
        """`assignments` is (n, 2N) for the caller's own n — chunking cannot shrink it. This box
        degrades to host RAM instead of failing, so an unmet request must say so."""
        ck = AssumptionChecker()
        ck.formula.init_formula(200_000, [[1, 2]])       # 400 kB of assignment per set
        with pytest.raises(MemoryError, match="assumption sets"):
            ck.check_assumptions([[1]] * 40_000, assignments=True)   # 16 GB asked for


@pytest.mark.gpu
def test_matches_the_reference_bcp():
    """Oracle check: conflict flags and implied literals against the pure-Python reference BCP,
    over random CNFs and random assumption sets, across a chunk boundary."""
    from tests.reference_bcp import lit_idx, ref_bcp

    rng = np.random.default_rng(7)
    for trial in range(5):
        nv = int(rng.integers(6, 14))
        clauses = [[int(v) * (1 if rng.random() < 0.5 else -1)
                    for v in rng.choice(nv, size=int(rng.integers(2, 4)), replace=False) + 1]
                   for _ in range(int(rng.integers(4, 20)))]
        ck = AssumptionChecker()
        ck.formula.init_formula(nv, clauses)
        sets = [[int(v) * (1 if rng.random() < 0.5 else -1)
                 for v in rng.choice(nv, size=int(rng.integers(1, 4)), replace=False) + 1]
                for _ in range(300)]
        ck.chunk_size = lambda n, max_lits=1: 41          # force several chunks
        r = ck.check_assumptions(sets, assignments=True)
        got_conflict = r["conflict"].cpu().numpy()
        got_assign = r["assignment"].cpu().numpy()
        for i, s in enumerate(sets):
            exp_conflict, exp_assigned = ref_bcp(clauses, [lit_idx(x) for x in s])
            assert bool(got_conflict[i]) == exp_conflict, (trial, i, s)
            if not exp_conflict:
                # confluence: the assigned-literal SET of a non-conflicted row is unique
                assert set(np.nonzero(got_assign[i])[0].tolist()) == set(exp_assigned), (trial, i, s)

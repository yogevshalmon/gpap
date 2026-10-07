"""AssumptionChecker: conflict and propagation answers for batches of assumption sets, the
chunking that hides K from the caller, input validation, and agreement with the reference BCP."""
import os
import random

import numpy as np
import pytest
import torch

from gpap.engine.checker import AssumptionChecker
from tests.conftest import TEST_DATA_DIR
from tests.support import random_assumptions, random_cnf

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------- assumption checker
def test_basic_check():
    checker = AssumptionChecker()
    # (x1 OR x2), (~x1 OR x3), (~x2 OR ~x3)
    checker.formula.init_formula(3, [[1, 2], [-1, 3], [-2, -3]])

    results = checker.check_assumptions([
        [1, 2, 3],     # clause 2 (~x2 OR ~x3) fully falsified -> conflict
        [1, -2, -3],   # clause 1 (~x1 OR x3) fully falsified -> conflict
    ])
    assert bool(results['conflict'][0])
    assert bool(results['conflict'][1])


def test_no_conflict():
    checker = AssumptionChecker()
    checker.formula.init_formula(2, [[1, 2], [-1, 2]])
    results = checker.check_assumptions([[2]])   # x2=TRUE satisfies both clauses
    assert not bool(results['conflict'][0])


def test_from_cnf_file():
    checker = AssumptionChecker(
        cnf_file=os.path.join(TEST_DATA_DIR, "simple_3v_4c.cnf"))
    assert checker.formula.num_vars == 3
    assert checker.formula.num_clauses == 4


def test_from_clauses_constructor():
    """`from_clauses` builds a checker in one call, like the C++ face's Formula::from_clauses."""
    ck = AssumptionChecker.from_clauses(3, [[1, 2], [-1, 3], [-2, -3]])
    r = ck.check_assumptions([[1, 2, 3], [2]])
    assert bool(r["conflict"][0]) and not bool(r["conflict"][1])


def test_propagation_count():
    checker = AssumptionChecker()
    # chain: (~x1 OR x2), (~x2 OR x3)
    checker.formula.init_formula(3, [[-1, 2], [-2, 3]])
    results = checker.check_assumptions([[1]])   # x1=TRUE -> x2 -> x3
    assert not bool(results['conflict'][0])
    assert results['num_propagated'][0].item() == 2


def test_state_arrays_are_opt_in():
    """`assignment` (n, 2N) and `sat_count` (n, M) are large, so they are returned only on
    request; the small `conflict` and `num_propagated` always are."""
    checker = AssumptionChecker()
    checker.formula.init_formula(3, [[1, 2], [-1, 3], [-2, -3]])
    assert set(checker.check_assumptions([[1], [2]])) == {"conflict", "num_propagated"}
    r = checker.check_assumptions([[1], [2]], assignments=True, sat_count=True)
    assert r['sat_count'].shape == (2, 3)
    assert r['assignment'].shape == (2, 6)


# ---------------------------------------------------------------- chunking
# Any number of sets, split into GPU-sized chunks internally. The user never names K.

def _formula(checker):
    checker.formula.init_formula(6, [[1, 2], [-1, 3], [-2, -3], [4, 5], [-4, 6], [-5, -6]])


def test_chunked_matches_single_chunk_exactly(monkeypatch):
    """Chunking is not observable: one chunk and many give the same results, in input order.

    Only `conflict` is compared on every row. A conflicted row's partial state depends on
    propagation order, and the chunk size can change the kernel (calibrate.dispatch_mode), so
    its other fields are compared only on surviving rows, whose fixpoint is unique."""
    ck = AssumptionChecker()
    _formula(ck)
    sets = random_assumptions(random.Random(0), 6, 5000, wmax=3)
    one = ck.check_assumptions(sets, assignments=True, sat_count=True)
    assert ck.chunk_size(len(sets), 3) >= len(sets), "precondition: this fits in one chunk"
    monkeypatch.setattr(ck, "chunk_size", lambda n, max_lits=1: 97)   # 52 chunks, uneven
    many = ck.check_assumptions(sets, assignments=True, sat_count=True)
    assert torch.equal(one["conflict"], many["conflict"]), "conflict is specified for EVERY row"
    live = ~one["conflict"]
    assert int(live.sum()) > 0, "precondition: some sets must survive"
    for key in one:
        assert torch.equal(one[key][live], many[key][live]), f"{key} on surviving rows"


def test_more_sets_than_any_chunk(monkeypatch):
    """Many more sets than one chunk holds complete correctly without the caller looping."""
    ck = AssumptionChecker()
    _formula(ck)
    monkeypatch.setattr(ck, "chunk_size", lambda n, max_lits=1: 64)
    sets = [[1, 2], [-1], []] * 200            # 600 sets, ~10 chunks
    r = ck.check_assumptions(sets)
    assert r["conflict"].shape == (600,)
    # the pattern repeats, so every third result must agree
    c = r["conflict"].tolist()
    assert c[0::3] == [c[0]] * 200 and c[1::3] == [c[1]] * 200 and c[2::3] == [c[2]] * 200


def test_chunk_size_is_bounded_and_positive():
    ck = AssumptionChecker()
    _formula(ck)
    assert ck.chunk_size(10, 3) == 10, "never larger than the input"
    assert ck.chunk_size(0, 1) == 1, "floor of 1 even for an empty ask (historical contract)"
    assert ck.chunk_size(10**9, 3) > 0
    assert ck.chunk_size(10**9, 100_000) >= 1, "a huge L must not drive it to zero"
    assert ck.chunk_size(10**9, 3) >= ck.chunk_size(10**9, 100_000), "L costs 8 bytes/row"


def test_engine_is_built_once_and_reused():
    """The engine is reused across calls (building it is most of a small call's cost) and
    rebuilt when the formula is replaced."""
    ck = AssumptionChecker()
    _formula(ck)
    ck.check_assumptions([[1]])
    eng = ck.engine
    ck.check_assumptions([[2]])
    assert ck.engine is eng
    ck.formula.init_formula(2, [[1, 2]])
    assert ck.engine is not eng


def test_ragged_and_empty_sets():
    ck = AssumptionChecker()
    _formula(ck)
    r = ck.check_assumptions([[], [1], [1, 2, 3, 4, 5, 6], [2]])
    assert not bool(r["conflict"][0]) and r["num_propagated"][0].item() == 0


# ---------------------------------------------------------------- input validation
# Literal validation is memory safety: see checker._flatten.

def _checker():
    ck = AssumptionChecker()
    ck.formula.init_formula(3, [[1, 2], [-1, 3], [-2, -3]])
    return ck


@pytest.mark.parametrize("lit", [99, -99, 4, -4, 0])
def test_out_of_range_literal_raises(lit):
    with pytest.raises(ValueError, match="out of range"):
        _checker().check_assumptions([[lit]])


def test_bad_literal_cannot_corrupt_another_row():
    """Literal 99 on a 3-variable formula is index 196, six bitmap words past row 0, i.e. in
    row 6. It must raise, and no other row may be touched."""
    ck = _checker()
    with pytest.raises(ValueError):
        ck.check_assumptions([[99]] + [[] for _ in range(7)])
    clean = ck.check_assumptions([[] for _ in range(8)], assignments=True)
    assert int(clean["assignment"].sum()) == 0, "no row may be assigned by an empty set"


def test_empty_input():
    """No sets means no launch and correctly shaped empty results."""
    ck = _checker()
    r = ck.check_assumptions([], assignments=True, sat_count=True)
    assert r["conflict"].shape == (0,) and r["num_propagated"].shape == (0,)
    assert r["assignment"].shape == (0, 6) and r["sat_count"].shape == (0, 3)


def test_oversized_result_is_refused_not_spilled():
    """`assignments` is (n, 2N) for the caller's n, which chunking cannot shrink. A request past
    the device budget must raise rather than spill to host RAM.

    n is derived from the live budget, which the guard compares against, so the request is too
    big on any card."""
    from gpap.engine.calibrate import gpu_mem_available
    ck = AssumptionChecker()
    ck.formula.init_formula(200_000, [[1, 2]])       # 400 kB of assignment per set
    row_bytes = 2 * 200_000
    n = int(gpu_mem_available(ck.formula.device) // row_bytes) + 1_000   # over the budget
    with pytest.raises(MemoryError, match="assumption sets"):
        ck.check_assumptions([[1]] * n, assignments=True)


def test_matches_the_reference_bcp():
    """Oracle check: conflict flags and implied literals against the pure-Python reference BCP,
    over random CNFs and random assumption sets, across a chunk boundary."""
    from tests.reference.bcp import lit_idx, ref_bcp

    rng = random.Random(7)
    for trial in range(5):
        nv = rng.randint(6, 13)
        clauses = random_cnf(rng, nv, rng.randint(4, 19), kmin=2, kmax=3)
        ck = AssumptionChecker()
        ck.formula.init_formula(nv, clauses)
        sets = random_assumptions(rng, nv, 300, wmin=1, wmax=3)
        ck.chunk_size = lambda n, max_lits=1: 41          # force several chunks
        r = ck.check_assumptions(sets, assignments=True)
        got_conflict = r["conflict"].cpu().numpy()
        got_assign = r["assignment"].cpu().numpy()
        for i, s in enumerate(sets):
            exp_conflict, exp_assigned = ref_bcp(clauses, [lit_idx(x) for x in s])
            assert bool(got_conflict[i]) == exp_conflict, (trial, i, s)
            if not exp_conflict:
                # confluence: a non-conflicted row's assigned-literal set is unique
                assert set(np.nonzero(got_assign[i])[0].tolist()) == set(exp_assigned), (trial, i, s)

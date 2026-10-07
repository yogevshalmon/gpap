"""The reference enumerator (tests/reference/combinations.py), checked against itertools. It
pins the core's candidate order the way tests/reference/bcp.py pins the kernels. Literals are
DIMACS. Pure Python: no library and no GPU.
"""
import pytest

from tests.reference.combinations import combinations


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

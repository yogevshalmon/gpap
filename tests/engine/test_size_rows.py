"""The batch rule itself (`size_rows`, cpp/src/memory.cpp), through its test driver and with no
device: every bound, the order of a tie, the GRANT a host's own plug reports (issue 72), which no
plug on this tree reports, and, with no limit and no grant, the arithmetic it replaced, request
for request. The driver is built beside the examples (scripts/build_core.sh).
"""
import random
import subprocess

import pytest

from tests.support import example_binary, need_binary

pytestmark = pytest.mark.core

DRIVER = example_binary("gpap_test_size_rows")
NONE, CAP, MEMORY, LIMIT, GRANT, BUFFER = 0, 1, 2, 3, 4, 5
INT32_MAX = 2**31 - 1


def _rule(*requests):
    """size_rows on each request (a dict of SizeRequest's fields), as (rows, bound, left,
    per_row) tuples."""
    need_binary(DRIVER)
    order = ("total", "memory", "grant", "limit", "held_other", "per_row", "fixed",
             "exact_per_row", "exact_fixed", "cap", "max_buffer", "widest")
    base = {"total": 0, "memory": -1, "grant": -1, "limit": 0, "held_other": 0, "per_row": 1,
            "fixed": 0, "exact_per_row": 0, "exact_fixed": 0, "cap": 0, "max_buffer": 0,
            "widest": 1}
    lines = [" ".join(str({**base, **r}[k]) for k in order) for r in requests]
    out = subprocess.run([str(DRIVER)], input="\n".join(lines) + "\n", capture_output=True,
                         text=True, check=True).stdout.split("\n")
    return [tuple(int(x) for x in line.split()) for line in out if line.strip()]


def _old_rows(memory, fixed, per_row, cap, max_buffer, widest):
    """The batch the engine sized before the rule (c26c889, engine_batch over rows_fitting):
    at least one row, always. Frozen here as the oracle."""
    budget = max(memory - fixed, 0)
    k = budget // max(per_row, 1) if budget > 0 else 0
    if cap > 0:
        k = min(k, cap)
    k = min(max(k, 1), INT32_MAX)
    if max_buffer > 0:
        k = min(k, max(max_buffer // max(widest, 1), 1))
    return k


def test_without_a_limit_the_batch_is_the_old_one_wherever_a_row_fits():
    """Over random requests with only the memory bound, the batch is the old arithmetic's
    wherever that fitted a row, and 0, not 1, wherever it did not (the user's Q4)."""
    rng = random.Random(11)
    reqs = []
    for _ in range(4000):
        per_row = rng.choice([1, 7, 1000, 123_457, 5 << 20])
        reqs.append({
            "memory": rng.choice([0, rng.randrange(1 << 40), rng.randrange(1 << 24),
                                  per_row * rng.randrange(4)]),
            "per_row": per_row,
            "fixed": rng.choice([0, 0, 73_400_320, rng.randrange(1 << 30)]),
            "cap": rng.choice([0, -1, 262_144, rng.randrange(1, 5000)]),
            "max_buffer": rng.choice([0, 0, 1 << 30, rng.randrange(1, 1 << 22)]),
            "widest": rng.choice([1, 4 * per_row, rng.randrange(1, 1 << 20)]),
        })
    for req, (rows, bound, _left, _pr) in zip(reqs, _rule(*reqs)):
        old = _old_rows(req["memory"], req["fixed"], req["per_row"], req["cap"],
                        req["max_buffer"], req["widest"])
        raw = max(req["memory"] - req["fixed"], 0) // req["per_row"]
        fits = raw >= 1 and (req["max_buffer"] <= 0 or req["max_buffer"] // req["widest"] >= 1)
        assert rows == (old if fits else 0), req
        assert bound in (CAP, MEMORY, BUFFER)


def test_a_host_grant_is_used_as_it_is():
    """Issue 72: a 1 GiB share sized as a device left 219 MB to rows; as a grant it leaves the
    whole GiB less what the call takes once, with the exact bound's own terms counted."""
    (rows, bound, left, per_row), = _rule({"grant": 1 << 30, "per_row": 1 << 20,
                                           "fixed": 1 << 22, "exact_per_row": 17,
                                           "exact_fixed": 1 << 20})
    assert bound == GRANT and per_row == (1 << 20) + 17
    assert left == (1 << 30) - (1 << 22) - (1 << 20)
    assert rows == left // per_row
    assert rows > 4 * _old_rows(int(((1 << 30) - 800_000_000) * 0.8), 1 << 22, 1 << 20, 0, 0, 1)


def test_a_grant_too_small_for_a_row_fits_none():
    """A share under one row is no batch, not one silent row (issue 72, Q4)."""
    (rows, bound, _left, _pr), = _rule({"grant": 700 << 20, "per_row": 800 << 20})
    assert (rows, bound) == (0, GRANT)


def test_a_limit_counts_what_gpap_holds_and_stops_at_the_device():
    """The LIMIT bound is the limit, at most the device, less what GPAP holds besides the rows
    the call reuses; a limit past the device is the device."""
    common = {"held_other": 100, "per_row": 10, "fixed": 50, "exact_per_row": 0,
              "exact_fixed": 0}
    (r1, b1, l1, _), (r2, b2, l2, _) = _rule({**common, "total": 0, "limit": 10_150},
                                             {**common, "total": 5_150, "limit": 10_150})
    assert (r1, b1, l1) == (1000, LIMIT, 10_000)
    assert (r2, b2, l2) == (500, LIMIT, 5_000)


def test_the_smallest_bound_wins_and_a_tie_goes_cap_limit_grant_memory():
    """At equal rows the report says the more useful thing: the cap (more memory would not help)
    before the limit, the limit before the default rule."""
    same = {"per_row": 10, "limit": 1000, "memory": 1000}
    out = _rule(same,
                {**same, "cap": 100},
                {**same, "memory": 500},
                {"per_row": 10, "grant": 1000, "memory": 1000},
                {"per_row": 10, "memory": 1000, "max_buffer": 40, "widest": 10})
    assert [(r, b) for r, b, _, _ in out] == [(100, LIMIT), (100, CAP), (50, MEMORY),
                                             (100, GRANT), (4, BUFFER)]


def test_no_bound_is_no_batch():
    """With nothing to size from (no memory bound, no grant, no limit) there is no batch."""
    (rows, bound, _left, _pr), = _rule({})
    assert (rows, bound) == (0, NONE)

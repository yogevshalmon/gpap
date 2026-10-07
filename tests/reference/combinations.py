"""Host-side reference enumerator for the core's combinatorial probe.

The sweep itself runs in cpp/src/probe.cpp; this exists ONLY so tests can pin that the core
produces the same candidates in the same order as an independent, obvious implementation — the
same role tests/reference/bcp.py plays for the kernels. It materializes C(P,k) tuples, which is
exactly why it is not in the shipping API: at k=3 over an unrolled circuit's free literals that is
~1e11 tuples.
"""
from __future__ import annotations

import numpy as np


def combinations(pool, size: int, *, distinct_vars: bool = True,
                 max_tuples: int | None = None) -> np.ndarray:
    """Every size-`size` combination of `pool` (DIMACS literals), as an (n, size) array.

    `distinct_vars` (default) drops any tuple holding two literals of the SAME variable: such a
    tuple is either a repeat or a direct contradiction, so it conflicts for reasons that have
    nothing to do with the formula. Combinations are ascending in pool position, matching the
    core's order, so the two paths can be compared candidate for candidate.
    """
    pool = np.ascontiguousarray(pool, dtype=np.int32).reshape(-1)
    if size < 1:
        raise ValueError(f"tuple size must be >= 1, got {size}")
    if size == 1:
        out = pool.reshape(-1, 1)
    elif len(pool) < size:
        out = np.empty((0, size), dtype=np.int32)
    elif size == 2:
        i, j = np.triu_indices(len(pool), k=1)              # the common case, directly
        out = np.stack([pool[i], pool[j]], axis=1)
    else:
        import itertools
        idx = np.fromiter(itertools.chain.from_iterable(
            itertools.combinations(range(len(pool)), size)), dtype=np.int64)
        out = (pool[idx.reshape(-1, size)] if idx.size
               else np.empty((0, size), dtype=np.int32))
    if distinct_vars and size > 1 and out.size:
        v = np.sort(np.abs(out), axis=1)
        out = out[(v[:, 1:] != v[:, :-1]).all(axis=1)]
    return out[:max_tuples] if max_tuples is not None else out

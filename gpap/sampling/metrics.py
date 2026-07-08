"""Sample-quality metrics (computed OFF the throughput timing path — see BENCHMARKING notes:
folding reconstruction/dedup into the timed run understates GPU throughput up to 10x)."""
import numpy as np


def distinct_projections(models, support):
    """Count DISTINCT support-projections among `models` (support = 1-based vars). Returns
    (n_distinct, n_models). This is the "unique solutions" verification metric — computed OFF the
    throughput timing path (the timed run uses collect=False), purely to confirm the sampler emits
    genuinely distinct solutions (not re-emitting the same model). Used for both GPAP and CMSGen on
    the SAME collected set so the distinct COUNTS are directly comparable."""
    if not models:
        return 0, 0
    seen = set()
    for m in models:
        s = set(m)
        seen.add(tuple(v if v in s else -v for v in support))
    return len(seen), len(models)


def two_wise_coverage(models, support):
    """% of sampling-var pairs whose all 4 (±,±) combos appear. support: 1-based vars."""
    S = len(support)
    if S < 2 or not models:
        return 0.0
    idx = {v: i for i, v in enumerate(support)}
    M = np.zeros((len(models), S), dtype=np.int8)
    for r, m in enumerate(models):
        s = set(m)
        for v in support:
            if v in s:
                M[r, idx[v]] = 1
    covered = total = 0
    for i in range(S):
        ci = M[:, i].astype(np.int16) << 1
        for j in range(i + 1, S):
            total += 1
            if np.unique(ci | M[:, j]).size == 4:
                covered += 1
    return 100.0 * covered / total if total else 0.0

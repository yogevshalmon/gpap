"""Sample-quality metrics (computed OFF the throughput timing path — see BENCHMARKING notes:
folding reconstruction/dedup into the timed run understates GPU throughput up to 10x).

Uniformity: `uniformity_stats` gives the cheap opt-in readout (`--uniformity fast|exact`,
default off — it costs an extra sampling pass). When the projected solution
space is small enough to ENUMERATE (enumerate_projections), it is an exact goodness-of-fit test
(chi-square p-value + total-variation distance vs the uniform distribution over all |S| solutions).
When the space is too large, no black-box statistic can CERTIFY uniformity — that is precisely why
Barbarik (Chakraborty & Meel, AAAI'19) needs conditional sampling — but skew is still DETECTABLE
distribution-free: the collision estimator p2 = sum c_i(c_i-1)/(n(n-1)) is unbiased for ||p||_2^2,
and under ANY uniform distribution ||p||_2^2 = 1/|S| <= 1/d (d = distinct seen <= |S|), so
l2_ratio = p2*d substantially above 1 certifies non-uniformity with NO knowledge of |S|.
For a real certificate on large spaces, run the Barbarik harness (eval/uniformity/barbarik_gpap.py).
"""
import math
import time

import numpy as np


def _gammq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) (Numerical-Recipes gser/gcf, via lgamma).
    Q(k/2, x/2) is the chi-square survival function — avoids a scipy dependency."""
    if x < 0 or a <= 0:
        raise ValueError("bad args to _gammq")
    if x == 0:
        return 1.0
    if x < a + 1.0:                                # series for P(a,x); Q = 1 - P
        ap, term, total = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-12:
                break
        p = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - p))
    b, c, d = x + 1.0 - a, 1e300, 1.0 / (x + 1.0 - a)   # Lentz continued fraction for Q(a,x)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    q = h * math.exp(-x + a * math.log(x) - math.lgamma(a))
    return max(0.0, min(1.0, q))


def chi2_sf(x: float, k: int) -> float:
    """P(Chi2_k >= x) — chi-square survival function (k = degrees of freedom)."""
    return _gammq(k / 2.0, x / 2.0)


def enumerate_projections(clauses, support, cap=4096, budget_s=2.0):
    """Enumerate DISTINCT support-projections of `clauses` (support: 1-based vars) with a SAT
    solver, stopping at `cap` projections or `budget_s` seconds (whichever first).

    Returns (projs, status): projs = set of 0/1 tuples ordered like `support`;
    status = "complete" (projs is the WHOLE projected solution set) | "capped" | "timeout"
    | "unavailable" (pysat missing). Only "complete" supports an exact uniformity test."""
    try:
        from pysat.formula import CNF
        from pysat.solvers import Solver
    except ImportError:                            # pragma: no cover
        return None, "unavailable"
    t0 = time.perf_counter()
    s = Solver(name="cadical195", bootstrap_with=CNF(from_clauses=clauses))
    projs = set()
    status = "complete"
    try:
        while s.solve():
            m = set(s.get_model())
            projs.add(tuple(1 if v in m else 0 for v in support))
            if len(projs) >= cap:
                status = "capped"
                break
            if time.perf_counter() - t0 > budget_s:
                status = "timeout"
                break
            s.add_clause([(-v if v in m else v) for v in support])
    finally:
        s.delete()
    return projs, status


EXACT_ALPHA = 1e-3
"""Significance for the exact readout's SKEWED verdict.

Matches eval/uniformity/fuzz_uniformity.py's flag threshold, whose controls calibrate it: 0/120
exact-uniform campaigns were flagged at this alpha, and a planted TV of 0.05 was caught 97% of the
time. A looser alpha would start calling honest samplers skewed on a single instance."""


def format_uniformity(u, verbose=False):
    """One-line human rendering of a uniformity_stats dict. verbose=True gives the full sentence
    (gpap sample); verbose=False a short table cell (gpap bench)."""
    if not u or u.get("mode") in (None, "empty", "error"):
        note = (u or {}).get("note", "")
        return f"n/a{' (' + note + ')' if verbose and note else ''}" if u else "n/a"
    if u["mode"] == "exact":
        if u.get("unsound_projections"):
            return f"UNSOUND ({u['unsound_projections']} bad projections)"
        p, tv = u.get("p"), u.get("tv", 0.0)
        # The exact mode used to print bare numbers while the WEAKER `fast` mode printed OK /
        # SKEWED — so the sensitive test looked less conclusive than the gross-skew one, and a
        # reader had to know that p=0.000 is bad news. Both modes lead with a verdict now.
        skewed = p is not None and p < EXACT_ALPHA
        word = "SKEWED" if skewed else ("OK" if p is not None else "n/a")
        if not verbose:
            return f"{'SKEW' if skewed else 'OK'} p={p:.2f}" if p is not None else f"TV={tv:.2f}"
        ptxt = (f"chi2({u['dof']})={u['chi2']:.0f} p={p:.3f}" if p is not None
                else f"chi2 p n/a ({u.get('note', 'n too small')})")
        tail = ("" if p is None else
                (f" — the sample is too uneven to come from a uniform sampler "
                 f"(p < {EXACT_ALPHA})" if skewed else
                 f" — no skew detectable at p < {EXACT_ALPHA}; NOT a proof of uniformity"))
        return (f"{word} TV={tv:.3f} {ptxt}  [exact: n={u['n']:,} over |S|={u['S']:,}]{tail}")
    # fast/collision mode (|S| unknown): repeat behaviour only — a gross-skew detector. The
    # follow-up hint depends on WHY |S| is unknown: in fast mode exact was never attempted
    # (suggest it); in exact mode enumeration was attempted and the space defeated it (only
    # barbarik can go further — suggesting --uniformity exact again would be nonsense).
    c, l2 = u.get("collisions", 0), u.get("l2_ratio")
    skewed = (l2 or 0) > 1.5
    if not verbose:
        return "OK" if not skewed else f"SKEW {l2:.1f}"
    st = u.get("enum_status", "not-attempted")
    if st == "not-attempted":
        hint = "gross-skew check only — --uniformity exact / barbarik for a sensitive test"
    elif st in ("capped", "timeout"):
        hint = "space too large to enumerate (>4096 solutions) — barbarik for certification"
    elif st == "skipped-large":
        hint = "formula too large for exact enumeration (M > 200k) — barbarik for certification"
    else:                                          # unavailable (pysat missing)
        hint = f"exact test infeasible ({st}) — barbarik for certification"
    if c == 0:
        floor = u.get("birthday_floor")
        scale = f" — consistent with uniform over >={floor:,} solutions" if floor else ""
        return f"OK — no repeated solution in {u['n']:,} samples{scale} ({hint})"
    if skewed:
        return (f"SKEWED — some solutions repeat far too often (l2_ratio {l2:.2f}, uniform ~1.0; "
                f"{u['distinct']:,} distinct / {u['n']:,} samples)")
    return (f"OK — repeat rate consistent with uniform (l2_ratio {l2:.2f}, "
            f"{u['distinct']:,} distinct / {u['n']:,} samples; {hint})")


def uniformity_stats(counts, n, S=None, n_unseen_known=None):
    """The uniformity readout over a collected sample.

    counts   : multiplicities of the DISTINCT observed projections (any order).
    n        : total samples (== sum(counts)).
    S        : EXACT projected-solution count when known (enumeration completed), else None.
    n_unseen_known : with S known, S - len(counts) (computed here if None).

    Returns a dict:
      mode "exact"     — chi2 statistic + p-value (all S cells, unseen included; dof S-1;
                         p reported only when n >= 5*S so the chi2 approximation holds) and
                         tv = total-variation distance to uniform (in [0,1]).
      mode "collision" — distinct/collisions + l2_ratio = p2_hat * d (see module docstring:
                         > ~1.5 certifies skew with NO knowledge of |S|; <= 1 is CONSISTENT with
                         uniform but not a certificate) + birthday_floor when collision-free
                         (a 95% lower bound on |S| implied by seeing no collision).
    """
    counts = np.asarray([c for c in counts if c > 0], dtype=np.int64)
    d = int(counts.size)
    n = int(n)
    out = {"n": n, "distinct": d, "collisions": n - d}
    if n == 0 or d == 0:
        out.update(mode="empty", note="no samples collected")
        return out
    if S is not None and S <= 1:
        out.update(mode="exact", S=int(S), tv=0.0, chi2=0.0, p=1.0, dof=0,
                   note="single-solution space")
        return out
    if S is not None:
        S = int(S)
        unseen = (S - d) if n_unseen_known is None else int(n_unseen_known)
        e = n / S                                   # expected per cell under uniform
        chi2 = float(((counts - e) ** 2 / e).sum() + unseen * e)
        dof = S - 1
        tv = float(0.5 * (np.abs(counts / n - 1.0 / S).sum() + unseen / S))
        out.update(mode="exact", S=S, chi2=chi2, dof=dof, tv=tv)
        if n >= 5 * S:
            out["p"] = chi2_sf(chi2, dof)
        else:
            out["p"] = None
            out["note"] = f"n < 5*S ({n} < {5 * S}) — chi2 p unreliable, TV reported"
        return out
    # |S| unknown: distribution-free collision readout.
    out["mode"] = "collision"
    if n < 2:
        out["note"] = "n too small"
        return out
    p2 = float((counts * (counts - 1)).sum() / (n * (n - 1)))   # unbiased ||p||_2^2 estimate
    out["l2_ratio"] = p2 * d if p2 > 0 else 0.0
    if n - d == 0:
        # No collision in n draws: under uniform-|S|, P(no collision) >= 95% forces |S| big.
        out["birthday_floor"] = int(n * (n - 1) / (2 * math.log(20)))
        out["note"] = "collision-free — consistent with uniform over a large space"
    elif out["l2_ratio"] > 1.5:
        out["note"] = ("l2_ratio > 1.5: NON-UNIFORM (||p||_2^2 exceeds every uniform "
                       "distribution consistent with the observed distinct count)")
    else:
        out["note"] = "consistent with uniform (not a certificate — see barbarik)"
    return out


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
    """% of sampling-var pairs whose all 4 (±,±) joint polarities appear in the sample — the
    combinatorial-testing COVERAGE metric the CMSGen paper reports (does the sample spread over
    the space?). NOT a uniformity test; opt-in (bench --twise) for paper comparisons.

    Vectorised: one (S,n)x(n,S) matmul gives the 11-counts per pair; the 10/01/00 counts follow
    from the column sums (the old per-pair Python loop was O(S^2 n) and dominated the quality
    pass on wide supports). support: 1-based vars."""
    S = len(support)
    if S < 2 or not models:
        return 0.0
    idx = {v: i for i, v in enumerate(support)}
    M = np.zeros((len(models), S), dtype=np.float32)
    for r, m in enumerate(models):
        s = set(m)
        for v in support:
            if v in s:
                M[r, idx[v]] = 1.0
    n = M.shape[0]
    c11 = (M.T @ M).round().astype(np.int64)       # sgemm; exact for counts < 2^24
    s1 = M.sum(axis=0).round().astype(np.int64)
    c10 = s1[:, None] - c11
    c01 = s1[None, :] - c11
    c00 = n - s1[:, None] - s1[None, :] + c11
    ok = (c11 > 0) & (c10 > 0) & (c01 > 0) & (c00 > 0)
    iu = np.triu_indices(S, k=1)
    total = iu[0].size
    return 100.0 * int(ok[iu].sum()) / total if total else 0.0

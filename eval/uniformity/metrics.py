"""Uniformity statistics for the evaluation layer — the goodness-of-fit machinery the
certification harnesses in this directory share.

Deliberately not part of the library: a uniformity number is a claim about a distribution, and
the honest ones need the whole solution set or conditional sampling, neither of which
`gpap sample` can do on the way past. The library samples; these tools make the claims.

    fuzz_uniformity.py   exact oracle: enumerate small formulas, chi2 + TV against uniform
    barbarik_gpap.py     the standard certificate (Chakraborty & Meel, AAAI'19), SPUR reference
    barbarik_screen.py   which instances that test can speak about at all
    barbarik_table.py    the eligibility rule + the two parameter sets, never pooled

Two statistics live here:

  `uniformity_stats(counts, n, S=...)` — with the exact projected-solution count in hand, a real
  goodness-of-fit test: chi-square p-value over all S cells plus total variation from uniform.

  `uniformity_stats(counts, n)` — without S no black-box statistic can certify uniformity, but
  skew is still detectable distribution-free: the collision estimator
  p2 = sum c_i(c_i-1)/(n(n-1)) is unbiased for ||p||_2^2, and under any uniform distribution
  ||p||_2^2 = 1/|S| <= 1/d for d distinct values seen, so l2_ratio = p2*d well above 1 certifies
  non-uniformity with no knowledge of |S|. Below 1 it is consistent with uniform, nothing more.

Measured results: docs/findings/UNIFORMITY_FINDINGS.md.
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

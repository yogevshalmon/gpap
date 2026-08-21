"""CPU preprocessing primitives for SAMPLING: unit closure + equivalent-literal substitution.

Both operations preserve the model set over the original variables (unit_closure) or preserve
it up to a recorded variable substitution (equivalent_literal_substitution + var_map), which is
exactly what the sampler's simplify_cnf / simplify_cnf_els need: shrink M and N so BCP is
faster and a larger K fits, without changing the set of valid support projections.

A GPU inprocessing-for-solve pipeline (vivify / subsumption / preprocess()) used to live here
and was retired — see docs/findings/INSIGHTS.md ("preprocessing-for-solve is retired").
"""
from collections import defaultdict

from gpap.engine.formula import ext_lit_to_lit_index, lit_index_to_ext_lit

# A trivially-UNSAT 2-clause formula, returned (same verdict) when preprocessing derives ⊥.
_UNSAT_CNF = [[1], [-1]]


def unit_closure(clauses, units):
    """Assert entailed units and unit-propagate at the formula level (equivalence-preserving).

    `units`: iterable of unit clauses ([lit]). Existing unit clauses in `clauses` are picked up
    too. Returns (new_clauses, assigned, status):
      new_clauses : unit clauses for every assigned literal + the simplified remaining clauses,
                    logically equivalent to (clauses ∧ units).
      assigned    : {var(1-based): bool}.
      status      : "ok" | "unsat" (a contradiction among entailed units ⇒ the formula is UNSAT).

    Vectorised (numpy CSR + the per-clause remaining-count/key-sum trick — the same sum trick
    as the GPU engine, on CPU): the old per-literal Python worklist (sets + occ dicts over every
    clause) cost ~2.5s per call on 1M-clause circuit unrolls, and the sampler pays it twice
    (construct + post-mining rebuild). Clause ORDER is preserved (units in assignment order,
    then survivors in original order); literals within a surviving clause come out sorted and
    de-duplicated (downstream consumers are order-insensitive: lit_sum/occ-CSR are sums/sets).
    """
    import numpy as np
    from itertools import chain

    M = len(clauses)
    lens0 = np.fromiter(map(len, clauses), dtype=np.int64, count=M)
    total = int(lens0.sum())
    flat = (np.array(list(chain.from_iterable(clauses)), dtype=np.int64) if total
            else np.empty(0, dtype=np.int64))
    cls_idx = np.repeat(np.arange(M, dtype=np.int64), lens0)
    # de-duplicate literals within each clause (parity with the old `set(c)`); sorts them too
    order = np.lexsort((flat, cls_idx))
    f, ci = flat[order], cls_idx[order]
    keep = np.ones(len(f), dtype=bool)
    keep[1:] = (ci[1:] != ci[:-1]) | (f[1:] != f[:-1])
    f, ci = f[keep], ci[keep]

    init_lits = [lit for u in units for lit in u]
    maxv = max(int(np.abs(f).max()) if len(f) else 0,
               max((abs(l) for l in init_lits), default=0))
    key = 2 * (np.abs(f) - 1) + (f < 0)                 # lit-index key (2N scheme)
    nk = 2 * maxv
    korder = np.argsort(key, kind="stable")
    csr_cls = ci[korder]                                # clause indices grouped by literal key
    offs = np.zeros(nk + 1, dtype=np.int64)
    if len(key):
        np.cumsum(np.bincount(key, minlength=nk), out=offs[1:])

    n_rem = np.bincount(ci, minlength=M)                # non-falsified lits per clause
    key_sum = np.bincount(ci, weights=key, minlength=M) # sum of their keys (float64 — exact)
    removed = np.zeros(M, dtype=bool)
    emitted = np.zeros(M, dtype=bool)                   # clauses already consumed as units
    val = np.zeros(maxv + 1, dtype=np.int8)             # var -> 0 / +1 / -1
    assigned_order: list[int] = []

    def push(lits):
        """Assign new lits; returns the fresh batch or None on contradiction."""
        batch = []
        for l in lits:
            v, s = abs(l), (1 if l > 0 else -1)
            if val[v]:
                if val[v] != s:
                    return None
                continue
            val[v] = s
            assigned_order.append(l)
            batch.append(l)
        return batch

    batch = push(init_lits + [c[0] for c in clauses if len(c) == 1])
    if batch is None:
        return list(_UNSAT_CNF), {}, "unsat"
    while True:
        seg_sat, seg_fal, fal_keys = [], [], []
        for l in batch:
            k = 2 * (abs(l) - 1) + (0 if l > 0 else 1)
            seg_sat.append(csr_cls[offs[k]:offs[k + 1]])        # clauses satisfied by l
            fk = k ^ 1
            seg = csr_cls[offs[fk]:offs[fk + 1]]                # clauses losing ¬l
            seg_fal.append(seg)
            fal_keys.append(np.full(len(seg), fk, dtype=np.int64))
        if seg_sat:
            removed[np.concatenate(seg_sat)] = True
            fal = np.concatenate(seg_fal)
            n_rem -= np.bincount(fal, minlength=M)
            key_sum -= np.bincount(fal, weights=np.concatenate(fal_keys).astype(np.float64),
                                   minlength=M)
        live = ~removed
        if bool((live & (n_rem == 0)).any()):                    # all literals falsified
            return list(_UNSAT_CNF), {}, "unsat"
        unit_mask = live & (n_rem == 1) & ~emitted
        if not unit_mask.any():
            break
        emitted[unit_mask] = True
        ks = key_sum[unit_mask].astype(np.int64)                 # the remaining lit IS the sum
        batch = push([(k // 2 + 1) * (1 if k % 2 == 0 else -1) for k in ks.tolist()])
        if batch is None:
            return list(_UNSAT_CNF), {}, "unsat"
        if not batch:
            break

    out = [[l] for l in assigned_order]                          # unit clauses for assigned vars
    alive = ~removed & (n_rem >= 2)
    if alive.any():
        m = alive[ci] & (val[np.abs(f)] == 0)                    # falsified lits drop out
        kf, kc = f[m].tolist(), ci[m]
        bounds = np.zeros(int(alive.sum()) + 1, dtype=np.int64)
        np.cumsum(np.bincount(kc, minlength=M)[alive], out=bounds[1:])
        out.extend(kf[a:b] for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()))
    return out, {abs(l): l > 0 for l in assigned_order}, "ok"


def equivalent_literal_substitution(clauses, edges, num_vars, *, rep_score=None, max_classes=None):
    """Equivalent-literal substitution (ELS): collapse mutually-implied literals, drop variables.

    `edges`: (E, 2) tensor/list of (src_lit_idx, dst_lit_idx) implications src ⇒ dst. Two
    literals are equivalent when each implies the other (both directions present as direct
    edges — pass a transitively-closed edge set, or, as simplify_cnf_els does, the direct
    binary-clause implications where mutual pairs ARE the binary equivalences); equivalence
    classes are merged with union-find and every member is rewritten to a single representative
    literal, which removes the other variables. Contrapositive edges are derived (x≡y entails
    ¬x≡¬y), so negations collapse consistently (sub[l^1] == sub[l]^1).

    Heuristic knobs (the "which equivalences / which representative" lever):
      rep_score   : optional {var0(0-based): float}. The class member whose VARIABLE has the
                    highest score is kept as the representative (ties → smallest var); the others
                    are eliminated. None → keep the smallest-variable member (default). A
                    per-VARIABLE score keeps skew-symmetry (sub[l^1] == sub[l]^1).
      max_classes : substitute only the largest `max_classes` equivalence classes (most variables
                    removed first); None → all.

    A class containing both `l` and `¬l` means the formula is UNSAT. Equisatisfiable: a model of
    the reduced CNF lifts to the original by setting each removed variable equal to its
    representative literal (`var_map`).

    Returns (new_clauses, var_map, stats):
      var_map : {removed_var(1-based): rep_literal(DIMACS)} — removed_var takes the rep's value.
      stats   : {status: "ok"|"unsat", n_classes, vars_removed, clauses_removed, lits_removed}.
    """
    n2 = 2 * num_vars
    parent = list(range(n2))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    elist = edges.tolist() if hasattr(edges, "tolist") else [list(e) for e in edges]
    eset = {(int(s), int(d)) for s, d in elist}
    for s, d in eset:
        if (d, s) in eset:                    # mutual implication ⇒ equivalent literals
            union(s, d)
            union(s ^ 1, d ^ 1)               # x≡y entails ¬x≡¬y (BCP may miss the contrapositive)

    members = defaultdict(list)
    for il in range(n2):
        members[find(il)].append(il)

    classes = [m for m in members.values() if len(m) > 1]
    if max_classes is not None and len(classes) > max_classes:
        # Budget: keep only the biggest classes — BUT a class C and its mirror ¬C must be kept or
        # dropped TOGETHER, else a variable is half-renamed (positive→rep, negative left) which
        # breaks skew-symmetry and is unsound. Group mirror pairs, keep whole groups by total size.
        root_of = {find(m[0]): m for m in classes}
        seen, pairs = set(), []
        for m in classes:
            r, rm = find(m[0]), find(m[0] ^ 1)
            key = frozenset((r, rm))
            if key in seen:
                continue
            seen.add(key)
            grp = [m] if (rm == r or rm not in root_of) else [m, root_of[rm]]
            pairs.append((sum(len(x) for x in grp), grp))
        pairs.sort(key=lambda t: t[0], reverse=True)
        kept = []
        for _, grp in pairs:
            if len(kept) >= max_classes:
                break
            kept.extend(grp)
        classes = kept

    def _canon(mem):
        if rep_score is None:
            return min(mem, key=lambda il: (il >> 1, il & 1))     # smallest var
        return max(mem, key=lambda il: (rep_score.get(il >> 1, 0.0), -(il >> 1)))  # highest score

    sub = list(range(n2))                     # literal index -> canonical (representative) literal
    n_classes = 0
    for mem in classes:
        seen = set()
        for il in mem:                        # both polarities of a var in one class ⇒ UNSAT
            v = il >> 1
            if v in seen:
                return list(_UNSAT_CNF), {}, {"status": "unsat", "n_classes": n_classes}
            seen.add(v)
        canon = _canon(mem)
        for il in mem:
            sub[il] = canon
        n_classes += 1

    var_map: dict[int, int] = {}
    for v in range(1, num_vars + 1):
        pos = 2 * (v - 1)
        if (sub[pos] >> 1) != (v - 1):        # this var's positive literal maps to another var
            var_map[v] = lit_index_to_ext_lit(sub[pos])

    lits_before = sum(len(c) for c in clauses)
    out: list[list[int]] = []
    for c in clauses:
        s: set[int] = set()
        taut = False
        for l in c:
            nl = lit_index_to_ext_lit(sub[ext_lit_to_lit_index(l)])
            if -nl in s:
                taut = True
                break
            s.add(nl)
        if not taut and s:
            out.append(list(s))

    stats = {"status": "ok", "n_classes": n_classes, "vars_removed": len(var_map),
             "clauses_removed": len(clauses) - len(out),
             "lits_removed": lits_before - sum(len(c) for c in out)}
    return out, var_map, stats

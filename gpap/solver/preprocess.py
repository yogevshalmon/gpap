"""GPU preprocessing pipeline: vivification + unit-closure + learned-clause subsumption.

Unlike clause *addition* (which perturbs CDCL chaotically — adding entailed-but-foreign
clauses lands the search in a different, unpredictable basin), every operation here is an
equivalence-preserving **reduction**: it shortens or deletes clauses the solver is already
built around, producing a smaller/tighter formula with the *same models*. This is exactly
the kind of work CaDiCaL already trusts (vivify / subsume), but GPAP's massively-parallel
BCP can do it far more exhaustively than a solver can afford under its tick budget.

Operations (each preserves the model set over the original variables):
  vivify          — GPUInprocessor.vivify_clauses: replace C by its shortest entailed prefix.
  unit_closure    — assert failed-literal units, BCP at the formula level (drop satisfied
                    clauses, strip falsified literals, cascade). Units are kept as unit clauses
                    so the result stays logically *equivalent* (not just equisatisfiable).
  apply_learned_subsumption — use discovered short clauses to subsume longer ones: a learned
                    clause D is added only if it subsumes ≥1 existing clause (which is then
                    dropped) — "the learned clause replaces a longer one", so no net growth.
  subsume_all     — generic forward subsumption + dedup + tautology removal (occurrence-pruned).

`preprocess()` wires them together and returns the reduced CNF + stats.
"""
import time
from collections import defaultdict

import torch

from gpap.clsdata.occ_list import ext_lit_to_lit_index, lit_index_to_ext_lit
from gpap.solver.inprocess import GPUInprocessor

# A trivially-UNSAT 2-clause formula, returned (same verdict) when preprocessing derives ⊥.
_UNSAT_CNF = [[1], [-1]]


def _frozen(clauses):
    """Clauses as frozensets, dropping tautologies (l and ¬l) and within-clause duplicates."""
    out = []
    for c in clauses:
        s = frozenset(c)
        if any(-l in s for l in s):
            continue  # tautology — always satisfied, drop
        if s:
            out.append(s)
    return out


def unit_closure(clauses, units):
    """Assert entailed units and unit-propagate at the formula level (equivalence-preserving).

    `units`: iterable of unit clauses ([lit]). Existing unit clauses in `clauses` are picked up
    too. Returns (new_clauses, assigned, status):
      new_clauses : unit clauses for every assigned literal + the simplified remaining clauses,
                    logically equivalent to (clauses ∧ units).
      assigned    : {var(1-based): bool}.
      status      : "ok" | "unsat" (a contradiction among entailed units ⇒ the formula is UNSAT).
    """
    val: dict[int, int] = {}        # var -> +1 / -1
    q: list[int] = []
    ok = True

    def enqueue(lit):
        nonlocal ok
        v, s = abs(lit), (1 if lit > 0 else -1)
        if v in val:
            if val[v] != s:
                ok = False
            return
        val[v] = s
        q.append(lit)

    for u in units:
        for lit in u:
            enqueue(lit)
    for c in clauses:
        if len(c) == 1:
            enqueue(c[0])
    if not ok:
        return list(_UNSAT_CNF), {}, "unsat"

    work = [set(c) for c in clauses if len(c) != 1]     # non-unit clauses (mutated in place)
    occ: dict[int, list[int]] = defaultdict(list)
    for idx, s in enumerate(work):
        for lit in s:
            occ[lit].append(idx)
    removed = [False] * len(work)

    head = 0
    while head < len(q) and ok:
        lit = q[head]
        head += 1
        for idx in occ.get(lit, ()):        # clauses with `lit` are satisfied
            removed[idx] = True
        for idx in occ.get(-lit, ()):       # clauses with `¬lit` lose that literal
            if removed[idx]:
                continue
            s = work[idx]
            s.discard(-lit)
            if len(s) == 0:
                ok = False
                break
            if len(s) == 1:
                enqueue(next(iter(s)))
    if not ok:
        return list(_UNSAT_CNF), {}, "unsat"

    out = [[v if val[v] == 1 else -v] for v in val]     # unit clauses for assigned vars
    for idx, s in enumerate(work):
        if removed[idx] or len(s) <= 1:                 # len<=1 ⇒ became a unit (already in `out`)
            continue
        out.append(list(s))
    return out, {v: (val[v] == 1) for v in val}, "ok"


def apply_learned_subsumption(clauses, learned):
    """Use discovered short clauses to subsume longer existing clauses ("replace" reduction).

    Each learned clause D (shortest first) is *added only if* it subsumes ≥1 strictly-longer
    existing clause (D ⊆ C); every such C is dropped. A learned clause that subsumes nothing is
    discarded (no net growth — the formula only shrinks). Equivalence-preserving. Returns
    (new_clauses, stats).
    """
    work = [set(c) for c in clauses]
    alive = [True] * len(work)
    occ: dict[int, set[int]] = defaultdict(set)
    for i, s in enumerate(work):
        for lit in s:
            occ[lit].add(i)

    n_sub = 0
    extra: list[frozenset] = []
    seen_extra: set[frozenset] = set()
    for D in sorted((frozenset(c) for c in learned), key=len):
        if not D or D in seen_extra:
            continue
        cands: set[int] | None = None              # existing clauses containing all of D
        for lit in D:
            s = occ.get(lit)
            if not s:
                cands = set()
                break
            cands = set(s) if cands is None else (cands & s)
            if not cands:
                break
        targets = [i for i in (cands or ()) if alive[i] and len(work[i]) > len(D)]
        if not targets:
            continue
        for i in targets:
            alive[i] = False
            for lit in work[i]:
                occ[lit].discard(i)
        n_sub += len(targets)
        extra.append(D)
        seen_extra.add(D)

    out = [list(work[i]) for i in range(len(work)) if alive[i]]
    out += [list(D) for D in extra]
    return out, {"subsumed": n_sub, "added": len(extra)}


def equivalent_literal_substitution(clauses, edges, num_vars, *, rep_score=None, max_classes=None):
    """Equivalent-literal substitution (ELS): collapse mutually-implied literals, drop variables.

    `edges`: (E, 2) tensor/list of (src_lit_idx, dst_lit_idx) implications src ⇒ dst (from
    `GPUInprocessor.discover_implications`, which BCPs to fixpoint — so the edge set is the
    *transitive closure*: x and y are equivalent iff BOTH x⇒y and y⇒x appear as direct edges).
    Two literals are equivalent when each implies the other; equivalence classes are merged with
    union-find and every member is rewritten to a single representative literal, which removes the
    other variables. Contrapositive edges (¬y⇒¬x) are produced by the same probe pass, so
    negations collapse consistently (sub[l^1] == sub[l]^1).

    Heuristic knobs (the "which equivalences / which representative" lever — selection, not
    discovery, is where prior inprocessing work found the signal):
      rep_score   : optional {var0(0-based): float}. The class member whose VARIABLE has the
                    highest score is kept as the representative (ties → smallest var); the others
                    are eliminated. None → keep the smallest-variable member (default). A
                    per-VARIABLE score keeps skew-symmetry (sub[l^1] == sub[l]^1). Use e.g. degree
                    to keep the "central" variable and eliminate the peripheral copies.
      max_classes : substitute only the largest `max_classes` equivalence classes (most variables
                    removed first); None → all. A budget knob, like the binary-injection budget.

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


def subsume_all(clauses):
    """Generic forward subsumption + dedup + tautology removal (occurrence/rarest-literal pruned).

    Drops any clause C for which some other clause D with |D| ≤ |C| satisfies D ⊆ C (D == C ⇒
    duplicate). Equivalence-preserving. Iterates *subsumers* shortest-first and finds the clauses
    they subsume in the occurrence list of the subsumer's rarest literal — any superset of D must
    contain every literal of D, so it appears in occ[l] for each l ∈ D (the rarest minimizes the
    scan).
    """
    work = _frozen(clauses)
    M = len(work)
    occ: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(work):
        for lit in c:
            occ[lit].append(i)
    alive = [True] * M
    for i in sorted(range(M), key=lambda j: len(work[j])):
        if not alive[i]:
            continue                                   # already subsumed by a shorter clause
        D = work[i]
        l_star = min(D, key=lambda l: len(occ[l]))     # rarest literal ⇒ fewest superset candidates
        for j in occ[l_star]:
            if j == i or not alive[j]:
                continue
            C = work[j]
            if len(C) < len(D):
                continue                               # too short to be a superset
            if len(C) == len(D):
                if C == D:
                    alive[j] = False                   # exact duplicate — keep i, drop j
            elif D <= C:
                alive[j] = False                       # D subsumes the strictly-longer C
    return [list(work[i]) for i in range(M) if alive[i]]


def _simplify_by_assignment(clauses, assigned):
    """Drop clauses satisfied by `assigned` and strip falsified literals (for discovered clauses)."""
    out = []
    for c in clauses:
        keep = []
        sat = False
        for lit in c:
            v = assigned.get(abs(lit))
            if v is None:
                keep.append(lit)
            elif (lit > 0) == v:
                sat = True
                break
        if not sat and keep:
            out.append(keep)
    return out


def preprocess(
    path: str,
    device: torch.device,
    *,
    do_vivify: bool = True,
    do_units: bool = True,
    do_subsume: bool = True,
    do_els: bool = True,
    vivify_max_len: int = 20,
    vivify_max_rounds: int | None = None,
    bin_vars: str = "auto",        # "auto" | "none" | "all"
    bin_pool: int = 256,
    k_cap: int | None = None,
    seed: int = 0,
):
    """Run the GPU reduction pipeline on a CNF file; return reduced clauses + stats.

    Pipeline (each step preserves satisfiability): GPU vivify → unit closure (failed-literal
    units) → equivalent-literal substitution (ELS, removes variables) → learned-binary
    subsumption → generic subsumption/dedup. Returns a dict with `clauses` (the reduced CNF),
    `nv`, before/after clause+literal counts, per-phase stats, `var_map` (removed_var → rep
    literal, for ELS model lifting), `gpu_ms`, and `status` ("ok"/"unsat").
    """
    import random

    # Read the clause literal lists once (GPUInprocessor builds the GPU formula from the same file).
    orig = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "cp%":
                continue
            lits = [x for x in (int(t) for t in line.split()) if x != 0]
            if lits:
                orig.append(lits)
    lits_orig = sum(len(c) for c in orig)

    inp = GPUInprocessor.from_cnf_file(path, device, k_cap=k_cap)
    N = inp.N

    t_gpu = time.perf_counter()
    phases: dict = {}

    units: list[list[int]] = []
    if do_units:
        units, s = inp.probe_failed_literals()
        phases["probe"] = s

    bins: list[list[int]] = []
    if do_subsume and bin_vars != "none":
        if bin_vars == "all":
            bvars = list(range(N))
        else:  # auto: exhaustive when cheap, else a degree-weighted sample
            from scripts.profile_engine import count_all_pairs
            if count_all_pairs(N) <= 4_000_000:
                bvars = list(range(N))
            else:
                rng = random.Random(seed)
                bvars = rng.sample(range(N), min(N, bin_pool))
        inp.reset_emitted()
        bins, s = inp.discover_binaries(bvars)
        phases["binaries"] = s

    edges = None
    if do_els:
        edges, els_units, s = inp.discover_implications()
        units = units + els_units if do_units else units   # probe ⊆ implication failed-literals
        phases["implications"] = s

    clauses = orig
    if do_vivify:
        clauses, s = inp.vivify_clauses(clauses, max_len=vivify_max_len, max_rounds=vivify_max_rounds)
        phases["vivify"] = s

    del inp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gpu_ms = (time.perf_counter() - t_gpu) * 1e3

    status = "ok"
    assigned: dict[int, bool] = {}
    var_map: dict[int, int] = {}
    if do_units:
        clauses, assigned, status = unit_closure(clauses, units)

    # ELS after unit-closure: assigned vars are gone from `clauses`, so substitution only touches
    # live variables; equivalences are entailed by the original formula so this stays equisat.
    if status == "ok" and do_els and edges is not None:
        clauses, var_map, s = equivalent_literal_substitution(clauses, edges, N)
        status = s["status"]
        phases["els"] = s

    if status == "ok" and do_subsume and bins:
        bins = _simplify_by_assignment(bins, assigned)
        clauses, s = apply_learned_subsumption(clauses, bins)
        phases["learned_subsume"] = s

    if status == "ok" and do_subsume:
        before = len(clauses)
        clauses = subsume_all(clauses)
        phases["subsume_all"] = {"removed": before - len(clauses)}

    if status == "unsat":
        clauses = list(_UNSAT_CNF)

    lits_final = sum(len(c) for c in clauses)
    return {
        "clauses": clauses, "nv": N, "status": status, "var_map": var_map,
        "n_clauses_orig": len(orig), "n_clauses_final": len(clauses),
        "lits_orig": lits_orig, "lits_final": lits_final,
        "n_units": len(units), "n_bins": len(bins), "n_vars_removed": len(var_map),
        "gpu_ms": gpu_ms, "phases": phases,
    }

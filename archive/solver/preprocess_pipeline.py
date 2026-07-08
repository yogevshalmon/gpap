"""ARCHIVED: the GPU preprocessing-for-solve pipeline (vivify + subsumption + preprocess()).

This was the inprocessing-era reduction pipeline built on GPUInprocessor (archive/solver/
inprocess.py). Retired with the rest of preprocessing-for-solve (see archive/README.md and
docs/findings/INSIGHTS.md); the two primitives the SAMPLER needs — unit_closure and
equivalent_literal_substitution — stayed behind in the package (gpap sampling preprocess
module) and are imported from there. Not maintained; imports here may be stale.
"""
import time
from collections import defaultdict

import torch

from gpap.engine.formula import ext_lit_to_lit_index, lit_index_to_ext_lit
from gpap.sampling.preprocess import _UNSAT_CNF, equivalent_literal_substitution, unit_closure
from archive.solver.inprocess import GPUInprocessor


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
            if N * (N - 1) * 2 <= 4_000_000:
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

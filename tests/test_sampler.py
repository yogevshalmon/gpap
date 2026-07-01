"""Soundness + simplification-legitimacy tests for the GPU BCP-completion sampler.

The load-bearing correctness claim: unit-propagation simplification (simplify_cnf) preserves the
set of valid SUPPORT PROJECTIONS, so sampling the simplified formula and reconstructing yields the
same distribution support as sampling the original. These tests enumerate the exact projection set
(via CaDiCaL) and check GPAP against it, with and without simplification."""
import random

import torch
from pysat.formula import CNF
from pysat.solvers import Solver

from gpap.solver.sampler import GPUSampler, simplify_cnf, simplify_cnf_els, _satisfies

DEV = torch.device("cpu")  # tests run on CPU for determinism/portability


def _enumerate_projections(clauses, support):
    """Exact set of valid support projections (0-based support) via CaDiCaL + blocking."""
    s = Solver(name="cadical195", bootstrap_with=CNF(from_clauses=clauses))
    out = set()
    while s.solve():
        m = set(s.get_model())
        proj = tuple((v + 1) if (v + 1) in m else -(v + 1) for v in support)
        out.add(proj)
        s.add_clause([-((v + 1) if (v + 1) in m else -(v + 1)) for v in support])
    s.delete()
    return out


def test_sound_basic():
    # (x1∨x2)∧(¬x1∨x3)∧(¬x2∨¬x3) → exactly 2 models {-1,2,-3},{1,-2,3}
    clauses = [[1, 2], [-1, 3], [-2, -3]]
    smp = GPUSampler.from_clauses(3, clauses, DEV, k_cap=64, seed=1)
    out = smp.sample(2000, decisions_per_round=1, max_chunks=40, collect=True,
                     verify_clauses=clauses)
    assert out["all_valid"], "produced a model that violates the formula"
    got = {tuple(m) for m in out["models"]}
    assert got.issubset({(-1, 2, -3), (1, -2, 3)})
    assert out["n_models"] > 0


def test_unsat():
    smp = GPUSampler.from_clauses(1, [[1], [-1]], DEV, k_cap=16, seed=1)
    out = smp.sample(10, max_chunks=5, collect=True, verify_clauses=[[1], [-1]])
    assert out["n_models"] == 0


def test_unsat_via_simplify():
    smp = GPUSampler.from_clauses(1, [[1], [-1]], DEV, k_cap=16, seed=1, simplify=True)
    out = smp.sample(10, max_chunks=5, collect=True)
    assert out["n_models"] == 0


def _controlled_formula(seed=3, S=6, nc=4, ng=10, add_forced=True, add_dontcare=True):
    """Small formula with a support of S vars, a few constraints, gate-defined non-support vars,
    optionally a unit-forced support var and a don't-care support var (in no clause)."""
    rng = random.Random(seed)
    clauses, sup, var = [], list(range(1, S + 1)), S + 1
    if add_forced:
        clauses.append([sup[0]])                       # force s1 = True (unit) → forced support var
    core = sup[1:-1] if add_dontcare else sup[1:]      # leave last support var in no clause
    for _ in range(nc):
        vs = rng.sample(core, min(3, len(core)))
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    for _ in range(ng):
        a, b = rng.sample(core, 2)
        g = var; var += 1
        sa = a if rng.random() < 0.5 else -a
        sb = b if rng.random() < 0.5 else -b
        clauses.append([-g, sa]); clauses.append([-g, sb]); clauses.append([g, -sa, -sb])
    return var - 1, clauses, [v - 1 for v in sup]       # 0-based support


def test_simplify_cnf_classifies_forced_and_dontcare():
    nv, clauses, sup = _controlled_formula()
    res = simplify_cnf(nv, clauses, sup)
    assert res is not None
    nv2, cl2, sup2, old_of_new, forced, free_support = res
    # s1 (var 1) was unit-forced → appears in `forced`, not in the compacted support
    assert any(abs(l) == 1 for l in forced)
    # the last support var was in no clause → don't-care → in free_support
    assert (sup[-1]) in free_support
    # no compacted clause references a variable outside 1..nv2
    assert all(1 <= abs(l) <= nv2 for c in cl2 for l in c)


def test_simplify_preserves_projection_set():
    """The decisive legitimacy check: valid support projections are identical with/without
    simplification, and GPAP-simplified covers exactly that set (sound + complete)."""
    nv, clauses, sup = _controlled_formula(seed=7)
    exact = _enumerate_projections(clauses, sup)
    assert len(exact) > 1

    # GPAP WITHOUT simplification
    smp_plain = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=512, seed=2, support=sup)
    op = smp_plain.sample(60_000, decisions_per_round=1, max_chunks=120, collect=True,
                          verify_clauses=clauses, support=sup)
    plain_projs = {tuple(l for l in m if abs(l) - 1 in set(sup)) for m in op["models"]}
    plain_projs = {tuple((v + 1) if (v + 1) in set(m) else -(v + 1) for v in sup)
                   for m in op["models"]}
    assert op["all_valid"]
    assert plain_projs.issubset(exact)

    # GPAP WITH simplification — reconstruct, verify vs ORIGINAL clauses, compare projection set
    smp_simp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=512, seed=2, support=sup,
                                       simplify=True)
    osr = smp_simp.sample(60_000, decisions_per_round=1, max_chunks=120, collect=True,
                          verify_clauses=clauses)
    assert osr["all_valid"], "simplified+reconstructed model violates ORIGINAL clauses"
    simp_projs = {tuple((v + 1) if (v + 1) in set(m) else -(v + 1) for v in sup)
                  for m in osr["models"]}
    assert simp_projs.issubset(exact), "simplification produced an INVALID projection (unsound)"
    # completeness: with enough samples both cover the full exact set
    assert simp_projs == exact, f"simplified missed projections: {exact - simp_projs}"
    assert plain_projs == exact, f"plain missed projections: {exact - plain_projs}"


def _equiv_formula(seed=5, S=6):
    """Formula with explicit binary equivalences: s2≡s3, s4≡s5 (support vars) and g≡s1 (a non-support
    gate var). ELS must collapse each pair to one variable, and reconstruction must keep them EQUAL."""
    rng = random.Random(seed)
    clauses, sup, var = [], list(range(1, S + 1)), S + 1
    for _ in range(4):                                  # base constraints among the support
        vs = rng.sample(sup, 3)
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    clauses += [[-2, 3], [-3, 2], [-4, 5], [-5, 4]]     # s2≡s3, s4≡s5
    g = var; var += 1
    clauses += [[-g, 1], [-1, g], [g, 6]]               # g≡s1, g used in a clause
    return var - 1, clauses, [v - 1 for v in sup]       # 0-based support


def test_els_removes_variables():
    nv, clauses, sup = _equiv_formula()
    res = simplify_cnf_els(nv, clauses, sup)
    assert res is not None
    nv_r, cl_r, sup_r, recon = res
    assert nv_r < nv, "ELS removed no variables despite explicit equivalences"
    # s3 (var3) and s5 (var5) are equivalent to s2/s4 → must map to a representative, not themselves
    rep_of = recon["rep_of"]
    assert rep_of[2][0] != 3 or rep_of[1][0] != 2, "s2/s3 equivalence not collapsed"
    assert all(1 <= abs(l) <= nv_r for c in cl_r for l in c)


def test_els_preserves_projection_set():
    """ELS-reduced sampling reconstructs models that (a) satisfy the ORIGINAL formula and (b) cover
    EXACTLY the valid support-projection set — the strongest soundness gate (an inconsistent
    reconstruction of equivalent vars would emit an invalid projection and fail the subset check)."""
    nv, clauses, sup = _equiv_formula(seed=8)
    exact = _enumerate_projections(clauses, sup)
    assert len(exact) > 1
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=512, seed=2, support=sup, els=True)
    out = smp.sample(80_000, decisions_per_round=1, max_chunks=160, collect=True,
                     verify_clauses=clauses)
    assert out["all_valid"], "ELS+reconstructed model violates ORIGINAL clauses (unsound)"
    projs = {tuple((v + 1) if (v + 1) in set(m) else -(v + 1) for v in sup) for m in out["models"]}
    assert projs.issubset(exact), f"ELS produced an INVALID projection: {projs - exact}"
    assert projs == exact, f"ELS missed projections: {exact - projs}"


def test_els_equivalent_vars_stay_equal():
    """The consistency requirement: s2≡s3 and s4≡s5 must be EQUAL in every reconstructed model."""
    nv, clauses, sup = _equiv_formula(seed=3)
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=512, seed=4, support=sup, els=True)
    out = smp.sample(20_000, decisions_per_round=1, max_chunks=60, collect=True)
    for m in out["models"]:
        s = set(m)
        assert (2 in s) == (3 in s), "equivalent vars s2,s3 reconstructed unequal"
        assert (4 in s) == (5 in s), "equivalent vars s4,s5 reconstructed unequal"


def test_fully_determined_formula_m0():
    """A formula that simplification reduces to ZERO clauses (everything unit-forced or equivalence-
    collapsed) must still sample without crashing — the BCP engine handles M=0 (empty clause set)."""
    # all-unit clauses → unit closure determines every var → 0 remaining clauses
    clauses = [[1], [-2], [3], [4]]
    sup = [0, 1, 2, 3]
    smp = GPUSampler.from_clauses(4, clauses, DEV, k_cap=64, seed=1, support=sup, els=True)
    assert smp.M == 0, "expected the formula to fully simplify to 0 clauses"
    out = smp.sample(500, decisions_per_round=4, max_chunks=10, collect=True, verify_clauses=clauses)
    assert out["n_models"] > 0
    assert out["all_valid"], "M=0 reconstructed model violates the original clauses"
    # forced support vars must be pinned (x1=T, x2=F, x3=T, x4=T) in every model
    for m in out["models"]:
        s = set(m)
        assert 1 in s and -2 in s and 3 in s and 4 in s


def test_b_larger_than_num_vars():
    """decisions_per_round B can exceed the (post-simplify) variable count on tiny formulas —
    _pick_decisions must clamp the topk instead of erroring."""
    clauses = [[1, 2], [-1, -2]]                       # 2 vars, 2 models
    smp = GPUSampler.from_clauses(2, clauses, DEV, k_cap=64, seed=1)
    out = smp.sample(200, decisions_per_round=8, max_chunks=10, collect=True, verify_clauses=clauses)
    assert out["n_models"] > 0 and out["all_valid"]


def test_simplify_dontcare_both_polarities():
    """A don't-care support var must take BOTH polarities across samples (uniform), not be pinned."""
    nv, clauses, sup = _controlled_formula(seed=11)
    dc = sup[-1]  # 0-based don't-care support var
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=512, seed=4, support=sup, simplify=True)
    out = smp.sample(20_000, decisions_per_round=1, max_chunks=60, collect=True)
    vals = {(1 if (dc + 1) in set(m) else 0) for m in out["models"]}
    assert vals == {0, 1}, "don't-care support var was not sampled in both polarities"

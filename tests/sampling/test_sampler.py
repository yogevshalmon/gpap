"""Soundness and simplification tests for GPUSampler.

Simplification (unit closure, ELS) must preserve the set of valid support projections. The
exact set is enumerated with CaDiCaL and compared with what GPAP samples, with and without
simplification. The simplify/ELS classification tests are host-only."""
import random

import pytest

from gpap.sampling.sampler import GPUSampler, simplify_cnf, simplify_cnf_els

from tests.support import DEV, enumerate_projections


@pytest.mark.gpu
def test_sound_basic():
    # (x1∨x2)∧(¬x1∨x3)∧(¬x2∨¬x3) → exactly 2 models {-1,2,-3},{1,-2,3}
    clauses = [[1, 2], [-1, 3], [-2, -3]]
    smp = GPUSampler.from_clauses(3, clauses, DEV, k_cap=64, seed=1)
    out = smp.sample(2000, decisions_per_round=1, max_chunks=40, collect=True,
                     verify_clauses=clauses, verify_sound=True)
    assert out["all_valid"], "produced a model that violates the formula"
    assert out["sound_check"] == "ok"
    assert {tuple(m) for m in out["models"]} == {(-1, 2, -3), (1, -2, 3)}


@pytest.mark.gpu
def test_wide_decision_rounds_are_accepted_and_sound():
    """B has no ceiling: the trail is sized by the variable count alone.

    Every trail push is deduplicated against the assignment bitmap, so a variable adds at most
    one entry for any B. `B >= |support|` is the exactly-uniform configuration (barbarik's `bsup`
    arm, tests/experiments/test_uniformity_fuzz.py), so a cap on B would remove it. B = 64/65
    straddle a former fixed trail size; N and 4N go past the variable count."""
    rng = random.Random(7)
    N, M = 200, 60                                  # loose: most assignments satisfy it
    clauses = [[v if rng.random() < 0.5 else -v
                for v in rng.sample(range(1, N + 1), 3)] for _ in range(M)]
    for B in (64, 65, N, 4 * N):
        smp = GPUSampler.from_clauses(N, clauses, DEV, seed=5, k_cap=4096)
        out = smp.sample(3000, decisions_per_round=B, collect=True, verify_clauses=clauses,
                         verify_sound=True)
        assert out["n_models"] > 0, f"B={B} produced nothing"
        assert out["all_valid"], f"B={B} emitted a row that violates the formula"
        assert out["sound_check"] == "ok", f"B={B} left an unsound row"


@pytest.mark.gpu
def test_config_portfolio_soft_lookahead():
    # 3-tuple configs (random_var, B, lookahead), including soft: every model satisfies the
    # formula, and the reported dominant config and breakdown keep the 3-tuple shape.
    clauses = [[1, 2], [-1, 3], [-2, -3]]
    smp = GPUSampler.from_clauses(3, clauses, DEV, k_cap=64, seed=1)
    out = smp.sample(2000, configs=[(True, 1, "soft"), (True, 8, False)], max_chunks=30,
                     collect=True, verify_clauses=clauses, verify_sound=True)
    assert out["all_valid"], "soft-lookahead portfolio produced an invalid model"
    assert out["sound_check"] == "ok"
    assert out["n_models"] > 0
    assert len(out["dominant_config"]) == 3                      # (rv, B, lookahead)
    assert any("la=soft" in k for k in out["config_breakdown"])  # label carries lookahead
    # backward-compat: a bare 2-tuple config still normalises (lookahead defaults off) and runs.
    out2 = smp.sample(500, configs=[(True, 4)], max_chunks=10, collect=True, verify_clauses=clauses)
    assert out2["all_valid"] and len(out2["dominant_config"]) == 3


@pytest.mark.gpu
def test_unsat():
    smp = GPUSampler.from_clauses(1, [[1], [-1]], DEV, k_cap=16, seed=1)
    out = smp.sample(10, max_chunks=5, collect=True, verify_clauses=[[1], [-1]])
    assert out["n_models"] == 0


@pytest.mark.gpu
def test_same_seed_draws_the_same_models():
    """The trajectory is a pure function of the seed and the order, so two samplers at one seed
    and batch return the same model set."""
    clauses = [[1, 2, 3], [-1, 4], [-2, -4], [3, -4, 5], [-5, 1]]
    runs = []
    for _ in range(2):
        smp = GPUSampler.from_clauses(5, clauses, DEV, k_cap=128, seed=17)
        out = smp.sample(500, decisions_per_round=2, max_chunks=8, collect=True,
                         verify_clauses=clauses)
        assert out["all_valid"]
        runs.append(sorted(tuple(m) for m in out["models"]))
    assert runs[0] and runs[0] == runs[1], "same seed produced different model sets"


@pytest.mark.gpu
def test_retired_lookahead_modes_are_refused():
    smp = GPUSampler.from_clauses(3, [[1, 2], [-1, 3]], DEV, k_cap=16, seed=1)
    for la in ("full", "orac", True):
        with pytest.raises(ValueError):
            smp.sample(10, configs=[(True, 1, la)], max_chunks=1)


@pytest.mark.gpu
def test_a_decision_round_never_straddles_the_support_boundary(tmp_path):
    """With a support that determines everything else, a huge B keeps success high: one
    decision round must not mix support and completion variables."""
    import os

    from gpap.circuits.bench import build_cnf, parse_bench, write_dimacs
    from tests.conftest import SEQ_DIR
    b, support, _, _ = build_cnf(parse_bench(os.path.join(SEQ_DIR, "s386.bench")), T=2, seed=1)
    p = tmp_path / "s386c2.cnf"
    write_dimacs(str(p), b, support)
    smp = GPUSampler.from_cnf_file(str(p), DEV, simplify=True, els=True, k_cap=512, seed=5)
    out = smp.sample(10 ** 9, configs=[(True, 4096, False)], max_chunks=2, collect=False,
                     verify_sound=True)
    assert out["sound_check"] == "ok"
    assert out["success_rate"] > 0.5, "huge-B success collapsed: phase boundary straddled?"


@pytest.mark.gpu
def test_unsat_via_simplify():
    smp = GPUSampler.from_clauses(1, [[1], [-1]], DEV, k_cap=16, seed=1, simplify=True)
    out = smp.sample(10, max_chunks=5, collect=True)
    assert out["n_models"] == 0


def _projections(models, support):
    """Full DIMACS models -> 0/1 projections over the 0-based `support`."""
    return {tuple(1 if (v + 1) in set(m) else 0 for v in support) for m in models}


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


@pytest.mark.gpu
def test_simplify_preserves_projection_set():
    """Sampling with and without simplification both cover exactly the valid support
    projections (sound and complete)."""
    nv, clauses, sup = _controlled_formula(seed=7)
    exact = enumerate_projections(clauses, [v + 1 for v in sup])
    assert len(exact) > 1

    # without simplification
    smp_plain = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=512, seed=2, support=sup)
    op = smp_plain.sample(60_000, decisions_per_round=1, max_chunks=120, collect=True,
                          verify_clauses=clauses, support=sup)
    plain_projs = _projections(op["models"], sup)
    assert op["all_valid"]
    assert plain_projs.issubset(exact)

    # with simplification: reconstruct, verify against the original clauses, compare sets
    smp_simp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=512, seed=2, support=sup,
                                       simplify=True)
    osr = smp_simp.sample(60_000, decisions_per_round=1, max_chunks=120, collect=True,
                          verify_clauses=clauses)
    assert osr["all_valid"], "simplified+reconstructed model violates ORIGINAL clauses"
    simp_projs = _projections(osr["models"], sup)
    assert simp_projs.issubset(exact), "simplification produced an INVALID projection (unsound)"
    # completeness: with enough samples both cover the full exact set
    assert simp_projs == exact, f"simplified missed projections: {exact - simp_projs}"
    assert plain_projs == exact, f"plain missed projections: {exact - plain_projs}"


def _equiv_formula(seed=5, S=6):
    """Formula with binary equivalences s2≡s3, s4≡s5 (support) and g≡s1 (a non-support gate).
    ELS must collapse each pair, and reconstruction must keep the pair equal."""
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
    # s3 and s5 must map to a representative, not to themselves
    rep_of = recon["rep_of"]
    assert rep_of[2][0] != 3 or rep_of[1][0] != 2, "s2/s3 equivalence not collapsed"
    assert all(1 <= abs(l) <= nv_r for c in cl_r for l in c)


@pytest.mark.gpu
def test_els_preserves_projection_set():
    """ELS-reduced sampling reconstructs models that satisfy the original formula and cover
    exactly the valid support projections. An inconsistent reconstruction of equivalent vars
    would emit an invalid projection and fail the subset check."""
    nv, clauses, sup = _equiv_formula(seed=8)
    exact = enumerate_projections(clauses, [v + 1 for v in sup])
    assert len(exact) > 1
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=512, seed=2, support=sup, els=True)
    out = smp.sample(80_000, decisions_per_round=1, max_chunks=160, collect=True,
                     verify_clauses=clauses)
    assert out["all_valid"], "ELS+reconstructed model violates ORIGINAL clauses (unsound)"
    projs = _projections(out["models"], sup)
    assert projs.issubset(exact), f"ELS produced an INVALID projection: {projs - exact}"
    assert projs == exact, f"ELS missed projections: {exact - projs}"


@pytest.mark.gpu
def test_els_equivalent_vars_stay_equal():
    """s2≡s3 and s4≡s5 are equal in every reconstructed model."""
    nv, clauses, sup = _equiv_formula(seed=3)
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=512, seed=4, support=sup, els=True)
    out = smp.sample(20_000, decisions_per_round=1, max_chunks=60, collect=True)
    assert out["models"], "no models to check — an ELS bug that emitted nothing would pass"
    for m in out["models"]:
        s = set(m)
        assert (2 in s) == (3 in s), "equivalent vars s2,s3 reconstructed unequal"
        assert (4 in s) == (5 in s), "equivalent vars s4,s5 reconstructed unequal"


@pytest.mark.gpu
def test_fully_determined_formula_m0():
    """A formula that simplifies to zero clauses (M=0) still samples, with forced vars pinned."""
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


@pytest.mark.gpu
def test_b_larger_than_num_vars():
    """B may exceed the variable count; the decision pick exhausts the order instead of
    erroring."""
    clauses = [[1, 2], [-1, -2]]                       # 2 vars, 2 models
    smp = GPUSampler.from_clauses(2, clauses, DEV, k_cap=64, seed=1)
    out = smp.sample(200, decisions_per_round=8, max_chunks=10, collect=True, verify_clauses=clauses)
    assert out["n_models"] > 0 and out["all_valid"]


@pytest.mark.gpu
def test_simplify_dontcare_both_polarities():
    """A don't-care support var takes both polarities across samples."""
    nv, clauses, sup = _controlled_formula(seed=11)
    dc = sup[-1]  # 0-based don't-care support var
    smp = GPUSampler.from_clauses(nv, clauses, DEV, k_cap=512, seed=4, support=sup, simplify=True)
    out = smp.sample(20_000, decisions_per_round=1, max_chunks=60, collect=True)
    vals = {(1 if (dc + 1) in set(m) else 0) for m in out["models"]}
    assert vals == {0, 1}, "don't-care support var was not sampled in both polarities"


# --------------------------------------------------------------------- portfolio configs

def test_portfolio_configs_are_tuple_compatible():
    """Portfolio `SamplingConfig` NamedTuples still behave as the `(random_var, B, lookahead)`
    triples that consumers read."""
    from gpap.sampling.pipeline import (CIRCUIT_PORTFOLIO, DEFAULT_PORTFOLIO, Lookahead,
                                        SOFT_CONFIG, SamplingConfig, VarOrder)
    for c in (*CIRCUIT_PORTFOLIO, *DEFAULT_PORTFOLIO, SOFT_CONFIG):
        assert isinstance(c, tuple) and len(c) == 3
        assert bool(c[0]) is (c[0] is VarOrder.RANDOM)     # the kernel flag reads `if random_var:`
        assert isinstance(c[1], int) and c[1] >= 1
        # The sampler tests `lookahead in ("soft", "fsoft")`; everything else must be falsy.
        assert (c[2] in ("soft", "fsoft")) == (c[2] is Lookahead.SOFT)
        assert bool(c[2]) == (c[2] is Lookahead.SOFT)
    assert SamplingConfig(VarOrder.STRUCTURED, 1) == (False, 1, Lookahead.NONE)
    assert CIRCUIT_PORTFOLIO[0].describe() == "random order, B=8"
    assert "soft-polarity" in SOFT_CONFIG.describe()


def test_soft_is_not_in_any_default_portfolio():
    """`soft` biases the distribution (docs/findings/UNIFORMITY_FINDINGS.md) and the bandit
    selects on throughput alone, so no default portfolio includes it."""
    from gpap.sampling.pipeline import (CIRCUIT_PORTFOLIO, DEFAULT_PORTFOLIO, Lookahead,
                                        SOFT_CONFIG, portfolio)
    for name in ("circuit", "default"):
        assert all(c[2] is Lookahead.NONE for c in portfolio(name)), \
            f"{name} portfolio must be uniformity-safe by default"
        assert SOFT_CONFIG in portfolio(name, allow_soft=True)
    assert all(c[2] is Lookahead.NONE for c in (*CIRCUIT_PORTFOLIO, *DEFAULT_PORTFOLIO))


@pytest.mark.gpu
def test_config_breakdown_keys_stay_legacy_formatted():
    """Results-JSON config keys keep the pre-enum format, so old and new rows compare."""
    from gpap.sampling.pipeline import portfolio
    smp = GPUSampler.from_clauses(4, [[1, 2], [-1, 3], [-2, -3], [2, 3, 4]], DEV, k_cap=256,
                                  seed=1)
    r = smp.sample(300, configs=portfolio("default"), collect=False, max_chunks=9)
    bd = r.get("config_breakdown")
    assert bd, "no config_breakdown to check — `or {}` would have passed vacuously"
    for k in bd:
        assert k.startswith("rv=True,") or k.startswith("rv=False,"), k


@pytest.mark.gpu
def test_timed_run_never_gathers_a_model_to_the_host():
    """The timed run never gathers a model row to the host (emission symmetry).

    Rivals are timed on generation alone, so `collect=False` asks the core for `keep_cap=0`.
    This is the GPAP half of eval/rivals/check_emission_symmetry.py. The spy sits on the core
    sampler's `bits`, where the gather happens; the `collect=True` run shows it can fire.
    """
    from gpap.sampling.pipeline import Lookahead
    smp = GPUSampler.from_clauses(4, [[1, 2], [-1, 3], [-2, -3], [2, 3, 4]], DEV, k_cap=256,
                                  seed=1)
    gathers, real = [], smp._smp.bits
    smp._smp.bits = lambda *a, **k: (gathers.append(a), real(*a, **k))[1]

    r = smp.sample(300, configs=[(True, 1, Lookahead.NONE)], collect=False, max_chunks=4)
    assert r["n_models"] > 0, "precondition: the timed run must have produced rows to gather"
    assert not gathers, "the timed run gathered model rows to the host"
    assert smp._smp.config_stats(0).kept_rows == 0, "the core kept rows for a timed run"

    r2 = smp.sample(50, configs=[(True, 1, Lookahead.NONE)], collect=True, max_chunks=4)
    assert r2["models"], "collect=True must still gather"
    assert gathers, "the spy never fired even on collect=True — it is watching the wrong call"

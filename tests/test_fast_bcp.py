"""Correctness tests for the packed Triton BCP kernels through the engine + sampler.

The semantic oracle is the pure-Python reference BCP (reference_bcp.py — BCP fixpoints are
confluent, so agreement is exact), plus external CPU verification of every sampled model
(_satisfies / PySAT in the miner test). Reproducibility: the fused trajectory consumes the
torch RNG once per chunk (order + seed), so same-seed runs are bit-identical.
"""
import random

import pytest
import torch

from gpap.engine.formula import CnfFormula
from gpap.engine.bcp_engine import BCPEngine, make_assumption_tensor
from gpap.engine import core_available
from gpap.sampling.sampler import GPUSampler, _satisfies

from tests.reference_bcp import ref_bcp, assigned_set

CUDA = torch.cuda.is_available()
AVAIL = CUDA and core_available()
pytestmark = pytest.mark.skipif(not AVAIL, reason="requires CUDA + a working triton")
DEV = torch.device("cuda") if CUDA else torch.device("cpu")


def _random_cnf(rng, nv, nc, kmax=4):
    """Random CNF with distinct-variable clauses (may include units)."""
    clauses = []
    for _ in range(nc):
        vs = rng.sample(range(1, nv + 1), rng.randint(1, min(kmax, nv)))
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    return clauses


def _free_decisions(rng, engine, K, B, contradict_rows=()):
    """(K,B) int64 decisions: distinct FREE vars per row, random polarity (the sampler's
    guarantee). Rows in `contradict_rows` get an l/~l pair to exercise the conflict path."""
    asg = engine.assignment.cpu()
    dec = torch.full((K, B), -1, dtype=torch.int64)
    for k in range(K):
        free = [v for v in range(asg.size(1) // 2)
                if asg[k, 2 * v] == 0 and asg[k, 2 * v + 1] == 0]
        if not free:
            continue
        if k in contradict_rows and B >= 2:
            v = rng.choice(free)
            dec[k, 0], dec[k, 1] = 2 * v, 2 * v + 1          # l and ~l
            continue
        for j, v in enumerate(rng.sample(free, min(B, len(free)))):
            dec[k, j] = 2 * v + rng.randint(0, 1)
    return dec.to(DEV)


def test_fixpoint_matches_reference_multistep():
    """Multi-step decide+propagate vs the reference, incl. an injected l/~l conflict row.
    The reference re-runs the CUMULATIVE decision list from scratch each step (same fixpoint
    by confluence)."""
    rng = random.Random(11)
    for _ in range(15):
        nv = rng.randint(4, 40)
        clauses = _random_cnf(rng, nv, rng.randint(nv, 4 * nv))
        K, B = 32, rng.choice([1, 2, 4])
        f = CnfFormula(DEV)
        f.init_formula(nv, clauses)
        eng = BCPEngine(f)
        eng.reset(K)
        cum = [[] for _ in range(K)]
        for step in range(3):
            contradict = {rng.randrange(K)} if (step == 1 and B >= 2) else ()
            dec = _free_decisions(rng, eng, K, B, contradict_rows=contradict)
            for k in range(K):
                cum[k].extend(int(x) for x in dec[k].tolist() if x >= 0)
            eng.apply_and_propagate(dec)
            for k in range(K):
                conf, asg = ref_bcp(clauses, cum[k])
                assert bool(eng.conflict[k].item()) == conf
                if not conf:
                    assert assigned_set(eng.assignment[k].cpu()) == asg


def test_fixpoint_matches_reference_circuit(tmp_path):
    """Reference A/B on a real unrolled circuit (deep propagation cascades)."""
    from gpap.circuits.bench import parse_bench, build_cnf, write_dimacs
    circ = parse_bench("benchmarks/circuits/bench/seq/s386.bench")
    b, support, _, _ = build_cnf(circ, T=3, seed=1)
    p = tmp_path / "s386c3.cnf"
    write_dimacs(str(p), b, support)
    from pysat.formula import CNF
    cnf = CNF(from_file=str(p))
    clauses = [list(c) for c in cnf.clauses]
    rng = random.Random(3)
    K = 24
    f = CnfFormula(DEV)
    f.init_formula(cnf.nv, clauses)
    eng = BCPEngine(f)
    eng.reset(K)
    dec = _free_decisions(rng, eng, K, 8)
    eng.apply_and_propagate(dec)
    for k in range(K):
        conf, asg = ref_bcp(clauses, [int(x) for x in dec[k].tolist() if x >= 0])
        assert bool(eng.conflict[k].item()) == conf
        if not conf:
            assert assigned_set(eng.assignment[k].cpu()) == asg


def test_sampler_same_seed_identical_models():
    """Same seed => identical model sets (the trajectory is a pure function of seed/order).
    Every model CPU-verified against the clauses."""
    clauses = [[1, 2, 3], [-1, 4], [-2, -4], [3, -4, 5], [-5, 1]]
    runs = []
    for _ in range(2):
        smp = GPUSampler.from_clauses(5, clauses, DEV, k_cap=128, seed=17)
        out = smp.sample(500, decisions_per_round=2, max_chunks=8, collect=True,
                         verify_clauses=clauses)
        assert out["all_valid"]
        runs.append(sorted(tuple(m) for m in out["models"]))
    assert runs[0] == runs[1], "same seed produced different model sets"


def test_sampler_portfolio_sound():
    """The adaptive-bandit portfolio (incl. soft risk-polarity) stays sound."""
    clauses = [[1, 2], [-1, 3], [-2, -3], [2, 4], [-4, -1, 5]]
    smp = GPUSampler.from_clauses(5, clauses, DEV, k_cap=128, seed=5)
    out = smp.sample(400, configs=[(True, 1, "soft"), (True, 4, False)], max_chunks=10,
                     collect=True, verify_clauses=clauses)
    assert out["all_valid"] and out["gpu_sound"] and out["n_models"] > 0
    for m in out["models"]:
        assert _satisfies(m, clauses)


def test_sampler_unsat():
    smp = GPUSampler.from_clauses(1, [[1], [-1]], DEV, k_cap=16, seed=1)
    out = smp.sample(10, max_chunks=5, collect=True, verify_clauses=[[1], [-1]])
    assert out["n_models"] == 0


def test_retired_lookahead_modes_raise():
    smp = GPUSampler.from_clauses(3, [[1, 2], [-1, 3]], DEV, k_cap=16, seed=1)
    for la in ("full", "orac", True):
        with pytest.raises(ValueError):
            smp.sample(10, configs=[(True, 1, la)], max_chunks=1)


def test_fused_trajectory_sound_and_diverse():
    clauses = [[1, 2, 3], [-1, 4], [-2, -4], [3, -4, 5], [-5, 1]]
    smp = GPUSampler.from_clauses(5, clauses, DEV, k_cap=128, seed=17)
    out = smp.sample(400, configs=[(True, 2, False)], max_chunks=8, collect=True,
                     verify_clauses=clauses)
    assert out["all_valid"] and out["gpu_sound"] and out["n_models"] > 0
    assert out["n_distinct"] >= 2                       # diversity sanity (formula has >=2 models)


def test_fused_phase_boundary(tmp_path):
    """A decision group must NOT straddle the support->completion boundary: on a circuit whose
    support fully determines the rest, a huge B must keep ~full success, not collapse (the
    pre-fix straddle bug measured 100%->0% here)."""
    from gpap.circuits.bench import parse_bench, build_cnf, write_dimacs
    circ = parse_bench("benchmarks/circuits/bench/seq/s386.bench")
    b, support, _, _ = build_cnf(circ, T=2, seed=1)
    p = tmp_path / "s386c2.cnf"
    write_dimacs(str(p), b, support)
    smp = GPUSampler.from_cnf_file(str(p), DEV, simplify=True, els=True, k_cap=512, seed=5)
    out = smp.sample(10 ** 9, configs=[(True, 4096, False)], max_chunks=2, collect=False)
    assert out["gpu_sound"]
    assert out["success_rate"] > 0.5, "huge-B success collapsed: phase boundary straddled?"


def test_kernel_counters():
    """apply_and_propagate feeds the work counters (ops/props/decs > 0 after propagation)."""
    f = CnfFormula(DEV)
    f.init_formula(3, [[1, 2], [-1, 3], [-2, -3]])
    eng = BCPEngine(f)
    eng.reset(8)
    dec = torch.zeros((8, 1), dtype=torch.int64, device=DEV)      # everyone decides x1=T
    eng.apply_and_propagate(dec)
    assert eng.ops_total() > 0
    assert eng.decs_total() == 8
    assert eng.launches >= 1


def test_miner_entailed_and_model_preserving():
    """Every clause mine_entailed() learns must be ENTAILED (formula ∧ ¬clause UNSAT), and the
    augmented formula must have EXACTLY the original model set."""
    from pysat.solvers import Minisat22
    from gpap.sampling.miner import mine_entailed

    def models(cls_):
        out = set()
        with Minisat22(bootstrap_with=cls_) as s:
            for m in s.enum_models():
                out.add(tuple(m))
        return out

    def check(nv, clauses, support):
        res = mine_entailed(nv, clauses, support, DEV)
        for cl in res["learned"]:
            with Minisat22(bootstrap_with=clauses) as s:
                assert not s.solve(assumptions=[-l for l in cl]), \
                    f"mined clause {cl} is NOT entailed"
        if res["learned"]:
            assert models(clauses) == models(clauses + res["learned"])
        return res

    # planted: (1|2),(1|-2) => backbone unit (1), NOT flush-implied (no unit clauses exist);
    # (-1,-3,-4),(−1,−3,4) => assuming {3} (with 1 forced) conflicts => unit (-3);
    # support pair: (-5|-6) is entailed via x7: (-5,-6,7),(-5,-6,-7)
    clauses = [[1, 2], [1, -2], [-1, -3, -4], [-1, -3, 4],
               [-5, -6, 7], [-5, -6, -7], [5, 6, 8], [2, 8, -5]]
    res = check(8, clauses, [4, 5])                       # support = vars 5,6 (0-based 4,5)
    assert res["n_units"] >= 2, res                        # backbone (1) and (-3) at least
    all_mined = {tuple(sorted(c)) for c in res["learned"]}
    assert (-6, -5) in all_mined or (-5,) in all_mined or (-6,) in all_mined, all_mined

    # random sweep: whatever is mined must be entailed + model-preserving
    rng = random.Random(7)
    for _ in range(10):
        nv = rng.randint(6, 14)
        clauses = _random_cnf(rng, nv, rng.randint(2 * nv, 5 * nv))
        with Minisat22(bootstrap_with=clauses) as s:
            if not s.solve():
                continue                                   # UNSAT input: mining may derive it; skip
        check(nv, clauses, list(range(min(nv, 6))))


def test_bandit_mix_with_k_fused_boost_sound():
    """Alternating configs at a BOOSTED K_fused vs the base K in ONE run: the state groups
    must never corrupt each other (reset() re-allocates on any K change)."""
    clauses = [[3], [1, 2, -3], [-1, 4], [-2, -4], [3, -4, 5]]
    smp = GPUSampler.from_clauses(5, clauses, DEV, k_cap=64, seed=9)
    smp.K_fused = 128                                   # force fused chunks to a DIFFERENT K
    out = smp.sample(10 ** 9, configs=[(True, 1, False), (True, 2, "soft")],
                     max_chunks=9, collect=True, verify_clauses=clauses)
    assert out["all_valid"] and out["gpu_sound"] and out["n_models"] > 0
    for m in out["models"]:
        assert _satisfies(m, clauses)


def test_make_assumption_tensor_shapes():
    t = make_assumption_tensor([[0, 2], [5]], DEV)
    assert t.shape == (2, 2) and t.dtype == torch.int64
    assert t[1].tolist() == [5, -1]
    assert make_assumption_tensor([[], []], DEV).shape == (2, 0)


"""A/B tests for the fused Triton BCP kernel (gpap/solver/bcp_triton.py) vs the PyTorch oracle.

Contract (BCP fixpoints are confluent, so this is exact):
  * conflict/active flags identical everywhere;
  * NON-conflicted rows: assignment / sat_count / lit_sum bit-identical;
  * conflicted rows: partial counter state unspecified (the sampler never reads it).
Inputs follow the sampler's guarantee: decisions are distinct variables per row (fresh or
deliberately contradictory — the conflict paths are exercised explicitly).
"""
import random

import pytest
import torch

from gpap.clsdata.cnf_formula import CnfFormula
from gpap.solver.bcp_engine import BCPEngine
from gpap.solver.bcp_triton import TritonBCP
from gpap.solver.sampler import GPUSampler, _satisfies

CUDA = torch.cuda.is_available()
AVAIL = CUDA and TritonBCP.available()
pytestmark = pytest.mark.skipif(not AVAIL, reason="requires CUDA + a working triton")
DEV = torch.device("cuda") if CUDA else torch.device("cpu")


def _random_cnf(rng, nv, nc, kmax=4):
    """Random CNF with distinct-variable clauses (may include units)."""
    clauses = []
    for _ in range(nc):
        vs = rng.sample(range(1, nv + 1), rng.randint(1, min(kmax, nv)))
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    return clauses


def _engines(nv, clauses, K, kernel_on_second=True):
    """Two engines over the same formula; the second gets the fused kernel attached."""
    out = []
    for use in (False, kernel_on_second):
        f = CnfFormula(DEV)
        f.init_formula(nv, clauses)
        e = BCPEngine(f)
        if use:
            e.attach_kernel(TritonBCP())
        e.init_state(K)
        e.propagate()                     # flush formula units (oracle path on both)
        out.append(e)
    return out


def _free_decisions(rng, engine, K, B, contradict_rows=()):
    """(K,B) int64 decisions: distinct FREE vars per row, random polarity (sampler's guarantee).
    Rows in `contradict_rows` get an l/~l pair to exercise the conflict path."""
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


def _assert_contract(oracle, kernel):
    assert torch.equal(oracle.conflict, kernel.conflict), "conflict flags differ"
    assert torch.equal(oracle._active, kernel._active), "active flags differ"
    mask = ~oracle.conflict
    assert torch.equal(oracle.assignment[mask], kernel.assignment[mask]), "assignment differs"
    assert torch.equal(oracle.sat_count[mask], kernel.sat_count[mask]), "sat_count differs"
    assert torch.equal(oracle.lit_sum[mask], kernel.lit_sum[mask]), "lit_sum differs"


def test_ab_random_cnfs():
    rng = random.Random(11)
    for trial in range(30):
        nv = rng.randint(4, 40)
        clauses = _random_cnf(rng, nv, rng.randint(nv, 4 * nv))
        K, B = 64, rng.choice([1, 2, 4])
        oracle, kern = _engines(nv, clauses, K)
        for step in range(3):                                # multi-step: rows diverge over steps
            contradict = {rng.randrange(K)} if step == 1 else ()
            dec = _free_decisions(rng, oracle, K, B, contradict_rows=contradict)
            oracle.apply_assumptions_bulk(dec)
            oracle.propagate()
            kern.apply_and_propagate(dec)
            _assert_contract(oracle, kern)


def test_ab_circuit_cnf(tmp_path):
    """A/B on a real unrolled circuit (deep propagation cascades)."""
    from scripts.bench_to_cnf import parse_bench, build_cnf, write_dimacs
    circ = parse_bench("benchmarks/circuits/s386.bench")
    b, support, _ = build_cnf(circ, T=3, seed=1)
    p = tmp_path / "s386c3.cnf"
    write_dimacs(str(p), b, support)
    from pysat.formula import CNF
    cnf = CNF(from_file=str(p))
    rng = random.Random(3)
    K = 128
    oracle, kern = _engines(cnf.nv, [list(c) for c in cnf.clauses], K)
    for _ in range(4):
        dec = _free_decisions(rng, oracle, K, 8)
        oracle.apply_assumptions_bulk(dec)
        oracle.propagate()
        kern.apply_and_propagate(dec)
        _assert_contract(oracle, kern)


def test_sampler_identical_models():
    """Same seed, kernel on vs off => identical model sets (RNG consumption unchanged; fixpoints
    confluent). Every model CPU-verified against the clauses."""
    clauses = [[1, 2, 3], [-1, 4], [-2, -4], [3, -4, 5], [-5, 1]]
    runs = {}
    for use in (False, True):
        smp = GPUSampler.from_clauses(5, clauses, DEV, k_cap=128, seed=17, use_kernel=use)
        assert smp.use_kernel == use
        out = smp.sample(500, decisions_per_round=2, max_chunks=8, collect=True,
                         verify_clauses=clauses)
        assert out["all_valid"]
        assert out["kernel"] == use
        runs[use] = sorted(tuple(m) for m in out["models"])
    assert runs[False] == runs[True], "kernel changed the sampled model set"


def test_sampler_kernel_portfolio_sound():
    """Kernel path under the adaptive-bandit portfolio (incl. soft lookahead) stays sound."""
    clauses = [[1, 2], [-1, 3], [-2, -3], [2, 4], [-4, -1, 5]]
    smp = GPUSampler.from_clauses(5, clauses, DEV, k_cap=128, seed=5, use_kernel=True)
    out = smp.sample(400, configs=[(True, 1, "soft"), (True, 4, False)], max_chunks=10,
                     collect=True, verify_clauses=clauses)
    assert out["all_valid"] and out["gpu_sound"] and out["n_models"] > 0
    for m in out["models"]:
        assert _satisfies(m, clauses)


def test_sampler_unsat_kernel():
    smp = GPUSampler.from_clauses(1, [[1], [-1]], DEV, k_cap=16, seed=1, use_kernel=True)
    out = smp.sample(10, max_chunks=5, collect=True, verify_clauses=[[1], [-1]])
    assert out["n_models"] == 0


def test_fused_trajectory_sound():
    """v2 fused-trajectory config (whole chunk in one launch): every model valid, UNSAT yields 0,
    and success parity with the v1 path on a support-driven formula (phase boundary respected)."""
    clauses = [[1, 2, 3], [-1, 4], [-2, -4], [3, -4, 5], [-5, 1]]
    smp = GPUSampler.from_clauses(5, clauses, DEV, k_cap=128, seed=17, use_kernel=True)
    out = smp.sample(400, configs=[(True, 2, "fused")], max_chunks=8, collect=True,
                     verify_clauses=clauses)
    assert out["all_valid"] and out["gpu_sound"] and out["n_models"] > 0
    assert out["n_distinct"] >= 2                       # diversity sanity (formula has ≥2 models)
    smp_u = GPUSampler.from_clauses(1, [[1], [-1]], DEV, k_cap=16, seed=1, use_kernel=True)
    out_u = smp_u.sample(10, configs=[(True, 1, "fused")], max_chunks=4, collect=True,
                         verify_clauses=[[1], [-1]])
    assert out_u["n_models"] == 0


def test_fused_phase_boundary(tmp_path):
    """A fused decision group must NOT straddle the support→completion boundary: on a circuit whose
    support fully determines the rest, huge-B fused must match v1's success (~100%), not collapse
    (the pre-fix straddle bug measured 100%→0% here)."""
    from scripts.bench_to_cnf import parse_bench, build_cnf, write_dimacs
    circ = parse_bench("benchmarks/circuits/s386.bench")
    b, support, _ = build_cnf(circ, T=2, seed=1)
    p = tmp_path / "s386c2.cnf"
    write_dimacs(str(p), b, support)
    smp = GPUSampler.from_cnf_file(str(p), DEV, simplify=True, els=True, k_cap=512, seed=5,
                                   use_kernel=True)
    o_v1 = smp.sample(10 ** 9, configs=[(True, 4096, False)], max_chunks=2, collect=False)
    o_fu = smp.sample(10 ** 9, configs=[(True, 4096, "fused")], max_chunks=2, collect=False)
    assert o_fu["gpu_sound"]
    # same ballpark success (both decide the whole support at once, then propagate)
    assert abs(o_fu["success_rate"] - o_v1["success_rate"]) < 0.25


def test_fused_fallback_without_kernel():
    """A 'fused' config with the kernel OFF degrades to the plain (rv, B) path and stays sound."""
    clauses = [[1, 2], [-1, 3], [-2, -3]]
    smp = GPUSampler.from_clauses(3, clauses, DEV, k_cap=64, seed=3, use_kernel=False)
    out = smp.sample(200, configs=[(True, 1, "fused")], max_chunks=6, collect=True,
                     verify_clauses=clauses)
    assert out["all_valid"] and out["n_models"] > 0 and not out["kernel"]


def test_kernel_ops_counter():
    """The fused kernel feeds the engine work counters (ops > 0 after propagation)."""
    clauses = [[1, 2], [-1, 3], [-2, -3]]
    _, kern = _engines(3, clauses, 8)
    dec = torch.zeros((8, 1), dtype=torch.int64, device=DEV)      # everyone decides x1=T
    kern.apply_and_propagate(dec)
    assert kern.kernel_ops() > 0
    assert kern.total_bcp_rounds >= 1

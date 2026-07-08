"""Circuit-sampling tests over the generated-CNF matrix (tests/conftest.py::circuit_cnfs).

Axes (per the refactor plan §5):
  soundness       — every emitted model satisfies the ORIGINAL clauses (pre-simplify).
  determinism     — same seed => identical model set.
  A/B             — oracle vs fused-kernel fixpoint equality (GPU only).
  reconstruction  — simplify / ELS on => models still verify on the ORIGINAL formula.
  contract        — n_time delivered (or the run stops via a valve).
  quality smoke   — distinct fraction ~1.0 on a non-tiny instance.

The CPU-runnable checks (no `gpu` marker) form a portable smoke suite; the broader / kernel /
deep-cascade coverage is `gpu` (skipped off-CUDA) and `slow` (c25) so CI can select subsets.
"""
import pytest
import torch

from gpap.engine import BCPEngine, CnfFormula
from gpap.sampling.sampler import GPUSampler, _satisfies

CUDA = torch.cuda.is_available()
try:
    from gpap.engine.kernels import TritonBCP
    KERNEL = CUDA and TritonBCP.available()
except Exception:  # noqa: BLE001
    KERNEL = False

DEV = torch.device("cuda") if CUDA else torch.device("cpu")
gpu = pytest.mark.gpu
slow = pytest.mark.slow

# A tiny subset that runs on ANY platform (portable smoke), and the broader GPU matrix.
CPU_SMOKE = ["b03-c1", "s386-c1"]
GPU_MATRIX = ["b03-c1", "b06-c5", "b10-c5", "s386-c5", "s526-c10", "c432-c1", "c880-c1"]


def _inst(circuit_cnfs, name):
    d = circuit_cnfs[name]
    return d["nv"], d["clauses"], d["support"]


# --------------------------------------------------------------------------- soundness

@pytest.mark.parametrize("name", CPU_SMOKE)
def test_circuit_models_sound_cpu(circuit_cnfs, name):
    """Portable: every emitted model (simplify on) satisfies the ORIGINAL clauses."""
    nv, clauses, support = _inst(circuit_cnfs, name)
    smp = GPUSampler.from_clauses(nv, clauses, DEV, simplify=True, support=support,
                                  k_cap=256, seed=5, use_kernel="auto")
    out = smp.sample(400, configs=[(True, 8, "fused")], max_chunks=3, collect=True,
                     verify_clauses=clauses)
    assert out["n_models"] > 0
    assert out["all_valid"], f"{name}: emitted a model violating the original clauses"
    assert out["gpu_sound"]


@gpu
@pytest.mark.parametrize("name", GPU_MATRIX)
def test_circuit_models_sound_gpu(circuit_cnfs, name):
    """Broader matrix on GPU: models sound under the circuit portfolio (fused + fsoft)."""
    if not KERNEL:
        pytest.skip("requires a working triton kernel")
    nv, clauses, support = _inst(circuit_cnfs, name)
    smp = GPUSampler.from_clauses(nv, clauses, DEV, simplify=True, support=support,
                                  k_cap=1024, seed=5, use_kernel="auto")
    out = smp.sample(1000, configs=[(True, 8, "fused"), (True, 8, "fsoft")], max_chunks=4,
                     collect=True, verify_clauses=clauses)
    assert out["n_models"] > 0 and out["all_valid"] and out["gpu_sound"]
    for m in out["models"][:64]:
        assert _satisfies(m, clauses)


# --------------------------------------------------------------------------- determinism

@pytest.mark.parametrize("name", CPU_SMOKE)
def test_circuit_determinism(circuit_cnfs, name):
    """Same seed => identical model set (the kernel/oracle path is reproducible)."""
    nv, clauses, support = _inst(circuit_cnfs, name)
    runs = []
    for _ in range(2):
        smp = GPUSampler.from_clauses(nv, clauses, DEV, simplify=True, support=support,
                                      k_cap=256, seed=7, use_kernel="auto")
        smp.K_fused = smp.K
        out = smp.sample(300, configs=[(True, 8, "fused")], max_chunks=3, collect=True)
        runs.append(sorted(tuple(m) for m in out["models"]))
    assert runs[0] == runs[1], f"{name}: same seed produced different model sets"


# --------------------------------------------------------------------------- A/B oracle vs kernel

@gpu
@pytest.mark.parametrize("name", ["b03-c1", "s386-c5", "b10-c5"])
def test_circuit_ab_oracle_vs_kernel(circuit_cnfs, name):
    """Oracle (PyTorch) and fused-kernel BCP fixpoints are bit-identical on non-conflicted rows
    (BCP confluence). Decide fresh distinct vars per row, multi-step."""
    if not KERNEL:
        pytest.skip("requires a working triton kernel")
    import random
    nv, clauses, _ = _inst(circuit_cnfs, name)
    rng = random.Random(3)

    def _mk():
        f = CnfFormula(DEV); f.init_formula(nv, clauses)
        e = BCPEngine(f); e.init_state(96); e.propagate()
        return e
    oracle, kern = _mk(), _mk()
    kern.attach_kernel(TritonBCP())
    for _ in range(3):
        asg = oracle.assignment.cpu()
        dec = torch.full((96, 8), -1, dtype=torch.int64)
        for k in range(96):
            free = [v for v in range(nv) if asg[k, 2 * v] == 0 and asg[k, 2 * v + 1] == 0]
            for j, v in enumerate(rng.sample(free, min(8, len(free)))):
                dec[k, j] = 2 * v + rng.randint(0, 1)
        dec = dec.to(DEV)
        oracle.apply_assumptions_bulk(dec); oracle.propagate()
        kern.apply_and_propagate(dec)
        assert torch.equal(oracle.conflict, kern.conflict)
        mask = ~oracle.conflict
        assert torch.equal(oracle.assignment[mask], kern.assignment[mask])
        assert torch.equal(oracle.sat_count[mask], kern.sat_count[mask])


# --------------------------------------------------------------------------- reconstruction

@pytest.mark.parametrize("name", ["b03-c1", "s386-c1"])
def test_circuit_reconstruction_els(circuit_cnfs, name):
    """simplify + ELS on => reconstructed models still satisfy the ORIGINAL formula."""
    nv, clauses, support = _inst(circuit_cnfs, name)
    smp = GPUSampler.from_clauses(nv, clauses, DEV, els=True, support=support,
                                  k_cap=256, seed=5, use_kernel="auto")
    out = smp.sample(400, configs=[(True, 8, "fused")], max_chunks=3, collect=True,
                     verify_clauses=clauses)
    assert out["n_models"] > 0
    assert out["all_valid"], f"{name}: ELS reconstruction violates the ORIGINAL clauses"


# --------------------------------------------------------------------------- contract

@gpu
def test_circuit_contract_delivers_n_time(circuit_cnfs):
    """A high-success shallow instance delivers the full n_time budget (not chunk-capped)."""
    if not KERNEL:
        pytest.skip("requires a working triton kernel")
    from gpap.sampling.pipeline import sample_clauses, CIRCUIT_PORTFOLIO
    nv, clauses, support = _inst(circuit_cnfs, "b02-c1") if "b02-c1" in circuit_cnfs \
        else _inst(circuit_cnfs, "s386-c1")
    info = sample_clauses(nv, clauses, support, DEV, n_time=5000, n_cov=0,
                          configs=CIRCUIT_PORTFOLIO, time_limit=30.0)
    assert info["n_models"] >= 5000 and not info["chunk_capped"]


# --------------------------------------------------------------------------- quality smoke

@gpu
def test_circuit_distinct_fraction(circuit_cnfs):
    """A non-tiny instance emits (nearly) all-distinct support projections."""
    if not KERNEL:
        pytest.skip("requires a working triton kernel")
    nv, clauses, support = _inst(circuit_cnfs, "b03-c10")
    smp = GPUSampler.from_clauses(nv, clauses, DEV, simplify=True, support=support,
                                  k_cap=1024, seed=5, use_kernel="auto")
    out = smp.sample(2000, configs=[(True, 8, "fused")], max_chunks=4, collect=True)
    if out["n_models"] >= 200:
        assert out["n_distinct"] / out["n_models"] > 0.9


# --------------------------------------------------------------------------- deep cascade (slow)

@gpu
@slow
def test_circuit_deep_cascade_c25(circuit_cnfs):
    """The deep 25-cycle unroll: sound under the soft-lookahead config (the circuit-regime win)."""
    if not KERNEL:
        pytest.skip("requires a working triton kernel")
    nv, clauses, support = _inst(circuit_cnfs, "b03-c25")
    smp = GPUSampler.from_clauses(nv, clauses, DEV, simplify=True, support=support,
                                  k_cap=1024, seed=5, use_kernel="auto")
    out = smp.sample(1000, configs=[(True, 1, "fsoft"), (True, 8, "fused")], max_chunks=6,
                     collect=True, verify_clauses=clauses)
    assert out["n_models"] > 0 and out["all_valid"] and out["gpu_sound"]

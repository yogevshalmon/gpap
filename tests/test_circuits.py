"""Circuit-sampling tests over the generated-CNF matrix (tests/conftest.py::circuit_cnfs).

Axes:
  soundness       — every emitted model satisfies the ORIGINAL clauses (pre-simplify).
  determinism     — same seed => identical model set.
  reference A/B   — kernel fixpoints == the pure-Python reference BCP (confluence).
  reconstruction  — simplify / ELS on => models still verify on the ORIGINAL formula.
  contract        — n_time delivered (or the run stops via a valve).
  quality smoke   — distinct fraction ~1.0 on a non-tiny instance.

ALL of these need CUDA + triton (the engine is packed-kernel-only); `slow` marks the deep
c25 unroll. The off-GPU portable smoke is the host-side logic suite (formula/preprocess/
circuits/simulate/metrics tests).
"""
import pytest
import torch

from gpap.engine import BCPEngine, CnfFormula, core_available
from gpap.sampling.sampler import GPUSampler, _satisfies

from tests.reference_bcp import ref_bcp, assigned_set

CUDA = torch.cuda.is_available()
KERNEL = CUDA and core_available()
pytestmark = pytest.mark.skipif(not KERNEL, reason="requires CUDA + a built libgpap")

DEV = torch.device("cuda") if CUDA else torch.device("cpu")
gpu = pytest.mark.gpu
slow = pytest.mark.slow

SMOKE = ["b03-c1", "s386-c1"]
GPU_MATRIX = ["b03-c1", "b06-c5", "b10-c5", "s386-c5", "s526-c10", "c432-c1", "c880-c1"]


def _inst(circuit_cnfs, name):
    d = circuit_cnfs[name]
    return d["nv"], d["clauses"], d["support"]


def _collecting_sampler(nv, clauses, support, *, k_cap, seed=5, **kw):
    """A sampler whose COLLECTED chunk is actually bounded by `k_cap`.

    THE TRAP: `k_cap` bounds K, but the batch a fused chunk runs is `batch_size`, which sizes
    itself from free VRAM — on these tiny circuit formulas it lands orders of magnitude higher
    (b10-c5 measured K=1024 but a 242,423-row batch). Under `collect=True` every completed row is
    reconstructed into python lists of ints (~36 B each), so a test asking for 1000 models peaked
    at **28 GB** of host RAM and 341 s, enough to push a 48 GB box into swap.

    `pin_batch` is the fix and exists precisely for this: it states a bound that holds, instead of
    hoping the boost happens to be off.
    """
    smp = GPUSampler.from_clauses(nv, clauses, DEV, support=support, k_cap=k_cap, seed=seed, **kw)
    smp.pin_batch(smp.K)
    return smp


# --------------------------------------------------------------------------- soundness

@pytest.mark.parametrize("name", SMOKE)
def test_circuit_models_sound(circuit_cnfs, name):
    """Every emitted model (simplify on) satisfies the ORIGINAL clauses."""
    nv, clauses, support = _inst(circuit_cnfs, name)
    smp = _collecting_sampler(nv, clauses, support, k_cap=256, simplify=True)
    out = smp.sample(400, configs=[(True, 8, False)], max_chunks=3, collect=True,
                     verify_sound=True,
                     verify_clauses=clauses)
    assert out["n_models"] > 0
    assert out["all_valid"], f"{name}: emitted a model violating the original clauses"
    assert out["sound_check"] == "ok"


@gpu
@pytest.mark.parametrize("name", GPU_MATRIX)
def test_circuit_models_sound_matrix(circuit_cnfs, name):
    """Broader matrix: models sound under the circuit portfolio (plain + soft)."""
    nv, clauses, support = _inst(circuit_cnfs, name)
    smp = _collecting_sampler(nv, clauses, support, k_cap=1024, simplify=True)
    out = smp.sample(1000, configs=[(True, 8, False), (True, 8, "soft")], max_chunks=4,
                     verify_sound=True,
                     collect=True, verify_clauses=clauses)
    assert out["n_models"] > 0 and out["all_valid"] and out["sound_check"] == "ok"
    for m in out["models"][:64]:
        assert _satisfies(m, clauses)


# --------------------------------------------------------------------------- determinism

@pytest.mark.parametrize("name", SMOKE)
def test_circuit_determinism(circuit_cnfs, name):
    """Same seed => identical model set (the kernel path is reproducible)."""
    nv, clauses, support = _inst(circuit_cnfs, name)
    runs = []
    for _ in range(2):
        smp = GPUSampler.from_clauses(nv, clauses, DEV, simplify=True, support=support,
                                      k_cap=256, seed=7)
        smp.pin_batch(smp.K)
        out = smp.sample(300, configs=[(True, 8, False)], max_chunks=3, collect=True)
        runs.append(sorted(tuple(m) for m in out["models"]))
    assert runs[0] == runs[1], f"{name}: same seed produced different model sets"


# --------------------------------------------------------------------------- reference A/B

@gpu
@pytest.mark.parametrize("name", ["b03-c1", "s386-c5", "b10-c5"])
def test_circuit_kernel_matches_reference(circuit_cnfs, name):
    """Kernel BCP fixpoints == the pure-Python reference on real circuit CNFs (confluence).
    Decide fresh distinct vars per row, check conflict flags + non-conflicted assignments."""
    import random
    nv, clauses, _ = _inst(circuit_cnfs, name)
    rng = random.Random(3)
    K = 24
    f = CnfFormula(DEV)
    f.init_formula(nv, clauses)
    eng = BCPEngine(f)
    eng.reset(K)
    asg = eng.assignment.cpu()
    dec = torch.full((K, 8), -1, dtype=torch.int64)
    for k in range(K):
        free = [v for v in range(nv) if asg[k, 2 * v] == 0 and asg[k, 2 * v + 1] == 0]
        for j, v in enumerate(rng.sample(free, min(8, len(free)))):
            dec[k, j] = 2 * v + rng.randint(0, 1)
    eng.apply_and_propagate(dec.to(DEV))
    for k in range(K):
        conf, ref_asg = ref_bcp(clauses, [int(x) for x in dec[k].tolist() if x >= 0])
        assert bool(eng.conflict[k].item()) == conf
        if not conf:
            assert assigned_set(eng.assignment[k].cpu()) == ref_asg


# --------------------------------------------------------------------------- reconstruction

@pytest.mark.parametrize("name", ["b03-c1", "s386-c1"])
def test_circuit_reconstruction_els(circuit_cnfs, name):
    """simplify + ELS on => reconstructed models still satisfy the ORIGINAL formula."""
    nv, clauses, support = _inst(circuit_cnfs, name)
    smp = _collecting_sampler(nv, clauses, support, k_cap=256, els=True)
    out = smp.sample(400, configs=[(True, 8, False)], max_chunks=3, collect=True,
                     verify_clauses=clauses)
    assert out["n_models"] > 0
    assert out["all_valid"], f"{name}: ELS reconstruction violates the ORIGINAL clauses"


# --------------------------------------------------------------------------- contract

@gpu
def test_circuit_contract_delivers_n_time(circuit_cnfs):
    """A high-success shallow instance delivers the full n_time budget (not chunk-capped)."""
    from gpap.sampling.pipeline import sample_clauses, CIRCUIT_PORTFOLIO
    nv, clauses, support = _inst(circuit_cnfs, "b02-c1") if "b02-c1" in circuit_cnfs \
        else _inst(circuit_cnfs, "s386-c1")
    info = sample_clauses(nv, clauses, support, DEV, n_time=5000, n_cov=0, verify_sound=True,
                          configs=CIRCUIT_PORTFOLIO, time_limit=30.0)
    assert info["n_models"] >= 5000 and not info["chunk_capped"]


# --------------------------------------------------------------------------- quality smoke

@gpu
def test_circuit_distinct_fraction(circuit_cnfs):
    """A non-tiny instance emits (nearly) all-distinct support projections."""
    nv, clauses, support = _inst(circuit_cnfs, "b03-c10")
    smp = _collecting_sampler(nv, clauses, support, k_cap=1024, simplify=True)
    out = smp.sample(2000, configs=[(True, 8, False)], max_chunks=4, collect=True)
    # assert the PRECONDITION rather than hiding behind it: `if n_models >= 200` alone means a
    # regression that stops the sampler producing anything passes this test silently.
    assert out["n_models"] >= 200, "produced too few models to judge distinctness"
    assert out["n_distinct"] / out["n_models"] > 0.9


# --------------------------------------------------------------------------- deep cascade (slow)

@gpu
@slow
def test_circuit_deep_cascade_c25(circuit_cnfs):
    """The deep 25-cycle unroll: sound under the soft-lookahead config (the circuit-regime win)."""
    nv, clauses, support = _inst(circuit_cnfs, "b03-c25")
    smp = _collecting_sampler(nv, clauses, support, k_cap=1024, simplify=True)
    out = smp.sample(1000, configs=[(True, 1, "soft"), (True, 8, False)], max_chunks=6,
                     verify_sound=True,
                     collect=True, verify_clauses=clauses)
    assert out["n_models"] > 0 and out["all_valid"] and out["sound_check"] == "ok"

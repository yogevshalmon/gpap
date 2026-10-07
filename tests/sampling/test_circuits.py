"""Circuit sampling over the generated CNFs (tests/conftest.py::circuit_cnfs).

Soundness against the original clauses, same-seed determinism, kernel fixpoints against the
reference BCP, ELS reconstruction, n_time delivery, and the distinct fraction. `slow` marks the
deep c25 unroll.
"""
import pytest
import torch

from gpap.engine import BCPEngine, CnfFormula
from gpap.sampling.sampler import GPUSampler, _satisfies

from tests.reference.bcp import ref_bcp, assigned_set
from tests.support import DEV

pytestmark = pytest.mark.gpu
slow = pytest.mark.slow

SMOKE = ["b03-c1", "s386-c1"]
GPU_MATRIX = ["b03-c1", "b06-c5", "b10-c5", "s386-c5", "s526-c10", "c432-c1", "c880-c1"]


def _inst(circuit_cnfs, name):
    d = circuit_cnfs[name]
    return d["nv"], d["clauses"], d["support"]


def _collecting_sampler(nv, clauses, support, *, k_cap, seed=5, **kw):
    """A sampler whose collected chunk is bounded by `k_cap`.

    `k_cap` bounds K, but a chunk runs `batch_size` rows, which is sized from free VRAM and is far
    larger on these small formulas. With `collect=True` every row is rebuilt as Python ints, so
    an unpinned batch can exhaust host RAM. `pin_batch` makes the bound hold.
    """
    smp = GPUSampler.from_clauses(nv, clauses, DEV, support=support, k_cap=k_cap, seed=seed, **kw)
    smp.pin_batch(smp.K)
    return smp


# --------------------------------------------------------------------------- soundness

@pytest.mark.parametrize("name", SMOKE)
def test_circuit_models_sound(circuit_cnfs, name):
    """Every emitted model (simplify on) satisfies the original clauses."""
    nv, clauses, support = _inst(circuit_cnfs, name)
    smp = _collecting_sampler(nv, clauses, support, k_cap=256, simplify=True)
    out = smp.sample(400, configs=[(True, 8, False)], max_chunks=3, collect=True,
                     verify_sound=True,
                     verify_clauses=clauses)
    assert out["n_models"] > 0
    assert out["all_valid"], f"{name}: emitted a model violating the original clauses"
    assert out["sound_check"] == "ok"


@pytest.mark.parametrize("name", GPU_MATRIX)
def test_circuit_models_sound_matrix(circuit_cnfs, name):
    """Models are sound under plain and soft configs on a broader matrix."""
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
    """The same seed gives the same model set."""
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

@pytest.mark.parametrize("name", ["b03-c1", "s386-c5", "b10-c5"])
def test_circuit_kernel_matches_reference(circuit_cnfs, name):
    """Kernel BCP fixpoints equal the pure-Python reference on circuit CNFs.

    Each row decides up to 8 distinct free vars; conflict flags and non-conflicted assignments
    must match."""
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
    """With ELS on, reconstructed models satisfy the original formula."""
    nv, clauses, support = _inst(circuit_cnfs, name)
    smp = _collecting_sampler(nv, clauses, support, k_cap=256, els=True)
    out = smp.sample(400, configs=[(True, 8, False)], max_chunks=3, collect=True,
                     verify_clauses=clauses)
    assert out["n_models"] > 0
    assert out["all_valid"], f"{name}: ELS reconstruction violates the ORIGINAL clauses"


# --------------------------------------------------------------------------- contract

def test_circuit_contract_delivers_n_time(circuit_cnfs):
    """A high-success shallow instance delivers the full n_time, without a valve firing."""
    from gpap.sampling.pipeline import sample_clauses, CIRCUIT_PORTFOLIO
    nv, clauses, support = _inst(circuit_cnfs, "s386-c1")
    info = sample_clauses(nv, clauses, support, DEV, n_time=5000, n_cov=0, verify_sound=True,
                          configs=CIRCUIT_PORTFOLIO, time_limit=30.0)
    assert info["n_models"] >= 5000 and not info["chunk_capped"]


# --------------------------------------------------------------------------- quality smoke

def test_circuit_distinct_fraction(circuit_cnfs):
    """A non-tiny instance emits (nearly) all-distinct support projections."""
    nv, clauses, support = _inst(circuit_cnfs, "b03-c10")
    smp = _collecting_sampler(nv, clauses, support, k_cap=1024, simplify=True)
    out = smp.sample(2000, configs=[(True, 8, False)], max_chunks=4, collect=True)
    # Assert the precondition, so a sampler that produces nothing fails instead of skipping.
    assert out["n_models"] >= 200, "produced too few models to judge distinctness"
    assert out["n_distinct"] / out["n_models"] > 0.9


# --------------------------------------------------------------------------- deep cascade (slow)

@slow
def test_circuit_deep_cascade_c25(circuit_cnfs):
    """The deep 25-cycle unroll is sound under the soft-lookahead config."""
    nv, clauses, support = _inst(circuit_cnfs, "b03-c25")
    smp = _collecting_sampler(nv, clauses, support, k_cap=1024, simplify=True)
    out = smp.sample(1000, configs=[(True, 1, "soft"), (True, 8, False)], max_chunks=6,
                     verify_sound=True,
                     collect=True, verify_clauses=clauses)
    assert out["n_models"] > 0 and out["all_valid"] and out["sound_check"] == "ok"

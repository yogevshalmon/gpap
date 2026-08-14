"""The VRAM-proportional K cap.

Throughput is ~linear in K, so the cap decides what a bigger GPU is allowed to do. It used to be
a hardcoded 49,152 in every driver, which pinned a 48 GB L40S or an 80 GB A100 to the 12 GB
laptop's batch size — a cross-GPU sweep then reproduced the laptop's numbers and could not answer
"does this scale with the GPU?" at all.

Two properties are gated here, and the FIRST one is the important one:

  * at or below the reference card's VRAM the cap is EXACTLY the historical 49,152, so no result
    measured on the reference box moves because of this change;
  * above it the cap grows with total VRAM, so the headroom is actually reachable.

These are pure arithmetic over a VRAM number — no GPU needed, which is the point: the property
must hold for hardware we do not have in front of us.

The cap keys off TOTAL VRAM, not free memory: a free-memory cap is not reproducible (it moved
between two runs in one session on an idle box), and a sweep whose K depends on what else was
resident is not a measurement. `test_auto_is_stable_across_calls` is that regression.
"""
import pytest
import torch

from gpap.engine.calibrate import (
    K_CAP_REF,
    K_CAP_VRAM_REF,
    default_k_cap,
    resolve_k_cap,
)

# The historical constant. Spelled out rather than imported so that editing K_CAP_REF cannot
# silently make this test agree with itself.
LEGACY_K_CAP = 49_152


def _cap_for_vram(total_bytes):
    """default_k_cap's arithmetic against an arbitrary card size (what that GPU would get)."""
    return max(K_CAP_REF, int(K_CAP_REF * total_bytes / K_CAP_VRAM_REF))


def test_reference_constant_is_the_historical_cap():
    assert K_CAP_REF == LEGACY_K_CAP


@pytest.mark.parametrize("frac", [0.1, 0.5, 0.9, 1.0])
def test_no_op_at_or_below_the_reference_card(frac):
    """A card no bigger than the reference box must get exactly the old cap — the guarantee that
    this change cannot move an existing number."""
    assert _cap_for_vram(K_CAP_VRAM_REF * frac) == LEGACY_K_CAP


@pytest.mark.parametrize("gpu,total,lo", [
    ("L4 24GB",     24e9,  1.8),
    ("L40S 48GB",   48e9,  3.7),
    ("A100 80GB",   80e9,  6.2),
    ("H200 141GB", 141e9, 10.9),
])
def test_scales_above_the_reference_card(gpu, total, lo):
    """...and a bigger card must actually get more. The targets we might actually run on, with a
    FLOOR rather than an equality so int() truncation is not asserted as behaviour."""
    cap = _cap_for_vram(total)
    assert cap >= LEGACY_K_CAP * lo, gpu
    assert cap <= LEGACY_K_CAP * (total / K_CAP_VRAM_REF) + 1, gpu


def test_monotone_in_vram():
    caps = [_cap_for_vram(K_CAP_VRAM_REF * m) for m in (0.5, 1.0, 2.0, 4.0, 10.0)]
    assert caps == sorted(caps)


def test_cpu_device_gets_the_reference_cap():
    """No CUDA budget to exploit — and it must not crash asking for one."""
    assert default_k_cap(torch.device("cpu")) == K_CAP_REF


def test_resolve_passes_explicit_values_through_untouched():
    """`auto` is an app-level default. An int must survive verbatim (a fixed cap is what a
    cross-GPU A/B needs), and None must stay None (GPUSampler's "no cap" contract)."""
    dev = torch.device("cpu")
    assert resolve_k_cap(1234, dev) == 1234
    assert resolve_k_cap(None, dev) is None
    assert resolve_k_cap("auto", dev) == default_k_cap(dev)


def test_gpusampler_defaults_to_auto():
    """`k_cap` defaults to "auto" so a direct GPUSampler user gets the same batch the pipeline
    would give them. It used to default to a hardcoded 16,384 — ~1/3 of the batch on this card,
    and wrong on EVERY card, since it does not scale with VRAM at all."""
    import inspect

    from gpap.sampling.sampler import GPUSampler
    assert inspect.signature(GPUSampler.__init__).parameters["k_cap"].default == "auto"


@pytest.mark.gpu
def test_core_and_python_caps_agree():
    """ONE rule, two implementations (calibrate.py and cpp/src/calibrate.cpp). They are asserted
    equal here rather than assumed equal — a C++ consumer and a Python consumer sizing the same
    formula differently on the same card is exactly the bug this cap exists to prevent."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from gpap.engine import _core
    if not _core.available():
        pytest.skip("needs a built libgpap")
    assert _core.default_k_cap(0) == default_k_cap(torch.device("cuda"))


@pytest.mark.gpu
def test_auto_is_at_least_the_reference_cap_on_this_gpu():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    assert default_k_cap(torch.device("cuda")) >= K_CAP_REF


@pytest.mark.gpu
def test_auto_is_stable_across_calls():
    """REGRESSION: the first cut keyed off free VRAM and drifted between calls, so two runs of
    the same sweep on the same box could use different K. Allocating in between must not move
    it."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    dev = torch.device("cuda")
    before = default_k_cap(dev)
    hog = torch.empty(int(3e8), dtype=torch.uint8, device=dev)   # ~300 MB of "something else"
    try:
        assert default_k_cap(dev) == before
    finally:
        del hog
        torch.cuda.empty_cache()
    assert default_k_cap(dev) == before


# ---------------------------------------------------------------- the batch-size API
# K and K_fused are two memory models; `batch_size` is the one number that says what runs, and
# `pin_batch` the one way to bound it. These gate that contract, because getting it wrong is not
# a wrong number — it is 28 GB of host RAM (see tests/test_circuits.py::_collecting_sampler).

def _tiny_sampler(k_cap):
    import torch as _t
    from gpap.sampling.sampler import GPUSampler
    clauses = [[1, 2], [-1, 3], [-2, -3], [2, 3, 4], [-4, 1]]
    return GPUSampler.from_clauses(5, clauses, _t.device("cuda"), k_cap=k_cap, seed=1)


@pytest.mark.gpu
def test_batch_size_follows_the_boost_state():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    smp = _tiny_sampler(1024)
    assert smp.batch_size == smp.K_fused, "at construction the fused boost applies"
    smp.K = smp.K // 2                       # below the construction value -> boost off
    assert not smp._boosted
    assert smp.batch_size == smp.K


@pytest.mark.gpu
def test_pin_batch_bounds_the_batch_even_when_the_boost_is_on():
    """THE regression. `K_fused = K` / `K = min(K, n)` only bound the batch when they happened to
    drop K below its construction value; at n >= K the boost stayed on and the chunk ran orders of
    magnitude wider than asked. pin_batch must hold in both states."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    smp = _tiny_sampler(1024)
    boosted = smp.batch_size
    assert boosted > smp.K, "precondition: this formula is small enough to be boosted"
    # the OLD idiom, at n == K: does not bound anything
    smp.K = min(smp.K, smp.K)
    assert smp.batch_size == boosted, "sanity: assigning K at its own value leaves the boost on"
    # the new one does
    assert smp.pin_batch(smp.K) == smp.K
    assert smp.batch_size == smp.K


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 7, 512, 100_000])
def test_pin_batch_is_exact_and_idempotent(n):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    smp = _tiny_sampler(1024)
    assert smp.pin_batch(n) == n and smp.batch_size == n
    assert smp.pin_batch(n) == n, "pinning twice must not drift"


@pytest.mark.gpu
def test_pinned_batch_restores_both_models():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    smp = _tiny_sampler(1024)
    before = (smp.K, smp.K_fused, smp.batch_size)
    with smp.pinned_batch(16):
        assert smp.batch_size == 16
    assert (smp.K, smp.K_fused, smp.batch_size) == before


@pytest.mark.gpu
def test_pipeline_reports_the_resolved_cap():
    """The provenance contract: a row must say which ceiling produced its K, or a small K on a
    cross-GPU row is ambiguous between "small GPU" and "cap bound it"."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from gpap.sampling.pipeline import sample_clauses
    dev = torch.device("cuda")
    clauses = [[1, 2], [-1, 3], [-2, -3], [2, 3, 4]]
    info = sample_clauses(4, clauses, None, dev, n_time=200, time_limit=5.0, mine=False)
    assert info["engine"]["k_cap"] == default_k_cap(dev)
    assert info["engine"]["K"] <= info["engine"]["k_cap"]

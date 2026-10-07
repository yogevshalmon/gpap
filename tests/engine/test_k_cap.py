"""The batch cap, the `GPAP_VRAM_BUDGET_MIB` ceiling, and the batch-size API.

The cap is one constant on every GPU, so `--k-cap auto` means the same batch on every machine.
It is a saturation limit, not a memory bound (`compute_k_safe` keeps a chunk inside VRAM), and
it must not depend on free memory, which is not reproducible.
"""
import os

import pytest
import torch

from gpap.engine.calibrate import K_CAP, default_k_cap, resolve_k_cap
from tests.conftest import REPO_ROOT

# Spelled out, not imported, so an edit to K_CAP fails here. Changing it changes every run's
# batch, so recorded sweeps stop being comparable.
EXPECTED_CAP = 262_144
REPO = REPO_ROOT


@pytest.fixture(autouse=True)
def _no_vram_cap(monkeypatch):
    """A `GPAP_VRAM_BUDGET_MIB` in the caller's shell would cap every budget in this file."""
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)


def test_the_cap_is_the_documented_constant():
    assert K_CAP == EXPECTED_CAP


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_cap_is_device_independent(device):
    """The same cap on every device; a CPU device answers without touching the driver."""
    assert default_k_cap(torch.device(device)) == EXPECTED_CAP


def test_resolve_passes_explicit_values_through_untouched():
    """An int passes through unchanged and None stays None (GPUSampler's "no cap"); only
    "auto" resolves to the default."""
    dev = torch.device("cpu")
    assert resolve_k_cap(1234, dev) == 1234
    assert resolve_k_cap(None, dev) is None
    assert resolve_k_cap("auto", dev) == default_k_cap(dev)


def test_gpusampler_defaults_to_auto():
    """A direct GPUSampler user gets the same batch as the pipeline."""
    import inspect

    from gpap.sampling.sampler import GPUSampler
    assert inspect.signature(GPUSampler.__init__).parameters["k_cap"].default == "auto"


@pytest.mark.gpu
def test_core_and_python_caps_agree():
    """calibrate.py and cpp/src/calibrate.cpp give the same cap, so C++ and Python callers size
    a formula the same way."""
    from gpap.engine import _core
    assert _core.default_k_cap(0) == default_k_cap(torch.device("cuda"))


@pytest.mark.gpu
def test_auto_is_stable_across_calls():
    """The cap does not depend on free VRAM: allocating between calls does not move it."""
    dev = torch.device("cuda")
    before = default_k_cap(dev)
    hog = torch.empty(int(3e8), dtype=torch.uint8, device=dev)   # ~300 MB of "something else"
    try:
        assert default_k_cap(dev) == before
    finally:
        del hog
        torch.cuda.empty_cache()
    assert default_k_cap(dev) == before


# ---------------------------------------------------------------- the VRAM budget ceiling
# `GPAP_VRAM_BUDGET_MIB` lets a larger card size the same K as a recorded run, since K is
# memory-bound on big formulas. It is a ceiling (a smaller card still reports less), applied once
# in gpap_mem_available so every consumer sees it.

@pytest.mark.gpu
def test_vram_budget_cap_is_a_ceiling_not_a_floor(monkeypatch):
    from gpap.engine import _core
    uncapped = _core.mem_available(0, 0.80)
    assert uncapped > 0
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", str(max(1, uncapped // (1 << 20) // 2)))
    assert _core.mem_available(0, 0.80) == pytest.approx(uncapped / 2, rel=0.01)
    # Above what the card has, the card binds: the cap never adds memory.
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", str(uncapped // (1 << 20) * 4))
    assert _core.mem_available(0, 0.80) == uncapped


@pytest.mark.gpu
@pytest.mark.parametrize("value", ["", "0", "-1", "bogus"])
def test_vram_budget_cap_ignores_meaningless_values(monkeypatch, value):
    """An empty, zero, negative or unparsable value means no cap, not a cap of zero (which
    would size K=1)."""
    from gpap.engine import _core
    uncapped = _core.mem_available(0, 0.80)
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", value)
    assert _core.mem_available(0, 0.80) == uncapped


def test_vram_budget_cap_reporter_matches_the_env(monkeypatch):
    """`vram_budget_cap` only reports the cap (the core clamps), but must parse it the same way
    so a row's provenance matches the budget that sized it."""
    from gpap.engine.calibrate import vram_budget_cap
    assert vram_budget_cap() == 0
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", "18120")
    assert vram_budget_cap() == 18120 << 20
    for junk in ("", "0", "-5", "bogus"):
        monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", junk)
        assert vram_budget_cap() == 0


#: Values that two plausible parsers read differently: `1e3` and `1024abc` under C's
#: numeric-prefix rule, `1_000` under Python's int(), and the huge ones overflow the MiB -> bytes
#: shift.
CAP_GRAMMAR_CASES = [("", 0), ("   ", 0), ("0", 0), ("-1", 0), ("bogus", 0),
                     ("1e3", 0), ("1024.5", 0), ("1024abc", 0), ("1_000", 0), ("0x400", 0),
                     ("2048", 2048), (" 2048 ", 2048), ("+2048", 2048),
                     ("17592186045440", 0), ("99999999999999999999", 0)]


@pytest.mark.parametrize(("value", "want_mib"), CAP_GRAMMAR_CASES)
def test_the_cap_grammar_is_one_grammar(monkeypatch, value, want_mib):
    """The Python reader of the cap. The engine clamps in C++ (`env_budget_cap`) while the rival
    harness and `gpap info` read the variable in Python; if they disagreed, only one side of a
    head-to-head would be capped. The core is held to the same table below."""
    from gpap.engine.calibrate import vram_budget_cap_mib
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", value)
    assert vram_budget_cap_mib() == want_mib


@pytest.mark.gpu
@pytest.mark.parametrize(("value", "want_mib"), CAP_GRAMMAR_CASES)
def test_the_core_reads_the_cap_grammar_the_same_way(monkeypatch, value, want_mib):
    """The core exposes the clamp only through its effect: a cap below free memory must bind
    exactly."""
    from gpap.engine import _core
    uncapped = _core.mem_available(0, 1.0)
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", value)
    got = _core.mem_available(0, 1.0)
    if want_mib and (want_mib << 20) < uncapped:
        assert got == want_mib << 20, f"the core parsed {value!r} differently"
    else:
        assert got == uncapped, f"the core capped on {value!r} where Python read no cap"


@pytest.mark.gpu
def test_the_vram_cap_actually_moves_the_sampler_batch(monkeypatch):
    """The budget cap reaches the sampler's K, not only `mem_available`.

    With no batch cap, the budget is set low enough that the memory bound decides K, and two
    budgets are compared."""
    from gpap.engine.calibrate import compute_k_safe, row_bytes
    smp = _tiny_sampler(None)                    # k_cap None: the memory bound decides
    wide, N, num_long = smp.K, smp.engine.num_vars, smp.engine.num_long
    # a budget admitting an eighth of what the uncapped card does
    mib = max(1, (wide // 8) * row_bytes(num_long, N) // (1 << 20))
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", str(mib))
    narrow = _tiny_sampler(None).K
    assert narrow < wide, f"the cap did not reach K ({narrow} vs {wide} at {mib} MiB)"
    assert narrow == compute_k_safe(num_long, N, mib << 20, None), \
        "K under a cap must be the memory model's own answer for exactly that budget"


def test_the_cpu_placeholder_budget_is_capped_too(monkeypatch):
    """The no-CUDA placeholder budget is capped too, so both sides agree on CPU-only hosts."""
    from gpap.engine.calibrate import gpu_mem_available
    uncapped = gpu_mem_available(torch.device("cpu"))
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", "1024")
    assert gpu_mem_available(torch.device("cpu")) == 1024 << 20 < uncapped


# ---------------------------------------------------------------- the batch-size API
# `batch_size` is the batch a chunk runs and `pin_batch` the way to bound it (an unbounded one
# can exhaust host RAM; see tests/sampling/test_circuits.py::_collecting_sampler). The per-row
# model is the engine's exact allocation list (cpp/src/calibrate.cpp).

def _tiny_sampler(k_cap):
    import torch as _t
    from gpap.sampling.sampler import GPUSampler
    clauses = [[1, 2], [-1, 3], [-2, -3], [2, 3, 4], [-4, 1]]
    return GPUSampler.from_clauses(5, clauses, _t.device("cuda"), k_cap=k_cap, seed=1)


@pytest.mark.gpu
def test_batch_size_is_the_calibrated_k():
    smp = _tiny_sampler(1024)
    assert smp.batch_size == smp.K == 1024, "the cap binds on a tiny formula"


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 7, 512, 100_000])
def test_pin_batch_is_exact_and_idempotent(n):
    smp = _tiny_sampler(1024)
    assert smp.pin_batch(n) == n and smp.batch_size == n
    assert smp.pin_batch(n) == n, "pinning twice must not drift"


@pytest.mark.gpu
def test_pinned_batch_restores():
    smp = _tiny_sampler(1024)
    before = (smp.K, smp.batch_size)
    with smp.pinned_batch(16):
        assert smp.batch_size == 16
    assert (smp.K, smp.batch_size) == before


@pytest.mark.gpu
def test_row_bytes_matches_the_engine_allocation_list():
    """The per-row model is the allocation list, restated here so that a buffer added to
    ensure_state() but not to calibrate.cpp fails the test."""
    from gpap.engine.calibrate import row_bytes, trail_cap
    for M, N in [(2215, 1494), (14505, 5701), (201437, 77151)]:
        want = (4 * M                              # packed clause word
                + 4 * ((2 * N + 31) // 32)         # assignment bitmap
                + 4 * trail_cap(N)                 # trail
                + 2 + 16 + 3)                      # flags + counters + selection masks
        assert row_bytes(M, N) == want, (M, N)


@pytest.mark.gpu
def test_trail_cap_is_the_variable_count_plus_one_wavefront():
    """The trail is num_vars entries (a variable is pushed at most once) plus one wavefront,
    because the wave paths append a whole wavefront with no bound test. It depends on nothing
    else, neither the decision batch nor the assumption width. Keep the slack in step with
    WAVE_W."""
    from gpap.engine.calibrate import trail_cap
    wave_w = 32                                  # cpp/tools/build_kernels.py::WAVE_W
    assert trail_cap(1494) == 1494 + wave_w
    assert trail_cap(0) == wave_w and trail_cap(7) == 7 + wave_w, "affine in N, one rule"
    import inspect
    assert list(inspect.signature(trail_cap).parameters) == ["num_vars"], \
        "no decision-batch or assumption-width parameter may come back"


@pytest.mark.gpu
def test_trail_cap_has_exactly_one_definition():
    """The engine's allocator and gpap_row_bytes use the same trail function, so the sizing
    model describes the real allocation."""
    import re
    src = open(os.path.join(REPO, "cpp", "src", "engine.cpp")).read()
    assert "gpap_trail_cap(" in src, "the engine must call the shared rule"
    assert not re.search(r"^\s*(constexpr|static).*kTrailSlack", src, re.M), \
        "engine.cpp must not carry a second copy of the slack"


@pytest.mark.gpu
def test_pipeline_reports_the_resolved_cap():
    """A row records the cap in force, so a small K can be traced to the cap or to a small
    GPU."""
    from gpap.sampling.pipeline import sample_clauses
    dev = torch.device("cuda")
    clauses = [[1, 2], [-1, 3], [-2, -3], [2, 3, 4]]
    info = sample_clauses(4, clauses, None, dev, n_time=200, time_limit=5.0, mine=False)
    assert info["engine"]["k_cap"] == default_k_cap(dev)
    assert info["engine"]["K"] <= info["engine"]["k_cap"]



"""The default batch cap.

Throughput is ~linear in K up to saturation, so the cap decides what a chunk is allowed to be.
It is ONE constant, the same on every GPU, and that device-independence is the property gated
here: `--k-cap auto` must mean the same batch on the laptop, the L4 and the A100, so two
machines' rows are comparable without anyone remembering to pin an override.

It used to scale with the card's total VRAM off a reference card's historical cap (49,152 at
12.0 GiB). That was removed because (a) it made `auto` machine-dependent, which is a trap in
a cross-GPU sweep, and (b) it had stopped doing anything anyway: a floor of 262,144 sat 5.3x
above the reference cap, so the proportional term could not win below ~68 GB of VRAM and every
card in reach got the floor regardless.

The cap is a SATURATION limit, not a memory bound — `compute_k_safe` is what keeps a chunk
inside VRAM. Measured: a 4x larger batch bought +2.8%, auto-vs-pinned across three GPUs ~1%.

It must also not key off FREE memory: that is not reproducible (it moved between two runs in
one session on an idle box), and a sweep whose K depends on what else was resident is not a
measurement. `test_auto_is_stable_across_calls` is that regression.
"""
import os

import pytest
import torch

from gpap.engine.calibrate import K_CAP, default_k_cap, resolve_k_cap

# Spelled out rather than imported so that editing K_CAP cannot silently make this test agree
# with itself. Moving it re-scales every future run's batch: change it deliberately, and expect
# recorded sweeps to stop being comparable.
EXPECTED_CAP = 262_144
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_cap_is_the_documented_constant():
    assert K_CAP == EXPECTED_CAP


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cap_is_device_independent(device):
    """THE property: same cap everywhere. A CPU device must answer without touching the driver,
    and a CUDA device must answer the same number — otherwise `auto` silently means a different
    experiment per machine."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    assert default_k_cap(torch.device(device)) == EXPECTED_CAP


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


# ---------------------------------------------------------------- the VRAM budget ceiling
# `GPAP_VRAM_BUDGET_MIB` exists so a reproduction on a LARGER card sizes the same K as the
# recorded one: K is memory-bound on big formulas (63 of the paper campaign's 141 rows sat below
# the cap), and a head-to-head whose rival is pinned to a recorded batch would otherwise compare
# a grown K against a fixed grant. It is a CEILING — a smaller card must still report less — and
# it is applied in ONE place, gpap_mem_available, so every consumer sees it.

@pytest.mark.gpu
def test_vram_budget_cap_is_a_ceiling_not_a_floor(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from gpap.engine import _core
    uncapped = _core.mem_available(0, 0.80)
    assert uncapped > 0
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", str(max(1, uncapped // (1 << 20) // 2)))
    assert _core.mem_available(0, 0.80) == pytest.approx(uncapped / 2, rel=0.01)
    # Above what the card has, the CARD binds: the cap can never hand out memory.
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", str(uncapped // (1 << 20) * 4))
    assert _core.mem_available(0, 0.80) == uncapped


@pytest.mark.gpu
@pytest.mark.parametrize("value", ["", "0", "-1", "bogus"])
def test_vram_budget_cap_ignores_meaningless_values(monkeypatch, value):
    """An unset, zero, negative or unparsable value must mean NO CAP, not a cap of zero: a
    budget of zero sizes K=1 and every run would silently crawl instead of failing."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from gpap.engine import _core
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    uncapped = _core.mem_available(0, 0.80)
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", value)
    assert _core.mem_available(0, 0.80) == uncapped


def test_vram_budget_cap_reporter_matches_the_env(monkeypatch):
    """`vram_budget_cap` only REPORTS; the clamp is the core's. Same parse, though, or a row's
    provenance would disagree with the budget that sized it."""
    from gpap.engine.calibrate import vram_budget_cap
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    assert vram_budget_cap() == 0
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", "18120")
    assert vram_budget_cap() == 18120 << 20
    for junk in ("", "0", "-5", "bogus"):
        monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", junk)
        assert vram_budget_cap() == 0


#: Values that read differently under two plausible parsers. `1e3` and `1024abc` are C's
#: longest-numeric-PREFIX rule; `1_000` is Python's int(); the huge ones overflow a MiB -> bytes
#: shift. Each is a way for one side of a comparison to be capped and the other not.
CAP_GRAMMAR_CASES = [("", 0), ("   ", 0), ("0", 0), ("-1", 0), ("bogus", 0),
                     ("1e3", 0), ("1024.5", 0), ("1024abc", 0), ("1_000", 0), ("0x400", 0),
                     ("2048", 2048), (" 2048 ", 2048), ("+2048", 2048),
                     ("17592186045440", 0), ("99999999999999999999", 0)]


@pytest.mark.parametrize(("value", "want_mib"), CAP_GRAMMAR_CASES)
def test_the_cap_grammar_is_one_grammar(monkeypatch, value, want_mib):
    """THE FAILURE THIS FORBIDS: C and Python disagreeing about a string. The engine clamps in
    C++ (`env_budget_cap`) and the rival harness plus `gpap info` read the same variable in
    Python, so a value one accepts and the other rejects caps ONE SIDE of a head-to-head — the
    inflation the cap exists to prevent, arrived at from the opposite direction.

    Asserted against the CORE, not only against the Python mirror: the reporter agreeing with
    itself is what the test above proves, and it is not the same claim."""
    from gpap.engine import _core
    from gpap.engine.calibrate import vram_budget_cap_mib
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", value)
    assert vram_budget_cap_mib() == want_mib
    if not _core.available() or not torch.cuda.is_available():
        return
    # The core exposes the clamp only through its effect, so compare the budget it reports with
    # the whole of free memory: a cap below that must bind at exactly the cap.
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    uncapped = _core.mem_available(0, 1.0)
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", value)
    got = _core.mem_available(0, 1.0)
    if want_mib and (want_mib << 20) < uncapped:
        assert got == want_mib << 20, f"the core parsed {value!r} differently"
    else:
        assert got == uncapped, f"the core capped on {value!r} where Python read no cap"


@pytest.mark.gpu
@pytest.mark.parametrize("cap_mib", [None, 2048, 4096])
def test_both_sides_of_a_head_to_head_get_the_same_budget(monkeypatch, cap_mib):
    """THE POINT OF THE CAP: the rival and the engine are handed the SAME bytes of the SAME card,
    so one GPAP_VRAM_BUDGET_MIB reproduces both sides of a comparison at once.

    They used to be two implementations of one idea — ours `(free - reserve) x 0.80`, the
    rival's `free x 0.80` — which differed by the driver reserve, 3.4% on the reference card. A
    single cap could then reproduce our budget or theirs, never both, and the artifact capped
    the rival 614 MiB below what the published campaign had granted it. `rival_vram_budget` now
    calls `gpu_mem_available`, so there is one number and nothing to keep in step."""
    from eval.rivals.rival_common import rival_vram_budget
    from gpap.engine.calibrate import gpu_mem_available
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if cap_mib is None:
        monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    else:
        monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", str(cap_mib))
    ours = gpu_mem_available(torch.device("cuda", 0))
    theirs = rival_vram_budget()
    assert ours == theirs, f"the two sides were sized apart: {ours:,} vs {theirs:,}"
    if cap_mib is not None:
        assert theirs <= cap_mib << 20, "the cap did not reach the rival's budget"


@pytest.mark.gpu
def test_the_vram_cap_actually_moves_the_sampler_batch(monkeypatch):
    """The cap has to reach K, not merely `mem_available`. Cheap to leave unproven and easy to
    get wrong: nothing else in the sampler reads the environment, so if the C++ side ever stopped
    consulting the model here, every capped reproduction would silently run at the full batch.

    A formula tiny enough that the batch CAP would bind is useless for this — it would answer the
    same number either way — so the test caps the BUDGET low enough that the analytic bound is
    what binds, and asks for two different budgets."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from gpap.engine.calibrate import compute_k_safe, row_bytes
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    smp = _tiny_sampler(None)                    # k_cap None: the MEMORY bound is what answers
    wide, N, num_long = smp.K, smp.engine.num_vars, smp.engine.num_long
    # a budget admitting an eighth of what the uncapped card does
    mib = max(1, (wide // 8) * row_bytes(num_long, N) // (1 << 20))
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", str(mib))
    narrow = _tiny_sampler(None).K
    assert narrow < wide, f"the cap did not reach K ({narrow} vs {wide} at {mib} MiB)"
    assert narrow == compute_k_safe(num_long, N, mib << 20, None), \
        "K under a cap must be the memory model's own answer for exactly that budget"


def test_rival_budget_honours_the_same_ceiling(monkeypatch):
    """BOTH sides or neither: capping only ours would hand the rival the whole card."""
    from eval.rivals.rival_common import rival_vram_budget
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", "1024")
    assert rival_vram_budget() <= (1024 << 20)
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    assert rival_vram_budget() > (1024 << 20) or not torch.cuda.is_available()


# ---------------------------------------------------------------- the batch-size API
# `batch_size` is the one number that says what a chunk runs, and `pin_batch` the one way to
# bound it. These gate that contract, because getting it wrong is not a wrong number — it is
# 28 GB of host RAM (see tests/test_circuits.py::_collecting_sampler).
#
# There used to be TWO models here (K plus a K_fused "boost"), and the boost existed only to
# recover headroom a 1.57x-over per-row model was wasting. The model is now the engine's exact
# allocation list (cpp/src/calibrate.cpp), so there is one number and nothing to boost.

def _tiny_sampler(k_cap):
    import torch as _t
    from gpap.sampling.sampler import GPUSampler
    clauses = [[1, 2], [-1, 3], [-2, -3], [2, 3, 4], [-4, 1]]
    return GPUSampler.from_clauses(5, clauses, _t.device("cuda"), k_cap=k_cap, seed=1)


@pytest.mark.gpu
def test_batch_size_is_the_calibrated_k():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    smp = _tiny_sampler(1024)
    assert smp.batch_size == smp.K == 1024, "the cap binds on a tiny formula"


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 7, 512, 100_000])
def test_pin_batch_is_exact_and_idempotent(n):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    smp = _tiny_sampler(1024)
    assert smp.pin_batch(n) == n and smp.batch_size == n
    assert smp.pin_batch(n) == n, "pinning twice must not drift"


@pytest.mark.gpu
def test_pinned_batch_restores():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    smp = _tiny_sampler(1024)
    before = (smp.K, smp.batch_size)
    with smp.pinned_batch(16):
        assert smp.batch_size == 16
    assert (smp.K, smp.batch_size) == before


@pytest.mark.gpu
def test_row_bytes_matches_the_engine_allocation_list():
    """The model IS the allocation list, so state it here too: if ensure_state() gains a buffer
    and calibrate.cpp does not, this fails rather than silently over-committing the GPU."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from gpap.engine.calibrate import row_bytes, trail_cap
    for M, N in [(2215, 1494), (14505, 5701), (201437, 77151)]:
        want = (4 * M                              # packed clause word
                + 4 * ((2 * N + 31) // 32)         # assignment bitmap
                + 4 * trail_cap(N)                 # trail
                + 2 + 16 + 3)                      # flags + counters + selection masks
        assert row_bytes(M, N) == want, (M, N)


@pytest.mark.gpu
def test_trail_cap_is_the_variable_count_plus_one_wavefront():
    """REGRESSION, twice over. The trail used to be sized by the width of whatever was last
    applied and only ever grew, so a formula with many unit clauses permanently inflated every
    later chunk (10.sk_1_46: 1,121 units => 2,623 instead of 1,566, +27% per-row for nothing).
    The decision batch B then survived that cleanup as a floor the ENGINE ignored but the sizing
    model still honoured — so a caller asking for B=5000 was charged for a trail that would
    never be allocated, and got a smaller K for it.

    Both are gone. The bound is EXACTLY num_vars (every push flips a bitmap bit, l and ~l share
    a word, so a variable contributes at most one entry — measured tight on both kernels), plus
    one WAVEFRONT of headroom: the vectorised append in the wave paths writes at
    `tail + prefix_sum` with no bound test, so a wavefront is the granularity in which a broken
    invariant would overshoot. Keep the slack in step with WAVE_W."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from gpap.engine.calibrate import trail_cap
    wave_w = 32                                  # cpp/tools/build_kernels.py::WAVE_W
    assert trail_cap(1494) == 1494 + wave_w
    assert trail_cap(0) == wave_w and trail_cap(7) == 7 + wave_w, "affine in N, one rule"
    import inspect
    assert list(inspect.signature(trail_cap).parameters) == ["num_vars"], \
        "no decision-batch or assumption-width parameter may come back"


@pytest.mark.gpu
def test_trail_cap_has_exactly_one_definition():
    """The engine's allocator and the exported sizing model must call the SAME function, or the
    per-row model stops describing the allocation — which is how the 1.57x-over model survived
    unnoticed. gpap_row_bytes' default trail term is that function, and it must equal what the
    engine reserves for a formula of the same size."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    import re
    src = open(os.path.join(REPO, "cpp", "src", "engine.cpp")).read()
    assert "gpap_trail_cap(" in src, "the engine must call the shared rule"
    assert not re.search(r"^\s*(constexpr|static).*kTrailSlack", src, re.M), \
        "engine.cpp must not carry a second copy of the slack"


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

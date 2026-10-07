"""The rivals are sized by the same VRAM budget as the engine (`rival_vram_budget` ->
`gpu_mem_available`), so one `GPAP_VRAM_BUDGET_MIB` caps both sides of a head-to-head."""
import pytest
import torch


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
    if cap_mib is None:
        monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    else:
        monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", str(cap_mib))
    ours = gpu_mem_available(torch.device("cuda", 0))
    theirs = rival_vram_budget()
    assert ours == theirs, f"the two sides were sized apart: {ours:,} vs {theirs:,}"
    if cap_mib is not None:
        assert theirs <= cap_mib << 20, "the cap did not reach the rival's budget"


def test_rival_budget_honours_the_same_ceiling(monkeypatch):
    """BOTH sides or neither: capping only ours would hand the rival the whole card.

    BOTH BRANCHES, because they carry the ceiling by different routes — the CUDA one through
    the core inside `gpap_mem_available`, the no-CUDA fallback on its own — so a test that only
    exercises whichever branch this machine takes proves nothing about the other. That is not
    hypothetical: the fallback shipped uncapped and stayed green on every box with a card until
    the first CPU-only CI run. Hence the forced-fallback half below, which fails on either kind
    of machine."""
    from eval.rivals.rival_common import RIVAL_VRAM_FALLBACK, rival_vram_budget
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    assert rival_vram_budget() > (1024 << 20) or not torch.cuda.is_available()
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", "1024")
    assert rival_vram_budget() <= (1024 << 20)

    # the no-CUDA fallback, forced: the ceiling binds it, and it is the CEILING doing it
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert rival_vram_budget() == 1024 << 20
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    assert rival_vram_budget() == RIVAL_VRAM_FALLBACK


def test_rival_budget_is_capped_when_the_core_cannot_load(monkeypatch):
    """The THIRD route, found inside an unpacked artifact before `setup` has built anything: CUDA
    is visible, so the CUDA branch is taken, the core call raises "libgpap not found", the
    `except` swallows it — and the last-resort return handed the rival the whole fallback with
    the ceiling ignored. Same asymmetry as the CPU-only case, on the machine a reviewer has the
    moment they unpack."""
    import gpap.engine.calibrate as calibrate
    from eval.rivals.rival_common import RIVAL_VRAM_FALLBACK, rival_vram_budget

    def no_core(*_a, **_k):
        raise RuntimeError("libgpap not found")

    monkeypatch.setattr(calibrate, "gpu_mem_available", no_core)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    assert rival_vram_budget() == RIVAL_VRAM_FALLBACK
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", "1024")
    assert rival_vram_budget() == 1024 << 20

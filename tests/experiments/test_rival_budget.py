"""The rivals are sized by the same VRAM budget as the engine (`rival_vram_budget` ->
`gpu_mem_available`), so one `GPAP_VRAM_BUDGET_MIB` caps both sides of a head-to-head."""
import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize("cap_mib", [None, 2048, 4096])
def test_both_sides_of_a_head_to_head_get_the_same_budget(monkeypatch, cap_mib):
    """The rival and the engine get the same budget, with or without the cap, so one
    `GPAP_VRAM_BUDGET_MIB` reproduces both sides of a comparison
    (docs/findings/ARTIFACT_FINDINGS.md §8)."""
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
    """The cap binds the rival's budget on both branches: CUDA (through the core) and the no-CUDA
    fallback. The fallback is forced below so the test covers it on any machine."""
    from eval.rivals.rival_common import RIVAL_VRAM_FALLBACK, rival_vram_budget
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    assert rival_vram_budget() > (1024 << 20) or not torch.cuda.is_available()
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", "1024")
    assert rival_vram_budget() <= (1024 << 20)

    # the no-CUDA fallback, forced: the cap binds it
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert rival_vram_budget() == 1024 << 20
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    assert rival_vram_budget() == RIVAL_VRAM_FALLBACK


def test_rival_budget_is_capped_when_the_core_cannot_load(monkeypatch):
    """With CUDA visible but no built library (an unpacked artifact before `setup`), the core call
    raises and the last-resort fallback must still honour the cap."""
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

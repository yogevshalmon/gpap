"""The soundness verdict of a recorded row: the tri-state, its reader and its fold.

A LEAF module on purpose — stdlib only, no numpy, no torch, no engine. The table generators
(`eval/artifact/paper_tables.py`, `verify_artifact.py`) read these off frozen JSONs on machines
with no GPU and no CUDA wheels, and reaching them through `gpap.sampling.pipeline` dragged the
whole sampler in, which made regenerating a table cost an 800 MB torch install. It sits at the
TOP level rather than under `gpap.sampling` because importing any module of that package first
executes its `__init__`, which constructs the sampler — the very import this exists to avoid.

There is still ONE definition: `pipeline` re-exports these names, so `from
gpap.sampling.pipeline import sound_verdict` — what every existing consumer writes — resolves
here.
"""
#: The verdict is tri-state, and "off" is load-bearing: a run that completed no row never called
#: the check, and reporting that as a pass would publish silence as evidence.
SOUND_OK, SOUND_BAD, SOUND_OFF = "ok", "UNSOUND", "off"

__all__ = ["sound_verdict", "merge_sound_check", "SOUND_OK", "SOUND_BAD", "SOUND_OFF"]


def sound_verdict(row) -> str:
    """The tri-state verdict of one recorded row, legacy schema included: a row with only
    `gpu_sound: bool` always checked, so the bool is a verdict; neither key means "off"."""
    verdict = row.get("sound_check")
    if verdict in (SOUND_OK, SOUND_BAD, SOUND_OFF):
        return verdict
    if "gpu_sound" in row:                       # legacy: the check was unconditional
        return SOUND_OK if row["gpu_sound"] else SOUND_BAD
    return SOUND_OFF


def merge_sound_check(*verdicts) -> str:
    """Fold several verdicts into one: UNSOUND dominates, then "ok", then "off". Any other
    order publishes an "ok" beside a non-zero `unsound_rows`."""
    seen = [v for v in verdicts if v]
    if SOUND_BAD in seen:
        return SOUND_BAD
    return SOUND_OK if SOUND_OK in seen else SOUND_OFF

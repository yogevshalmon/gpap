"""What a recorded row MEANS, shared by every printer: the soundness tri-state, its reader and
its fold, plus the one rendering of a success rate.

A LEAF module on purpose — stdlib only, no numpy, no torch, no engine. The table generators
(`eval/artifact/paper_tables.py`, `verify_artifact.py`) read these off frozen JSONs on machines
with no GPU and no CUDA wheels, and reaching them through `gpap.sampling.pipeline` dragged the
whole sampler in, which made regenerating a table cost an 800 MB torch install. It sits at the
TOP level rather than under `gpap.sampling` because importing any module of that package first
executes its `__init__`, which constructs the sampler — the very import this exists to avoid.

There is still ONE definition: `pipeline` re-exports the VERDICT names, so `from
gpap.sampling.pipeline import sound_verdict` — what every existing consumer writes — resolves
here. `fmt_pct` is imported from this module directly, by `gpap/cli.py` and `gpap/benchmark.py`;
it is not re-exported, because no consumer predates it.
"""
#: The verdict is tri-state, and "off" is load-bearing: a run that completed no row never called
#: the check, and reporting that as a pass would publish silence as evidence.
SOUND_OK, SOUND_BAD, SOUND_OFF = "ok", "UNSOUND", "off"

__all__ = ["sound_verdict", "merge_sound_check", "fmt_pct",
           "SOUND_OK", "SOUND_BAD", "SOUND_OFF"]


#: Widest string `fmt_pct` can return. The bench table has a fixed column, so this is a contract,
#: not a note: "%.2g" of a rare-pin rate is "0.00019%", 8 characters, and it ran into the next
#: column. Every branch below is checked against it.
PCT_WIDTH = 6


def fmt_pct(frac) -> str:
    """A success rate as a percentage, never flooring a real one to "0%": a 0.4%-success instance
    that delivered 20,000 models printed "success 0%", which reads as a failed run. Here rather
    than in either printer because `gpap sample` and the `gpap bench` table print the same
    quantity, and a rounding rule that lives in two places is one that will disagree in two.

    At most `PCT_WIDTH` characters, which is what makes it usable in a table — but the small end
    keeps its MAGNITUDE ("4e-05%", not "<0.01%"), because a rare-pinned instance lives down there
    and the order of magnitude is the story on it (docs/findings/CLI_DELIVERY_FINDINGS.md §7 has
    the recorded range). The exact rate is `success_rate` in the row.
    """
    v = 100.0 * frac
    if v != v:                                   # nan: no rate was measured, do not print "0%"
        return "nan%"
    if v >= 1.0 or v == 0.0:
        return f"{v:.0f}%"
    if v >= 0.01:
        return f"{v:.2g}%"                       # 0.01% .. 0.99%, never wider than PCT_WIDTH
    # Below 0.01%: one significant digit and an exponent still fits. A denormal would not
    # ("1e-300%" is 7), and a rate that small cannot come out of models/attempts — but the
    # contract is checked rather than argued.
    small = f"{v:.0e}%"
    return small if len(small) <= PCT_WIDTH else "<0.01%"


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

# GPAP documentation

| you want to… | read |
|---|---|
| understand how the engine works — state layout, the two kernels, the memory model, what is guaranteed | [ENGINE.md](ENGINE.md) |
| use GPAP from Python or C++ | [PYTHON_API.md](PYTHON_API.md) · [../cpp/README.md](../cpp/README.md) |
| measure GPAP, or compare it against another sampler, without fooling yourself | [BENCHMARKING.md](BENCHMARKING.md) |
| reproduce a paper's experiments end to end | [EVAL_RUNBOOK.md](EVAL_RUNBOOK.md), then that paper's folder under [papers/](papers/) |

**[`findings/`](findings/)** — the evidence. Every design decision in the code is backed by a
measurement, and the measurement (date, GPU, per-instance table) lives here, never in a source
comment. [findings/INSIGHTS.md](findings/INSIGHTS.md) is the index of settled results — dated,
one paragraph each, with a note wherever a later measurement moved a number; the
`*_FINDINGS.md` files are the deep dives, and
[findings/CLEAN_MODE_FINDINGS.md](findings/CLEAN_MODE_FINDINGS.md) carries the design and the
lemmas of CLEAN mode. Retired directions stay recorded so they are not re-tried without new
evidence.

**[`papers/`](papers/)** — one folder per submission: what that paper evaluates, which rules its
generator enforces, and the transcription of every number it prints, so the generated tables can be
held to the PDF. **No generated table and no `.tex` section is committed** — a paper stands on its
own, its LaTeX lives in its own tree, and the rows and tables live outside the repository in
`$GPAP_RESULTS`. The talk decks are there too: paper material, not documentation of the library.

Two conventions hold everywhere: **code carries the rule, docs carry the evidence** (a comment
says what the code does and why it is correct; the number that settled it lives here), and
**absolute rates are never comparable across power states or GPUs** — only ratios measured in
one session are ([BENCHMARKING.md](BENCHMARKING.md) §1).

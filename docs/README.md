# GPAP documentation

Four kinds of document live here. Start with the one that matches what you are doing.

| you want to… | read |
|---|---|
| understand how the engine works — state layout, the two kernels, the memory model, what is guaranteed | [ENGINE.md](ENGINE.md) |
| use GPAP from Python or C++ | [PYTHON_API.md](PYTHON_API.md) · [../cpp/README.md](../cpp/README.md) |
| measure GPAP, or compare it against another sampler, without fooling yourself | [BENCHMARKING.md](BENCHMARKING.md) |
| reproduce the paper's experiments end to end | [EVAL_RUNBOOK.md](EVAL_RUNBOOK.md), then [paper/README.md](paper/README.md) |

**[`findings/`](findings/)** — the evidence. Every design decision in the code is backed by a
measurement, and the measurement (date, GPU, per-instance table) lives here, never in a source
comment. [findings/INSIGHTS.md](findings/INSIGHTS.md) is the index of settled results — dated,
one paragraph each, with ⚠️ notes where a later measurement moved a number; the `*_FINDINGS.md`
files are the deep dives. Retired directions stay recorded so they are not re-tried without new
evidence.

**[`plans/`](plans/)** — design documents for the larger changes, each with its status at the top.
Kept because the reasoning (why the alternative was rejected, which lemma the gate checks) does
not fit in a commit message.

**[`paper/`](paper/)**, **[`slides/`](slides/)** — generated tables and the deck. Nothing in them
is typed by hand; [paper/README.md](paper/README.md) says how to regenerate them from the raw
results, which live outside the repository (`$GPAP_RESULTS`).

Two conventions hold everywhere: **code carries the rule, docs carry the evidence** (a comment
says what the code does and why it is correct; the number that settled it lives here, with its
date and GPU); and **absolute rates are never comparable across power states or GPUs** — only
ratios measured in one session are ([BENCHMARKING.md](BENCHMARKING.md) §1).

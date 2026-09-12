# GPAP documentation

| you want to… | read |
|---|---|
| understand how the engine works — state layout, the two kernels, the memory model, what is guaranteed | [ENGINE.md](ENGINE.md) |
| use GPAP from Python or C++ | [PYTHON_API.md](PYTHON_API.md) · [../cpp/README.md](../cpp/README.md) |
| measure GPAP, or compare it against another sampler, without fooling yourself | [BENCHMARKING.md](BENCHMARKING.md) |
| reproduce the paper's experiments end to end | [EVAL_RUNBOOK.md](EVAL_RUNBOOK.md), then [paper/README.md](paper/README.md) |

**[`findings/`](findings/)** — the evidence. Every design decision in the code is backed by a
measurement, and the measurement (date, GPU, per-instance table) lives here, never in a source
comment. [findings/INSIGHTS.md](findings/INSIGHTS.md) is the index of settled results — dated,
one paragraph each, with a note wherever a later measurement moved a number; the
`*_FINDINGS.md` files are the deep dives, and
[findings/CLEAN_MODE_FINDINGS.md](findings/CLEAN_MODE_FINDINGS.md) carries the design and the
lemmas of CLEAN mode. Retired directions stay recorded so they are not re-tried without new
evidence.

**[`paper/`](paper/)** — the results record: which campaign the published numbers come from, the
rules its generator enforces, and the generated tables themselves. Nothing in it is typed by hand;
[paper/README.md](paper/README.md) says how to regenerate it from the raw results, which live
outside the repository (`$GPAP_RESULTS`). The paper's LaTeX and the talk decks live there too —
they are paper material, not documentation of the library.

Two conventions hold everywhere: **code carries the rule, docs carry the evidence** (a comment
says what the code does and why it is correct; the number that settled it lives here), and
**absolute rates are never comparable across power states or GPUs** — only ratios measured in
one session are ([BENCHMARKING.md](BENCHMARKING.md) §1).

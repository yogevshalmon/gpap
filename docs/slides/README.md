# Slides

`gpap_overview.tex` — the project told as the paper we are about to write. **The engine is the
result; the applications prove it is useful** — that ordering is deliberate and drives the whole
deck: NobleCount (FMCAD'26) → the GPU engine (~10⁹ propagations/s, ~300× a CPU core) → sampling as
the application → uniformity. **26 pages: 22 talk slides, then 4 generated result tables.**

Three primer frames exist because the audience is not assumed to know the tooling: **GPU
SMs/warps/occupancy** (before the architecture frame, since the whole scalar-vs-WAVE design is an
occupancy argument), **the GPU rival samplers** (before the head-to-head), and **what "uniform"
means** (before the two uniformity frames).

`backup_slides.tex` — **a separate deck**, compiled on its own (7 pages), not `\input` anywhere.
It holds what an advisor may ask about but the talk should not carry: the engine feature list, the
two-machine (4080 vs L4) table with the CPU-dependence analysis, the two reporting corrections we
made to our own headline, the status/roadmap, the cross-GPU rival check, and the loose numbers.

`gpap_style.tex` — the preamble both decks `\input`, so colours, macros and theme cannot drift
apart.

```bash
pdflatex backup_slides.tex && pdflatex backup_slides.tex
```

Visual-first: the architecture is carried by TikZ figures (the NobleCount count+key idea, the
packed clause word with its self-detecting `atomic_add`, one-assignment-per-warp with the
scalar/WAVE lane picture and the dispatch crossover, the library stack, K trajectories, the
target-rarity distribution, the uniformity regime bars). Text is captions, not prose.

Build:

```bash
pdflatex gpap_overview.tex && pdflatex gpap_overview.tex   # twice, for the TOC
```

Dependencies are deliberately minimal so it also compiles on Overleaf as-is: base `beamer`
(Boadilla theme), `booktabs`, `xcolor`, `tikz` (`positioning`, `arrows.meta`). No `metropolis`, no
external images — every figure is TikZ.

Builds clean: **26 pages** (22 talk slides, then 4 generated result tables). Build artifacts (`.aux`, `.log`, `.nav`, `.snm`, `.out`, `.toc`, …) are
gitignored under `docs/slides/`; the `.pdf` is kept.

## Where the numbers come from — and why they cannot drift

`l4_results.tex` (the appendix frames) is **generated**, never edited:

```bash
R=${GPAP_RESULTS:-../../gpap-results}/gpap_experiments_l4_20260805
eval/artifact/slide_tables.py --cmsgen $R/gpap_vs_cmsgen_l4 --rivals $R/gpap_vs_gpu_rivals_l4 \
    --uniformity $R/uniformity_l4/uniformity_campaign.json --out docs/slides/l4_results.tex
```

`slide_tables.py` **imports its aggregation from `eval/artifact/paper_tables.py`** rather than restating
it, so a deck number and a paper number are the same computation. Both apply the same exclusions
(vacuous pins, zero-search cells) and the same symmetric zero ledger. This matters: an earlier
deck carried `47.7×` because the two had separate copies of the rule.

Prose sources: `docs/findings/L4_EVAL_FINDINGS.md` (the current eval),
`docs/findings/INSIGHTS.md` (settled results index),
`docs/findings/UNIFORMITY_FINDINGS.md`, `docs/ENGINE.md`.
Raw data lives outside the repo in `$GPAP_RESULTS` (default `../gpap-results`).

The cross-GPU rival frame ("Does that verdict survive a different GPU?") is generated from
`eval/rivals/rivals_summary.py <4080-dir> <l4-dir> --variant raw`, which pairs the two machines on the
`(instance, variant)` cells BOTH produced — that is why its 4080 column can differ from the
single-machine frame before it (Recurrent: $599\times$ paired over 3 instances vs $1{,}200\times$
over all 7).

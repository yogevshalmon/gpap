# Slides

`gpap_overview.tex` — the project as the paper to come: NobleCount (FMCAD'26) → the GPU engine (~10⁹ propagations/s, ~300× a CPU core) → sampling → uniformity. **The engine is the
result; the applications prove it useful**, and that ordering drives the deck. 26 pages: 22 talk
slides, 4 generated tables, and three primer frames (GPU SMs/warps/occupancy, the rival samplers,
"uniform") before the frames that need them.

`backup_slides.tex` — a separate deck (7 pages), compiled on its own, never `\input`: the engine
feature list, the two-machine (4080 vs L4) table and its CPU-dependence analysis, the two reporting
corrections to our own headline, status/roadmap and the cross-GPU rival check.

`gpap_style.tex` — the preamble both decks `\input`, so colours, macros and theme cannot drift.

```bash
pdflatex gpap_overview.tex && pdflatex gpap_overview.tex   # twice, for the TOC
pdflatex backup_slides.tex && pdflatex backup_slides.tex
```

Dependencies are minimal, so it compiles on Overleaf as-is: `beamer` (Boadilla), `booktabs`,
`xcolor`, `tikz` (`positioning`, `arrows.meta`) — no `metropolis`, no external images, every figure
TikZ, text as captions. Build artifacts are gitignored; the `.pdf` is kept.

## Where the numbers come from

`l4_results.tex` (the appendix frames) is **generated**, never edited:

```bash
R=${GPAP_RESULTS:-../../gpap-results}/gpap_experiments_l4_20260805
eval/artifact/slide_tables.py --cmsgen $R/gpap_vs_cmsgen_l4 --rivals $R/gpap_vs_gpu_rivals_l4 \
    --uniformity $R/uniformity_l4/uniformity_campaign.json --out docs/slides/l4_results.tex
```

`slide_tables.py` imports its aggregation from `eval/artifact/paper_tables.py` rather than
restating it, so a deck number and a paper number are one computation under the same exclusions and
zero ledger. An earlier deck carried `47.7×` because the two had separate copies of the rule.

Prose sources: `docs/findings/L4_EVAL_FINDINGS.md`, `INSIGHTS.md`, `UNIFORMITY_FINDINGS.md`,
`docs/ENGINE.md`. Raw data lives outside the repo in `$GPAP_RESULTS` (`../gpap-results`).

The cross-GPU rival frame comes from `eval/rivals/rivals_summary.py <4080-dir> <l4-dir> --variant
raw`, which pairs the machines on the cells both produced — hence its 4080 column can differ from
the single-machine frame (Recurrent: $599\times$ paired over 3 instances, $1{,}200\times$ over
all 7).

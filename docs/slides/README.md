# Slides

`gpap_overview.tex` — the project told as the paper we are about to write: NobleCount (FMCAD'26)
→ the GPU generalization (sum-trick clause word, one assignment per warp) → sampling as the
application → uniformity → status and open items. 26 frames, the last two are backup.

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

Builds clean (20 pages, one 0.7pt overfull box). Build artifacts (`.aux`, `.log`, `.nav`, `.snm`,
`.out`, `.toc`, …) are gitignored under `docs/slides/`; the `.pdf` is kept.

Sources for the numbers: `docs/findings/INSIGHTS.md` (settled results index),
`docs/findings/UNIFORMITY_FINDINGS.md`, `docs/ENGINE.md`, and
`.experiments/full_eval/<gpu>/plots/summary.md` (the 357-row sweep).

The cross-GPU rival frame ("Does that verdict survive a different GPU?") is generated from
`eval/rivals_summary.py <4080-dir> <l4-dir> --variant raw`, which pairs the two machines on the
`(instance, variant)` cells BOTH produced — that is why its 4080 column can differ from the
single-machine frame before it (Recurrent: $599\times$ paired over 3 instances vs $1{,}200\times$
over all 7).

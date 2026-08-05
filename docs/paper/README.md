# Paper experiment section — data, tables, and how to regenerate them

Everything in this directory is **generated from the experiment JSONs**. Nothing is typed by
hand, and that is deliberate: an earlier version of the deck carried a sequential headline of
`47.7×` that silently included four instances whose output pins are tautologies. The real number
is `36.5×`. The only durable fix is to derive every published figure from the data with the
exclusions applied in exactly one place.

| file | what it is |
|---|---|
| `RESULTS.md` | every table, plus the caveats that must appear in the text |
| `tables.tex` | the same tables as `booktabs` LaTeX, ready to `\input` |
| `../../artifacts/gpap_experiments_l4_<date>.tar.gz` | the raw JSONs + MANIFEST + verification |

## Regenerate

```bash
eval/paper_tables.py \
    --cmsgen     .experiments/cmsgen_l4_2026-08-03 \
    --cmsgen-ref .experiments/full_eval/nvidia-geforce-rtx-4080-laptop-gpu \
    --rivals     .experiments/l4_rivals \
    --recheck    .experiments/recheck \
    --uniformity .experiments/uniformity/*.json \
    --soft       .experiments/uniformity/soft_profile.json \
    --out-md docs/paper/RESULTS.md --out-tex docs/paper/tables.tex

eval/verify_artifact.py .experiments/cmsgen_l4_2026-08-03 \
    --rivals .experiments/l4_rivals --uniformity '.experiments/uniformity/*.json'

eval/make_artifact.sh          # -> artifacts/gpap_experiments_l4_<date>.tar.gz
```

## The four rules the generator enforces

These exist because each one, left implicit, produces a number that a referee would reject.

1. **Vacuous pins are excluded from sequential geomeans.** `b02`, `b05`, `b07`, `s526` have
   constant last-cycle outputs at T=25, so pinning them is a tautology and every sampler scores
   ~100%. Including them inflates the *rare* column — the one we tell people to quote — by up to
   1.9×, and is nearly free at random pins.
2. **A ratio against a sampler that produced nothing is not a ratio.** Where CMSGen emitted zero
   samples inside the budget the ratio is ∞; those cells are counted as outright wins and
   excluded from every geomean, which makes the published figures *conservative*. The count is
   reported in its own column.
3. **A geomean over fewer than three instances is flagged, not quoted.** On HWMCC, CMSGen scores
   zero on 5 of 7 (T=25) and 6 of 7 (T=50), so the surviving "geomean" rests on 2 and 1 instance.
   Report the win count instead.
4. **A GPAP/CMSGen ratio without its CPU is meaningless.** CMSGen is single-core, so the ratio is
   as much a property of the host as of the sampler: the same suite reads 48.5× on an RTX 4080
   laptop (i7-13800H) and 113× on a Colab L4 box whose CPU is 0.41× as fast — and GPAP itself is
   0.96× between them. `gpap bench` records `cpu` in every JSON for this reason.

## What the experiments are

| experiment | question it answers |
|---|---|
| GPAP vs CMSGen | throughput against the SOTA uniform-ish CPU sampler, patched for fair timing |
| GPAP vs 3 GPU rivals | throughput against the gradient-descent GPU samplers, on their metric |
| uniformity vs an exact oracle | is what we emit actually uniform — including the `soft` config |
| soft portfolio profile | what excluding the Barbarik-rejected `soft` config costs in throughput |
| reproducibility re-check | do the numbers reproduce on a fresh machine of the same type |

The uniformity campaign carries **both controls** — an exact-uniform negative and a known-TV
positive. Without them a "no skew detected" result is uncalibrated, and `verify_artifact.py`
fails the artifact if either is missing.

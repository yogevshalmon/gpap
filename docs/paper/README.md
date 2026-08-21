# Paper experiment section — data, tables, and how to regenerate them

Everything in this directory is **generated from the experiment JSONs**. Nothing is typed by
hand, and that is deliberate: an earlier version of the deck carried a sequential headline of
`47.7×` that silently included four instances whose output pins are tautologies. The real number
is `36.5×`. The only durable fix is to derive every published figure from the data with the
exclusions applied in exactly one place.

`eval/artifact/slide_tables.py` **imports** its aggregation from `eval/artifact/paper_tables.py` rather than
restating it, so the deck cannot drift away from the paper. When the generator learned to count
our own zeros, the deck learned it in the same commit.

## Where the data lives

The raw results are **outside the repo**, in `$GPAP_RESULTS` (default `../gpap-results`, a
sibling of the checkout) — they are too heavy to track. Set the variable if yours is elsewhere.

| file | what it is |
|---|---|
| `RESULTS.md` | every table, plus the caveats that must appear in the text |
| `tables.tex` | the same tables as `booktabs` LaTeX, ready to `\input` |
| `SOFT_PRICE.md` / `soft_price.tex` | what the uniformity guarantee costs, paired sweep vs sweep |
| `$GPAP_RESULTS/gpap_experiments_l4_<date>.tar.gz` | raw JSONs + MANIFEST + SHA-256 + verification |

## Regenerate

```bash
R=${GPAP_RESULTS:-../gpap-results}/gpap_experiments_l4_20260805

eval/artifact/paper_tables.py \
    --cmsgen     $R/gpap_vs_cmsgen_l4 \
    --cmsgen-ref $R/gpap_vs_cmsgen_4080_reference \
    --rivals     $R/gpap_vs_gpu_rivals_l4 \
    --recheck    $R/reproducibility_recheck_l4 \
    --uniformity $R/uniformity_l4/uniformity_campaign.json \
    --soft       $R/uniformity_l4/soft_profile.json \
    --out-md docs/paper/RESULTS.md --out-tex docs/paper/tables.tex

eval/artifact/soft_price_tables.py --safe $R/gpap_vs_cmsgen_l4 --soft $R/gpap_vs_cmsgen_l4_softarm \
    --out-md docs/paper/SOFT_PRICE.md --out-tex docs/paper/soft_price.tex

eval/artifact/slide_tables.py --cmsgen $R/gpap_vs_cmsgen_l4 --rivals $R/gpap_vs_gpu_rivals_l4 \
    --uniformity $R/uniformity_l4/uniformity_campaign.json --out docs/slides/l4_results.tex

eval/artifact/verify_artifact.py $R/gpap_vs_cmsgen_l4 --rivals $R/gpap_vs_gpu_rivals_l4 \
    --uniformity $R/uniformity_l4/*.json

eval/artifact/make_artifact.sh          # -> $GPAP_RESULTS/gpap_experiments_l4_<date>.tar.gz
```

## The six rules the generator enforces

These exist because each one, left implicit, produces a number that a referee would reject. Rules
1 and 5 are two faces of the same principle: **a cell that measures the instance rather than the
sampler does not belong in a geomean.**

1. **Vacuous pins are excluded from sequential geomeans.** `b02`, `b05`, `b07`, `s526` have
   constant last-cycle outputs at T=25, so pinning them is a tautology and every sampler scores
   ~100%. Including them inflates the *rare* column — the one we tell people to quote — by up to
   1.9×, and is nearly free at random pins.
2. **A ratio against a sampler that produced nothing is not a ratio.** Where CMSGen emitted zero
   samples inside the budget the ratio is ∞; those cells are counted as outright wins and
   excluded from every geomean, which makes the published figures *conservative*.
3. **The zero ledger is symmetric.** This is the rule that was missing, and it flattered us: a
   CMSGen-zero row carries `ratio = inf` and was branched on, while a GPAP-zero row carries
   `ratio = 0.0` and a both-zero row `ratio = None` — both falsy, both silently dropped. 38
   instances CMSGen could not sample were shown as wins; **3 we could not sample and 11 neither
   could were invisible**. Table 1 now carries `CMSGen=0`, `GPAP=0` and `both=0`, and they sum
   with the geomean's instances to the instance count.
4. **A geomean over fewer than three instances is flagged, not quoted.** On HWMCC, CMSGen scores
   zero on 5 of 7 (T=25) and 6 of 7 (T=50), so the surviving "geomean" rests on 2 and 1 instance.
   Report the win count instead.
5. **Zero-search cells are excluded.** EPFL `ctrl`, `dec`, `square` and sk\_ `tableBasedAddition`
   emit at 235–240M samples/s with *no decisions and no implications*: after propagation the
   support is a free hypercube, so no search happens. Worse, the best-encoder rule **prefers**
   them, because they carry their family's highest ratio. Leaving them in reads EPFL-rare `29.0×`
   instead of `9.4×`. Detected per row (`paper_tables.zero_search`), not by a name list.
6. **A GPAP/CMSGen ratio without its CPU is meaningless.** CMSGen is single-core, so the ratio is
   as much a property of the host as of the sampler: the same suite reads 46.2× on an RTX 4080
   laptop (i7-13800H) and 98.9× on a Colab L4 box whose CPU is 0.41× as fast — and GPAP itself is
   0.96× between them. `gpap bench` records `cpu` in every JSON for this reason.

## What the experiments are

| experiment | question it answers |
|---|---|
| GPAP vs CMSGen | throughput against the SOTA uniform-ish CPU sampler, patched for fair timing |
| GPAP vs 3 GPU rivals | throughput against the gradient-descent GPU samplers, on their metric |
| **propagation rate** | is the speedup the engine or the benchmark? (Table 3) |
| uniformity vs an exact oracle | is what we emit actually uniform — including the `soft` config |
| **soft-arm sweep** | the whole matrix re-run with `--allow-soft`: what uniformity costs us |
| reproducibility re-check | do the numbers reproduce on a fresh machine of the same type |

The uniformity campaign carries **both controls** — an exact-uniform negative and a known-TV
positive. Without them a "no skew detected" result is uncalibrated, and `verify_artifact.py`
fails the artifact if either is missing.

### The propagation table answers the obvious objection

Sample-rate ratios span 8×–636×, which invites "you picked your benchmarks". The counters answer
it: GPAP holds **~10⁹ implications/s on every family** (geomean 1.07×10⁹, spread 1.41× over the nine
families with ≥3 instances) against CMSGen's 1.7–5.4×10⁶, and the identity

```
samples/s ratio = (implications/s ratio) / (implications-per-sample ratio)
```

holds to within 1% on 9 of 9 families, and `eval/artifact/verify_artifact.py` re-derives it from the raw
rows rather than trusting the generator. The engine buys a flat 197–631× (geomean 306×);
rejection sampling spends part of it back (1.0× on ISCAS-85, 5.6× on sequential T=25, 53.7× on
EPFL); the remainder is the published ratio. ⚠️ Every column must be computed over the *same* rows — filtering each
independently lets an instance that propagates hard but emits nothing (EPFL `bar`) into the
numerator and not the denominator, breaking the identity by 3×.

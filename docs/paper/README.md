# Paper experiment section — data, tables, and how to regenerate them

Everything in this directory is **generated from the experiment JSONs**. Nothing is typed by
hand, and that is deliberate: an earlier version of the deck carried a sequential headline of
`47.7×` that silently included four instances whose output pins are tautologies. The real number
is `36.5×`. The only durable fix is to derive every published figure from the data with the
exclusions applied in exactly one place.

`eval/artifact/slide_tables.py` **imports** its aggregation from `eval/artifact/paper_tables.py`
rather than restating it, so the deck cannot drift away from the paper. When the generator
learned to count our own zeros, the deck learned it in the same commit.

## The campaign these tables report

**`$GPAP_RESULTS/l4_abc_20260902`** (default `../gpap-results/l4_abc_20260902`, a sibling of the
checkout — the raw data is too heavy to track). Measured 2026-09-02 on a Colab **NVIDIA L4** at
repo commit `fc83dcd`, engine build `libgpap 0.3.0+f653e35cc6bd3f94` — the build id every row
records and the header of `RESULTS.md` re-derives from the rows. Its scope is converged, on
purpose:

* **Circuit families only** — seq (19) / comb (10) / epfl (20) / hwmcc (7). No sk stage.
* **`abc:fast` encoder only** — no tseitin arm, so there is no encoder A/B and no per-instance
  best-encoder question. One row per (instance, target, depth).
* **Uniformity = Barbarik CM_test only** (Table 4b), run locally (RTX 4080) on the same
  `abc:fast`-encoded circuits.

⚠️ **This campaign predates the converged protocol** ([../EVAL_RUNBOOK.md](../EVAL_RUNBOOK.md)
§0, 2026-09-03) and its tables are superseded by the next one. Under the protocol there is ONE
campaign directory, the rival stages (`rivals_*.json`): each row is the rival, the GPAP arm and
the CMSGen arm on the same CNF, and `paper_tables.py --rivals <dir>` reads Tables 1, 2 and 3
off it (`--cmsgen` is then optional and supplementary; `verify_artifact.py --rivals <dir>` is
the gate). The commands below regenerate the tables of the campaign named above.

| campaign path | content |
|---|---|
| `gpap_vs_cmsgen_l4_abc/` | 12 stage JSONs + `plots/` (figures, `sps_scatter_all.png`, `summary.md`) |
| `gpap_vs_gpu_rivals_l4_abc/` | 5 rival JSONs + `rivals_steps_calibration.json` + `plots/` (per-rival dumbbells, `sps_scatter_rivals.png`) |
| `barbarik_abc/` | per-arm JSONL (`circ_abc_{spur,cmsgen,pipe,b8}.jsonl`) + `eligible.txt` + logs |
| `VERIFICATION.txt` | `verify_artifact.py` over the shipped copy (exit 0) |
| `README.md` | the campaign's own protocol record (VM history, incidents, deviations) |

## Files in this directory

| file | what it is |
|---|---|
| `RESULTS.md` | every table, plus the caveats that must appear in the text |
| `tables.tex` | the same tables as `booktabs` LaTeX, ready to `\input` |

The WAVE-vs-SCALAR kernel A/B table (`KERNEL_AB.md` / `kernel_ab.tex`,
`eval/artifact/kernel_ab_tables.py`) was removed rather than left stale: its data is the
2026-08-23 pre-shuffle-schedule engine, which the 2026-08-27 wave rewrite superseded
(`docs/findings/WAVE_SHUFFLE_FINDINGS.md` — the tie band's crossover moved and must be
re-derived, not guessed). The dispatch-rule evidence lives in
`docs/findings/KERNEL_AB_FINDINGS.md`; regenerate the table from a fresh `eval/kernel_ab.py`
campaign against the current engine before citing a kernel A/B number.

## Regenerate

```bash
R=${GPAP_RESULTS:-../gpap-results}/l4_abc_20260902

eval/artifact/paper_tables.py \
    --cmsgen   $R/gpap_vs_cmsgen_l4_abc \
    --rivals   $R/gpap_vs_gpu_rivals_l4_abc \
    --barbarik $R/barbarik_abc/circ_abc_*.jsonl \
    --out-md docs/paper/RESULTS.md --out-tex docs/paper/tables.tex

eval/artifact/verify_artifact.py $R/gpap_vs_cmsgen_l4_abc \
    --rivals $R/gpap_vs_gpu_rivals_l4_abc          # the publishability gate

eval/eval_plots.py $R/gpap_vs_cmsgen_l4_abc        # figures -> <dir>/plots/
eval/eval_plots.py $R/gpap_vs_gpu_rivals_l4_abc    # rival figures, incl. the unique/s scatter
```

The two all-cells-at-once scatters (`sps_scatter_all.png` for CMSGen, `sps_scatter_rivals.png`
for the GPU rivals) answer the question the geomeans hide — how the instances are distributed
around a ratio — and both import their cell classification from `paper_tables`
(`cms_arm_ran`/`zero_search`, `rival_cells`), so a figure cannot show a cell its table screens.

The header of `RESULTS.md` carries two commits and they mean different things: the **measured
engine** is the per-row build id (unioned over every stage — more than one value means mixed
engines), while `tables generated at` is only the HEAD the generator ran under. Re-running the
block above at a later commit moves the second, never the first.

## What the experiments are

| experiment | question it answers |
|---|---|
| GPAP vs CMSGen (Table 1) | throughput against the SOTA uniform-ish CPU sampler, patched for fair timing |
| GPAP vs 3 GPU rivals (Table 2) | throughput against the gradient-descent GPU samplers, on their metric |
| propagation rate (Table 3) | is the speedup the engine or the benchmark? |
| Barbarik CM_test (Table 4b) | the standard conditional-sampling uniformity certificate |

**Absent by design, and the absence is load-bearing**: no reproducibility re-check arm has been
run against this campaign's engine yet, so `RESULTS.md` carries no Table 6 — an un-run check is
neither a pass nor a failure, and the generator only prints the table when the arm exists.

## The nine rules the generator enforces

These exist because each one, left implicit, produces a number that a referee would reject.
Rules 1, 5 and 6 are three faces of the same principle: **a cell that measures the instance
rather than the sampler does not belong in a geomean.** The incident record behind each rule is in
`docs/findings/` (RIVAL_FAIRNESS, L4_EVAL, UNIFORMITY).

1. **Vacuous pins are excluded from sequential geomeans.** `b02`, `b05`, `b07`, `s526` have
   constant last-cycle outputs at T=25, so pinning them is a tautology and every sampler scores
   ~100%. Including them inflates the *rare* column — the one we tell people to quote.
2. **A ratio against a sampler that produced nothing is not a ratio.** Where CMSGen emitted zero
   samples inside the budget the ratio is ∞; those cells are counted as outright wins and
   excluded from every geomean, which makes the published figures *conservative*.
3. **The zero ledger is symmetric.** A CMSGen-zero row carries `ratio = inf`, a GPAP-zero row
   `0.0`, a both-zero row `None` — the last two are falsy, and branching on `ratio` alone once
   showed 38 of their zeros as wins while hiding all of ours. Table 1 carries `CMSGen=0`,
   `GPAP=0` and `both=0`, and they sum with the geomean's instances to the instance count.
4. **Absent is not zero.** A row whose CMSGen arm never ran (a campaign resumed on a VM that
   lost the binary) or whose subprocess crashed (`cms_error`) is neither a win nor a zero: it is
   screened per row, named in the caveats, and `verify_artifact.py` FAILS a stage that mixes
   armed and armless rows. The producer refuses too — `gpap bench --resume` will not append rows
   with a different arm than the file holds.
5. **Zero-search cells are excluded.** EPFL `ctrl`, `dec`, `square`, `cavlc` emit at hundreds of
   M samples/s with *no decisions and no implications*: after propagation the support is a free
   hypercube, so no search happens. Detected per row (`paper_tables.zero_search`), not by a name
   list. Leaving them in would read EPFL-rare `257.0×` instead of `108.9×` on this campaign.
6. **A solution space smaller than the sample request is enumerated, not sampled.** ISCAS-85
   `c17` has 5 inputs = 32 points; both tools saturate it and the ratio becomes a race between
   startup costs. DEMOTIC's own Table II says the same thing by carving it out — "a minimum of
   1000 distinct solutions (**except c17**)" — and that 1000 is the threshold
   `paper_tables.tiny_space` uses (`2**support < 1000`, read off the rival's own `num_inputs` on
   a Table 2 row). It is applied to BOTH tables, and it does not flatter us: it removes a
   45,520× cell from Table 2 (DEMOTIC-native would otherwise read ~6.5× instead of 2.7×) while
   *returning* 1,348× → 1,620× to Table 1's combinational row, where the tiny space made us look
   worse. A screen that only ever helped would be the one to distrust.
7. **Every screen applies to every table.** Vacuous pins were screened in Table 1 from the start
   but not in Table 2 — invisible while no vacuous instance was in the rival set, and live the
   moment that set became each rival's own published one (Recurrent's Table II includes `s526`).
   Both tables now share `zero_search`, `tiny_space` and `VACUOUS_PINS`, and both ledgers sum.
8. **A geomean over fewer than three instances is flagged, not quoted.** On HWMCC, CMSGen scores
   zero on 2–3 of 7 instances depending on depth and target; the ledger columns carry what the
   thinned geomean cannot.
9. **A GPAP/CMSGen ratio without its CPU is meaningless.** CMSGen is single-core, so the ratio
   is as much a property of the host as of the sampler: this campaign's Xeon is 0.41× the
   reference laptop's CPU while GPAP itself moves 0.96× between the cards. `gpap bench` records
   `cpu` in every JSON for this reason — quote it with every ratio.

### The propagation table answers the obvious objection

Sample-rate ratios in Table 1 span two orders of magnitude, which invites "you picked your
benchmarks". The counters answer it: GPAP holds **2.07×10⁹ implications/s** with a spread of
only 1.61× across the 12 family rows, against CMSGen's 3.39×10⁶ (its own `--statsevery`
counters) — a **611× engine advantage that is essentially independent of the benchmark** — and
the identity

```
samples/s ratio = (implications/s ratio) / (implications-per-sample ratio)
```

closes on 12 of 12 family rows; `eval/artifact/verify_artifact.py` re-derives it from the raw
rows rather than trusting the generator. What varies between families is how much propagation a
conflicted trajectory wastes (rejection cost), not the propagator.

### Table 4b — what a Barbarik verdict does and does not say

Parameters are the CMSGen paper's own certification setting (ε=0.3, η=1.8, δ=0.1 ⇒ 1,872
samples per instance). Of 92 generated instances, 68 pass the feasibility screen and **65 are
eligible** — eligibility is bucketed on the **negative control's verdict** (SPUR, an exactly
uniform sampler, must Accept), never on the screen's own estimate. Results: `spur` 65/65 Accept,
`cmsgen` 65/65, `gpap:b8` 63 Accept / 0 Reject / 2 no-verdict, `pipe:circuit` 60 / 0 / 5 —
**zero GPAP Rejects, parity with CMSGen up to coverage**. Three readings matter:

* **The guarantee is asymmetric**: a Reject is unconditional; an Accept means *not caught*
  under Barbarik's non-adversality assumption, never "proven uniform".
* **"No verdict" bounds coverage, not uniformity**: Barbarik's kernel enlarges the sampling set
  while pinning the support — adversarial for a rejection sampler by construction — so an order
  GPAP cannot fill is neither verdict.
* The full control story (including the positive control that must be Rejected) is
  `docs/findings/UNIFORMITY_FINDINGS.md`; `eval/uniformity/barbarik_table.py` is the one
  aggregator and refuses to pool parameter settings.

## Comparability notes

* GPAP is the **2026-09-01 core-sampler engine** (RNG moved into libgpap): same-seed model
  *counts* are not comparable across that boundary; *rates* are. Cross-checking these tables
  against the 20260823/20260827 full-matrix campaigns compares two engines, not two protocols.
* `eval/artifact/make_artifact.sh` (the tar-with-MANIFEST packager) still speaks the
  full-matrix campaign layout; it refuses stale or mismatched arms, so it cannot silently
  package this campaign. Until it learns this layout, the campaign directory itself — with
  `VERIFICATION.txt` and its own `README.md` inside — is the organized artifact of record.

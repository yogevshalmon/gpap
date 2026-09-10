# Artifact-evaluation findings

The evidence behind `ARTIFACT.md` and `eval/artifact/reproduce.sh`. Measured 2026-09-09 against
the frozen campaign `l4_protocol_20260904` (protocol v1.0, `docs/EVAL_RUNBOOK.md` §0). Nothing
here changes the protocol; the artifact layer only chooses which cells a reviewer runs and under
which two documented shortcuts.

## 1. 24 GB is a hard requirement, and the grants file proves it

Counted over `eval/rivals/grants/l4_protocol_20260904.json`:

| tool | cells | pinned | recorded failures | at the card's memory ceiling | granted the ladder top |
|---|---|---|---|---|---|
| DEMOTIC | 22 | 22 | 0 | 0 | 2 |
| Recurrent-CircuitSAT | 120 | 92 | 28 | **92** | 0 |

"At the memory ceiling" means the entry records an OOM one rung above the rung it was granted —
i.e. the reference L4's 24 GB is what stopped it, not the ladder and not the wall. **Every
pinned Recurrent cell is in that state**, so a card with less memory walks down on essentially
all of them, and a walked-down rung hands the rival less parallelism than the paper gave it,
which inflates our ratio. `compare_campaign.py` counts those cells and warns; `grant_source`
records it per row.

DEMOTIC's cells are memory-free by contrast: none sits at a ceiling, and two ran at the ladder
top (which `paper_tables` already names, since a top rung that completes first try means the
ladder bound the grant).

## 2. The other direction: a bigger card silently favours us

63 of the campaign's 141 measured rows carry `engine.K` **below** the 262,144 batch cap — they
are VRAM-bound, not cap-bound. On a larger card those Ks grow while the rival stays pinned to
its recorded batch, so an uncapped reproduction on an A100-80GB compares a grown K against a
fixed grant. Hence `GPAP_VRAM_BUDGET_MIB`, clamped inside `gpap_mem_available` (the one funnel
every consumer already calls, C++ and Python alike) and mirrored in `rival_vram_budget`, with
`reproduce.sh --vram-cap l4` defaulting to the budget the published run had:

    (23.45 GB free - 800 MB reserve) x 0.80 = 18.12 GB = 17,280 MiB

23.45 GB free is what all 142 rows record, to two decimals. The cap is a ceiling only, so it is
inert on smaller cards.

## 3. What a pinned reproduction costs, and what the rows do not time

Per cell, a grants-pinned run pays the rival's one recorded run (`wall_s`), our arm at three
seeds (`n_models / sps_e2e_with_delivery` plus `distinct_s`), and the CMSGen arm
(`cms_wall_s`). Summed over the campaign (`eval/artifact/estimate_wall.py`):

| | recorded failures pinned | re-measured |
|---|---|---|
| full campaign, 142 cells | 9.8 h | 19.1 h |

The 28 recorded failures are 9.3 h of the 19.1 — half a full reproduction spent re-deriving rows
that say "no measurement". That is why the artifact pins them by default and says so in every
row (`not_remeasured`); `--remeasure-failures` is the faithful run.

**The rows do not time everything.** Measured on the four smoke cells (RTX 4080 Laptop, 12 GB,
where two DEMOTIC cells also walked their grant down):

| stage | model | measured | residual |
|---|---|---|---|
| demotic native (c432) | 5.7 s | 26 s | +20 s |
| demotic comb rare (c432) | 5.8 s | 26 s | +20 s |
| recurrent table (b03) | 34.9 s | 49 s | +14 s |
| recurrent rare (b03) | 35.2 s | 47 s | +12 s |

So the unrecorded work is a per-cell **constant**, not a fraction: on the cheapest cells it is
four times the recorded wall. A percentage-only model underestimates exactly the tiers made of
many cheap cells.

Part of that constant scales with the netlist. The ABC simplify plus its equivalence check,
timed on the same box:

| circuit | gates | `prepare_variant` |
|---|---|---|
| b03 | 152 | 0.2 s |
| b13 | 342 | 0.2 s |
| b22 | 29,897 | 7.5 s |
| b17 | 32,192 | 9.5 s |
| b18 | 114,561 | 32.1 s |
| b19 | 231,266 | 71.9 s |

≈ 0.31 ms per gate, linear. This matters because the reviewer tier keeps b17–b19 and b22 —
their rival side is free (a pinned failure) while their preparation is a minute each. The
estimator therefore models the overhead as `per-cell constant + per-gate + fraction`, with
`--cpu-scale` for a slower host (the terms are CPU-bound; the Colab Xeon runs CMSGen at ~0.53x
this box, measured on the smoke cells' `cms_sps`).

Estimates that come out of it, for the reference L4 at `--cpu-scale 1.9`:

| tier | cells | pinned failures | re-measured |
|---|---|---|---|
| smoke | 4 | 4 min | 4 min |
| subset | 66 | 2.7 h | 5.6 h |
| full | 142 | 12.0 h | 22.3 h |

## 4. The reviewer tier's cell set is chosen by the clock

`eval/artifact/subset.json`: a sequential instance is kept when both its T=25 tasks together
cost ten minutes or less on the reference card under the artifact's defaults. That keeps 22 of
30 (dropping b06, b14, b15, b20, b21, s15850.1, s35932, s38584) and all 11 ISCAS-85 under both
tasks: 66 of 142 cells for 23% of the wall. The rule is mechanical and about the clock alone,
which is the point — a hand-picked subset could select for cells we win. The selection is
committed, so an estimate and a run cannot describe different cell sets, and it is regenerated
only by `estimate_wall.py --write-subset`.

**It is a different population, and it does not flatter us.** The published rows restricted to
the subset:

| stage | cells | Table 1 (ss) subset / full | Table 2 subset / full |
|---|---|---|---|
| demotic native | 11/11 | 1,243.7x / 1,243.7x | 2.2x / 2.2x |
| demotic comb rare | 11/11 | 230.4x / 230.4x | ~7.8e5x / ~7.8e5x |
| recurrent table, T=25 | 22/30 | 541.8x / 820.4x | 10.3x / 21.5x |
| recurrent rare, T=25 | 22/30 | 41.1x / 78.0x | 176.8x / 173.2x |

Both DEMOTIC rows are reproduced whole; the two Recurrent rows lose their eight most expensive
instances and read LOWER on three of four numbers. So a reviewer's subset run is compared cell
by cell against these rows, never against the paper's geomeans, which are over all 142 cells.

## 5. The reviewer tier, run for real (2026-09-09/10)

`eval/artifact/reproduce.sh subset` on a rented L4 — the same card the campaign was measured on —
mirrored to `$GPAP_RESULTS/paper-artifact`. 66 cells, 2 h 50 min wall across three VM lifetimes,
`--vram-cap l4` and recorded failures pinned (the artifact's defaults).

**Every cell agreed.** `compare_campaign.py`: PASS on all four stages, 66/66 cells reaching the
same verdict against both CMSGen and the rival, 66/66 rows sound and replayed through the
original netlist, **44 grants honoured at the recorded rung and none walked down**, 8 pinned
failures (b17/b18/b19/b22 under each task) marked `not_remeasured`. The gate over the fresh rows
read 18 ok / 2 warnings / 0 failures — the same two warnings the published campaign carries.

Same card, so the rates land where they should: ×0.98–×1.02 on every arm (GPAP, CMSGen and the
rival), against the ×1.85 CPU factor and ×0.64 handicapped rival the 4080 smoke run showed.
Table by table, against the published rows restricted to the same cells:

| stage | Table 1 (ss) fresh / reference | Table 2 fresh / reference |
|---|---|---|
| demotic native | 1,298.8x / 1,243.7x | 2.2x / 2.2x |
| demotic comb rare | 234.3x / 230.4x | 7.8e5x / 7.8e5x |
| recurrent table, T=25 | 550.6x / 541.8x | 10.2x / 10.3x |
| recurrent rare, T=25 | 42.5x / 41.1x | 178.1x / 176.8x |

Within 1-4% on every row, over the same `n` — a whole re-measurement of the pipeline, not a
re-read of the data.

**And the estimate held.** The model applied to the FRESH rows says 2.42 h; the estimate made
beforehand from the reference rows said 2.44 h — under 1% apart, so the per-cell cost model
transfers between campaigns. The 2 h 50 min actually spent is +17% on that, and the gap is the
rented VM rather than the work: two re-bootstraps and two in-flight cells lost to hourly
reclaims. On a machine that is not taken away, expect the model's number.

## 6. A cross-card reproduction, demonstrated

The smoke tier on an RTX 4080 Laptop (12 GB) against the L4 rows:

* 4/4 cells: same formula, sound, replayable, **same verdicts** — PASS.
* Speed factors: GPAP ×1.06–1.09, CMSGen ×1.79–1.92 (the host CPU, not the GPU), the rival
  ×1.03–1.05 where its grant was honoured and **×0.64 where it was not** — the walked-down
  cells, exactly the handicap §1 predicts for a card under 24 GB.

That is the shape of the evidence a reviewer on other hardware can produce: the verdicts
reproduce, the rates are about their machine, and the one place the comparison could quietly
flatter us is counted and named.

## 7. Tier 1 needs no GPU, and now no CUDA wheels either

`paper_tables.py` and `verify_artifact.py` reached the soundness verdict through
`gpap.sampling.pipeline`, which imports the sampler, torch and numpy — so regenerating a table
in the TACAS evaluation VM would have meant an 800 MB install for a pure JSON reader. The
reader moved to `gpap/verdict.py`, a stdlib-only leaf that `pipeline` re-exports (one
definition, every existing import site unchanged). Gated by
`tests/test_paper_tables.py::test_the_table_generators_import_without_torch_or_numpy`, which
asserts it in a subprocess with those modules blocked — this interpreter has them installed and
would import them happily, so an in-process check would prove nothing.

Regenerated output is byte-identical to the tracked tables apart from the date and commit line
in each header.

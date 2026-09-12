# Artifact-evaluation findings

> **Status:** settled.
> **Measured:** 2026-09-09/10 against the frozen campaign `l4_protocol_20260904` — NVIDIA L4
> (24 GB, Colab) for the reference rows, RTX 4080 Laptop (12 GB) for the smoke tier.
> **Rule lives in:** `GPAP_VRAM_BUDGET_MIB`, clamped once inside `gpap_mem_available`
> (`cpp/src/calibrate.cpp`); `eval/artifact/reproduce.sh --vram-cap`;
> `eval/artifact/compare_campaign.py`; `eval/artifact/subset.json`.
> **Gated by:** `tests/test_k_cap.py`.
> **Record only:** §8, the review post-mortem on gates that could not fail.

The evidence behind `ARTIFACT.md` and `eval/artifact/reproduce.sh` (protocol v1.0,
`docs/EVAL_RUNBOOK.md` §0). Nothing here changes the protocol; the artifact layer only chooses
which cells a reviewer runs and under which two documented shortcuts.

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
fixed grant. Hence `GPAP_VRAM_BUDGET_MIB`, clamped inside `gpap_mem_available` — the one funnel
every consumer already calls, C++ and Python alike — with `reproduce.sh --vram-cap l4`
defaulting to the budget the published run had:

    (23.45 GB free - 800 MB reserve) x 0.80 = 18.12 GB = 17,280 MiB

23.45 GB free is what all 142 rows record, to two decimals. The cap is a ceiling only, so it is
inert on smaller cards.

**And it covers both sides, because there is only one budget now.** `rival_vram_budget` used to
be a second implementation of the same idea — `free x 0.80`, with no driver reserve — which read
17,894 MiB where ours read 17,281 on the very same card. One number could therefore reproduce
our budget or the rival's, never both: capping both at 17,280 handed the rival 614 MiB less than
the campaign had granted it, and a rung that just fit could OOM and walk down, inflating our
ratio in exactly the direction this section is about. It now calls `gpap_mem_available` itself,
so the two sides are the same bytes of the same card by construction
(`tests/test_k_cap.py::test_both_sides_of_a_head_to_head_get_the_same_budget`).

Changing it costs nothing in the record: the modelled budget only ever picks a rung through
`run_at_batch`'s truncation fallback — "`batch`, the rung our VRAM model predicts, decides
nothing", the grant being the largest rung that COMPLETES — and **no cell in the campaign, 0 of
120, ever reached that fallback**. In a grants-pinned reproduction it cannot: the pinned rung is
the top of the ladder.

## 3. What a pinned reproduction costs, and what the rows do not time

Per cell, a grants-pinned run pays the rival's one recorded run (`wall_s`), our arm at three
seeds (`n_models / sps_e2e_with_delivery` plus `distinct_s`), and the CMSGen arm
(`cms_wall_s`). Summed over the campaign (`eval/artifact/estimate_wall.py`):

| | recorded failures pinned | re-measured |
|---|---|---|
| full campaign, 142 cells | 10.5 h | 19.8 h |

The 28 recorded failures are 9.3 h of the 19.8 — half a full reproduction spent re-deriving rows
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
| subset | 66 | 2.9 h | 5.8 h |
| full | 142 | 12.7 h | 23.0 h |

A cell that delivers NO models is not a free cell — it ran its three seeds to the harness's 60 s
per-seed wall and stopped. Pricing those at zero (11 cells of the campaign, 7 of them spending
three full seeds) is what the first version of this table did, and it is why the numbers above
are a little higher than the ones the reviewer tier was originally quoted.

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

**Every cell agreed.** `compare_campaign.py`: PASS on all four stages, **66/66 cells reaching
the same verdict against CMSGen and 58/58 against the rival**, 66/66 rows sound and replayed
through the original netlist, **44 grants honoured at the recorded rung and none walked down**,
8 pinned failures (b17/b18/b19/b22 under each task) marked `not_remeasured`. The gate over the
fresh rows read 18 ok / 2 warnings / 0 failures — the same two warnings the published campaign
carries.

The two denominators are the point. A cell whose rival side was PINNED agrees with the reference
by construction — the fresh row is a verbatim copy — so counting it is counting the copy, and the
comparator now drops those 8 from the rival tally rather than folding them in. Their CMSGen arm
really ran, which is why that side keeps all 66. The first reading of this run said "66/66
against both"; it was 58 measurements and 8 tautologies, disclosed a sentence later but summed
into one number.

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

**And the estimate held, twice over.** At the host's measured CPU factor (`--cpu-scale 1.9`,
from the smoke cells' `cms_sps`) the model reads 2.86 h off the REFERENCE rows and 2.85 h off
the FRESH ones — under 1% apart, so the per-cell cost model transfers between campaigns rather
than describing the run it was fitted to. The 2 h 50 min (2.83 h) actually spent is inside 1% of
both, and that number already contains two VM re-bootstraps and two in-flight cells lost to
hourly reclaims, so the pure work is if anything slightly under the estimate.

An earlier reading of this run put the estimate at 2.44 h and called the 2 h 50 a +17% overrun
attributable to the VM. That was the cost model's own gap, not the VM's: zero-model cells were
being priced at zero (§3), and the reviewer tier contains two of them.

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

## 8. What a review of the artifact layer found (2026-09-10)

Three reviews over the four commits. Nothing moved a published number; what they found was the
artifact layer's own gates being weaker than their prose. The four that mattered:

**A gate that passes what it did not check is worse than no gate.** `compare_campaign.py` could
print PASS over a campaign that had barely run. A deterministic field recorded on one side and
absent on the other was skipped as "not a mismatch"; a `sound_check` of `"off"` — which also
means *requested but never reached* — satisfied the safety tier; and a stage whose fresh file
held zero rows crashed the report instead of failing it. All three now fail, and
`--expect-cells`, which `reproduce.sh` fills in per tier, fails a run that compared fewer cells
than the tier set out to produce. Four deliberately broken fixtures (fields stripped, one cell
per stage, an empty stage, every arm delivering nothing) were used to check the new behaviour;
three now exit non-zero and the fourth reports the flips in its headline instead of claiming
"same verdicts".

**A completion marker must be earned.** `reproduce.sh` does not `set -e`, and printed
`ARTIFACT_TIER_COMPLETE` unconditionally after the tables and the comparison — so a measurement
driver that died at cell 3 produced a "Publishable" verdict, a PASS over whatever had been
written, and a marker the Colab supervisor reads as DONE. Every step now records into a failure
ledger, the marker is conditional, and the ledger reaches the exit code.

**A pinned cell agrees with itself.** A rival failure honoured from the grants file is copied
verbatim into the fresh row, so counting it as agreement counts the copy. The comparator now
drops those cells from the rival tally and reports them separately (§5).

**One environment variable, two parsers.** `GPAP_VRAM_BUDGET_MIB` was read by `strtoll` in C++
(longest numeric PREFIX) and by `int()` in Python (all or nothing). `1e3` therefore capped the
ENGINE to 1 MiB while the rival ran with the whole card, and `1_000` split the other way — the
two-sides-capped-apart failure the cap exists to prevent, reached from the opposite direction. A
huge value overflowed the MiB→bytes shift into a small positive cap that bound while `gpap info`
called it inert. Both sides now share one narrow grammar (blanks, an optional `+`, digits) and
answer *no cap* outside it, pinned by `tests/test_k_cap.py::test_the_cap_grammar_is_one_grammar`,
which asserts against the CORE and not only against the Python mirror.

**Two budgets under one cap.** The rival's budget was a second implementation — the same 0.80
of the same card, minus the driver reserve we take and it did not — so `--vram-cap l4` capped
the two sides 3.4% apart and could not reproduce both. Folded into `gpap_mem_available` (§2).

**And the cap was not in the results cache key.** `gpap_cache_key` hashed the CNF, the build id,
the sampling layer, the ABC fingerprint and the GPU name — so an uncapped run and a capped one
shared cache entries, and `--vram-cap l4` could serve numbers measured with the cap off while
printing the cap in its own provenance. The cap is part of the key now.

**The branch the developer's box takes is not the branch CI takes (2026-09-10).** Folding the
two budgets into `gpap_mem_available` moved the clamp into the core — and left the two no-CUDA
branches behind it uncapped: the rival's 5 GiB fallback, and our own 8 GB CPU placeholder. Both
are placeholders on a machine that cannot run a comparison, but the asymmetry is the one the cap
exists to remove, and the new-in-the-same-commit test could not see it: on any box with a card
it took the CUDA branch and passed. CPU-only CI found it on the first run. Both branches now
read the ceiling through the same reader, and the test forces the fallback
(`monkeypatch.setattr(torch.cuda, "is_available", ...)`) so it fails on either kind of machine.
The general form is the older lesson in one more disguise: a test proves nothing about a branch
it did not take, and "which branch" can be a property of the hardware rather than of the input.

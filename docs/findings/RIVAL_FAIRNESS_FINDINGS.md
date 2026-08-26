# Rival-comparison fairness: what we were measuring, and what we should have been

2026-08-24/25, RTX 4080 Laptop (local calibration) + NVIDIA L4 (the recorded campaign).
Three audits of the GPU head-to-heads, run because a comparison you have not attacked yourself is
one a reviewer will attack for you. All three found something, and **all three had been flattering
GPAP** — which is the expected direction: every unexamined default in a harness we wrote is a
default we chose.

The rules these settled live in `docs/EVAL_RUNBOOK.md` §6 and `docs/BENCHMARKING.md` §4b/§4c;
the numbers live here.

---

## 1. Emission symmetry — is anyone timed while DELIVERING solutions?

GPAP's timed run emits nothing (`collect=False` leaves `keep_rows` False, so the `var_bits`
gather never fires) and CMSGen is timed under the eval patch's `--nowrite` for exactly that
reason — a stock binary writing every sample cost 8.2x of its wall on 27.sk_3_32. The same has to
hold for the GPU rivals, or the ratio is partly a measurement of their PCIe traffic.

Re-derive with `python eval/rivals/check_emission_symmetry.py --results <campaign>`: it AST-scans
the function each rival's own `timer(...)` wraps for host transfers and IO, checks every
`.sample()` in our timed window passes `collect=False`, and audits `cms_mode` over the rows.

| tool | inside its timed region | verdict |
|---|---|---|
| DEMOTIC | `run_back_prop_comb/seq` — pure GPU descent | clean |
| HTS | `run_back_prop` — pure GPU descent | clean |
| CMSGen | `--nowrite` on 737/737 recorded rows | clean |
| GPAP | `collect=False` ⇒ `keep_rows` False | clean |
| Recurrent | `.cpu()` of per-depth candidates, `run.py:186` | **exception, below** |

DEMOTIC and HTS count and dedup their solutions *after* their own timer stops, so their published
metric excludes it entirely. Note the asymmetry that runs against US and is likewise left alone:
our timed loop pays a per-chunk on-device soundness check no rival performs at all.

**The exception, measured** (4080, `--num_steps 10`): Recurrent copies its per-depth candidate
assignments to the host inside its timed loop — `|PI| x sum(d+1 for d in start_point..T-1)`
strided copies, 0.5-4.0 GiB per run.

| instance | batch | metered egress | % of `model_runtime` | unique with copy on/off |
|---|---|---|---|---|
| b03 | 100k | 0.46 s | 0.70% | identical (1,995,117) |
| s386 | 100k | 0.84-0.91 s | 1.32-1.42% | identical (2,134,454) |
| s1196 | 30k | 0.80 s | 0.88% | identical (662,936) |

Removing the copy outright (snapshots kept on device — their `_check_solution` already does
`par.to(self.device)` on each) leaves `num_unique_solutions` bit-identical. On s386, 4 interleaved
repetitions: removing it saves **0.89 s (1.41%)** where a meter around the block reads **0.84 s
(1.32%)** — agreement to 0.05 s inside a ~2 s spread.

⚠️ **One un-repeated on/off pair is worthless at this scale.** The first pairs read -0.23 s,
+3.11 s and +0.06 s against meters of 0.46/0.91/0.80 s, because run-to-run spread IS the effect
size. The meter is the instrument; the A/B calibrates it.

**Decision: not corrected.** One rival, ~1%, and deducting it would RAISE our ratio — so an
uncorrected row under-reports GPAP rather than flattering it, and the rival's rate stays exactly
the one it publishes. The instrumentation was built, validated and reverted; only the bound is
documented.

---

## 2. The parallelism grant — the batch handed to a rival IS its score

Measured on the 4080. `model_runtime` is nearly flat across these, i.e. the GPU is nowhere near
saturated at the small end:

| rival | instance | batch | unique/s | factor |
|---|---|---|---|---|
| HTS | b03-c25-rare | 100 -> 100,000 | 1 -> 294 | **295x** |
| DEMOTIC | c1355 | 1k -> 1M | 1.2k -> 662k | **564x** |
| DEMOTIC | c432 | 1k -> 300k | 2.5k -> 654k | **258x** |
| Recurrent | s386 T=25 | 10k -> 300k | 3.8k -> 69.9k | **18.6x** |

So the batch is the same knob `K` is for us, and GPAP sizes `K` at 0.80 of free VRAM. Two things
were wrong before 2026-08-25, both in our favour: the circuit rivals' budget was a **fixed 5 GiB
whatever the card** (a fifth of a 24 GB L4 while GPAP took ~19 GB of it), and **HTS ran at batch
100/1000**.

**Their own defaults cannot serve as the rule.** All three argparse defaults are `--batch_size
10`, a placeholder; the published demo scripts are per-instance one-liners with a 1000x spread on
comparably tiny circuits — DEMOTIC 1,000 on c17's six gates, Recurrent 1,000,000 on s27's ten,
HTS 100 on a 75-variable CNF. Transplanted onto our instances those hand DEMOTIC ~1/500 and HTS
~1/295 of what their tool actually reaches, inflating our ratios by the same factor. A share of
the machine is the only quantity that means the same thing for both sides.

### Why a static memory model is not enough

Peak VRAM per gate-row is not a constant — measured for DEMOTIC at batch 300k:

| instance | gates | peak GiB | B/gate-row |
|---|---|---|---|
| c432 | 160 | 1.03 | 23.0 |
| c1355 | 546 | 3.05 | 20.0 |
| c2670 | 1193 | 4.88 | 14.6 |
| c3540 | 1669 | 5.40 | 11.6 |

Part of the cost is per-ROW, not per-gate, so any single factor over-charges one end. At 20 B the
model rejects c2670 at batch 1M, which really needs ~16 GiB and fits an L4 fine. The naive
`n_gates * T * batch * 4` estimate undercounts real peak by 2-6x, tool-dependent (Recurrent
s386@300k: 4.44 GiB estimated, 8.36 measured; HTS: 8.4 B per clause-row).

### The rule

Largest batch that fits **0.80 of free VRAM**, then two bounded checks: step DOWN if the driver
refuses the size, and probe ONE rung up, kept only if it is also FASTER. The speed condition is
the spill guard — too big does not always error (Recurrent on s386 runs at 1M and is *slower*
than at 300k). On the L4 campaign all three branches were load-bearing:

| outcome | count |
|---|---|
| up-probe recovered an under-grant | 92 |
| driver refused a rung | 82 |
| bigger rung ran but was SLOWER (spilling) | 18 |

A timeout is NOT a memory refusal and must not walk the ladder: HTS's three tseitin cells stepped
1M -> 1,000 at the full 300 s cap on every rung, ~40 min per cell to record an error they had
after the first run. Two rungs, then record it.

### What it cost us

Full L4 re-run, 2026-08-25/26, 74 rows, 0 unsound, 0 zero-search, pinned commits, 17.5 GiB
budget on every row (0.80 of the L4's free VRAM):

| stage | old grant | new grant | shift |
|---|---|---|---|
| DEMOTIC, their ISCAS-85 pins | 4.9x | **4.6x** | 1.1x lower |
| DEMOTIC, our rare pins (miter) | 6,176.4x | **4,344.1x** | 1.4x lower |
| Recurrent-CircuitSAT, their table | 8.4x | **6.1x** | 1.4x lower |
| Recurrent-CircuitSAT, our rare pins | 1,507.2x | **660.3x** | 2.3x lower |
| HTS, our CNFs | 6,369.5x | **602.9x** | **10.6x lower** |

The correction is concentrated on HTS exactly as the local calibration predicted, because it was
the only rival being run far below its share: per cell its rate rose 10-274x (b10-c25-abc
4 -> 985 unique/s at batch 1,000 -> 300,000). **Three HTS cells also left the zero column**
(b13-abc, s386-abc, s386-tseitin): at batch 1,000 it found no solutions at all and we scored
them as outright wins; at its proper share it finds some, so the geomean now stands on 13 cells
instead of 10. DEMOTIC barely moved — the old fixed budget happened to serve it about right.

⚠️ **Only the rival side moved, and that was checked rather than assumed.** GPAP's own arm
reproduces across the two campaigns to within draw noise — median new/old 0.94-1.00x per stage,
worst single cell 1.29x — so the shift is attributable to the grant, not to drift on our side.
Where our arm does differ it is slightly LOWER, which makes the new ratios marginally
conservative.

---

## 3. Preprocessing — GPAP's was outside the headline rate, CMSGen's was inside its own

`ratio` divides GPAP's steady-state sampling rate by CMSGen's samples-per-subprocess-wall — but
CMSGen's wall already contains ITS parse and setup. Across the 329 timed L4 rows, GPAP's
preparation (parse + construct + ELS + warmup + mining + rebuild) is a **median 20.7% / mean
27.4% of its end-to-end wall** (max 99%):

| phase | median | mean |
|---|---|---|
| construct | 0.15 s | 1.45 s |
| mine | 0.18 s | 0.92 s |
| rebuild | 0.14 s | 2.35 s |
| parse / warmup / recheck | ~0.01 s each | — |

Suite geomean **165.0x steady-state vs 96.7x e2e — the headline was 1.71x the like-for-like
number** (EPFL 2.4-2.8x, sk 2.2x). `ratio_e2e` was already recorded; only the table omitted it.
Table 1 now prints `e2e` and `pre%` beside the steady-state column. Circuit ENCODING is in
neither — it is `t_enc`, an encoder cost, reported per instance.

---

## 4. Making a re-run affordable

Of the 2026-08-23 campaign's ~3.3 h rival stage, the three rivals' own subprocesses were **64
min**. The rest was our side — the GPAP arm (3 seeds x 18 cells for Recurrent alone), the
`start_point` probes, `sim_baseline` — and all of it was re-paid whenever a stage re-ran for a
RIVAL-side reason, which is exactly what fixing §2 required.

Both are cached on disk under `<output dir>/.gpap_cache`: the **GPAP arm** keyed on the sampled
file's BYTES + the task + the engine build + the GPU name, and Recurrent's **start_point probe**
keyed on circuit + depth (it is pure, yet each probe pays a full process launch and model build).
A probe that TIMED OUT is deliberately not cached — that is about the machine, not the circuit.

Two traps found while building it:

- `cached_gpap(..., lambda: None)` as a read-only lookup POISONS the cache: it stores `None`, and
  every later lookup is then a false hit that silently skips the work. Hence `cache_peek` /
  `cache_store` with a `CACHE_MISS` sentinel.
- `colab_supervisor.pull_results` validated EVERY mirrored file as JSON, so the cache tarball was
  downloaded and deleted on every cycle — any non-JSON artifact was silently discarded. It now
  validates by type and still rejects a download caught mid-write, which is what the check was for.

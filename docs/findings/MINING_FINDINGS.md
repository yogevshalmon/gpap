# Entailed-clause mining — measured results

Results and provenance for `gpap.mining` (the API lives in the module; this file holds the
numbers, so the docstrings can stay about behaviour). Measured on an RTX 4080 Laptop unless
stated otherwise.

## The primitive moved into the core (2026-08-18)

The sweep — enumeration, chunking, budget — now lives in `cpp/src/probe.cpp` behind
`gpap_probe_combinations` / `gpap_probe_tuples`, with a tier-1 `gpap::Prober` and a thin Python
face (`gpap.engine.probe`). Python keeps the STRATEGY: which widths, over which pool, the
width-1 fixpoint, and the subsumption filter that drops candidates whose clause is already known.
That split is deliberate — a second copy of a strategy in two languages drifts, which is the same
failure the batch-sizing collapse removed from the K model.

Gated on the mined clause SET, not a count: identical sha1 on s386-c25 (143/18/0), b03-c25
(1007/8/0 at a 200k ceiling, 1007/8/7 at 2M) and b09-c25 (106/0/4), before and after the port.

**`max_tuples` IS A WORK BUDGET, not a refusal.** A width with more candidates than the budget
probes that many and reports itself not exhaustive; every conflict it found is a genuine entailed
clause regardless of what went unprobed, so partial results are kept. Refusing outright is opt-in
(`refuse_over_max`) for a caller that wants a proof or nothing. Measured on s386-c25 at
`--pool all --width 3`: **2,459 clauses in 0.42s out of an 84,059,717,360-candidate space**
(0.000238% of it) — where the same command previously OOM-killed the process, and then, under a
first "refuse over the ceiling" cut, returned nothing at all.

**Three host-side defects the port removed, all measured on plain7824 (N=7,820, M=18,930,
122,289,160 pairs at width 2):**

| | before | after |
|---|---|---|
| `--seconds 5` sweep | 67.95s, **0 probed** | 5.10s, **16,410,368 probed** (13.4%) |
| host memory | 0.91 GiB candidate array | one chunk |
| stats | `n_mined: 0` and nothing else | candidates/probed/coverage/rate per width |

1. *The ceiling was checked against an already-materialized enumeration*, so it could not protect
   against the case it exists for — at `pool=all` width 3 that array is ~1e11 tuples (~2 TB). The
   count is now exact and computed without generating anything (`gpap_probe_count`: a binomial, or
   an elementary-symmetric DP when same-variable tuples are filtered), saturating at INT64_MAX so
   a wrapped value cannot compare below the budget.
2. *A candidate pre-filter (`_drop_subsumed`) cost 65.3s and dropped 0 of 122,289,160 candidates —
   0.0000%.* It could not do better: width 1 runs first and `assume` folds its units into the
   flush, removing those variables from the pool, so what remains to subsume is just the formula's
   existing binary clauses. DELETED.
3. ...but subsumption itself was NOT redundant. `keep()` checked only for an EXACT duplicate,
   which let **1,504 subsumed clauses** through on b03-c25 (1,511 ternaries where 7 are useful):
   a clause implied by a shorter one already present adds nothing and slows every later
   propagation. It is now checked against every subset.

## Subsumed conflicts are filtered IN THE ENGINE (2026-08-19)

**The redundant conflicts are the overwhelming majority, so filtering them on the host is too
late.** At mining's real width-3 state on b03-c25 (original + the 1,015 clauses mined at widths
1-2), a sweep over the support finds **59,160 conflicts of which 59,153 are subsumed and 7 are
useful** — the result buffer holds 8,451x more than anyone wants, and every one of them crosses
the bus before being discarded.

The engine already knows, and for free: **a conflict reached with ZERO implications was caused by
a clause the formula already has.** No implication means the assumption literals alone (plus what
the formula's units had already falsified) emptied that clause, so the clause the tuple would
prove is implied by one already present. The kernel's existing implication counter IS the test —
no extra pass, no clause identity needed. `gpap_probe_options::skip_subsumed`.

MEASURED, same instance and width:

| | conflicts reported | dropped in-engine | result buffer | wall |
|---|---|---|---|---|
| `skip_subsumed=0` | 59,160 | 0 | 693.3 KiB | 0.56s |
| `skip_subsumed=1` | 1,535 | 57,625 | **18.0 KiB** | 0.36s |

**SOUND but INCOMPLETE, and both halves matter.** Cross-tabulated against exact subsumption over
all 59,160 conflicts: `impl==0` has **0 false positives** (it never drops one of the 7 useful
clauses) and catches 57,625 of 59,153 redundant ones (97.4%). The 1,528 it misses are the
order-dependent cases where propagation derived some implication before reaching the dead clause.
So the engine filter is the cheap bulk pass and `keep()`'s subset test is the exact remainder;
neither replaces the other, and mining's OUTPUT is identical with the filter on or off (gated).

**`keep()` is not deduplication.** Mining cannot produce the same clause twice — candidates within
a width are distinct combinations, different widths give different lengths, and width 1's fixpoint
folds each proven unit into the flush, removing that variable from the pool before the next round.
The `clause in known` test is the |C|-sized case of SUBSUMPTION against clauses the formula already
holds (original + everything mined so far), which is why it cannot be dropped.

**Generation is NOT the bottleneck** (measured, because it looked like one): the core's paged
sweep runs 4M tuples in 6.40s against 6.73s for probing the same tuples pre-materialized — i.e.
generation is free next to the probe, and numpy's 22M tuples/s "advantage" was generation-alone
compared against generation+probe. The real cost is that probing n tuples initializes n x M clause
words: each chunk's `reset` broadcasts ~1.2 GiB of packed state, and the full 122M x 18,930 sweep
would be ~9 PB. There is nothing to win by parallelizing or GPU-ifying the enumerator.

**Pool semantics are DIMACS literals**, not variables — so a caller can ask about one polarity
only. `gpap mine --pool` takes `all`, `support`, or an explicit list.


## What it does

Assume a set of literals, propagate, and see whether BCP conflicts. If it does, the formula
entails the negation of that conjunction — a clause. Sweeping every k-subset of a literal pool
therefore yields every entailed clause of width k that BCP alone can see. Every clause is
entailed, so the model set is **exactly** unchanged; the clauses only make later propagation
stronger and prune doomed search earlier.

## Effect on solution sampling (2026-07-03, vs a CMSGen reference)

Mining flipped 7 of 8 losing instances to wins on the "service" family — support variables that
are individually free but jointly constrained through deep cascades, where random completions die
from LATE BCP conflicts:

| instance | before | after |
|---|---|---|
| ActivityService2 | 0.02x | 1.13x |
| IterationService | 0.12x | 2.10x |
| PhaseService | 0.07x | 1.12x |
| UserServiceImpl | 0.32x | 6.68x |
| ProjectService3 | 0.38x | 4.24x |
| ProcessBean | 0.70x | 2.56x |
| c3540 | 0.72x | 1.82x |

Cost was 1–10 s, one-time. `NotificationServiceImpl2` stayed a loss (0.00x → ~0.16x): its joint
constraint is beyond ternary reach, and neither deeper widths nor full lookahead converted it —
both raise success but not throughput.

Restarts were tried first and rejected: retrying tripled success **and** wall. Clauses pay
because each one is proven once and then prunes every one of the millions of later trajectories.

## Width 1 is the lever

The backbone sweep (width 1, iterated to a fixpoint) dominates the win on circuit unrolls —
hundreds of units on every service instance, and units additionally shrink the formula through
the sampler's `simplify`. b03 unrolled 25 cycles: best-config completion **4.5% → 99%** after its
1,007 units. The wide widths are instance-specific: b09-c25 (|sup|=25) gets 4 ternaries worth
1.4x for 0.7 s, s386-c25 (|sup|=175) gets 18 binaries worth 1.48 s for ~2 s, while wide-support
unrolls typically get 0 binaries.

## Candidate generation was the bottleneck, not BCP (2026-08-14)

Measured per tier, before the vectorised rewrite:

| instance | \|sup\| | width-2 candidates | enumeration | GPU probe |
|---|---|---|---|---|
| s386-c25 | 175 | 60,900 | 898 ms | 240 ms |
| b03-c25 | 100 | 19,800 | 280 ms | 96 ms |

The pure-Python `itertools.combinations` × `itertools.product` enumeration cost **2.9–3.7× the
GPU probe it fed**. Replacing it with array operations produced the identical candidate set
**525× faster** (60,900 pairs: 857 ms → 1.6 ms).

End-to-end effect at the production 5 s budget, identical clause sets on all three:

| instance | old wall | new wall | speedup |
|---|---|---|---|
| s386-c25 | 5.01 s (budget-bound) | 1.23 s | 4.07x |
| b03-c25 | 5.01 s (budget-bound) | 0.71 s | 7.06x |
| b09-c25 | 0.68 s | 0.13 s | 5.30x |

Two of the three were previously hitting the wall-clock budget, i.e. the old tier gating was
protecting against a cost that was mostly avoidable. **The `pairs="auto"` thresholds (80,000
candidate pairs / 40,000 triples) were calibrated against the slow enumeration and are now
loose** — they survive as `pipeline.STAGE1_MAX_TUPLES` for continuity, and are worth re-tuning
against probe cost alone.

## Widths must be mined in order, on the strengthened formula

A clause proven at width 2 can make a width-3 assumption conflict that would not have conflicted
without it. Mining width 3 against the *original* formula loses real clauses — measured **4
ternaries instead of 7** on b03-c25. `mine_entailed` therefore rebuilds the probe formula after
any width that proved something (units are folded in place instead, which is cheaper).

## Incremental unit folding

Rebuilding the formula per width-1 iteration dominated wall on large instances: **532 s →
seconds** on an unrolled circuit once `LiteralProbe.assume()` folded new units into the cached
post-flush state instead of rebuilding the occurrence lists.

## Interaction with the ABC encoder

`dc2`/`fraig` already removes the backbone that width 1 mines: **0–45 units on ABC-encoded CNFs
vs 59–1007 on Tseitin ones** for the same 5 s budget. The sampling pipeline therefore disables
mining automatically on CNFs carrying `c encoder abc` provenance (`--mine on` forces it).

## Correctness

Every clause is checked for entailment against an independent SAT solver, and the augmented
formula is checked to have exactly the original model set — on planted formulas and on random
CNFs (`tests/test_mining.py`). The 2026-08-14 rewrite was gated on producing **bit-identical
clause sets** to the previous implementation on s386-c25, b03-c25 and b09-c25.

## Reporting: implications/s, and what a truncated sweep actually covers (2026-08-19)

`MiningResult.implications` / `.implications_per_s` and the CLI's `--stats` now carry the ENGINE's
work, not just the yield. Clauses/s and probes/s both depend on the instance — how many clauses it
entails, how fast a tuple conflicts — while implications/s is hardware work, so it is the number
that compares across instances, GPUs and engine changes. Banked before the probe is rebuilt
between widths, or the rebuild silently drops a width's counters (gated).

`keep()` is SKIPPED at width 1, where it is provably vacuous: a mined unit (u) comes from probing
~u, and ~u was probed only because it was FREE, so neither polarity was settled — had (u) been a
formula unit, or a unit mined in an earlier fixpoint round (those are folded in by `assume`), ~u
would not have been free. Measured overlap between formula units and mined units: 0 on b03-c25,
s386-c25, b09-c25. Worth ~0.1% of wall (0.04-0.28 ms) — recorded for the invariant, not the speed.

⚠️ **A truncated sweep probes the first N candidates in LEXICOGRAPHIC order, which is not a
uniform sample of the space.** It is not as skewed as it first looks — the prefix already touches
100% of the pool's literals, because pool[0] pairs with everything — but pair coverage is
concentrated on the earliest literals. Measured on plain7824, 200,000 of 122,289,160 pairs:
the lex prefix derives **16 implications**, a random sample of the same size derives **90** (5.6x),
both finding 0 conflicts. So a work budget spent on the lex prefix explores less structure than the
same budget spent randomly. Not changed here, because the lex order is what makes the enumeration
resumable and gate-comparable against the numpy reference — but a `--sample random` mode for
truncated sweeps is the obvious follow-up, and it should be measured on an instance where mining
actually finds something before being adopted.

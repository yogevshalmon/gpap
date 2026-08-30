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

## Redundancy: three notions, and we were filtering the wrong one (2026-08-19)

A mined clause can be useless in two quite different ways, and until now only the first was tested:

| | test | removes (b03-c25, widths 1-3) |
|---|---|---|
| **subsumed** | some clause we already have is a SUBSET of it | 21 of 1,508 (1.4%) |
| **derivable** | can propagation do ALL of the clause's forcing jobs? | 942 of 1,551 (61%) |

They barely overlap — of the derivable clauses on b03-c25, **0** are subsumed — so the filter we
had was catching a rounding error and missing the phenomenon.

This is not a new problem. Probing for binary clauses is **hyper-binary resolution**, and
[Heule et al., *Revisiting Hyper Binary Resolution*](https://www.cs.cmu.edu/~mheule/publications/NHBR.pdf)
states it plainly: "most HBR clauses are implied by the binary implication graph," and the
recommendation is to keep only the non-transitive ones. CryptoMiniSat and Lingeling run
**transitive reduction** during probing for exactly this reason.

### A clause does k JOBS, and all of them have to be covered before it can go

⚠️ **The obvious version of the derivable test is wrong, and it was wrong here first.** A clause
`(l1|...|lk)` is not one constraint but k of them: when the other k-1 literals are false it forces
the remaining one, and which job fires depends on the search. The first implementation dropped a
clause as soon as **one** job was covered. It has to be **all** of them.

The two are not close, because propagation is not contrapositive-complete. Base `(-1|5)`, `(-1|6)`,
`(-5|-6|2)` propagates `1 => 2` (both 5 and 6 arrive, the ternary goes unit) but NOT `-2 => -1`:
that step would have to conclude "not both 5 and 6", a disjunction, and unit propagation can only
ever conclude a single literal. So `(-1|2)` has one live direction and one covered one.

Measured, mined at widths 1-3 and judged against input + mined units:

| instance | wide mined | one-job rule | all-jobs rule (shipped) | **live jobs the weak rule discards** |
|---|---|---|---|---|
| b03-c25 | 1,551 | 1,479 (95.4%) | 942 (60.7%) | **537 (34.6%)** |
| b09-c25 | 9,457 | 5,153 (54.5%) | 1,922 (20.3%) | **3,231 (34.2%)** |
| s386-c25 | 50 | 50 (100%) | 50 (100%) | 0 |

A THIRD of the mined wide clauses. **Never a soundness question** — every mined clause is entailed,
so the model set is unchanged whatever is dropped — only propagation strength.

**Why the literature's rule does not transfer.** Transitive reduction runs on the binary implication
graph, where a binary clause IS both of its edges, so the relation is symmetric by construction and
one direction really does imply the other. Our reference is the PROPAGATION relation, which also
carries implications squeezed out of long clauses going unit, and those are one-way. Getting the
aggressive rule right means building the BIG, not reusing the propagation sweep — parked as future
work below.

### ...but DERIVABLE IS NOT USELESS, and that decided the default

A derivable clause adds nothing to what propagation can REACH — but it shortens how long the reach
takes, and GPAP's sampler is propagation-bound. Ablation, pinned config, k-cap 20,000, seed 5,
model counts bit-identical across repeats:

| | no mining | width 1 only | + widths 2,3 |
|---|---|---|---|
| b03-c25 | 120k/s | 485k/s (4.05x) | 625k/s (**+29%**) |
| s386-c25 | 91k/s | 119k/s (1.31x) | 170k/s (**+43%**) |
| b09-c25 | 290k/s | 296k/s (1.02x) | 305k/s (+3%) |

The wide tiers are mostly derivable and still worth 1.03-1.43x. So "logically redundant" and
"useless" are different questions, and only a consumer knows which one it is asking.

**Shipped**: `mine_entailed(drop="none"|"subsumed"|"derivable")` / `gpap mine --drop`, defaulting to
`subsumed` — the strongest filter is available but never the default, because for our own sampler
it is a regression.

### The engine-side filter was deleted

`skip_subsumed` dropped a conflict the assumed literals alone caused. Sound, and it survived a
300-formula fuzz — but it was worth **0.0-0.3%** of the result buffer on every instance measured
(re-measured after an earlier note claiming 97.4% failed to reproduce: b03-c25's support yields 198
width-3 conflicts, not the 59,160 recorded). Against that it cost an int32 per row, a branch in the
innermost loop of both propagate kernels, and a two-paragraph explanation of why the two kernels
saw it differently. Removed 2026-08-19; mined sets bit-identical after (`770cbdc66db7` b03-c25,
`a8a18d8c6357` b09-c25, `bf45d34cdb1a` s386-c25). Redundancy is a property of a CLAUSE against a
FORMULA — it belongs where the formula is known, not in the sweep.

## The closure is the primitive, not the k-subset sweep (2026-08-19)

Probe a literal at width 1 and BCP gives two things: a conflict bit (a FAILED literal, i.e. a unit)
and the CLOSURE — everything that literal implies. The closure is what three applications want:

    failed literals     the conflict bit                              (the backbone; the lever)
    hyper-binary        every `m` in closure(l) IS the clause (~l|m)
    equivalent literals l == m exactly when each is in the other's closure

`gpap/implications.py`. One launch, one row per literal — on a CPU each literal is a separate
propagation, which is why solvers probe only roots of the binary implication graph.

**This also explains the 96%.** A binary `(~a|~b)` found by probing the PAIR is the same clause
width-1 hands you directly when `~b` is in closure(a) — at `P` cost instead of `C(P,2)`. The k-subset
sweep's unique contribution is the residue that genuinely needs the pair (56 of 1,487 on b03-c25).

**Equivalences: strictly stronger than what the sampler ships.** `sampler.py` builds ELS edges from
BINARY CLAUSES only. Closure-based is a superset — verified 0 of 7,856 binary mutual pairs missing
on b03-c25 — and finds equivalences that only exist through long clauses:

| | binary clauses (today) | BCP closures | |
|---|---|---|---|
| b03-c25 | 1,964 vars | 2,598 vars | **1.32x** |
| s386-c25 | 973 vars | 1,120 vars | 1.15x |
| b09-c25 | 1,882 vars | 2,082 vars | 1.11x |

~7 ms CPU against ~250-350 ms GPU, so it only pays when the width-1 sweep is running anyway — which
for mining it always is. NOT yet wired into the sampler; that is a behaviour change needing its own
gate. (An earlier note records 2.81x more equivalences than CaDiCaL's `decompose` — a different
baseline; do not carry that number into this comparison.)

⚠️ The closure is dense, `P x 2N` bits — 11 MB on b03-c25 but ~3.5 GB on a big HWMCC unroll — so
`implications()` refuses a pool it cannot hold rather than reserving it. A chunked reducer form is
the obvious extension.

## The RESET is the bottleneck, not the propagate (2026-08-19)

Measured by stepping one probe chunk through its phases with a device sync between each
(`reset` = the state broadcast, `apply` = BCP to fixpoint, then the flag/counter readout), wave
kernel, RTX 4080 Laptop:

| instance | M | rows/chunk | reset | apply | readout | reset share |
|---|---|---|---|---|---|---|
| b09-c25 | 15,327 | 99,064 | 27.9 ms | 27.4 ms | 0.7 ms | 50% |
| b03-c25 | 14,505 | 104,409 | 22.6 ms | 9.2 ms | 0.6 ms | 70% |
| b12-c50 | 218,857 | 6,988 | 25.1 ms | 4.5 ms | 0.4 ms | 84% |
| 6s215rb0-c25 | 65,873 | 21,806 | 25.5 ms | 1.1 ms | 0.5 ms | **94%** |

**The reset cost is a CONSTANT per chunk, ~26 ms, and it does not depend on the instance.** That
is structural rather than coincidental: the batch is sized as `K ≈ 0.80·free / row_bytes` and
`row_bytes ≈ 4M`, so `rows · 4M ≈ 0.80 · free VRAM` — every chunk broadcasts the whole memory
budget, 5.5–5.8 GiB here, at ~200–250 GB/s. **Probing throughput is therefore bounded by
`budget/(4M) / 26 ms` tuples per second no matter how fast BCP is**, and making the propagate
kernel faster (as the dispatch fix does) only moves the remaining share.

### What a chunk actually modifies

A probe assigns the k assumed literals plus whatever propagates, and touches one clause word per
occurrence of each. Measured over 20,000 rows, against the M words the broadcast writes:

| instance | trail/row (k=2) | words touched | vs M | crossover (M / mean occ) |
|---|---|---|---|---|
| b03-c25 | 127 | 359 | 2.5% | 5,120 |
| b09-c25 | 454 | 1,293 | 8.4% | 5,376 |
| b12-c50 | 737 | 2,143 | 1.0% | 75,303 |
| 6s215rb0-c25 | 6.6 | 16 | **0.02%** | 26,536 |

So the broadcast restores 40x–4,000x more state than the chunk dirtied.

### The design that follows: restore by TRAIL, not by broadcast

Undoing the arithmetic (add back `1 + (neg << SHIFT)` per falsified occurrence) is exact — the
packed word is an abelian group, so order does not matter and `atomic_add` handles the races the
same way propagate does. But it needs to know exactly WHICH subtractions happened, and on a
CONFLICTED row it cannot: both kernels break out mid-occurrence-list, and wave breaks mid-wave, so
the stop position is a (literal, offset) pair the kernel does not record.

**Rewriting the baseline value is strictly better than undoing it.** The set of clause words a row
can have touched is exactly `∪ occ[l]` over the row's trail — and the trail is already recorded,
with its length already in `nprops`. So:

    for each row, for each trail literal l, for each c in occ[l]:  packed[row][c] = packed_flushed[c]

- **No new kernel state.** `nprops` and the trail buffer already exist.
- **Superset-safe.** It covers the partially-subtracted literal a conflicted row stopped inside,
  so conflicted rows need no special case — which is what killed the undo variant.
- **Idempotent**, so clauses shared between two trail literals are written twice harmlessly: plain
  stores, no atomics, no dedup.
- The assignment bitmap restores the same way (one word per trail literal); flags are a 2-byte fill.

Preconditions, all checkable in the engine: the rows must be the ones the last `reset` sized (grow
⇒ broadcast the new rows), the baseline must not have moved (`assume_units` / `recache_flush` ⇒
broadcast), and **exactly one `apply` may have run since the reset** — the kernel rewrites the
trail from index 0, so a second apply loses the first one's record. The probe path is exactly
reset → apply → read.

### Expected win, bounded by measurements already in hand

`apply` walks the same occurrence lists as the restore would, and does strictly more (dependent
fixpoint rounds, unit/conflict detection, atomics). So the measured `apply` is an UPPER bound on
the restore, giving a floor on the win: **b03-c25 ≥1.7x, b12-c50 ≥3.3x, 6s215rb0-c25 ≥10.3x,
b09-c25 ~1.0x.**

⚠️ **REGIME-DEPENDENT, and the boundary is not the 1.0 you would guess.** The traffic ratio is
`mean_trail · mean_occ / M`, but the restore's scattered 4-byte stores pull 32-byte sectors while
the broadcast streams, so up to 8x amplification applies to one side only — treat the crossover as
~1/8, not 1. Probing sits at 0.0002–0.084 (win); a SAMPLING trajectory assigns 28–56% of all
variables and sits at **0.32–0.62** (loss). It must therefore be a decision the engine makes from
the trail length, not an unconditional change to `gpap_engine_reset` — and any implementation has
to be gated the way every K change is, since it alters what a chunk costs.

### Alternatives weighed

- **Smaller chunks**: no. Reset and apply both scale linearly in rows, so the ratio is invariant.
- **Overlapping reset with apply on two streams**: bounded by ~1.3x even in theory, and both
  phases are memory-bound so they contend. Strictly worse than not doing the work.
- **Fusing the broadcast into the BCP kernel**: saves nothing. The state is K·M and cannot live in
  cache; it has to reach DRAM either way.
- **Copy-on-write row state** (a shared baseline plus a small per-row delta) is the asymptotically
  right answer — it removes the K·M materialization entirely, and would also lift K, which is
  currently `budget/4M` purely because of state nobody reads. In-repo prior art (`SparseBCPEngine`,
  see INSIGHTS) measured up to 31x on exactly this low-touch regime. Much larger change; the
  trail-restore gets most of it for a fraction of the risk.

> **STATUS: SHIPPED 2026-08-21, as the FUSED variant — CLEAN mode.** The restore did not end up
> a separate pass at the next reset: `gpap_engine_apply_clean` runs the identical fixpoint and
> then rewrites its own touched words back to the baseline INSIDE the same launch (trail still
> in registers, touched words still in L2), and `gpap_engine_reset` skips the K·M broadcast when
> the rows are provably clean ("reset-lite"). Probing/mining and flags-only checking ride it;
> plain apply / trajectory / recache keep full semantics and full broadcasts; `GPAP_CLEAN=0`
> restores exact legacy behaviour. Design + soundness lemmas + gates + first measurements
> (b03-c25 e2e 2.60x; b12-c50 GPU-side 20.9 -> 4.7 ms per chunk = 4.4x; the quiet-box protocol
> is still owed): `docs/plans/RESET_RESTORE_PLAN.md` §9. The trail-length and touched-words
> analysis above is what sized it; the conflicted-row analysis below is why it rewrites the
> baseline instead of undoing arithmetic.

## The width-2 ceiling was calibrated against the OLD reset cost (2026-08-21)

`STAGE1_MAX_TUPLES[2] = 80,000` dates from when every probe chunk paid a full K·M state
broadcast. With that gone (CLEAN mode, §"The RESET is the bottleneck"), the ceiling — not the
clock — is what truncates the production sweep, and it truncates precisely the wide-support
instances. Support-literal pool, `MINE_SECONDS = 5.0`, RTX 4080 Laptop:

| instance | \|sup\| | width-2 candidates | 80k ceiling | uncapped | verdict |
|---|---|---|---|---|---|
| b03-c25 | 100 | 19,900 | 933 cl, 0.41s, exhaustive | 933 cl, 0.19s | ceiling never bound |
| s386-c25 | 175 | 61,075 | 42 cl, 0.24s, exhaustive | 42 cl, 0.24s | ceiling never bound |
| b04-c25 | 275 | 150,975 | 260 cl, 0.46s, **truncated** | **262 cl, 0.40s, exhaustive** | +2 clauses, FASTER |
| b10-c25 | 275 | 150,975 | 59 cl, 0.37s, **truncated** | 59 cl, 0.37s, exhaustive | proof, same clauses |
| s1196-c25 | 350 | 244,650 | 0 cl, 0.38s, **truncated** | 0 cl, 0.40s, exhaustive | proof, same clauses |

Uncapping is free — it never costs more than 0.02 s and sometimes costs LESS (the capped run
still pays the same per-chunk overheads for a partial answer). What it buys is mostly the
`exhausted` flag: "no width-2 entailed clause over this support exists that BCP can see" is a
PROOF, where a truncated sweep licenses nothing and, per the lex-prefix note below, is not even
an unbiased sample. Raised to **1,000,000** (exhaustive to \|sup\| ≈ 700); past that
`MINE_SECONDS` is the real bound, which is the correct behaviour.

### Validated e2e on an L4 (2026-08-21)

Full seq family, T=25, target rare, tseitin, n_time 50k / 60 s cap, circuit portfolio, seed 5 —
NEW (CLEAN mode + the 1M ceiling) against OLD (`GPAP_CLEAN=0` + the 80k ceiling) **on the same
VM, same session**, 19 instances each, 0 errors and 0 unsound in both arms. Data:
`$GPAP_RESULTS/clean_mode_l4_20260821/`.

**16/19 instances have BIT-IDENTICAL model counts**, and all three that differ are exactly the
ones whose mined clause SET changed — i.e. CLEAN mode on its own moved nothing, which is what the
state-identity gate promises. Where the ceiling bound, the tune is a clear win:

| instance | mined u/b/t NEW → OLD | M_eff | K | succ | models |
|---|---|---|---|---|---|
| s832-c25 | 211/3/46 → 202/2/**4457** | **23,554 → 28,039** | **146,910 → 128,108** | 0.0102 → 0.0067 | +25 (**1.75× throughput**) |
| s1196-c25 | 381/**10**/0 → 381/**4**/0 | 29,846 → 29,840 | ~equal | **0.818 → 0.782** | **+14.8%** |
| b14-c25 | **917**/0/0 → 65/0/0 | 525,066 → 528,422 | 5,576 → 5,539 | 0.0007 → 0.0006 | +12 ⚠ bandit picked B=1 vs B=8, confounded |

⚠️ **s832 is the instructive one, and it runs against the intuition that a bigger ceiling means a
bigger formula.** Exhausting width 2 (404,550 candidates, well inside the new ceiling) proves a
few good binaries and leaves less budget for width 3; the OLD arm truncated width 2 at 80k, spent
the remainder on width 3, and kept **4,457 ternaries** that inflated the effective formula. Since
`K ≈ budget/4M`, the smaller mined formula bought a **15% larger batch** and 1.75× the
throughput. Cheaper mining and a better result: the ceiling was not trading cost for quality,
it was buying the wrong width.

Whole-family steady-state geomean NEW/OLD **1.064×** (14 instances, vacuous pins excluded) — most
rows sit at 1.02-1.05× because mining is a small share of e2e wall on an instance the ceiling
never bound.

⚠️ **The tempting next step is WRONG: do not widen the POOL.** Mining every free literal instead
of the support finds 100-1000× more clauses — b03-c25 131,380 vs 933, s386-c25 289,933 vs 42 —
all sound and all entailed. Sampling the augmented formula is a catastrophe, because `K ≈
budget/4M` and M grows with them:

| instance | M before → after | e2e sampling |
|---|---|---|
| b03-c25 | 14,502 → 146,326 | **292,343 → 502 models/s (582× SLOWER)** |
| s386-c25 | 12,652 → 304,322 | 144,755 models/s → **zero models** |

The keep-if-it-helps gate rejects them, correctly. Width 3 over the support pool is also poor
value (+4 clauses on b09-c25, nothing elsewhere, up to 4.9 s). The support pool at widths 1-2 is
the right ask; only its ceiling was stale.

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
loose** — they survive as `pipeline.STAGE1_MAX_TUPLES`. RE-TUNED 2026-08-21 after the reset cost
collapsed: width 2 raised to 1,000,000, width 3 left alone (it earns nothing on the support
pool). See "The width-2 ceiling was calibrated against the OLD reset cost" below.

## Widths must be mined in order, on the strengthened formula

A clause proven at width 2 can make a width-3 assumption conflict that would not have conflicted
without it. Mining width 3 against the *original* formula loses real clauses — measured **4
ternaries instead of 7** on b03-c25. `mine_entailed` therefore rebuilds the probe formula after
any width that proved something (units are folded in place instead, which is cheaper).

## Incremental unit folding

Rebuilding the formula per width-1 iteration dominated wall on large instances: **532 s →
seconds** on an unrolled circuit once `LiteralProbe.assume()` folded new units into the cached
post-flush state instead of rebuilding the occurrence lists.

## Interaction with the ABC encoder — the gate was REMOVED (2026-08-19)

`dc2`/`fraig` really does remove most of the backbone width 1 mines: **0–45 units on ABC-encoded
CNFs vs 59–1007 on Tseitin ones** for the same budget. From 2026-07-10 the pipeline read that as a
reason to disable mining automatically on CNFs carrying `c encoder abc` provenance.

**That was the wrong inference, and the unit COUNT was the wrong proxy — it never measured the
sampling rate.** Measured over 24 instances x 2 encoders, mining is worth MORE on ABC-encoded CNFs
than on Tseitin ones: **geomean 3.28x vs 2.05x** (s386-c25 14.3x, s832-c25 5.5x, b10-c25 4.0x).
WHICH units are found matters more than how many. The gate is gone; `MINE_SECONDS` bounds the cost
and the keep-if-it-helps guard below bounds the risk.

A second gate went with it: `MINE_MIN_CLAUSES` was an inline **3000**, justified by "even 0.3 s of
probing would dominate the e2e wall" — which is not what mining costs down there (0.01–0.04 s
measured). Because ABC-encoded CNFs are SMALL by construction, that threshold excluded precisely
the instances ABC produces. At **500** it is worth 2.68x on b03-c25/abc (M=2,087) and 1.65x on
b09-c25/abc (M=2,882).

## Mined clauses are kept only if they PAY (2026-08-20)

Mining is sound but not free: every clause enlarges the formula, and the batch is sized as
`K ~ budget/row_bytes` with `row_bytes ~ 4M`, so **a bigger M buys a smaller K**. On an instance
that already completes trajectories there is no success left to win and the batch loss is all that
happens — measured on b06-c25/abc, where 5 mined units cost **0.55x** end-to-end.

Since `K x M` is roughly the memory budget however big the formula is, a chunk costs about the same
wall before and after mining, and the thing that moves is **models per chunk = K x success**. Both
terms are already in hand (the batch from the rebuilt sampler, the completion rate from the warmup
the pipeline already runs), so the comparison costs no extra GPU work. If it does not beat the
pre-mining sampler, the clauses are dropped and the original is rebuilt.

⚠️ It is a PROXY, not a measurement: the warmup pins `K <= 1024`, so its success rate is read at a
batch the timed run will not use, and holding chunk wall constant ignores that a longer trail costs
more per row. The honest alternative is a second timed run, which costs more than the clauses are
worth on exactly the instances where this fires. One deliberate asymmetry: when the un-mined
sampler completes NOTHING in the warmup, the clauses are kept regardless — there is no baseline to
be worse than, and mining is the only thing that has ever converted those (b12-c25/abc: 516 ->
20,031 models).

Since 2026-08-27 mined width-2 clauses are **K-free** — binary clauses are implication edges with
no per-row state ([BINARY_OCC_FINDINGS.md](BINARY_OCC_FINDINGS.md)) — so the proxy naturally keeps
more of them; it retunes itself, because `smp.K` on the rebuilt sampler already charges only
`num_long` (measured: 6s215rb0's gain rose 1.43 → 2.44 and texasparsesysp3 flipped from revert to
keep, with no constant touched).

## FUTURE WORK — what is deliberately not built (2026-08-20)

Mining is now a bounded few seconds of a sampling run, sound, and self-limiting. Everything below
was investigated far enough to size it and then parked, so the shipping path stays simple.

| | what | why parked |
|---|---|---|
| **Transitive reduction** | the real HBR filter, over the binary implication graph | needs the BIG as a separate structure; the propagation relation cannot answer it (see the k-JOBS section). Would make `implications.hyper_binary` usable as a preprocessor output rather than an analysis product |
| **Aggressive `derivable`** | one covered direction is enough | same reason — correct only on a symmetric relation. Ours is conservative: 1.6-2.7x fewer clauses dropped, never unsound |
| ~~**Trail-superset restore**~~ | kill the per-chunk state broadcast (41-94% of probe wall) | **SHIPPED 2026-08-21 as CLEAN mode**, fused into the probe launch (`docs/plans/RESET_RESTORE_PLAN.md` §9). The SAMPLER stays on the broadcast: it sits on the losing side of the crossover |
| **Chunked closure** | sweep a pool larger than VRAM instead of refusing it | nothing in the pipeline hits the limit; `implications` refuses loudly |
| **Closure-based ELS** | strictly stronger equivalences than the sampler's binary-clause edges (1.11-1.32x more vars) | a behaviour change to the shipping sampler, so it needs its own uniformity gate |
| **`hyper_binary` size guard** | the output is ~1000x the formula (b03-c25: 14M clauses vs M=14,505) | documented instead; it is an analysis product and nothing ships on it |

The Python mining/probing path also has not had a line-by-line review since the C++ port — noted
by the user, not yet done.

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

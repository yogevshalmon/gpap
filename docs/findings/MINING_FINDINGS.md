# Entailed-clause mining — measured results

> **Status:** settled — shipped 2026-08-19/21.
> **Measured:** 2026-08-19 to 2026-08-21, RTX 4080 Laptop unless a section says otherwise (one
> L4 validation run).
> **Rule lives in:** `gpap/mining.py`, `gpap/sampling/pipeline.py` (`STAGE1_MAX_TUPLES`,
> `MINE_MIN_CLAUSES`, `MINE_SECONDS`), `cpp/src/probe.cpp`.
> **Gated by:** `tests/test_mining.py`, including `test_mining_is_kernel_invariant`.
> **Record only:** the "follow-ups, deliberately not built" table.

Results and provenance for `gpap.mining` (`gpap mine`, and the mining stage of the default
sampling pipeline). CLEAN mode, which removed the probe path's reset bottleneck, has its own
record: [CLEAN_MODE_FINDINGS.md](CLEAN_MODE_FINDINGS.md).

## What it does

Assume a set of literals, propagate, and see whether BCP conflicts. If it does, the formula
entails the negation of that conjunction — a clause. Sweeping every k-subset of a literal pool
yields every entailed clause of width k that BCP alone can see. Every clause is entailed, so the
model set is exactly unchanged; the clauses only make later propagation stronger and prune doomed
trajectories earlier.

Correctness: every mined clause is checked for entailment against an independent SAT solver and
the augmented formula is checked to have exactly the original model set, on planted formulas and
random CNFs (`tests/test_mining.py`). Every rewrite of the sweep was gated on bit-identical
mined-clause sets on s386-c25, b03-c25 and b09-c25.

## The sweep lives in the core (2026-08-18)

Enumeration, chunking and the budget live in `cpp/src/probe.cpp` (`gpap_probe_combinations` /
`gpap_probe_tuples`, tier-1 `gpap::Prober`, Python face `gpap.engine.probe`). Python keeps the
strategy: widths, pool, the width-1 fixpoint and the host-side redundancy filter. One copy of the
strategy, because two drift. Gated on identical sha1 of the mined set before and after the port
(s386-c25 143/18/0; b03-c25 1007/8/0 at a 200k ceiling, 1007/8/7 at 2M; b09-c25 106/0/4).

The port removed three host-side defects, measured on plain7824 (N=7,820, M=18,930, 122,289,160
pairs at width 2): a `--seconds 5` sweep went from 67.95 s with 0 probed to 5.10 s with
16,410,368 probed (13.4%); a 0.91 GiB candidate array became one chunk; stats now report
candidates/probed/coverage/rate per width. The candidate count is computed exactly without
generating anything (`gpap_probe_count`: a binomial, or an elementary-symmetric DP when
same-variable tuples are filtered, saturating at INT64_MAX) — at `pool=all` width 3 the
materialized array is ~1e11 tuples (~2 TB) and the process was OOM-killed reaching the old
comparison. A candidate pre-filter that cost 65.3 s and dropped 0 of 122,289,160 candidates was
deleted: width 1 runs first and folds its units into the flush, so nothing is left to
pre-subsume. Subsumption of the result is not redundant: an exact-duplicate check let 1,504
subsumed clauses through on b03-c25; it is now checked against every subset.

## Budgets: `max_tuples` is a work budget

A width with more candidates than `max_tuples` probes that many and reports `exhausted=False`;
every conflict found is entailed regardless, so partial results are kept (`refuse_over_max` opts
into all-or-nothing). s386-c25 at `--pool all --width 3`: 2,459 clauses in 0.42 s out of an
84,059,717,360-candidate space, where a first "refuse over the ceiling" cut returned nothing.

Rules from the width-1 budget (details in [CLI_DELIVERY_FINDINGS.md](CLI_DELIVERY_FINDINGS.md)):

- The budget binds at width 1 too, spent across that width's fixpoint rounds. A per-width dict
  must give width 1 an explicit `0` (`pipeline.STAGE1_MAX_TUPLES`): the fallback is
  `DEFAULT_MAX_TUPLES`, which silently capped the backbone on the default sampling path.
- Width 1's `candidates`/`probed` are sums over `rounds[1]` fixpoint rounds (s1196-c25:
  35,298 + 35,216 = 70,514) and are not comparable to the pool size; proving convergence always
  costs one final round that finds nothing. Never sum candidates across widths.
- The non-exhaustive reason comes from the probe's own `stopped`: width 1 is as often cut by the
  deadline as by the ceiling.

A truncated sweep probes a lexicographic prefix, which is not a uniform sample: on plain7824,
200,000 of 122,289,160 pairs derived 16 implications as the lex prefix and 90 as a random sample
of the same size. Kept lexicographic because it is resumable and gate-comparable against the
numpy reference; a `--sample random` mode is the follow-up.

`keep()` is skipped at width 1, where it is provably vacuous: a mined unit (u) comes from probing
~u, which was probed only because it was free. Measured overlap between formula units and mined
units: 0 on b03-c25, s386-c25, b09-c25.

`MiningResult.implications` / `.implications_per_s` carry the engine's work — the one number that
compares across instances, GPUs and engine changes — banked before the probe is rebuilt between
widths.

## Redundancy: three notions (2026-08-19)

`mine_entailed(drop="none"|"subsumed"|"derivable")` / `gpap mine --drop`, each stronger than the
last, default `subsumed`:

| | test | removes (b03-c25, widths 1-3) |
|---|---|---|
| subsumed | some clause already present is a subset of it | 21 of 1,508 (1.4%) |
| derivable | propagation covers all of the clause's forcing jobs | 942 of 1,551 (61%) |

They barely overlap: 0 of the derivable clauses on b03-c25 are subsumed. Probing for binary
clauses is hyper-binary resolution, and
[Heule et al., *Revisiting Hyper Binary Resolution*](https://www.cs.cmu.edu/~mheule/publications/NHBR.pdf)
states that most HBR clauses are implied by the binary implication graph; CryptoMiniSat and
Lingeling run transitive reduction during probing for that reason.

### A clause does k jobs, and all of them must be covered before it can go

`(l1|...|lk)` is k constraints: when the other k-1 literals are false it forces the remaining one.
Dropping on one covered job is wrong because propagation is not contrapositive-complete: base
`(-1|5)`, `(-1|6)`, `(-5|-6|2)` propagates `1 => 2` but not `-2 => -1` (that step would conclude a
disjunction). Mined at widths 1-3, judged against input + mined units:

| instance | wide mined | one-job rule | all-jobs rule (shipped) | live jobs the weak rule discards |
|---|---|---|---|---|
| b03-c25 | 1,551 | 1,479 (95.4%) | 942 (60.7%) | 537 (34.6%) |
| b09-c25 | 9,457 | 5,153 (54.5%) | 1,922 (20.3%) | 3,231 (34.2%) |
| s386-c25 | 50 | 50 (100%) | 50 (100%) | 0 |

A third of the mined wide clauses. Never a soundness question, only propagation strength. The
literature's rule does not transfer: transitive reduction runs on the binary implication graph,
which is symmetric by construction, while our reference is the propagation relation, which also
carries one-way implications squeezed out of long clauses going unit.

### Derivable is not useless, and that decided the default

A derivable clause adds nothing to what propagation can reach but shortens how long the reach
takes, and the sampler is propagation-bound. Ablation, pinned config, k-cap 20,000, seed 5, model
counts bit-identical across repeats:

| | no mining | width 1 only | + widths 2,3 |
|---|---|---|---|
| b03-c25 | 120k/s | 485k/s (4.05x) | 625k/s (+29%) |
| s386-c25 | 91k/s | 119k/s (1.31x) | 170k/s (+43%) |
| b09-c25 | 290k/s | 296k/s (1.02x) | 305k/s (+3%) |

The wide tiers are mostly derivable and still worth 1.03-1.43x, so the strongest filter is
available but never the default. Drop `derivable` for a consumer that wants a minimal formula (a
CDCL preprocessor); keep it for one that wants speed.

### The reference formula is the baseline-reduced one

`_Subsumption` tests against the formula reduced by the engine's baseline, not the input clauses:
mining folds proven units into the flush, so an input (a|b|c) whose c the baseline falsified is a
binary constraint during the sweep and proves a clause no input clause subsumes (17 of them on
b03-c25). Testing against the raw input made the output kernel-dependent (3,096 clauses under
scalar vs 3,113 under wave); `test_mining_is_kernel_invariant` gates it.

### No filter in the engine

`skip_subsumed` (drop a conflict the assumed literals alone caused) was sound but worth 0.0-0.3%
of the result buffer while costing an int32 per row and a branch in both propagate loops.
Removed 2026-08-19; mined sets bit-identical after. Redundancy is a property of a clause against
a formula, so it belongs where the formula is known.

## The closure is the primitive

Probing a literal at width 1 gives a conflict bit (a failed literal) and the closure, everything
it implies. Failed literals are the backbone; every m in closure(l) is the clause (~l|m); l ≡ m
exactly when each is in the other's closure. One launch, one row per literal. The k-subset
sweep's unique contribution is the residue that genuinely needs the pair (56 of 1,487 binaries on
b03-c25). Closure-based equivalences are a strict superset of the sampler's binary-clause ELS
edges (b03-c25 2,598 vs 1,964 vars, 1.32x; s386-c25 1.15x; b09-c25 1.11x) — not wired in, since
it is a sampler behaviour change needing its own uniformity gate. The closure is dense, one byte
per (pool literal, literal) pair (82.5 MiB on b03-c25); anything rebuilding it must check the
size first. The module that packaged this (`gpap/implications.py`) was removed 2026-09-01,
unused; the `derivable` level's `_derivable_mask` moved into `gpap/mining.py`.

## Effect on sampling

- Service family (2026-07-03, vs a CMSGen reference): mining flipped 7 of 8 losing instances to
  wins (UserServiceImpl 0.32x → 6.68x, ProjectService3 0.38x → 4.24x), at 1–10 s one-time.
  NotificationServiceImpl2 stayed a loss: its joint constraint is beyond ternary reach. Restarts
  were tried first and rejected (retrying tripled success and wall).
- Width 1 is the lever: b03-c25 best-config completion 4.5% → 99% after its 1,007 units. Wide
  widths are instance-specific (b09-c25 gets 4 ternaries worth 1.4x; s386-c25 18 binaries).
- Widths must be mined in order on the strengthened formula: width 3 against the original formula
  found 4 ternaries instead of 7 on b03-c25. `mine_entailed` rebuilds after any width that proved
  something; units are folded in place (`LiteralProbe.assume()`), which took one large unroll
  from 532 s to seconds.
- Candidate generation was a bottleneck (2026-08-14): pure-Python enumeration cost 2.9–3.7x the
  GPU probe it fed; vectorised it is 525x faster (60,900 pairs: 857 ms → 1.6 ms), e2e 4.07x /
  7.06x / 5.30x on s386 / b03 / b09 with identical clause sets.

### Mining is worth more on ABC CNFs than on Tseitin (2026-08-19)

`dc2`/`fraig` removes most of the backbone: 0–45 units on ABC-encoded CNFs vs 59–1007 on Tseitin
for the same budget, which the pipeline once read as a reason to disable mining on ABC
provenance. The unit count is the wrong proxy: over 24 instances x 2 encoders, mining is worth
geomean 3.28x on ABC CNFs vs 2.05x on Tseitin (s386-c25 14.3x, s832-c25 5.5x, b10-c25 4.0x). The
gate is gone, and `MINE_MIN_CLAUSES` moved from 3000 to 500 (mining costs 0.01–0.04 s there; at
500 it is worth 2.68x on b03-c25/abc, M=2,087, and 1.65x on b09-c25/abc, M=2,882).

### Keep-if-it-helps (2026-08-20)

Every clause of width ≥ 3 enlarges `num_long`, and `K ~ budget/4M_long`, so a bigger formula buys
a smaller batch (5 mined units cost 0.55x on b06-c25/abc). `K x M` is roughly constant, so chunk
wall is roughly constant and models per chunk = `K x success` is the throughput proxy; both terms
are in hand from the rebuilt sampler and the warmup. If it does not beat the pre-mining sampler
the clauses are dropped and the original rebuilt. It is a proxy (the warmup pins K <= 1024). When
the un-mined warmup completed nothing the clauses are kept regardless: mining is the only thing
that has converted those (b12-c25/abc 516 → 20,031 models). Mined width-2 clauses are K-free
since binary clauses became implication edges ([BINARY_OCC_FINDINGS.md](BINARY_OCC_FINDINGS.md)),
and the proxy retunes itself: 6s215rb0's gain rose 1.43 → 2.44 and texasparsesysp3 flipped from
revert to keep with no constant touched.

### Do not widen the pool

Mining every free literal instead of the support finds 100-1000x more entailed clauses (b03-c25
131,380 vs 933; s386-c25 289,933 vs 42), and sampling the augmented formula is a catastrophe:
b03-c25 M 14,502 → 146,326, 292,343 → 502 models/s (582x slower); s386-c25 M 12,652 → 304,322,
zero models. The keep-if-it-helps gate rejects them. Width 3 over the support pool is also poor
value (+4 clauses on b09-c25, up to 4.9 s). The support pool at widths 1-2 is the right ask.

## The reset was the bottleneck (2026-08-19)

Stepping one probe chunk through its phases: the state broadcast cost a constant ~26 ms per chunk
(50–94% of probe wall on b09-c25 / b03-c25 / b12-c50 / 6s215rb0-c25), because `rows · 4M ≈ 0.80 ·
free VRAM` is rewritten every chunk while a probe modifies 0.02–8.4% of those words. The fix is
CLEAN mode (shipped 2026-08-21): the launch restores its own touched words from the baseline and
`reset` skips the broadcast when rows are provably clean. Design, lemmas, gates and measurements
(b03-c25 e2e 2.60x; b12-c50 chunk 20.9 → 4.7 ms): [CLEAN_MODE_FINDINGS.md](CLEAN_MODE_FINDINGS.md).
The sampler stays on the broadcast: a trajectory assigns 28–56% of all variables and sits on the
losing side of the crossover.

## The width-2 ceiling (2026-08-21)

`STAGE1_MAX_TUPLES[2] = 80,000` was calibrated against the old reset cost. It truncated exactly
the wide-support instances (b04/b10 150,975 candidates, s1196 244,650) for no saving: uncapped
never costs more than 0.02 s extra and sometimes less, and buys the `exhausted` proof. Raised to
1,000,000 (exhaustive to |sup| ≈ 700); past that `MINE_SECONDS` is the bound.

Validated on an L4 (seq family, T=25, rare, tseitin, 19 instances, new vs `GPAP_CLEAN=0` + the
80k ceiling on the same VM; `$GPAP_RESULTS/clean_mode_l4_20260821/`): 16/19 bit-identical model
counts, the three that differ being exactly those whose mined set changed; family geomean
1.064x. s832-c25 is the instructive one: exhausting width 2 proved a few good binaries and left
less budget for width 3, where the old arm had kept 4,457 ternaries; the smaller formula bought
a 15% larger batch and 1.75x throughput. A bigger ceiling does not mean a bigger formula.

## Follow-ups (deliberately not built)

| | why parked |
|---|---|
| Transitive reduction (the real HBR filter) | needs the binary implication graph as a separate structure; the propagation relation cannot answer it |
| Aggressive `derivable` (one covered direction) | correct only on a symmetric relation; ours drops 1.6-2.7x fewer clauses, never unsound |
| Trail-superset restore | shipped 2026-08-21 as CLEAN mode ([CLEAN_MODE_FINDINGS.md](CLEAN_MODE_FINDINGS.md)); the sampler stays on the broadcast |
| Chunked closure (pool larger than VRAM) | nothing in the pipeline hits the limit |
| Closure-based ELS (1.11-1.32x more vars) | a sampler behaviour change; needs its own uniformity gate |
| `hyper_binary` size guard (~1000x the formula) | analysis product only; documented |
| `--sample random` for truncated sweeps | measure on an instance where mining finds something first |
| Line-by-line review of the Python mining path since the C++ port | noted by the user, not done |

# GPU BCP-completion sampling — findings

> **Status:** HISTORICAL — the sampler's development log, kept for its rules and its negative
> results. The engine, the portfolios and every absolute number have moved since, so **no rate or
> ratio from this file is quotable**. Current numbers: [`docs/paper/`](../paper/README.md) and
> [RIVAL_FAIRNESS_FINDINGS.md](RIVAL_FAIRNESS_FINDINGS.md).
> **Measured:** 2026-06 → 2026-07-04, on the PyTorch engine and the first Triton kernels. No
> card is named anywhere in this file, deliberately.
> **Rule lives in:** the surviving decisions are in `gpap/sampling/pipeline.py` (the portfolio
> definitions and the per-instance config rule, which cites this file).
> **Record only:** the seven retired configurations and their verdicts — that code is deleted,
> and this table is the sole reason not to rebuild them.

Code: [`gpap/sampling/sampler.py`](../../gpap/sampling/sampler.py), harness
[`gpap/benchmark.py`](../../gpap/benchmark.py), tests
[`tests/sampling/test_sampler.py`](../../tests/sampling/test_sampler.py).

## The algorithm: GPAP = GPU-parallel CMSGen

CMSGen (Golia–Soos–Chakraborty–Meel, FMCAD'21) is, per its Algorithm 1 *UniformLikeWitness*: pick
an unassigned variable at random, assign a polarity uniformly, unit-propagate, repeat — "a
randomized variation on CDCL, random var + random polarity heuristics, no restarts". **GPAP runs
exactly this, K trajectories in parallel on the GPU.** The one difference: CMSGen is built on
CryptoMiniSat and *backtracks* on conflict, while GPAP has no search and **discards** the
conflicted trajectory. Every fully-assigned, non-conflicted trajectory is therefore a valid model.
Sampling is two phases (`GPUSampler.sample`): fix the independent support (`c ind`), then complete
the remaining free variables, and project onto the support.

**`sat_count > 0` means satisfied only at a full assignment.** At a complete assignment
non-falsified ⟺ true, so the GPU check is correct *because* it is gated on fully-assigned rows; on
a partial assignment it means only "not yet violated" — never use it mid-propagation. It is in
fact redundant (a row goes inactive the instant a clause hits `sat_count == 0`); the load-bearing
independent check is the CPU `_satisfies()` in the tests.

## Preprocessing that is legitimate here

CMSGen disables preprocessing because variable/clause **elimination** changes the solution space.
Ours is not elimination:

- **Unit propagation** (`simplify_cnf`) only pins variables already forced in every model, so it is
  strictly model-preserving. The exact set of valid support projections is identical with and
  without it, and reconstructed models satisfy the original clauses
  (`tests/sampling/test_sampler.py::test_simplify_preserves_projection_set`). Don't-care support variables —
  those in no remaining clause — are sampled uniformly, which preserves the projection
  distribution.
- **ELS** (`simplify_cnf_els`) collapses binary-implication SCCs, removing variables. It is
  support-projection-preserving **iff reconstruction is consistent**: every removed variable equals
  its representative in every model, so removed variables — *don't-cares included* — are
  reconstructed from their representative and never randomized independently, else an equivalent
  pair could be assigned unequal values and the projection would be invalid. Default since
  2026-08-19 (`--no-els` restores the old behaviour).
- **Entailed-clause mining** probes the formula against itself (backbone units, then support pairs
  and triples); the learned clauses are entailed, so the model set is unchanged, but the sampler's
  own propagation now kills doomed trajectories early. [MINING_FINDINGS.md](MINING_FINDINGS.md).

Profiling explains why all three pay: propagation is 79–86% of chunk wall and setup ~0.5%, so the
lever is anything that shrinks the formula — it hits twice, faster propagation *and* a bigger K.
(The same measurement kills "recycle dead trajectories": there is no setup cost to amortize.)

## The configuration lessons that still shape the code

- **B = 8 decisions per round** is the throughput optimum and the default on every face: fewer
  rounds while success barely drops, where B ≥ 16 begins to over-constrain. Uniformity later added
  the other half of the story — B ≥ |support| is exactly uniform, since deciding the whole support
  in one round *is* rejection sampling ([UNIFORMITY_FINDINGS.md](UNIFORMITY_FINDINGS.md)).
- **Structured decision order** (`random_var=False`) decides variables in index order, which on
  program-synthesis formulas follows the gate/definition DAG, so propagation determines the gates
  instead of the search conflicting on them. With B=1 it takes the "service" family from ~0 to
  real, at 2-wise coverage equal to CMSGen's and with polarity still random.
- **The right configuration is formula-dependent**, which is why there is a bandit rather than a
  default: the sk_ niche wants B=8 in any order, the service family wants structured order with
  B=1, and circuits differ again. The bandit runs ONE config per chunk at full K and picks on
  measured models/wall (decaying-ε greedy), never permanently committing. A `static` portfolio that
  runs every config each chunk with K split was measured and rejected: one slow config's
  K-independent round count drags every chunk.
- **Every config must be cold-started once before the sample target can end the run.** On a
  large-K GPU a single chunk can already exceed the request, so the timing loop would otherwise
  exit after the first (arbitrary) config and report the exploration-blended rate. The reported
  number is the winning config's own steady-state rate, and a tiny warmup chunk precedes the timed
  run to kill first-chunk JIT skew.
- **A throughput bandit is blind to mode collapse.** Pure risk-argmin polarity is row-deterministic
  when risks never tie, so all K trajectories replicate one path — on one instance, 70 distinct
  solutions in 6,144 samples while CMSGen was fully distinct on the same CNF. Fixed by ε-greedy
  polarity (hash-forced random bit, ε=1/8) in both soft kernels. Measure distinctness; the raw rate
  cannot see this.
- **Unique-solution counting stays off the timed path**: the timed run generates and emits nothing,
  and dedup by support projection is a separate, separately reported step.

## Configurations measured and retired

| retired | verdict |
|---|---|
| `soft` risk polarity in the default portfolios | wins on tight/deep circuits by success rate, but Barbarik-rejects; opt-in via `--allow-soft` |
| full-propagation support lookahead | raises feasibility exactly as predicted and still loses: ~3× the propagations and half the K |
| 1-step `_immediate_conflict` lookahead | cheap and useless — it cannot see the deep cascades that kill the low-`supOK` family |
| the original soft weights (`w(2)=4`) | a semantic misreading: `sat_count` counts non-falsified literals, so a clause going 2→1 becomes a **unit and propagates**, which is how a trajectory completes. Conflict-only weights are ≈ baseline, i.e. no instantaneous counter encodes the propagation closure — random polarity (CMSGen's Algorithm 1) is near-optimal |
| gate/definability (Arjun-style) propagation | cannot help the service family: its residual has ≥200 solutions after the support is fixed, so it is genuinely under-determined, not gate-defined |
| in-kernel retry-on-conflict | tripled per-slot success and tripled wall with it — conflicts die late, so a failed attempt costs a full trajectory. It also taxed the accepted path ~10% by wrapping it: experimental paths get their own kernel |
| `static` portfolio (all configs per chunk, K split) | the slow config's round count drags every chunk |
| recycling dead trajectories | setup is 0.5% of wall; nothing to amortize |

## The niche, and the honest losses

GPAP wins when **both** hold: fixing the `c ind` support lets propagation determine the rest
(`freeAftSup ≈ 0`), and a random support assignment is usually feasible (`supOK` high). Where it
does not:

- **The "service" family** (ActivityService, PhaseService, IterationService, ProcessBean, …):
  `freeAftSup` in the hundreds, so completion needs real search — exactly what CMSGen's CDCL
  provides and GPAP lacks. Structured order plus mining converted most of them later; the residual
  (NotificationServiceImpl2 class) needs in-search clause learning, and no amount of restarts or
  probing substitutes.
- **7/71/80.sk**: `freeAftSup` = 0 but the *support itself* self-conflicts through deep cascades,
  so most random support assignments are infeasible. Only lookahead sees this, and lookahead does
  not pay.
- **The Baital config family** (busybox/buildroot/adder): no `c ind` at all, so there is no small
  support and the instance is out of the niche by construction.

The general shape: GPAP's advantage is throughput, and it is spent on rejection. That is the same
statement the propagation-rate identity makes quantitatively for the paper campaign — the engine
buys a large constant factor and the algorithm gives part of it back by discarding conflicted
trajectories.

## Methodology notes that survived

- **CMSGen is the right fast competitor**; UniGen3 gives the uniformity guarantee but runs
  ApproxMC counting first and is not a throughput comparison. Metrics follow the literature:
  samples/s plus 2-wise coverage (a spread metric, not a uniformity one).
- **Soundness is checked twice**: on the GPU over completed rows, and on the CPU against the
  original clauses in the tests.
- **Timing must be symmetric.** GPAP's timed run emits nothing while stock CMSGen writes every
  sample, which is a real handicap — see [RIVAL_FAIRNESS_FINDINGS.md](RIVAL_FAIRNESS_FINDINGS.md)
  §7 and [../BENCHMARKING.md](../BENCHMARKING.md) §4b for the patched-binary protocol that fixed it.
- Cross-GPU behaviour is in [CROSS_GPU_FINDINGS.md](CROSS_GPU_FINDINGS.md).

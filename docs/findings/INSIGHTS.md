# INSIGHTS — settled results index

> **Reader:** contributors and researchers — every settled result in a line or two, with the file that holds its evidence.

One entry per settled result, newest first within a section, each pointing to its evidence. A
direction that was measured and lost is listed so it is not re-tried without new evidence; its
code may be deleted, the entry stays. A new measurement adds an entry here. Each findings file's
status block says when and on which card it was measured; read it before quoting a number.

**↻ re-evaluate:** measured on an engine many times slower than today's (the pure-PyTorch engine
of June, or the kernels before the 2026-08 gains: binary clauses as edges, the shuffle schedule,
CLEAN mode), where speed decided the outcome. The current engine is reason enough to measure it
again before relying on it, or before calling the direction closed.

**Current evidence** — the rule is live and the numbers are the engine's:

| topic | file |
|---|---|
| binary clauses as edges (2026-08-27) | [BINARY_OCC_FINDINGS.md](BINARY_OCC_FINDINGS.md) |
| the WAVE shuffle lane schedule (2026-08-27) | [WAVE_SHUFFLE_FINDINGS.md](WAVE_SHUFFLE_FINDINGS.md) |
| CLEAN mode: design and lemmas | [CLEAN_MODE_FINDINGS.md](CLEAN_MODE_FINDINGS.md) |
| the C++ core, AOT kernel build | [CPP_BACKEND_FINDINGS.md](CPP_BACKEND_FINDINGS.md) |
| what a bigger GPU buys | [CROSS_GPU_FINDINGS.md](CROSS_GPU_FINDINGS.md) |
| entailed-clause mining and redundancy | [MINING_FINDINGS.md](MINING_FINDINGS.md) |
| uniformity: Barbarik, TV, the B knob | [UNIFORMITY_FINDINGS.md](UNIFORMITY_FINDINGS.md) |
| rival-comparison fairness, and protocol v1.0 | [RIVAL_FAIRNESS_FINDINGS.md](RIVAL_FAIRNESS_FINDINGS.md) |
| what `--out` cost, and what the console says | [CLI_DELIVERY_FINDINGS.md](CLI_DELIVERY_FINDINGS.md) |
| reproducing the campaign, and the Docker images | [ARTIFACT_FINDINGS.md](ARTIFACT_FINDINGS.md) |

**Historical record** — kept so a settled question is not re-opened. **Quote no rate from these.**

| topic | file | why it is here |
|---|---|---|
| WAVE vs scalar by formula size (2026-08-23) | [KERNEL_AB_FINDINGS.md](KERNEL_AB_FINDINGS.md) | the `M ≥ 2048` rule is current; its numbers predate the shuffle schedule |
| the 2026-08 L4 campaign | [L4_EVAL_FINDINGS.md](L4_EVAL_FINDINGS.md) | superseded by protocol v1.0; its screens survive in `paper_tables.py` |
| the sampler's development log | [GPU_SAMPLING_FINDINGS.md](GPU_SAMPLING_FINDINGS.md) | the only record of seven retired configurations, judged on the June engine ↻ |
| GPU inprocessing for CDCL, retired | [ELS_INPROCESSING_FINDINGS.md](ELS_INPROCESSING_FINDINGS.md) | the only record of a deleted subsystem, measured on the pure-PyTorch engine ↻ |
| porting to AMD / Intel / Apple | [PORTABILITY.md](PORTABILITY.md) | an analysis; nothing in it is implemented |

## Engine & memory model

- **Three path roots, one owner each (2026-09-13).** `.gpap/` holds machinery and all scratch
  (`gpap/paths.py`), `.experiments/` a run's output and `$GPAP_RESULTS` the record
  (`eval/paths.py`); nothing else names them, and shell asks Python instead of re-deriving them
  (`tests/repo/test_suite_hygiene.py`).
- **A printed number means one thing (2026-09-12).** The CLI prints each phase as it happens,
  and a rate renders once (`gpap/verdict.py::fmt_pct`). [CLI_DELIVERY](CLI_DELIVERY_FINDINGS.md)
- **The sampling loop lives in the core (2026-09-01).** `cpp/src/sample.cpp` owns chunks, the
  bandit and the accounting; both faces marshal over it (`tests/sampling/test_face_parity.py`).
- **A zeroed options struct means every default (2026-08-31).** A field whose useful default is
  on is named for its opt-out; `mode` defaulting to scalar had cost raw-ABI callers up to 2.20×.
- **A soundness verdict needs evidence (2026-08-31).** `sound_check` is `off` unless rows were
  checked, and its wall is billed apart. [CLI_DELIVERY](CLI_DELIVERY_FINDINGS.md)
- **Key a cache on `gpap_build_id()`, never `gpap_version()` (2026-08-31).** The version stayed
  0.1.0 while the engine got 1.8× faster.
- **No two shipped kernel binaries alike (2026-08-31).** 6 of 33 cubins were duplicates; compare
  executable sections, not bytes (`tests/engine/test_kernel_table.py`).
- **WAVE's lane schedule is warp-shuffle based: 1.80× at the same K (2026-08-27).**
  [WAVE_SHUFFLE](WAVE_SHUFFLE_FINDINGS.md)
- **Binary clauses are implication edges (2026-08-27).** Only clauses of length 3 or more carry
  per-row state: 1.42× at the same K, and twice the K. [BINARY_OCC](BINARY_OCC_FINDINGS.md)
- **CLEAN mode (2026-08-21).** A probe launch restores what it touched instead of re-broadcasting
  every chunk: 4.4× per chunk on `b12-c50`. [CLEAN_MODE](CLEAN_MODE_FINDINGS.md)
- **A default belongs at the ABI, not in a wrapper (2026-08-19).** `GPAP_MODE_AUTO` resolves
  inside `gpap_engine_reset`.
- **The wave/scalar dispatch keys off M, not the batch (2026-08-18).** ↻ Wave when `M >= 2048`
  or the batch is under the resident warps; the crossover is to be re-measured on the shuffle
  engine. [KERNEL_AB](KERNEL_AB_FINDINGS.md)
- **Wave and scalar may differ on the counters of conflicted rows, never on models
  (2026-08-18).** Gate on models.
- **The engine is libgpap, one library for Python and C++ (2026-07-28):** a performance tie with
  the JIT path. [CPP_BACKEND](CPP_BACKEND_FINDINGS.md)
- **Portability (2026-07-29):** AMD is a small port, Apple a kernel rewrite.
  [PORTABILITY](PORTABILITY.md)
- **The assignment bitmap (2026-07-14):** one `atomic_or` per push is dedup and contradiction
  detection, +40–50%.
- **The SUM-packed clause word (2026-07-13):** one `atomic_add` is the whole clause update, and
  its return value detects unit and conflict race-free.
- **The engine is packed-kernel-only (2026-07-13).** The PyTorch path is deleted; the oracle is
  `tests/reference/bcp.py`. The WIDE 8|24 word covers N ≤ 8.4M.
- **The batch comes from the memory model, never a runtime probe (2026-07-01).** Allocator
  fragmentation, not tensor size, had set the old small K.
- **This WSL2 box spills to host RAM instead of failing (2026-05-31).** An oversized K collapses
  throughput silently; validate K changes under a kill-switch.
- **An entry point that applies literals must skip those already TRUE (2026-07-03),** or
  `occ[lit]` is decremented twice. Gated by a regression test.

## Sampler, mining and uniformity

- **Barbarik: the shipped portfolio Accepts 65/65 with zero Rejects (2026-10-13),** as do pinned
  `b8` and CMSGen; `soft` is Rejected on 64/65 and opt-in.
  [UNIFORMITY_FINDINGS.md](UNIFORMITY_FINDINGS.md) §4b
- **Uniformity claims live in `eval/uniformity/`, never in the library (2026-09-01):** exact
  where the solution set is enumerable, Barbarik for certificates.
- **`B >= |support|` is exactly uniform (2026-07-30):** rejection sampling, yield = `p_accept`.
  [UNIFORMITY_FINDINGS.md](UNIFORMITY_FINDINGS.md) §3
- **The bandit is the default's TV cost (2026-07-30, 2026-08-28):** it picks B=1 for throughput;
  `--portfolio b8` holds the floor for ~10%. A TV claim must name a config. §4c
- **Uniformity vs CMSGen splits by regime (2026-07-30):** circuits favour GPAP, random k-CNF over
  all variables favours CMSGen; never quote an aggregate. §2
- **Skew is instance-dependent, and CMSGen shows it on the same CNF (2026-07-11):** a property of
  propagation-completion sampling, not a bug.
- **A K-parallel sampler needs forced randomness (2026-07-04):** risk-argmin polarity made all K
  rows one path; ε-greedy, ε = 1/8.
- **Bandit mechanics (2026-07-01/06):** cold-start every config once, report the best config's
  steady-state rate, decay ε-greedy.
- **A losing lever stays a bandit option:** `soft` lookahead is 0% success on `sk_` instances and
  10% → 88% on tight circuits.
- **Mining (2026-08-19/21):** entailed clauses are kept only if they pay (models per chunk =
  K × success); `subsumed` is the default redundancy, and backbone units are the top lever.
  [MINING](MINING_FINDINGS.md)
- **Simplification (2026-06/07):** ↻ unit propagation is model-preserving and the big lever
  (about 4.5× over plain); ELS adds 1.48×, sound when reconstruction is class-consistent. On a
  faster engine preprocessing is a larger share of the wall, so re-measure what ELS buys.

## Circuit encoding and benchmarks

- **The ABC encoder (2026-07-10)** ↻ shrinks unrolled CNFs about 5×, but helps CMSGen almost
  as much; `fast` optimizes before the unroll. Mining helps more on ABC CNFs than on Tseitin ones
  (2026-08-19).
- **Rare targets (2026-07-11):** ↻ at random pins plain simulate-and-filter beat every CNF
  sampler, so comparisons use `--target rare`. `--sim-check` is the encoder-independent
  soundness gate.
- **Degenerate cells leave the geomean:** a vacuous pin (`p_accept` 1, decided by the row,
  RIVAL_FAIRNESS_FINDINGS.md §17d) or a zero-search cell (a free hypercube after propagation)
  measures the instance, not the sampler.
- **New circuit families (2026-07-12):** of 231 HWMCC'17 AIGs only 7 fire at T=25; EPFL splits
  into GPAP-only wins, hard-for-both arithmetic and rejection-easy cells.

## The comparison and the paper's campaign

- **The paper reports the 2026-10-12 re-run.** With Recurrent's host copy deducted, its Table 2
  ratio fell 21.5→18.7×, and GPAP loses four small circuits on its own table.
  [RIVAL_FAIRNESS_FINDINGS.md](RIVAL_FAIRNESS_FINDINGS.md) §17e
- **Protocol v1.0, frozen 2026-09-06:** publish only quantities each tool defines; a review fixes
  bugs and prose, never a metric, configuration, screen, budget or cell set. §17,
  [PROTOCOL.md](../papers/tacas-27/PROTOCOL.md)
- **Three tools on one constraint (2026-09-03):** Table 1 is read off Table 2's rows. §14
- **Every fairness audit had been flattering us (2026-08-24 → 09-06):** the rival's batch and
  steps, CMSGen's floored counts, the encoder argmax, the screen order, the probe cap. §2–§16e
- **Superseded, quote nothing:** `l4_protocol_20260903` and `l4_protocol_20260904` (the
  September tables), the August family sweeps, the first rival runs (2026-07-12/13), and every
  GPAP-vs-CMSGen ratio before 2026-08-26 (CMSGen was charged for writing its samples until
  2026-07-27 and its counts were floored until 2026-08-26).

## Measurement methodology (the traps that actually bit)

- **The batch is not part of the seed:** pin the batch and one config before comparing counts
  across runs; same-seed repeats measure timing noise only.
- **A 30 s run cannot rank two builds (2026-08-18):** ↻ the bandit is a coin flip there;
  pin `--portfolio` or compare `implications_per_s`.
- **The SM clock moves within a run (2026-07-29):** cross-run rates cannot resolve a few
  percent; use interleaved A/Bs in one session ([BENCHMARKING.md §1b](../BENCHMARKING.md)).
- **Power state (2026-07-07):** the laptop 4080 can sit at a 40 W cap; check
  `nvidia-smi -q -d POWER` first.
- **Emission is not free (2026-07-27):** writing its samples cost CMSGen 5.3×; no tool is timed
  while delivering.
- **The distinct count stays off the timing path:** `unique_sps_est = raw × distinct_frac`.
- **Benchmark GPU work against the solver's own C operation,** never a pysat re-implementation,
  which inflated a speedup 50–100×.
- **`--n-time` is a delivery target:** quote the delivered count.
- **A missing instance is a bug, not a gap (2026-09-03):** a refusal is a row with its reason.
- **A bigger GPU is worth ~1.8×, tracking SM count; 4× the batch buys +2.8% (2026-08-03).** ↻
  [CROSS_GPU](CROSS_GPU_FINDINGS.md)

## The artifact

- **A reproduction is asked for verdicts, not rates (2026-09-09):** rates describe the card.
  [ARTIFACT_FINDINGS.md](ARTIFACT_FINDINGS.md) §6
- **Another card can only flatter us, and both directions are counted (2026-09-09):**
  `GPAP_VRAM_BUDGET_MIB`, and the walked-down grants. §1, §2
- **The rows do not time a per-cell constant (2026-09-09):** the cost model is constant + per
  gate + fraction. §3
- **A gate that cannot fail is not a gate (2026-09-10):** make silence fail. §8
- **A leak gate knows only the shapes it was given (2026-09-13):** generators print
  `display_path`, and `check_leaks.sh` holds every shape that shipped. §9
- **The image compiles the engine for the GPU it finds (2026-10-19):** one linked without a GPU
  would load CI's stub `libcuda`. §10
- **PID 1 drops the signals it has no handler for, and `grep -q` under `pipefail` can fail its
  own pipeline (2026-10-19).** §11

## Measured and lost — do not retry without new evidence

A ↻ entry lost on a much slower engine; today's engine is the new evidence to measure it again.

- Pure-PyTorch BCP: latency-bound by a launch and a sync per round; the fused kernel replaced it
  (2026-07-02).
- The XOR-packed clause word: the SUM word's single `atomic_add` beat its `atomic_cas` loop
  everywhere (2026-07-13).
- A 1-byte (MONO) clause state: 0.93× of WIDE (2026-07-08). ↻ Duplicate-literal padding:
  unsound.
- In-kernel retry on conflict: success ×3, wall ×3 (2026-07-03). ↻
- W = 64 as instruction-level parallelism: 0.918× (2026-08-27).
- Clause renumbering for occ locality: topological emission is already near-optimal
  (2026-07-13). ↻
- Shared-memory staging of the row state (2026-07-13). ↻
- A runtime memory probe to size K (2026-07-01).
- Full-propagation lookahead: more success, less throughput (~3× propagations, half the K). ↻
- The `orac` config: 0 of 64 wins (2026-07-06). ↻
- UCB-style bandit bonuses: 0.52× on small budgets (2026-07). ↻
- `soft` in a default portfolio: Barbarik-Rejected (2026-07-30).
- Pinning GPAP to `b8` for the comparison: measured, not adopted
  (RIVAL_FAIRNESS_FINDINGS.md §12).

## Retired application directions

- **Preprocessing and inprocessing for solving (2026-06-29):** ↻ a CPU incremental prober was
  7–15× faster than GPU discovery, and the reduction helped CaDiCaL but not kissat. That was the
  pure-PyTorch engine, before the fused kernels and CLEAN-mode probing: the first retired
  direction to re-measure. [ELS_INPROCESSING](ELS_INPROCESSING_FINDINGS.md)
- **Where GPAP wins (2026-06/07):** ↻ sampling, cube-and-conquer cubing, and one-shot UNSAT
  proving on high-k random near-threshold instances. The novelty is many independent assumption
  sets propagated incrementally in one fused kernel. Only sampling has been measured on the
  current engine.

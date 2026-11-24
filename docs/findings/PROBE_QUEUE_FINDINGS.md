# The job queue — many sets on few rows

> **Reader:** contributors and researchers — why probe sweeps and flags-only checks run on a queue, and what was measured. A dated research log, not a user guide.
>
> **Status:** settled — probe sweeps and flags-only checks run on the queue, on both backends
> (issue 58).
> **Measured:** 2026-11-19 and 2026-11-20 on Colab L4s (24 GB), main 54aa441 with the queue
> (§3 to §6); 2026-11-20 on a laptop's RTX 4080 (12 GB), a sample of the SAT Competition 2024
> main track and the application gates (§10, §11); 2026-11-24 on the same laptop, the shipped
> host side against the first one (§12). §4 to §11 ran the first host side, which §12 shows the
> shipped one matches in every result and outruns. The OpenCL prototype of 2026-11-14 is in
> [OPENCL_FINDINGS.md](OPENCL_FINDINGS.md) §10.
> **Rule lives in:** `cpp/kernels/bcp.py` (`_bcp_queue_kernel`), `cpp/kernels/bcp.cl`,
> `cpp/src/gpap_internal.hpp` and `cpp/src/engine.cpp` (`QueuedCall`), `cpp/src/probe.cpp`
> (`sweep_on_queue`), `cpp/src/check.cpp` (`check_on_queue`).
> **Gated by:** `tests/engine/test_queue.py`.
> **Record only:** §8, what was looked at and left.

The design is in [ENGINE.md](../ENGINE.md) §6b. Data, drivers and logs:
`$GPAP_RESULTS/probe_queue_20261118/`.

## 1. What a chunked sweep pays for

A chunked sweep gives each assumption set its own row, and a launch ends with its slowest row.

- **Depth is very uneven.** At width 1 the median probe implies 2 to 14 literals and the
  deepest 155 to 50,854, on the eight formulas of §3 ([OPENCL_FINDINGS.md](OPENCL_FINDINGS.md)
  §10 has the distribution). A chunk waits for its deepest row.
- **Every chunk starts with a reset.** After a CLEAN launch the reset skips the packed words but
  still copies the bitmap to every row: 3.05 to 3.24 ms per chunk on the formulas past 300k
  clauses, against launches of 0.73 ms (`hyp`, width 1) to 13.9 ms (`s35932` at 30 cycles,
  width 3).

## 2. Why a row can be reused without a reset

A queued row runs set after set. After each one it restores the packed words it touched, as
CLEAN mode does ([CLEAN_MODE_FINDINGS.md](CLEAN_MODE_FINDINGS.md), lemmas L1 to L4), and the
bitmap words of the literals on its trail. The bitmap restore is complete:

- The baseline is a BCP fixpoint. A literal that is false there was taken out of every count and
  sum by the flush, so no long clause can force it; and a binary edge onto it made its other end
  false in the baseline too, so that end is never assigned. Hence no set can force a literal
  that is false in the baseline.
- So a bit that a set turns on belongs to a literal the row assigned, which is on its trail, or
  to a literal whose negation the row assigned: the WAVE lane that finds a contradiction sets
  its bit before it looks. A literal and its negation share a bitmap word, so the words of the
  trail's literals cover every bit.
- A seed that is already false conflicts the set before it sets anything.

## 3. Before the engine work: a throwaway kernel (2026-11-19)

A Triton JIT kernel built from the engine's own `_propagate`, `_wave_propagate` and
`_clean_packed`, against the engine as it was. Eight formulas from 1.5k to 1.75M clauses, widths
1 to 3, 262,144 random tuples each, best of 2.

- **Exact.** The flags equalled the engine's, the implied counts too on the tuples that do not
  conflict, and every row equalled the baseline afterwards.
- **Rate, queue over chunks:** 2.93 / 2.21 / 1.91 at widths 1 / 2 / 3 (geometric means), from
  1.10 to 15.63.
- **Rows.** However many rows are asked for, 1,392 get work on the L4 (58 SMs times 24 one-warp
  programs); the driver's resident-warp number there is 2,784. 1,024 rows are within a few
  percent of the plateau.
- **Launch size.** Launches of rows × 32 tuples cost 5% to 25% against one launch over all the
  tuples on the formulas past 300k clauses: a launch has a fixed cost.

## 4. The engine: queue against chunks (2026-11-20)

`GPAP_QUEUE=1` against `GPAP_QUEUE=0` on the same work, one engine alive at a time, arm order
alternated, best repetition kept (3 on CUDA, 2 on OpenCL and on the Hardware set). Per formula:
262,144 random tuples of 1, 2 and 3 literals (`gpap_probe_tuples`); 65,536 random cubes of 8, 16
and 32 literals (`gpap_check_sets`, flags only); and width 1 over every free literal to its
fixpoint, with the units folded in, which is what mining runs first. In every cell both arms
reported the same conflicts, implied counts and units.

**CUDA, seventeen formulas** of the repository's families, 961 to 1.75M clauses (twelve past
100k):

| queue over chunks | width 1 | width 2 | width 3 | cubes of 8 | 16 | 32 | width 1 to its fixpoint |
|---|---|---|---|---|---|---|---|
| geometric mean, 17 formulas | 2.53 | 2.16 | 1.80 | 1.61 | 1.39 | 1.34 | 2.37 |
| geometric mean, past 100k clauses | 2.57 | 2.06 | 1.73 | 1.61 | 1.39 | 1.33 | 2.94 |
| smallest | 1.20 | 1.15 | 1.05 | 1.08 | 1.08 | 1.08 | 1.00 |

The largest gains are on the two formulas at 1.7M clauses: `b14` at 40 cycles 7.0 / 5.4 / 4.2
by width and 8.9 to the fixpoint (4.07 s to 0.46 s), `s35932` at 30 cycles 4.1 / 2.6 / 2.0 and
4.8. The fixpoint's 1.00 is a formula whose rounds have no more literals than rows, so both arms
run them as chunks (§7).

**OpenCL, eight of them** (the NVIDIA driver's OpenCL on the same L4):

| queue over chunks | width 1 | width 2 | width 3 | cubes of 8 | 16 | 32 | width 1 to its fixpoint |
|---|---|---|---|---|---|---|---|
| geometric mean, 8 formulas | 2.79 | 2.46 | 2.07 | 1.68 | 1.55 | 1.46 | 3.11 |
| geometric mean, past 100k clauses (5) | 3.70 | 2.87 | 2.37 | 1.64 | 1.46 | 1.32 | 4.07 |

The one formula under 2,048 clauses, `registerlesSwap`, lost here (0.79 to 0.89), on the SCALAR
kernel; §6 is why.

**CUDA, the SatDiv Hardware BMC set.** Of its eighteen formulas (three circuits), the engine
refuses the six `cpipe` ones: a 301-literal clause and 850k literals fit neither packed word.
The other twelve, 8k to 798k clauses:

| queue over chunks | width 1 | width 2 | width 3 | cubes of 8 | 32 | width 1 to its fixpoint |
|---|---|---|---|---|---|---|
| geometric mean, 12 formulas | 3.76 | 3.31 | 2.62 | 2.25 | 1.36 | 3.15 |
| smallest | 3.19 | 2.50 | 2.15 | 1.70 | 1.17 | 1.19 |

The formulas of one circuit are close to one another, so this is two circuits more than twelve
samples.

## 5. Rows and launches

- **Rows:** as many as the device runs at once (the driver's resident-warp number on CUDA),
  within the memory budget. A row past what the device runs takes no set (§3).
- **Blocks:** the first launch takes a chunk's sets, the second eight times that, and later ones
  grow eightfold while launches take under 50 ms and fit the deadline, which hides the fixed cost
  of §3 (§12). The first host side sized each launch from the last one's rate instead; §12 is why
  that went.

## 6. The kernel on the queue (2026-11-20)

Above 2,048 clauses a chunk runs WAVE, and the queue first took the chunk's kernel, so SCALAR
below that. On the eleven formulas under 2,048 clauses of the repository's families (961 to
1,891 clauses), each kernel pinned, both backends, best of 3:

| geometric mean, 11 formulas | width 1 | width 2 | width 3 | cubes of 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| CUDA: WAVE over SCALAR, on the queue | 1.21 | 1.27 | 1.27 | 1.26 | 1.13 | 1.03 |
| CUDA: the queue on SCALAR over chunks | 1.17 | 1.10 | 1.05 | 1.04 | 1.08 | 1.12 |
| OpenCL: WAVE over SCALAR, on the queue | 1.86 | 1.94 | 1.90 | 1.78 | 1.46 | 1.14 |
| OpenCL: the queue on SCALAR over chunks | 0.80 | 0.84 | 0.88 | 0.91 | 0.95 | 1.09 |

On WAVE the queue beat chunks in every cell, by at least 1.08 on CUDA and 1.01 on OpenCL; on
SCALAR it fell to 0.93 and 0.56. The reason is the unit the rows are counted in: the queue takes
as many rows as the device runs warps at once, and a WAVE row is one warp on both backends,
while a SCALAR row is one working lane of a warp on CUDA and a single work item on OpenCL, where
4,096 rows fill 128 warps. A chunk does not have the problem: it runs the memory batch.

So the queue runs WAVE on every formula unless a kernel is pinned; the dispatch rule still
chooses for chunks. As shipped, the queue beat chunks on these eleven formulas by 1.20 to 1.44
on CUDA and 1.27 to 1.68 on OpenCL (geometric means by column), smallest cell 1.10 and 1.02.

## 7. The rules the measurements set

- **Rows and blocks:** §5.
- **Kernel:** WAVE, unless pinned (§6).
- **Small sweeps:** a sweep with no more sets than rows runs as a chunk. Every row would take
  one set, which is a chunk with more bookkeeping: a sweep of 430 tuples a round on a
  1.5k-clause formula took 1.6 ms queued against 1.2 ms in a chunk on 2026-11-19.

## 8. Looked at and left (record only)

- **Deepest first**, ordering the tuples by their literals' width-1 depths (§3): -10% to +13%.
  Not built.
- **The implied count per set** is returned by flags-only checks; the probe sweep does not use
  it yet (issue 45 would).
- **The resident count** on an L4 is twice the programs that run (§3). The extra rows cost
  memory and nothing else.
- **Memory below the resident count.** On the largest formulas the memory budget gives the
  queue fewer rows than the card could run (§10); the trail is the first thing to shrink there
  (issue 50).
- **Checks that read row state** (assignments, sat counts) stay in chunks: a queued row is back
  at the baseline when its set ends.
- **Follow-ups, each its own issue:** OpenCL's one-buffer limit (68), the sampler on a queue
  (69), probe tuples built on the device (70).

## 9. The checks

- The tests were written first: against the engine without the queue, 19 of 20 failed.
- Thirteen planted bugs, one line each, in the Triton kernel, the engine and the OpenCL kernels
  (a restore dropped, flags zeroed, the implied count off, the queue on a chunk's kernel, a pin
  that does not reach the queue, a clean claim kept past a plain launch, the queued work not
  counted): all thirteen fail the tests.
- The 27 kernels that existed before have the same machine code after (executable sections
  compared).
- The default suite passes on CUDA and OpenCL on the L4; on PoCL 7.2's CPU driver the CI
  selection passes, and on Oclgrind 21.10 every engine and mining test that needs no CUDA
  passes with no data race or out-of-bounds report.
- A test of a restore needs a formula larger than a few bitmap words: a job that assigns as many
  literals as its row has words rewrites the whole row, which heals a planted bug. The tests that
  run a sweep after a trajectory or a CLEAN launch use 640 variables for that reason.

## 10. A sample of the SAT Competition 2024 main track (2026-11-20, a laptop's RTX 4080)

A WSL2 laptop's RTX 4080 (12 GB; 58 SMs and the L4's resident-warp number, 2,784). Thirty-six of
the 363 main-track formulas the engine accepts, six from each clause-count band, picked with a
fixed seed. The work of §4, best of 2 with the arm order alternated; the four past 13M clauses
took 16,384 tuples and 4,096 cubes a width. Width 1 to its fixpoint ran on formulas with at most
2M free literals, for at most 20 rounds. In every cell both arms reported the same results.

| queue over chunks, geometric mean | formulas | width 1 | width 2 | width 3 | cubes of 8 | 16 | 32 | width 1 to its fixpoint |
|---|---|---|---|---|---|---|---|---|
| under 100k clauses | 6 | 8.54 | 6.49 | 5.20 | 3.97 | 3.56 | 3.62 | 5.93 (1) |
| 100k to 300k | 6 | 5.50 | 4.62 | 4.17 | 4.44 | 3.96 | 3.53 | 7.00 (5) |
| 300k to 1M | 6 | 1.90 | 1.80 | 1.71 | 1.63 | 1.53 | 1.39 | 2.17 (5) |
| 1M to 3M | 6 | 2.15 | 2.01 | 1.89 | 1.49 | 1.44 | 1.45 | 2.31 (4) |
| 3M to 10M | 6 | 2.70 | 2.00 | 1.71 | 1.25 | 1.17 | 1.13 | 4.78 (2) |
| past 10M | 6 | 5.19 | 4.41 | 3.93 | 2.25 | 1.94 | 1.72 | 2.89 (3) |
| all | 36 | 3.73 | 3.14 | 2.79 | 2.22 | 2.04 | 1.92 | 3.50 (20) |
| smallest cell | | 1.10 | 1.04 | 1.03 | 1.06 | 1.01 | 1.01 | 1.01 |

The fixpoint column counts the formulas whose rounds ran on the queue (§7).

- **The least gain is where the sets are alike:** probes that imply almost nothing (`EDP3`,
  `Nb44T6`, the `circuit_*` family), where no row waits for a deepest one, and probes that are
  heavy beside a launch's fixed cost (`oisc`, `sokoban`, `REGRandom`).
- **Memory sets the rows on the biggest.** From 1.1M clauses on, this card's memory gives the
  queue fewer than 2,784 rows, and from 1.6M fewer than the 1,392 programs it runs, down to 79.
  The queue still wins there, most on shallow probes: `g2-T93` 21.8 at width 1 with 197 rows,
  `T105` 12.6 with 222. With so few rows a chunked sweep is thousands of chunks, each with its
  reset and readback and each waiting for its deepest set.
- **OpenCL agrees.** 27 of the formulas, up to 4.4M clauses, through PoCL's CUDA driver (WSL2
  offers no NVIDIA OpenCL): 2.96 / 2.72 / 2.56 at widths 1 / 2 / 3, cubes 1.78 / 1.73 / 1.68,
  the fixpoint 2.98 (16 formulas); smallest cell 1.01, the same results in every cell.

Data: `$GPAP_RESULTS/probe_queue_20261118/local/`; the census of the track is in `census/`
beside it.

## 11. The applications on the queue (2026-11-20, the laptop's RTX 4080)

The command line with `GPAP_QUEUE=1` against `GPAP_QUEUE=0`, everything else the same.

- **`gpap mine` with no deadline** (widths 1 to 3, 200k tuples a width): byte-identical clause
  files on all 41 formulas (the 29 of `benchmarks/sampling` and the 12 of §10 under 300k
  clauses).
- **`gpap mine` at its defaults** (10 s) on the 15 of §10 from 300k to 4.4M clauses: the same
  clauses on all 15. Where the deadline binds (`REGRandom`), the queue probes 160k width-2
  tuples in it against 90k.
- **`gpap sample`, the no-change gate** of [BENCHMARKING.md](../BENCHMARKING.md) §3 (`b8`,
  `--k-cap 20000`, seed 5, mining on) on the 29 formulas of `benchmarks/sampling`: every run that
  ends on its model count delivers the same models in the same order (25 formulas, and a 26th
  has no model on either arm). Runs that stop on the time limit test nothing: two chunked runs
  of one formula under a 60 s limit disagree with each other too.

Data: `$GPAP_RESULTS/probe_queue_20261118/local/gates/`.

## 12. The host side, simplified (2026-11-24, the laptop's RTX 4080)

The kernels are those of §6; the host side around them changed ([ENGINE.md](../ENGINE.md) §6b
has the result). The bitmap has no clean claim of its own: a queued call copies it at its start,
so `packed_clean_rows` is the one claim. Blocks follow §5, with no rate. The launch buffers come
out of the memory budget, in bytes, before the rows are sized.

**The A/B.** The work of §10 on the same 36 formulas and pools of literals (the fixpoint two
rounds), on the queue only, against the first host side: one library per process (`GPAP_LIB`),
in the order first, new, new, first. Each design's time is the mean of its two runs; best-of-2
would favour the first slot, about 3% faster on the cells under 0.1 s.

| first design / new | width 1 | width 2 | width 3 | cubes of 8 | 16 | 32 | fixpoint (28) | all 244 cells |
|---|---|---|---|---|---|---|---|---|
| geometric mean, 36 formulas | 1.24 | 1.23 | 1.19 | 1.11 | 1.12 | 1.10 | 1.07 | 1.15 |

- **Identical results** in all four runs of every cell, and 1.22 times the first design past
  300k clauses (160 cells).
- **The largest gains are where sets are deep and rows few.** The first design's rate pacing
  kept blocks near one set a row once a set took 50 ms, so every launch waited for its deepest
  set. Width 1 on `oisc` (25M clauses, 81 rows) went from 110 s in 194 launches to 17.8 s in 29;
  a launch of eight sets a row still takes 0.61 s there, about one chunk. The fixpoint on
  `REGRandom-K4-L3` went from 46 s to 6.5 s.
- **The second block matters.** A version that grew blocks only while launches took under 50 ms
  left those formulas at one set a row (the fixpoint on `REGRandom-K4-L2` 0.49 s to 1.58 s) and
  was 1.02 times the first design overall.
- **The smallest cells** (under 0.1 s) are within run-to-run noise, up to 1.3 times either way.
- **`gpap mine` with no deadline** (§11) gives byte-identical clause files on all 41 formulas,
  the same as the first design's.
- **Planted bugs**, each failing `tests/engine/test_queue.py`: no bitmap copy at the start of a
  call, packed words trusted without the claim, a trajectory that keeps the claim, a first block
  of 16 chunks, flags read back in reverse order. Reserving the launch buffers as rows rather
  than bytes gave a formula of 8 variables one row (its batch is `gpap_default_k_cap`, not a
  memory figure); a test holds it.

Data: `$GPAP_RESULTS/probe_queue_20261118/design_ab_20261124/` (`design_ab2.jsonl` the shipped
version, `design_ab.jsonl` the one of the second-block bullet, the drivers, `gates/`).

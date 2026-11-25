# Memory — what the default batch takes, and why it stays

> **Reader:** contributors and researchers — how much of a GPU the default batch takes, what more rows would buy, and what lies outside GPAP's own count. A dated research log, not a user guide.
>
> **Status:** settled for issue 59 on branch mem-default-59 (2026-11-25): the batch cap is 64
> rows per resident warp (§8), and the default budget is all free memory less a 1 GiB reserve
> the user can set, kept at every allocation on CUDA (§9). §1–§6 describe the rule before it
> (0.80 · (free − 800 MB), a fixed cap of 262,144): more rows than that default bought no rate
> (§4), and a sampling run gives the sampler's rows back before it mines (issue 76, §6).
> **Measured:** 2026-11-25 on a Colab T4 (15 GB, driver 580.82), an Athena A100-SXM4 (40 GB,
> driver 595.71) and a 12 GB laptop card under WSL2 (§8, §9 only), on main 48f4575 (§1–§6) and
> 814d75d (§8, §9). Not on an L4 (§7).
> **Rule lives in:** `cpp/src/calibrate.cpp` (`mem_budget`, `kBatchRowsPerWarp`),
> `cpp/src/memory.hpp` (`kDefaultMemReserve`), `cpp/src/memory.cpp` (`size_rows`, `mem_take`),
> `cpp/src/engine.cpp` (`engine_batch`), `gpap/sampling/pipeline.py`
> (`_free_rows_for_mining`).

The rule is [ENGINE.md](../ENGINE.md) §7. Data, drivers and logs:
`$GPAP_RESULTS/memdefault_20261125/` (§1–§7) and `$GPAP_RESULTS/memrule_20261125/` (§8, §9).

## 1. The question

Without a limit, a batch takes 0.80 · (free − 800 MB) of a CUDA card. Issue 59 asked whether
that leaves throughput on the table, and what the buffers outside the rows take. Three readings
answer it: what the device loses beyond what GPAP counts (§2), how much headroom a card that
fails loudly needs (§3), and what more rows buy (§4). The rival harness takes its memory budget
from the same rule (`gpap_mem_available`, `eval/rivals/rival_common.py`), so a change to the
rule changes both sides' budget at once
([RIVAL_FAIRNESS_FINDINGS.md](RIVAL_FAIRNESS_FINDINGS.md)).

## 2. What lies outside GPAP's count

One process per formula: torch's context first, then an engine and every entry point once (the
sampler, a check with assignments in chunks, a flags-only check and a probe sweep on the queue).
The device's free memory (`torch.cuda.mem_get_info`) against GPAP's count (`gpap_mem_held`):

| card | after the engine | after every entry point |
|---|---|---|
| T4 | 1.7–1.9 MiB | 3.1–3.7 MiB |
| A100 | 1.7–1.9 MiB | 4.9–6.0 MiB |

`27.sk_3_32` and `55.sk_3_46`, at their default batch of 262,144 rows. Kernel code, the driver's
per-launch state and allocation rounding together are a few MiB: the ledger is the memory.

## 3. The headroom a native card needs

A sampler pinned to the rows that leave a margin X free beside what GPAP holds, then every entry
point, X from 512 MiB down to 0, in a fresh process each: every run completed on both cards
(4.1M rows of `27.sk_3_32` on the A100). The rows were sized with a per-row estimate a few dozen
bytes over the real one, so the "0" run still left roughly 0.1 GB (T4) and 0.3 GB (A100) free: a
card that fails loudly needs no more than that. The default's 800 MB and 20% are not a safety
margin there; they leave room for other work.

## 4. What more rows buy

The sampler, one config (b8), seed 5, nothing leaving the GPU (`keep_cap=0`), pinned to a ladder
of batches: R, 2R, 4R, ... (R = the card's resident warps: 1,280 on the T4, 6,912 on the A100)
up to the default batch, and, where free memory set the default, the rows that fit in free
memory less 1 GiB: 1.23× the default on every such formula, about what a fraction near 1 could
give. Each batch ran 4 s (T4) or 5 s (A100), at least 3 chunks, twice, the ladder up then down.
Attempts per second over the default batch's:

T4 (rows at the default: up to 11.8 GB of 15):

| instance | default K | set by | 16R | 32R | 64R | 1.23× |
|---|---|---|---|---|---|---|
| 10.sk_1_46 | 262,144 | cap | 1.01 | 1.02 | 1.02 | |
| 27.sk_3_32 | 262,144 | cap | 0.97 | 0.96 | 0.97 | |
| 55.sk_3_46 | 262,144 | cap | 0.98 | 1.00 | 0.95 | |
| b03 c25 | 262,144 | cap | 0.98 | 0.99 | 0.99 | |
| i2c c1 | 262,144 | cap | 0.93 | 0.96 | 0.99 | |
| s1196 c25 | 102,788 | memory | 1.03 | 0.96 | 0.98 | 1.04 |
| b12 c25 | 44,971 | memory | 1.06 | 1.04 | | 1.05 |
| div c1 | 24,899 | memory | 1.00 | | | 1.03 |
| b14 c10 | 11,402 | memory | | | | 0.96 |
| s38584 c10 | 8,306 | memory | | | | 0.94 |
| s15850 c25 | 7,790 | memory | | | | 0.99 |
| hyp c1 | 6,650 | memory | | | | 0.92 |
| b17 c10 | 3,567 | memory | | | | 0.97 |
| b18 c5 | 2,005 | memory | | | | 1.01 |

A100 (rows at the default: up to 32.9 GB of 40):

| instance | default K | set by | 16R | 32R | 1.24× |
|---|---|---|---|---|---|
| 10.sk_1_46 | 262,144 | cap | 0.99 | 1.00 | |
| 27.sk_3_32 | 262,144 | cap | 0.96 | 0.98 | |
| 55.sk_3_46 | 262,144 | cap | 0.97 | 0.99 | |
| b03 c25 | 262,144 | cap | 0.97 | 1.00 | |
| s1196 c25 | 262,144 | cap | 0.83 | 0.95 | |
| i2c c1 | 262,144 | cap | 0.94 | 0.99 | |
| b12 c25 | 125,712 | memory | 1.09 | | 1.10 |
| div c1 | 69,608 | memory | | | 1.03 |
| b14 c10 | 31,886 | memory | | | 1.02 |
| s38584 c10 | 23,234 | memory | | | 0.99 |
| s15850 c25 | 21,794 | memory | | | 1.06 |
| hyp c1 | 18,601 | memory | | | 1.05 |

b17 c10 and b18 c5, whose sources are fetched rather than committed, ran on the T4 only.

- **More rows than the default buy nothing that the noise does not swamp:** 0.92–1.05 on the T4,
  0.99–1.10 on the A100, with the same instance on either side of 1.00 from one card to the
  other, and a smaller batch beating the default as often (b12 at 16R: 1.06 and 1.09). Raising
  the 0.80 buys no rate: what it changes is who holds the memory.
- **The rate plateaus by about 32R.** Below the cap, rows still buy rate up to 16R–32R (i2c, a
  small formula with short trajectories, is the slowest to saturate); a formula whose rows the
  memory bound sets is at its plateau from 4R–8R on.
- **A check with assignments in chunks is flat in its batch:** 0.96–1.08 (T4) and 0.98–1.01
  (A100) of the default's sets per second, R to the default, on s1196 c25, s15850 c25 and
  s38584 c10, and b18 c5 on the T4.

## 5. A cap per card

The cap is one number on every GPU, so `--k-cap auto` resolves to the same batch on every card
wherever the cap binds (`cpp/src/calibrate.cpp`). A cap at a multiple of the resident warps
frees memory where rows no longer buy rate: at 64R, 82k rows on a T4 instead of 262,144. Below an
A100 it is smaller than 262,144 (64R is 442k there), it saves memory mostly on small formulas
whose rows are small anyway (262,144 rows of `27.sk_3_32` hold 2.6 GB), §4's ladder reads 0.95 at
64R on the T4 for `55.sk_3_46` and 1.00 at 32R, and it makes the default batch differ from card
to card. The user chose it on 2026-11-25 (issue 59); the multiple comes from a measurement on
more cards, appended here. *Measured in §8: 64.*

## 6. Mining beside the sampler (issue 76)

A sampling run builds mining's engine while the sampler holds its warmup rows (up to 1024). The
branch gives those rows back first on every run; main did so only under a limit. `gpap bench`,
main against the branch, each with its own library, mining on, one config (b8), seed 5, the batch
pinned at 49,152, 10 s; and as a user runs it (the bandit portfolio, the library's own batch).
Three reps a cell, the arm order alternated.

On the A100 and on the T4, in all 10 cells:
- **the mined clauses are identical** (264 to 24,700 units, every clause kept);
- **the sampler's rate is the same:** 0.97–1.02 of main's on the A100 and 0.99–1.05 on the T4,
  best rep against best rep, but for one count-terminated cell that ends in a fraction of a
  second (`27.sk_3_32` on the T4, 1.13);
- **mining and the rebuild take the same time** (for example 0.057–0.094 s and 0.011–0.014 s on
  `27.sk_3_32` on the A100, either arm). The end-to-end rate scatters (0.85–1.07 on the A100,
  0.99–1.07 on the T4) only because the short runs end in a fraction of a second and the
  sampler's construction varies by 0.2–0.3 s from run to run, on either arm.

So trimming costs nothing measurable, and mining sizes from the formula and the card on every
run, which also keeps it from fitting no row under a tight limit.

## 7. Not measured

- **A laptop under WSL2**, which pages device memory to the host instead of failing
  ([BENCHMARKING.md](../BENCHMARKING.md) §2): its default is unchanged, so its spill point was
  not needed for this decision. Measure it before any rule that takes more of a WSL2 card.
  *Measured in §9.*
- **An L4.** Colab's L4 assignment failed three times in a row on 2026-11-25 (502).
- **OpenCL** throughput against rows: OpenCL's batch is often set by the device's one-buffer
  limit instead ([OPENCL_FINDINGS.md](OPENCL_FINDINGS.md) §12).
- **OpenCL's default share beside other engines.** The branch takes what GPAP's other engines
  hold off the device's total before the share (`engine_batch`). A lone engine's batch is
  main's, and beside others the batch can only shrink, so it cannot spill where main did not
  ([BENCHMARKING.md](../BENCHMARKING.md) §2). Its rate was not measured.

## 8. The cap's multiple of the resident warps

The sampler, one config (b8), seed 5, nothing leaving the GPU (`keep_cap=0`), pinned to 4R, 8R,
16R, 24R, 32R, 48R, 64R, 96R and 128R rows (R = the card's resident warps), plus 262,144 and the
library's own batch, every batch that fits in free memory less 2 GiB. Each ran 3–4 s and at least
3 chunks, three passes (up, down, up); a batch's rate is its median pass. 16 formulas (the six
`sk` instances, five sequential circuits unrolled, five EPFL circuits). Over the formulas whose
ladder reached 128R (the others are memory-bound well below it), attempts per second over the
best batch's, mean (lowest):

| card | R | formulas | 16R | 32R | 48R | 64R | 96R | 128R | 262,144 is |
|---|---|---|---|---|---|---|---|---|---|
| laptop, 12 GB (WSL2) | 2,784 | 7 | 0.89 (0.83) | 0.95 (0.91) | 0.96 (0.93) | 0.96 (0.95) | 0.98 (0.97) | 0.99 (0.97) | 94R |
| A100 40 GB | 6,912 | 11 | 0.94 (0.89) | 0.96 (0.88) | 0.97 (0.92) | 0.97 (0.88) | 0.99 (0.88) | 0.98 (0.95) | 38R |
| T4 | 1,280 | 13 | 0.93 (0.91) | 0.96 (0.94) | 0.96 (0.89) | 0.97 (0.93) | 0.97 (0.91) | 0.98 (0.95) | 205R |

- **There is no knee.** Past 32R each doubling buys 1–2% on average, for twice the rows' memory
  and twice a chunk's wall; one formula (`55.sk_3_46`) swings by ±6% from rung to rung on every
  card.
- **64R** is within 3–4% of the best batch on average and within 5–12% on every formula; 128R
  would buy another 1–2%. The cap is 64 rows per resident warp: 81,920 on the T4, 178,176 on the
  laptop, 442,368 on the A100. On the T4 the five smallest formulas ran best at 262,144 (205R)
  and give up 1–2% at 64R; on the A100 the cap rises past 262,144, where 64R read 0.97 against
  0.96 at 38R.
- **Where the warp count is unknown** (OpenCL), the cap stays 262,144.

## 9. The reserve

The sampler (as §8) pinned to the rows that leave X free beside what GPAP holds, X from 3 GiB to
0, one fresh process per X: a warm-up batch at a quarter of the library's measures the real
bytes of a row, then the batch is reset to the X rows and runs three windows of 6 s. Rate: the
best window over the 3 GiB run's. Three memory-bound formulas: `hyp` c1 (1.8 MB a row), `b14`
c10 (1.0 MB), `b18` c5 (5.9 MB). On the laptop the Windows counter of the NVIDIA adapter's
shared (host) memory was sampled every 2 s, and a run that raised it by 1 GiB would have been
killed.

| card | formula | 2 GiB | 1 GiB | 512 MiB | 256 MiB | 0 |
|---|---|---|---|---|---|---|
| laptop (WSL2) | hyp c1 | 1.00 | 0.99 | 0.95 | 0.96 | 0.94 |
| laptop (WSL2) | b14 c10 | 1.00 | 1.01 | 1.01 | 1.02 | 0.99 |
| laptop (WSL2) | b18 c5 | 1.06 | 1.05 | 1.06 | 1.07 | **0.81** |
| T4 | hyp c1 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 |
| T4 | b14 c10 | 1.05 | 1.06 | 1.06 | 1.06 | 1.05 |
| T4 | b18 c5 | 1.01 | 1.00 | 0.99 | 0.98 | 0.98 |
| A100 | hyp c1 | 1.00 | 1.01 | 1.01 | 1.01 | 1.01 |
| A100 | b14 c10 | 0.98 | 0.98 | 0.99 | 0.99 | 0.98 |
| A100 | b18 c5 | 1.00 | 0.97 | 0.96 | 0.95 | 0.95 |

- **A native card needs no reserve for GPAP itself.** Every run completed, the last with 1–21 MiB
  free, at the same rate within noise (the A100's `b18` drifts with its batch, from 1 GiB on).
  §3 found the same with every entry point.
- **Under WSL2 nothing fails.** The calls after the reset took 200–300 MiB more on the laptop
  than X (the "256 MiB" runs left 62–76 MiB free, the "0" runs none), and the adapter's shared
  usage rose by 50–130 MiB in the runs that left under 300 MiB free. `b18` at 0 ran at 0.81: a
  silent spill, as §7 feared. With 60–80 MiB free every formula held its rate.
- **So the reserve is for what GPAP cannot see**: the desktop and other processes, the caller's
  tensors, kernels loaded at a first launch. The default is 1 GiB, several times what GPAP needed
  anywhere here, and the user sets it (`--mem-reserve`, `GPAP_MEM_RESERVE`). The sizing keeps
  all of it; on CUDA every allocation keeps half of it, so a pinned batch past the card fails as
  memory instead of paging, and allocation rounding or a first launch's kernel code does not.
- **What the new default changes.** Against the old rule (0.80 · (free − 800 MB)), all free
  memory less 1 GiB gives memory-bound formulas 22–24% more rows (`hyp` c1: 4,865 → 5,922 on the
  laptop, 6,650 → 8,153 on the T4, 18,601 → 23,093 on the A100). §4 measured 1.23× the default
  batch at 0.92–1.10 of its rate: the change uses the card, it does not speed it up.
- **As a user runs it** (`gpap sample`, the library's batch, b8, seed 5, 20 s, mining off, two
  reps in alternating order, main 814d75d against the branch): on the laptop, 21–22% more rows
  (GPAP's peak 10.5 GB of 12 instead of 8.6), the adapter's shared usage flat at 100–117 MiB in
  either arm, attempts per second 0.98–1.01 of main's; on the A100, 24% more rows (peak 38.1 GiB
  of 40 instead of 30.7), 0.99–1.00 of main's rate; on the T4, 23% more rows (13.5 GiB of 15
  instead of 11.0) at 1.00 (`hyp`) and 0.97 (`b14`, both reps), and `b18` now at the 64R cap
  (81,920 rows, 1.00). The no-change gate (BENCHMARKING.md §3: a batch pinned at 20,000, b8,
  seed 5, mining on) gave identical model counts and mined clauses on six cells on all three
  cards, and mining's own time is unchanged (locally: i2c c1 153 against 152 ms, b12 c25 1.00 s
  either way, 15 runs).

Not measured: an L4; OpenCL (whose default keeps its share of the total); two processes sharing a
card; a desktop under load beside the WSL2 runs.

## 10. The readout slice beside a memory-bound batch

`check_sets` with assignments (a (rows, 2N) readout) and flags only, on two random 3-CNFs built
to be memory-bound, the library's own chunks, the laptop card (WSL2), main 814d75d against the
branch, median of 5-9 calls after a warm-up (calls on WSL2 vary by up to 40% from run to run):

| formula | main | branch, 32-row slice | branch, slice set aside |
|---|---|---|---|
| N = 200,000, M = 700,000, 20,000 sets | 1.17-1.45 s (6 runs) | 1.76 s | 1.30-1.35 s |
| N = 2,000, M = 1,500,000, 60,000 sets | 1.64-1.66 s (6 runs) | 1.70 s | 1.63-1.64 s |

- **A batch that fills the default budget left the readout its 32-row minimum** (the one slice
  every bound counts): many small slices instead of main's 256 MiB, 30-50% slower on the large
  formula. The default budget now sets a slice aside, a sixteenth of it, at most a whole slice;
  a limit keeps the 32-row minimum its tight room relies on.
- **The sizing also counts back the staging the engine holds** (its decision matrix, slice and
  selection lists), as it counts back the rows. Without that, each call sized a few rows smaller
  than the last, found its own buffers did not fit beside the rows it kept, and gave the rows
  back: a flags-only check took 284 ms instead of 1-2 ms, and calls alternated between two
  batch sizes. A kept surplus under a sixteenth of the rows, or a quarter of the reserve, is
  now left alone.
- Flags only: 1-2 ms on both trees for the large formula, 1.52 s for the other.

Data: `$GPAP_RESULTS/memrule_20261125/local_4080/readbench.txt` (`scripts/readbench.py`).


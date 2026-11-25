# Memory — what the default batch takes, and why it stays

> **Reader:** contributors and researchers — how much of a GPU the default batch takes, what more rows would buy, and what lies outside GPAP's own count. A dated research log, not a user guide.
>
> **Status:** open — the default is being changed (GitHub issue 59; the user's decision of
> 2026-11-25: a batch cap of c times the resident warps, and all free memory less a reserve).
> Settled here: more rows than today's default buy no rate (§4), and a sampling run gives the
> sampler's rows back before it mines (issue 76, §6).
> **Measured:** 2026-11-25 on a Colab T4 (15 GB, driver 580.82) and an Athena A100-SXM4 (40 GB,
> driver 595.71), the step-2 branch of the memory-limit work, on main 48f4575. Not on a laptop
> under WSL2, nor on an L4 (§7).
> **Rule lives in:** `cpp/src/calibrate.cpp` (`mem_budget`, `kBatchCap`), `cpp/src/engine.cpp`
> (`engine_batch`), `cpp/src/memory.cpp` (`size_rows`), `gpap/sampling/pipeline.py`
> (`_free_rows_for_mining`).

The rule is [ENGINE.md](../ENGINE.md) §7. Data, drivers and logs:
`$GPAP_RESULTS/memdefault_20261125/`.

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
more cards, appended here.

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
- **An L4.** Colab's L4 assignment failed three times in a row on 2026-11-25 (502).
- **OpenCL** throughput against rows: OpenCL's batch is often set by the device's one-buffer
  limit instead ([OPENCL_FINDINGS.md](OPENCL_FINDINGS.md) §12).
- **OpenCL's default share beside other engines.** The branch takes what GPAP's other engines
  hold off the device's total before the share (`engine_batch`). A lone engine's batch is
  main's, and beside others the batch can only shrink, so it cannot spill where main did not
  ([BENCHMARKING.md](../BENCHMARKING.md) §2). Its rate was not measured.

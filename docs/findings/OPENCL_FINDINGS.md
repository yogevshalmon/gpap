# The OpenCL backend — the GPU socket and the OpenCL kernels

> **Reader:** contributors and researchers — what was built to run the engine through OpenCL,
> what was checked and measured, and what is open. A dated research log; the guide is
> [OPENCL.md](../OPENCL.md).
>
> **Status:** current. Correct on CPU OpenCL devices, through PoCL's CUDA driver and on NVIDIA's
> own driver. The current rates are §12 (an L4) and §13 (a laptop CPU); §14 shows the CUDA path
> lost nothing. §7 to §10 are the steps that led there.
> **Measured:** 2026-11-13 to 2026-11-16. A WSL2 laptop with an RTX 4080 Laptop (12 GB) and an
> i7-13800H: CUDA, PoCL 3.0 and 7.2 (CPU and CUDA drivers), Oclgrind 21.10. Colab L4s with
> NVIDIA's driver 580.82 (OpenCL 3.0) for §8, §9, §10 and §12.
> **Rule lives in:** `cpp/src/gpu_api.hpp` (the socket), `cpp/src/gpu_cuda.cpp` and
> `cpp/src/gpu_opencl.cpp` (the plugs), `cpp/kernels/bcp.cl` (the kernels).
> **Gated by:** `tests/engine/test_opencl_backend.py`, and every `engine` test under
> `GPAP_BACKEND=opencl` (in CI on PoCL's CPU driver, with no NVIDIA driver).

Origin: running entailed-clause mining inside a host that loads OpenCL C source and cannot load
Triton's binaries. [PORTABILITY.md](PORTABILITY.md) §6 had proposed the interface.

## 1. What was built (2026-11-13)

- **The socket.** Every device operation of the engine goes through `gpap::Gpu`: allocate,
  copy, launch a kernel by key with the Triton signature's arguments, and the rest. The CUDA
  plug holds what the engine used to call directly and is the only file that includes `cuda.h`.
- **The OpenCL plug** opens the OpenCL library at run time, so the build needs nothing new. Its
  first version had the scalar fixpoint kernel and the support kernels; WAVE and CLEAN followed
  the same day (§9). The trajectory (sampling) kernel is not ported and is refused by name.

## 2. The refactor changed nothing on CUDA

The no-change gate of [BENCHMARKING.md](../BENCHMARKING.md) §3, baseline build against this
one: `gpap_sample`, batch pinned at 20,000, one configuration, 20,000 models, seeds 5 and 6, on
`10.sk_1_46`, `27.sk_3_32`, `55.sk_3_46` and `7.sk_4_50`. All eight model files are
byte-identical, and the tests that passed before pass unchanged. Nor did it cost speed (§14).

## 3. The OpenCL kernels give the reference's answers, and CUDA's

On PoCL's CPU driver, compared exactly: conflict flags on every row and, on rows without a
conflict, the assignment and per-clause counts against the pure-Python reference; the raw
packed words and implication counts against the CUDA backend; the conflicting tuples of widths
1 and 2 against a brute-force enumeration; and the mined clause sets of `mine_entailed` and
`gpap mine` on both backends. A conflicted row may differ in its implication count, as the
kernel contract allows: only its flag is defined.

## 4. The tests can fail

Of thirteen bugs planted in `bcp.cl` one at a time (the sum field not updated, a contradiction
missed, a row offset dropped, a reduction that drops partial sums, and others), twelve fail a
test. The thirteenth, skipping the long clauses in `sound_rows`, cannot be caught here: the
scalar kernel never empties a long clause (it turns unit first, and pushing its last literal
conflicts).

## 5. Platform facts

- **WSL2 exposes no NVIDIA OpenCL**; the card is reachable through PoCL's CUDA driver (§7).
- **PoCL 3.0's CPU device:** OpenCL C 1.2, largest work group 4096, largest buffer 16 GiB of
  45 GiB, no sub-groups. The first engine of a process takes about 0.7 s, compiling `bcp.cl`.
- **The OpenCL constants** declared by hand in `gpu_opencl.cpp` equal the headers' values (all
  fifteen, checked against PyOpenCL's build).

## 6. Open

- The bit gathers' shape (64 columns per work item) is unmeasured.
- Sampling has no OpenCL kernel.
- No host other than libgpap's plug has loaded `bcp.cl`.
- Three WAVE details differ from Triton, unmeasured: a conflict stops the other lanes at the
  end of the wave, not at the next round; lanes find their segment by a linear search of the
  published bounds; the CLEAN restore deals whole trail literals to lanes where Triton flattens
  their segments.

## 7. On the GPU through PoCL's CUDA driver, and under Oclgrind (2026-11-13)

Three things had to change before a kernel ran on the card, none in `bcp.cl`'s logic:

- *A second LLVM in the process breaks PoCL's compiler.* With Oclgrind (LLVM 13) registered
  beside PoCL (LLVM 19), PoCL aborts on every kernel; `scripts/setup_opencl.sh` gives the
  loader a directory with PoCL alone.
- *PoCL's CUDA driver needs NVIDIA's `libdevice`,* which GPAP takes from the triton wheel.
- *A launch dimension past the first holds at most 65,535 groups.* The sampler's default batch
  failed; every launch is now one-dimensional (`GPAP_GRID_ITEM`). The CPU device had hidden it.

With CUDA pinned to its scalar kernel, the two scalar kernels do the same work on every row,
conflicted ones included, so implications per second is the same quantity on both backends.
Oclgrind with race detection reports nothing, and reports a removed row guard when one is
planted; its uninitialized-value plugin aborts on LLVM's `freeze`.

## 8. NVIDIA's own OpenCL: the first rates (2026-11-13, Colab L4)

NVIDIA's driver reports the L4 as OpenCL C 1.2, largest work group 1024, largest buffer 5.5 GiB
of 22 GiB. Method (`eval/profiling/backend_ab.py`): probing work, the same random literal pairs
at one pinned batch for every arm, best of three with the order alternated; 89 instances from
the five families unrolled 10 cycles, every arm reporting the same conflicting pairs. Geometric
means. Data: `$GPAP_RESULTS/opencl_l4_20261113/`.

- **The broadcast copies 4 columns per work item.** OpenCL scalar over CUDA scalar: 0.68 with
  16 columns, 0.92 with 4; 1, 2 and 4 are equal and 8 loses (0.88, 0.91, 0.88, 0.68 on 12
  instances). The reset was the difference: 2.10 of CUDA's time with 16, 0.93 with 4.
- **The work-group size does not matter** (256 against 64: 1.01).
- **CLEAN before WAVE.** The scalar-only backend ran at 0.40 of CUDA with WAVE and CLEAN (0.17
  past 100k clauses). On CUDA, CLEAN was worth 1.4 to 1.6 overall and 2.2 to 2.8 from 20k
  clauses; WAVE 1.4 overall and 1.9 past 100k.
- The scalar launch fell behind past 100k clauses (1.73 of CUDA's time); explained in §9.

## 9. The CLEAN and WAVE kernels (2026-11-13, Colab L4)

`bcp.cl` got the eight fixpoint kernels of `bcp.py`: scalar and WAVE, plain and CLEAN, both
packed splits. A WAVE row is a work group of 32 with Triton's atomics; what replaces the warp
shuffles is in `bcp.cl`. Its lanes race for the trail's order, so a conflicted row's counters
are not repeatable; the contract leaves that row unspecified. Of 16 planted bugs, 15 fail a
test; the last (not flagging the lane that empties a clause) cannot be caught, because the
lane that made the clause unit conflicts first.

§8's method on 15 of its instances across the size range (data:
`$GPAP_RESULTS/opencl_l4_wave_20261113/`):

- **OpenCL with WAVE and CLEAN ran at 0.73 of CUDA with both** (parity below 2k clauses, 0.56
  past 100k), against 0.33 for the scalar kernel alone. The ports are worth on OpenCL nearly
  what they are on CUDA: 2.2 against 2.8 times the plain scalar kernel.
- **A scalar row gets one work item, not a warp.** A warp per row brings the scalar launch past
  100k clauses from 1.58 to 1.07 of CUDA's time, so sharing a warp was §8's gap; but it loses
  below 2k clauses, where the scalar kernel is dispatched, and WAVE beats it above.
- **The dispatch threshold holds:** with CLEAN, WAVE is 1.5 to 2.8 times the scalar kernel from
  2k clauses up, and level below.
- The WAVE launch still took about 1.5 times CUDA's (2 past 100k clauses); found in §10.

## 10. Probing big formulas: where the time goes, and two changes (2026-11-14, Colab L4)

Widths 1 to 3 on formulas of 100k to 1.75M clauses, random tuples over the free literals, each
backend at its own batch. Data and scripts: `$GPAP_RESULTS/opencl_probe_ideas_20261114/`.

- **A chunk waits for its deepest row.** At width 1 the median probe implies 3 to 14 literals,
  the deepest 1,300 to 51,000, and the deepest 1% do 16% to 43% of the work. Sorting the tuples
  by a depth the host can compute (the sum of the literals' width-1 depths; rank correlation
  0.98 or more) gains up to 5.5 on OpenCL and 2.4 on CUDA, the most on the largest formulas.
- **OpenCL's batch is about half of CUDA's from 300k clauses up:** one buffer's size limit caps
  the trail (§12).
- **Change 1, kept: the lanes take their entries in step.** Lane i takes entries i, i + 32, ...
  of the flattened frontier, so all lanes issue their accesses together; before, they walked
  the segments one after the other, a memory round trip each. OpenCL over CUDA, geometric mean
  of six instances at each one's own batch: 0.47 to 0.81 (0.93 at one shared batch). Barriers
  cost nothing measurable (two extra per wave: 1.00).
- **Change 2, taken out: a probe queue.** A row pulls its next tuple as it finishes one and
  restores what it touched, so no row waits for a chunk. 2.0 to 3.2 times CUDA in chunks
  (geometric means by width), up to 9 on the largest; the rate levels off at about 1,000 rows.
  Removed on 2026-11-15 because `bcp.cl` follows the Triton kernels and Triton has no queue:
  GitHub issue 58 holds the design and the numbers, and the prototype is a patch in the data
  directory.

## 11. The review of the branch, and the gates it adds (2026-11-15)

A review of kernels, plugs, engine, tests and CI against the rule that `bcp.cl` matches the
Triton kernels in what a caller gets and in each kernel's logic. What it changed:

- **Launch shapes are declared:** the WAVE kernels carry `reqd_work_group_size(32, 1, 1)`.
- **The plug's three launch levers are gone,** their values settled by §8 and §9; one of them
  gave wrong answers without an error.
- **No NVIDIA driver needed:** libgpap opens libcuda at run time, so it builds and loads without
  it, and the Python faces and `gpap mine` / `gpap info` run on OpenCL without CUDA.
- **One automatic kernel choice per engine,** shared by reset and the sampler's pin.
- **CI can no longer pass without running a kernel:** a `bcp.cl` that does not compile fails the
  OpenCL tests instead of skipping them, and CI runs every `engine` test on PoCL's CPU driver.

Checked on the WSL2 laptop:

| check | result |
|---|---|
| `gpap mine --width 1 2 3` on s386, b03, b09 unrolled 25 cycles, CUDA against OpenCL | byte-identical clause files (193, 2,558 and 9,563 clauses) |
| widths 2 and 3 on s386 and b03 unrolled 5 cycles, both splits, WAVE and CLEAN (a new test) | the same conflicting tuples |
| the no-change gate on CUDA against the base commit (`gpap sample`, `gpap mine`) | identical model files and mined clauses |
| `pytest`, CUDA | 625 passed, 39 skipped |
| `GPAP_BACKEND=opencl pytest` | 508 passed, 156 skipped (104 sampling refused by name, 39 for want of `abc`, 13 `cuda_only`) |
| as CI: no CUDA, PoCL's CPU driver, `-m "(engine or opencl or core) and not slow"` | 179 passed |
| Oclgrind, every engine and mining test that needs no CUDA | 122 passed, no report |

Each new test was seen to fail on a planted bug.

## 12. Probing widths 1 to 3 past 100k clauses, after the review (2026-11-15, Colab L4)

The tree of §11 on an L4 with NVIDIA's own OpenCL, CUDA against OpenCL, both with WAVE and
CLEAN (`eval/profiling/backend_ab.py`): the same random tuples of 1, 2 and 3 free literals on
both, best of three. Fifteen formulas from 103k to 1.75M clauses: b14, b15, s15850.1, s35932,
s38584 (10 cycles), s35932 at 30 and b14 at 40 cycles; EPFL div, hyp, mem_ctrl; HWMCC 6s215rb0,
bj08amba2g4f3, bobsynthor, pdtswvibs8x8p0, texasparsesysp3 (10 cycles). Each backend at its own
batch, sized on a fresh engine (GitHub issue 56), then both at one shared batch. All 90 rows
reported the same conflicting tuples on both backends.

OpenCL's rate over CUDA's, tuples per second, geometric mean over the 15:

| batch | width 1 | width 2 | width 3 | the two past 1M clauses |
|---|---|---|---|---|
| each backend's own | 0.83 | 0.85 | 0.86 | 0.65 to 0.70 |
| one shared batch | 0.96 | 0.95 | 0.95 | 0.94 to 0.97 |

**The kernel is at parity; the batch is the gap.** At its own batch OpenCL has 0.59 of CUDA's
rows (0.45 to 0.52 on the sequential circuits, 0.67 or 0.68 on the rest): on all fifteen the
device's single-buffer limit (5.5 GiB, a quarter of the L4) caps the trail, the widest per-row
buffer at 4 bytes per variable. CUDA run at that batch keeps 0.86 to 0.90 of its rate.

**Kept as is (decided 2026-11-15).** Two ways past the gap are untried: the trail in more than
one buffer, and the probe queue (issue 58), after which the batch matters little (§10).

Data, scripts and logs: `$GPAP_RESULTS/opencl_l4_perf_20261115/`.

## 13. On the laptop's CPU, against its GPU (2026-11-15)

The tree of §11 on the WSL2 laptop: OpenCL through PoCL 7.2's CPU driver on the i7-13800H (20
threads; its device memory is WSL's 43 GiB of RAM) and through PoCL 7.2's CUDA driver on the
RTX 4080, each against CUDA on that GPU. §12's fifteen formulas, widths 1 to 3, one shared
batch (the GPU's), best of three; all 131 rows passed the checks.

OpenCL's rate over CUDA's on the laptop GPU, tuples per second, geometric mean over the 15:

| OpenCL on | kernel | width 1 | width 2 | width 3 |
|---|---|---|---|---|
| the GPU, PoCL's CUDA driver | WAVE + CLEAN | 0.87 | 0.89 | 0.89 |
| the CPU | SCALAR + CLEAN | 0.27 | 0.25 | 0.25 |
| the CPU | WAVE + CLEAN, the automatic choice | 0.13 | 0.11 | 0.11 |

- **The CPU probes at a quarter of the laptop GPU** with SCALAR (0.15 to 0.53 by formula).
- **On the CPU SCALAR is the faster kernel at every size**, but the automatic choice, made for
  GPUs, takes WAVE from 2,048 clauses: WAVE runs at 0.43 to 0.48 of SCALAR on the fifteen, and
  at 0.24 to 0.49 on each of the 24 seq formulas at width 2 (942 to 587k clauses).
- **More rows buy the CPU little, and its budget is the host's RAM.** Uncapped, the core
  budgets 0.8 of the device's memory; this run capped it (`GPAP_VRAM_BUDGET_MIB=16384`). At its
  own batch under that cap, twice the GPU's rows, SCALAR gains 4%.
- **PoCL's CUDA driver** gives 0.87 to 0.89 of CUDA at equal rows, below NVIDIA's own OpenCL
  on the L4 (§12).
- **Kept as advice (decided 2026-11-15).** The automatic choice and the budget stay as they
  are; [OPENCL.md](../OPENCL.md) tells a CPU user to pin SCALAR and cap the budget.

Data, scripts and logs: `$GPAP_RESULTS/opencl_local_cpu_20261115/`.

## 14. The CUDA path is as fast as before the socket (2026-11-16)

The base commit (9bb462d) against the tree of §13, each with its own libgpap, on the CUDA
backend of the WSL2 laptop's RTX 4080; the two arms back to back with the order alternated
([BENCHMARKING.md](../BENCHMARKING.md) §1b), each scored by its best repetition.

| work | cells | branch over base | check |
|---|---|---|---|
| probing, widths 1 and 2, WAVE + CLEAN, 65,536 random tuples at a pinned batch | div, bobsynthor, s35932 (10 cycles), b14 (40 cycles): 172k to 1.69M clauses | geometric mean 1.001 (0.980 to 1.014), 3 reps | the same conflicting tuples on both |
| sampling, `gpap bench`, `--portfolio b8`, seed 5, `--k-cap 49152`, 20,000 models | 10.sk_1_46, 27.sk_3_32, 55.sk_3_46, 7.sk_4_50; b03 and b12 at 25 cycles | geometric mean 1.03 (0.86 to 1.17), 3 reps | the same model counts on both |

The sampling cells are short, and one arm's own repetitions spread by up to 40%. The two
farthest from 1, b03 (0.86) and 7.sk_4_50 (1.17), were run eight more times each: best of eight,
0.99 and 0.94; median of eight, 1.17 and 1.06. The sign changes with the statistic, so neither
is a difference. Data, scripts and logs: `$GPAP_RESULTS/cuda_nochange_perf_20261116/`.

# Changelog

> **Reader:** users and packagers of GPAP — what each release adds, changes or breaks.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The C library
follows the rule in [CONTRIBUTING.md](CONTRIBUTING.md): an ABI break moves its soname, an additive
change does not.

## Unreleased

### Engine
- The WAVE kernel's atomics are relaxed, and two barriers per propagate order a row's threads
  instead of a GPU-wide fence around every atomic. With WAVE running, the sampler is 1.4 to 1.5
  times as fast on an A100 and 1.2 times on a T4, and the queued sweeps 1.1 times on the A100;
  every result is the same ([docs/findings/KERNEL_AB_FINDINGS.md](docs/findings/KERNEL_AB_FINDINGS.md),
  issue 51). The scalar kernels and the dispatch rule are unchanged.
- Probe sweeps (`gpap_probe_*`, `LiteralProbe`, mining) and flags-only assumption checks
  (`gpap_check_sets` without assignments or sat counts) run on a job queue, on both backends: a
  few rows of state, each taking set after set and restoring what it touched, so a launch no
  longer waits for its deepest set. Same results in the same order, 1.8 to 3.8 times the
  chunked sweep at widths 1 to 3 (geometric means over four benchmark sets;
  [docs/findings/PROBE_QUEUE_FINDINGS.md](docs/findings/PROBE_QUEUE_FINDINGS.md), issue 58).
  Mining, which has a time budget, can prove more clauses in it.
- The queue runs the WAVE kernel unless `GPAP_KERNEL` or `gpap_engine_set_mode` pins one.
  `GPAP_QUEUE=0`, or `GPAP_CLEAN=0`, runs the sweeps in chunks as before.
- Changed meanings: `seconds` in `gpap_probe_options` is checked between launches, and `batch`
  (like `chunk` of `gpap_check_sets`) is the rows of state a launch uses. After a queued call
  the engine is as after `gpap_engine_reset` to the queue's rows, so a readout reads the
  baseline rather than the last chunk.
- Additive ABI, same soname: `gpap_engine_queue_stats` (launches, sets and the kernel the queue
  ran); in Python, `Engine.queue_stats()` and `queue_mode()`.

### Memory
- A memory limit: the most GPU memory GPAP's engines hold on a device, summed over engines
  (issues 59 and 72). `--mem-limit 8G` on `gpap sample`, `mine`, `bench` and `info`;
  `GPAP_MEM_LIMIT` for every face; `gpap.set_mem_limit` and `gpap.engine.memory` in Python;
  `gpap::set_mem_limit` / `gpap::memory` in C++; `gpap_set_mem_limit`, `gpap_parse_mem_limit`,
  `gpap_mem_limit`, `gpap_mem_limit_source`, `gpap_mem_held`, `gpap_mem_peak` and
  `gpap_mem_reset_peak` in C. Sizes take binary units (`8G`, `512M`) or a share of the device
  (`75%`); a malformed limit is an error. Every batch is sized under it, and an allocation past
  it fails with `GPAP_ERR_MEMORY` instead of spilling. On OpenCL the limit takes the place of the
  default share of the device's total. Without a limit the batch is the one it was.
- What GPAP holds is counted buffer by buffer, as formula bytes and row bytes:
  `gpap_engine_mem_held`, Python `Engine.mem_held`. `gpap_engine_batch_bound` (Python
  `batch_bound`) says what set the last batch: the cap, free memory, the limit, a host's grant,
  the device's one-buffer limit, the work, the rows the device runs at once, or a pin. A
  sampling run's stats carry `batch_bound`, `mem_limit` and `mem_peak`.
- Changed: where not even one row fits, a call fails with `GPAP_ERR_MEMORY` (Python
  `MemoryError`, as `GpapMemoryError`) and says what a row needs, instead of running a batch of
  one row. `gpap_check_chunk_size` returns 0 then, and the C++ `AssumptionChecker::chunk_size`
  throws. The job queue declines where its launch buffers leave no row, and the call runs in
  chunks, where it used to run one queued row.
- A reset that the device, or the limit, refuses part-way leaves the engine with no rows (K
  reads 0) rather than with rows whose buffers were freed; the next reset that fits takes rows
  again.
- `gpap_engine_trim` (Python `.trim()`) gives an idle engine's rows back and keeps its formula.
  Under a limit, a sampling run trims its sampler before mining, so mining fits beside it
  (issue 76).
- The GPU interface reports memory in one call, `Gpu::memory`, which says whether free memory
  is measured (CUDA), the device's total (OpenCL) or a host's exact grant.
- Additive ABI, same soname: the functions above, `gpap_bound`, `gpap_limit_source` and
  `gpap_mem_kind`.
- The job queue takes off the budget what a call's own launches can allocate, for its sets and
  its widest set, not a fixed worst case of about 70 MB, so a tight limit or a small grant keeps
  it (issue 72).
- Under a limit or a grant, a batch smaller than the engine's last gives the extra rows back, so
  the room its sizing counted for the call's later buffers is there; without one the buffers
  are kept for the next bigger batch, as before.
- Fixed: on OpenCL, a buffer the device cannot back now fails when it is allocated, with
  `GPAP_ERR_MEMORY`, as on CUDA. NVIDIA's runtime backs a buffer only at its first use, so a
  batch pinned past the card failed at the first launch as a driver error and left the engine
  unusable (issue 79).
- On OpenCL, which cannot measure free memory, the default share of the device's total is taken
  after what GPAP's other engines hold there, as CUDA's measured free memory already is: a second
  engine beside a big one sizes to what is left instead of planning on the whole device.
- A sampling run gives the sampler's rows back before every mining call, with no limit too, so
  mining sizes from the formula and the card rather than from what the warmup rows leave
  (issue 76).
- C++: `gpap::reset_peak` and `gpap::parse_mem_limit`, the C ABI's last two memory functions.
- Changed (issue 59): without a limit, the batch is sized from all free memory less a memory
  reserve, 1 GiB by default, instead of 0.80 · (free − 800 MB); every bound now counts what a
  call allocates after it is sized, so no fraction is held back, and a readout slice grows only
  into the room the budget leaves. Memory-bound formulas get about a quarter more rows
  ([docs/findings/MEMORY_FINDINGS.md](docs/findings/MEMORY_FINDINGS.md) §9). Set the reserve
  with `--mem-reserve` (`sample`, `mine`, `bench`, `info`), `GPAP_MEM_RESERVE`,
  `gpap.set_mem_reserve`, `gpap::set_mem_reserve` or `gpap_set_mem_reserve`, in the limit's
  grammar; `gpap_mem_reserve`, `gpap.engine.memory()["reserve"]` and a sampling run's
  `mem_reserve` report it. Under a limit on CUDA the limit is no longer cut to 80% of free
  memory. OpenCL keeps its share of the total, less the reserve.
- Changed: on CUDA an allocation that would leave less than half the reserve free fails with
  `GPAP_ERR_MEMORY` (the sizing keeps all of it), so a pinned batch past the card fails instead of paging to host memory,
  which WSL2 did silently.
- Changed: the batch cap is 64 rows per warp the card keeps resident, instead of 262,144 on
  every card (82k on a T4, 178k on a 12 GB laptop card, 442k on an A100; §8 of the same
  findings); 262,144 where the warps are unknown (OpenCL). `--k-cap auto` now differs per card.
- Changed: the sampler sizes its batch at the start of every run, from the memory free then,
  instead of once at creation; a pinned batch stays. `gpap_sampler_set_k(s, k)` (C++
  `Sampler::set_k`) now pins for any k > 0, the sized batch too, and `k <= 0` unpins: a caller
  that saves `gpap_sampler_k` and restores it by value leaves the sampler pinned. Python's
  `pinned_batch` restores the pin state.
- Changed: `gpap_mem_available`'s `fraction` scales the engine's own budget (after
  `GPAP_VRAM_BUDGET_MIB` caps it), and every face defaults to 1 (was 0.80):
  `gpap::mem_available`, `_core.mem_available`, `calibrate.gpu_mem_available`. A caller that
  still passes 0.80 gets four fifths of the new budget, not the engine's. The rival harness
  takes the engine's budget.
- Additive ABI, same soname: `gpap_set_mem_reserve`, `gpap_mem_reserve`; `gpap::Memory` gains
  `reserve`. The changed meanings above (`gpap_sampler_set_k`, `gpap_sampler_k`,
  `gpap_mem_available`'s fraction, `gpap_default_k_cap`) keep every signature; a caller that
  relied on the old ones needs the changes listed there.

### Sampling
- Fixed: a run's `phases` sum to `wall_total_s` in a fresh process too. The wire map is billed
  to `parse`, the library's first load to `construct`, and the run's report (provenance, memory,
  rates) to `output`; on an A100 node they had left about 15 ms of the first run unbilled.

### Evaluation
- `eval/colab/colab_supervisor.py` drives the VM through `colab ssh` and needs google-colab-cli
  0.8 or later, which removed the `console` it used; it refuses an older CLI at start
  ([eval/colab/README.md](eval/colab/README.md)). The TACAS 2027 artifact keeps its own copy.

## 0.4.0 — 2026-11-23

An OpenCL backend, and fixes to the engine, its C++ face and mining. libgpap is 0.4.0 with the
same soname: no existing symbol or struct changed. Callers note two changed behaviours, both below:
`gpap_check_sets` without sat counts refuses a later packed readout, and C++
`AssumptionChecker::check` returns flags only by default.

### Engine
- An OpenCL backend, selected with `GPAP_BACKEND=opencl` ([docs/OPENCL.md](docs/OPENCL.md)). It
  runs batched assumption checking, probing, mining and the readouts on an OpenCL device (a GPU
  of any vendor, or a CPU), from OpenCL C kernels (`cpp/kernels/bcp.cl`) compiled on the
  device: the scalar and the WAVE propagate kernel, each with CLEAN mode. It does not sample:
  `gpap sample` and `gpap_sampler_create` refuse up front. `GPAP_OPENCL_LIB` names the OpenCL
  library. CUDA stays the default, the reference and the fast path.
- libgpap no longer links `libcuda`: it opens the NVIDIA driver at run time
  (`GPAP_CUDA_LIB` names another), so it builds and loads without it, and the CI stub is gone.
  The CLI and the Python faces run on the OpenCL backend without a CUDA device.
- The engine reaches the GPU through one internal interface (`cpp/src/gpu_api.hpp`) with a plug
  per GPU library. Nothing changes on CUDA: same-seed runs give the same models. A CUDA tensor
  given to an OpenCL engine goes through a host copy.
- Additive ABI, same soname: `gpap_backend`, `gpap_engine_backend`, `gpap_engine_device_name`.
  `GPAP_ERR_CUDA` now means a failed GPU driver call on either backend; the Python faces' texts
  for codes 2 and 5 read "GPU driver error" and "missing kernel".
- `scripts/setup_opencl.sh` installs a pinned OpenCL runtime without root where the system has
  none (PoCL: its CPU driver, or its CUDA driver for WSL2).
- `mine_entailed` raises when its engine cannot be built, instead of returning no clauses.
- A clause that repeats a literal is built from its distinct literals: `(x | x | y)` propagates
  like `(x | y)`, and `(x | x)` is the unit `x`. The repeat used to keep the clause one slot
  short of a unit, so BCP missed its implication (issue 57). Clause count and order are
  unchanged; `total_lits`, clause sizes and the clause read-back count distinct literals.
- A later call on one engine (`gpap_check_sets`, a probe sweep, a sampler) sizes the batch the
  first call did. It used to size from what the first call left free: about a sixth of the batch,
  and half the probing rate on a 1.2M-clause formula (issue 56).
- `gpap_check_sets` that returns assignments but no sat counts takes the CLEAN kernel, as a
  flags-only call does: up to 1.4x faster, identical results. Afterwards the engine refuses a
  packed readout (`gpap_sat_rows`, `gpap_check_sound`) until its next reset, as after any CLEAN
  launch (issue 42). The CLEAN contract in the findings now matches the header: bitmap readouts
  are served after a CLEAN launch.

### Sampling
- Mining no longer builds a sampler while the previous one is alive, which left the new batch
  about a thousand rows short and could revert clauses that paid (issue 48).
- Mined units, binaries and ties are kept; only a find with a wider clause can be reverted. The
  `mined` report gives the batch and completion rate before and after.
- Mining records its probe engines' resets (`resets_full`, `resets_lite`) in the `mined` report
  and `gpap mine --stats`; `BCPEngine.reset_stats()` and `LiteralProbe.reset_stats()` read them.
  A lite count above 0 is the record that CLEAN mode ran (issue 43).

### C++
- `gpap::AssumptionChecker::check(sets)` returns the conflict flags and implication counts only,
  as the Python face does; `check(sets, /*assignments=*/true)` adds what `value()` and
  `literals()` read. The default used to be `true`, which cost n x 2N bytes and the plain kernel
  on every call (issue 44). `value()` on a flags-only result throws and says which argument to set.
- `gpap::Engine::check_sets` takes assumption sets flat (DIMACS literals and n + 1 offsets), as
  the C ABI and Python do, and `AssumptionChecker::check` runs on it. Both can return per-clause
  sat counts (`sat_count`). Additive: same soname (issue 46).
- The wrapper's own refusals (`gpap::Error::own`) no longer carry the core's last error message,
  which was stale there; `gpap_check_sets` names every argument it refuses.

## 0.3.1 — 2026-10-23, the TACAS 2027 artifact

The engine is unchanged: libgpap is still 0.3.0, and the sampler is the one that measured the
paper's rows. The paper is on arXiv ([arXiv:2610.28805](https://arxiv.org/abs/2610.28805)), and
`CITATION.cff` cites it.

### Installation
- A Docker image of the tool (`Dockerfile`; [docs/DOCKER.md](docs/DOCKER.md)). It compiles the
  engine for the GPU it runs on the first time, and keeps it in a cache volume.

### The TACAS 2027 artifact
- The artifact is a Docker image, a layer over the tool image (`eval/artifact/Dockerfile`).
  `eval/artifact/make_image.sh` builds both from the artifact archive, and
  `eval/artifact/check_image.sh` gates them. The guide leads with it.
- `reproduce.sh tables` and each GPU tier write one `SUMMARY.txt`, the lines to check, when they
  end, and everything else under `details/`. A bare `compare` prints the whole comparison.
- `compare_campaign.py` names the cells whose verdict changed in its closing block, the part a
  tier's summary shows.
- `make_artifact.sh` accepts a relative output directory and a relative `PYTHON`, both taken
  from where they are typed. It used to lose the uniformity rows and refuse to package. It also
  writes the venue's upload, a zip of the AE README and the licence, and a `.sha256` of the
  source archive.
- `reproduce.sh tables` prints one line per check. `RESULTS.md` is now the paper's tables as the
  paper prints them, with the ratios its text quotes and every cell not counted named under each
  table; the analysis behind the paper's wording moved to `paper_tables.py --out-notes`, which
  the artifact does not run. The archive no longer carries the emission audit, which the paper
  does not cite; `eval/rivals/check_emission_symmetry.py` still runs it. `paper_tables.py` lost
  the tables the paper does not use (the exact-oracle uniformity, the soft-portfolio price, the
  re-check and the `gpap bench` family matrix).
- The AE README is the evaluation guide, and the image gates read it; `ARTIFACT.md` holds the
  detail behind it and repeats none of its commands.

## 0.3.0 — 2026-10-17, the first public release

The Python package starts at 0.3.0, the version of `libgpap` it ships with; earlier versions were
never published.

### Installation
- `gpap info`, and `eval/artifact/reproduce.sh check`, report a missing torch in one line
  instead of a traceback.
- Package metadata: the MIT licence as an SPDX expression, and the supported Python versions
  (3.10 to 3.13, all tested in CI).

### The engine
- `libgpap`, one C++ library: Boolean constraint propagation over thousands of independent
  partial assignments of one formula, in one GPU launch, each to its own fixpoint.
- Two entry points: batched assumption checking (`AssumptionChecker`, `LiteralProbe`) and fused
  randomized trajectories, the basis of sampling.
- A C ABI (`cpp/include/gpap/gpap.h`), a header-only C++ wrapper (`gpap.hpp`) and a Python face
  (`gpap.engine`) over the same implementation.
- Kernels written in Triton and compiled into the library ahead of time. The runtime needs only
  the NVIDIA driver; building needs no CUDA toolkit.

### Applications
- `gpap sample`: solutions of a CNF formula or a circuit (`.bench`, AIGER), with simplification,
  equivalent-literal substitution, entailed-clause mining, an optional ABC encoder, and soundness
  and simulation checks ([docs/SAMPLING.md](docs/SAMPLING.md)).
- `gpap mine`: clauses a formula already implies, found under a work budget.
- `gpap bench`: a benchmark driver, with an optional CMSGen comparison.

### Evaluation
- `eval/`: campaign drivers, and harnesses that run CMSGen and the GPU samplers on the same
  constraints as GPAP ([docs/EVAL_RUNBOOK.md](docs/EVAL_RUNBOOK.md)).
- The TACAS 2027 submission's evaluation and artifact, in
  [docs/papers/tacas-27/](docs/papers/tacas-27/README.md).

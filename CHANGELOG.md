# Changelog

> **Reader:** users and packagers of GPAP — what each release adds, changes or breaks.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The C library
follows the rule in [CONTRIBUTING.md](CONTRIBUTING.md): an ABI break moves its soname, an additive
change does not.

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

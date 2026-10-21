# Changelog

> **Reader:** users and packagers of GPAP — what each release adds, changes or breaks.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The C library
follows the rule in [CONTRIBUTING.md](CONTRIBUTING.md): an ABI break moves its soname, an additive
change does not.

## Unreleased

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
  writes the venue's upload, a zip of the AE README and the licence.

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

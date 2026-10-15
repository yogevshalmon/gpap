# Changelog

> **Reader:** users and packagers of GPAP — what each release adds, changes or breaks.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The C library
follows the rule in [CONTRIBUTING.md](CONTRIBUTING.md): an ABI break moves its soname, an additive
change does not.

## 0.3.0 — the first public release (unreleased)

The Python package starts at 0.3.0, the version of `libgpap` it ships with; earlier versions were
never published.

### Installation and the artifact
- `gpap info`, and `eval/artifact/reproduce.sh check`, report a missing torch in one line
  instead of a traceback, on machines that only regenerate the paper's tables.
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
- `eval/`: the experiments behind the TACAS 2027 paper, and
  [`eval/artifact/reproduce.sh`](eval/artifact/reproduce.sh) to reproduce them
  ([ARTIFACT.md](ARTIFACT.md)).

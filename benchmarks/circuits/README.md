# Circuit benchmarks — provenance, licenses, citations

Committed here is a **curated subset** (small files only); `scripts/fetch_benchmarks.sh` fetches
the full upstream sets into `.gpap/benchmarks/` (beside the repo) for breadth runs.

| directory | suite | source | license / terms | cite |
|---|---|---|---|---|
| `bench/comb/` | ISCAS-85 (c432…c7552) | classic distribution (via the DEMOTIC instance set) | de-facto public research benchmarks | Brglez & Fujiwara, ISCAS 1985 |
| `bench/seq/s*.bench` | ISCAS-89 | classic distribution | de-facto public research benchmarks | Brglez, Bryan & Kozminski, ISCAS 1989 |
| `bench/seq/b*.bench` | ITC'99 (I99T) | [cad-polito-it/I99T](https://github.com/cad-polito-it/I99T) | **EUPL-1.2** (see upstream LICENSE) | Corno, Sonza Reorda & Squillero, *IEEE D&T* 17(3), 2000 |
| `epfl/` | EPFL combinational suite (arithmetic + random_control, AIG) | [lsils/benchmarks](https://github.com/lsils/benchmarks) | freely usable research benchmarks (MIT-licensed repo) | Amarù, Gaillardon & De Micheli, IWLS 2015 |
| `hwmcc17/` | HWMCC'17 single-property AIGs (curated) | [fmv.jku.at/hwmcc17](https://fmv.jku.at/hwmcc17/) | competition benchmarks, free for research use | Biere, van Dijk & Heljanko, *HWMCC 2017* (FMCAD'17 report) |

Notes for the sampling use case:

- **ITC'99**: plain `b<k>.bench` variants (not `_opt`/`_C`/`_r`). Committed up to b15 (≤336 KB);
  the big ones (b17–b22, 1.3–14.5 MB each) are NOT committed — fetch them with the script. At
  T=25, b02/b05/b07/s526 have **vacuous** output pins and b01/b09 are near-vacuous (see
  docs/findings/INSIGHTS.md) — exclude from headline comparisons.
- **EPFL** circuits are combinational: conversion clamps to T=1; pin a SINGLE output
  (`--pin-outputs 1`, the DEMOTIC-style setup) — pinning all outputs of an arithmetic block
  over-constrains to a near-point solution set.
- **HWMCC'17**: each AIG has ONE output (the property). The committed 7 are exactly the
  instances (of the 231 with ≤30k ANDs) whose property output is **non-constant under random
  stimuli at T=25** (fires with frequency 3e-4 … 0.97, measured with
  `gpap.circuits.simulate`) — for every other instance the output pin is vacuous and the
  instance is a free-hypercube throughput test, not constrained sampling. Pinning the RARE
  value of the property bit (`--target rare`) means "sample diverse witness traces".
  The full 300-instance sweep is `.gpap/benchmarks/hwmcc17_sweep.json` after running
  the fetch script.

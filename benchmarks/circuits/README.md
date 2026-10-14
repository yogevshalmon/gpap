# Circuit benchmarks — provenance, licenses, citations

> **Reader:** anyone using or redistributing the circuit benchmarks — where each set comes from, its licence, and what to cite.

A **curated subset** (small files) is committed. `scripts/fetch_benchmarks.sh` fetches the 20
EPFL circuits (the same ones) and the whole HWMCC'17 set into `.gpap/benchmarks/`, and ITC'99
`b17`–`b22` into `bench/seq/`.

| directory | suite | source | license / terms | cite |
|---|---|---|---|---|
| `bench/comb/` | ISCAS-85, **all 11** (c17, c432…c7552) | classic distribution (the DEMOTIC set) | none stated | Brglez & Fujiwara, ISCAS 1985 |
| `bench/seq/s*.bench` | ISCAS-89 (the 8 in Recurrent-CircuitSAT's Table II, plus `s35932`) | classic distribution; `s208.1`, `s4863`, `s15850.1`, `s35932`, `s38584` byte-identical to the copies in the DEMOTIC and Recurrent-CircuitSAT repositories | none stated | Brglez, Bryan & Kozminski, ISCAS 1989 |
| `bench/seq/b*.bench` | ITC'99 (I99T) | [cad-polito-it/I99T](https://github.com/cad-polito-it/I99T) | **EUPL-1.2** — text in [LICENSE-ITC99.EUPL-1.2.txt](LICENSE-ITC99.EUPL-1.2.txt), shipped because the licence requires it of a redistributor. `b15.bench` differs from I99T only by its final empty line | Corno, Sonza Reorda & Squillero, *IEEE D&T* 17(3), 2000 |
| `epfl/` | EPFL combinational suite (arithmetic + random_control, AIG) | [lsils/benchmarks](https://github.com/lsils/benchmarks) | MIT — notice in [epfl/LICENSE](epfl/LICENSE) | Amarù, Gaillardon & De Micheli, IWLS 2015 |
| `hwmcc17/` | HWMCC'17 single-property AIGs | [fmv.jku.at/hwmcc17](https://fmv.jku.at/hwmcc17/) | none stated by the distributor | Biere, van Dijk & Heljanko, *HWMCC 2017* (FMCAD'17 report) |

Notes:

- **Which circuits, and why.** The committed set is what the GPU-rival papers evaluate: ISCAS-85 in
  full (DEMOTIC, arXiv 2502.08086 Table II) and Recurrent-CircuitSAT's 8 ISCAS-89 instances (arXiv
  2502.21226 Table II), whose `s35844` is no ISCAS-89 circuit — its row (12 PI / 278 PO / 19,253
  gates / 1,452 DFF) is `s38584`, which we ship, with `s35932` added.
- **The fetch script installs into `bench/seq/`**, not only `.gpap/benchmarks/`: every driver lists
  only that directory, so an instance elsewhere is silently absent. `b17`–`b22`
  (1.3–14.5 MB) land there, gitignored. There is no `b16`: the set is
  b01-b15 + b17-b22 = 21 circuits, 30 sequential with the 9 ISCAS-89 files, and the campaign runs
  all 30.
- **The big ITC'99 circuits are a task limit, not a size limit** (2026-09-02): they encode inside
  the packed envelope (T=25 `abc:fast`: b21 252k clauses, b17 442k, under `s35932`'s 319k), but at
  a rare pin the target has 0 hits in 917,504 simulated sequences
  (p_accept < 1.1e-6) and GPAP discards conflicted trajectories, so nothing is emitted. Depth is
  the lever: b20 gives 5,574 models at T=5, nothing at T=10 or T=25.
- **Our four long-standing ISCAS-89 files** (`s386`, `s526`, `s832`, `s1196`) differ byte-wise from
  the rivals' copies but are verified equivalent: identical PI/PO/DFF/gate counts, identical OUTPUT
  order (which their positional tables index), identical content modulo comments, whitespace
  and line order. The rivals get our file.
- **ITC'99**: plain `b<k>.bench` variants (not `_opt`/`_C`/`_r`), committed up to b15 (≤336 KB). Some
  instances have vacuous pins at some depths (every input sequence satisfies the target); the
  evaluation screens them per row ([docs/BENCHMARKING.md](../../docs/BENCHMARKING.md) §8).
- **EPFL** circuits are combinational: conversion clamps to T=1; pin a single output
  (`--pin-outputs 1`, DEMOTIC-style), since all outputs of an arithmetic block over-constrain to a
  near-point set.
- **HWMCC'17**: one output per AIG, the property. The committed 7 are those (of the 231 with ≤30k
  ANDs) whose property output is non-constant under random stimuli at T=25 (3e-4 … 0.97,
  `gpap.circuits.simulate`); elsewhere the pin is vacuous. The sweep that chose them is not committed.

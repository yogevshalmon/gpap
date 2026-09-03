# Circuit benchmarks — provenance, licenses, citations

Committed here is a **curated subset** (small files only); `scripts/fetch_benchmarks.sh` fetches
the full upstream sets into `.gpap/benchmarks/` (beside the repo) for breadth runs.

| directory | suite | source | license / terms | cite |
|---|---|---|---|---|
| `bench/comb/` | ISCAS-85, **all 11** (c17, c432…c7552) | classic distribution (via the DEMOTIC instance set) | de-facto public research benchmarks | Brglez & Fujiwara, ISCAS 1985 |
| `bench/seq/s*.bench` | ISCAS-89 (the 8 in Recurrent-CircuitSAT's Table II) | classic distribution; `s208.1`, `s4863`, `s15850.1`, `s35932` taken from the rivals' own pinned checkouts | de-facto public research benchmarks | Brglez, Bryan & Kozminski, ISCAS 1989 |
| `bench/seq/b*.bench` | ITC'99 (I99T) | [cad-polito-it/I99T](https://github.com/cad-polito-it/I99T) | **EUPL-1.2** (see upstream LICENSE) | Corno, Sonza Reorda & Squillero, *IEEE D&T* 17(3), 2000 |
| `epfl/` | EPFL combinational suite (arithmetic + random_control, AIG) | [lsils/benchmarks](https://github.com/lsils/benchmarks) | freely usable research benchmarks (MIT-licensed repo) | Amarù, Gaillardon & De Micheli, IWLS 2015 |
| `hwmcc17/` | HWMCC'17 single-property AIGs (curated) | [fmv.jku.at/hwmcc17](https://fmv.jku.at/hwmcc17/) | competition benchmarks, free for research use | Biere, van Dijk & Heljanko, *HWMCC 2017* (FMCAD'17 report) |

Notes for the sampling use case:

- **WHICH circuits are here, and why**: the committed set is exactly what the GPU-rival papers
  evaluate on, so the head-to-head instance list follows from their protocols rather than from a
  choice of ours — ISCAS-85 in full (DEMOTIC, arXiv 2502.08086 Table II) and the 8 ISCAS-89
  instances of Recurrent-CircuitSAT's Table II (arXiv 2502.21226). Their table prints `s35844`,
  which is not an ISCAS-89 circuit; `s35932` is the only `s359xx` branch in their `run.py` and is
  what we ship. The other ~39 ISCAS-89 circuits ship inside both rival checkouts under
  `.gpap/src/*/data/ISCAS/` but appear in NEITHER paper, so they are not committed here.
  ITC'99 `b17`/`b19`/`b22` are in Recurrent's table but are 1.3-14.5 MB; `scripts/fetch_benchmarks.sh`
  fetches them and **installs them into `bench/seq/` alongside the committed circuits** (they are
  gitignored, so `git status` is unchanged). That install step is what makes them visible: every
  driver builds its instance list by listing that ONE directory, so a big instance sitting only in
  `.gpap/benchmarks/` is silently absent — no error, no skip line, just a sweep six instances
  short. They were in the 2026-09-02 campaign only because they had been copied over by hand.
  (There is no `b16` — the suite jumps b15 -> b17, so the full set is b01-b15 + b17-b22 = 21
  circuits, 29 sequential in all with the eight ISCAS-89 files.)
- ⚠️ **The "stops at b15" note below is SUPERSEDED (2026-09-03): the campaign now runs all 29.**
  What kept the big instances out was partly the task (below, still true and still measured) and
  partly three silent-truncation bugs that made them look untouchable — the coverage run's host
  budget, the fetch-vs-driver directory mismatch above, and a variant failure that wrote no row.
  See docs/findings/INSIGHTS.md. They are included now and a cell that produces nothing is
  reported as producing nothing, which is the honest version of the paragraph below.
- **Why the campaign USED TO stop at b15, MEASURED (2026-09-02) — it is not a size limit.** The large
  ITC'99 circuits encode fine and sit inside the packed envelope: at T=25 `abc:fast` gives
  b21 252k clauses (4 s), b20 255k (6 s), b22 386k (9 s), b17 442k (11 s) — b20/b21 are
  *smaller* than `s35932` (319k), which samples at 10k/s. What defeats them is the TASK, not the
  bytes: at T=25 with every output pinned, `--target rare` picks a pattern with **0 hits in
  917,504 simulated sequences** (p_accept < 1.1e-6), and GPAP discards conflicted trajectories
  rather than backtracking, so it emits nothing (`b1` and `b8` portfolios both, over 90 s).
  ⚠️ **Size does not predict this**: at `--target random` the two LARGEST produce models
  (b17 1,024 at 21/s, b22 783 at 13/s) while the two smallest produce none (b20, b21).
  ⚠️ **Depth is the lever**: b20 at a rare pin gives 5,574 models at T=5 (M=6,983) and nothing
  at T=10 or T=25. So these instances are reachable — just not at the depth this protocol fixes.
  Including them at T=25/50 would mostly enlarge the "neither sampler could touch" column that
  `b14`/`b15` already mark; if they are ever added it must be all six at both targets, never the
  two that happen to work.
- **Our four long-standing ISCAS-89 files** (`s386`, `s526`, `s832`, `s1196`) came from a
  different distribution than the rivals' copies and differ byte-wise, but were VERIFIED
  equivalent: identical PI/PO/DFF/gate counts, identical OUTPUT order (which is what their
  positional target tables index), and identical content after normalising comments, whitespace
  and line order. The rivals are handed our file either way, so both sides always see one circuit.

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

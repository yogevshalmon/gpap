# GPAP Refactor Plan — from experiment pile to publishable package

Instructions for the refactor session agent. Written 2026-07-08 on branch `gpu_bcp_sampling`
(all tests green: 78/78; latest wins: wide 8|24 kernel geomean 1.19× on big circuits).
Goal: a light, friendly, production/publication-shaped package whose center of gravity is
**the engine** (K-parallel GPU BCP — the novelty) and whose flagship app is **circuit SAT
sampling**, while keeping the engine reusable for the next apps we will find.

---

## 0. Ground rules for the refactor session

1. **Measure before touching.** First action: record a baseline benchmark JSON
   (`scripts/sample_benchmark.py` on the seq c25+c50 circuit sets, `--use-kernel auto` and
   `wide`) and keep it out of the refactor's way. After every phase, re-run and compare:
   worst single-instance GPAP/s ≥ 0.9× baseline, suite geomean ≥ 0.95×. Same-seed model
   sets must be bit-identical where the change claims "no semantic change".
2. **Check GPU state before benchmarking**: `nvidia-smi -q -d POWER` — this box has shown a
   40 W cap (default 80 W). Ratios A/B in the same state are valid; absolute numbers are not
   comparable across states. Any K-sizing change runs under the nvidia-smi kill-switch
   (`used > 10.8 GB → pkill`; the box SPILLS to host RAM, it does not OOM).
3. **Tests green after every phase** (`PYTHONPATH=. .venv/bin/python -m pytest tests/ -q`).
4. **Do not commit without the user's explicit go** — propose a commit at each phase
   boundary and wait.
5. **Insights are never deleted.** Code may go; findings move to `docs/findings/` (Phase 2).
6. **Do not re-litigate settled findings** (they cost days to establish):
   - Runtime per-batch memory PROBES are unreliable → fixed 4×-floor K calibration stays.
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` must be set before torch inits CUDA.
   - `orac` (high-B python-engine lookahead) lost 0/64 on circuits — stays out of the portfolio.
   - `mono` (1-byte count-only kernel) is measured 0.93× vs `wide`: the unit-time clause scan
     costs more than the extra K buys. Negative result — code is deletable, insight stays.
   - Soft lookahead is regime-dependent (win on some circuits, loss elsewhere) → it stays a
     bandit config, never a global default.
   - `--n-time` is a delivery contract; safety valves are `--time-limit` + zero-progress stop.

---

## 1. Target architecture

```
gpap/
  engine/                  # THE NOVELTY — reusable, app-agnostic
    formula.py             # CnfFormula + OccurrenceList (merge clsdata/, keep vectorized ctor)
    bcp_engine.py          # PyTorch reference engine: SLIMMED (see §3) — oracle + CPU fallback
    kernels.py             # all Triton kernels, unified (see §4)
    calibrate.py           # K sizing, VRAM model, expandable_segments setup (moved from sampler)
  sampling/                # THE FLAGSHIP APP
    sampler.py             # GPUSampler (trajectories, bandit, soft-stop) — engine consumer only
    miner.py               # entailed-clause mining (it exists FOR sampling today)
    preprocess.py          # unit_closure / simplify_cnf / ELS + model reconstruction
    pipeline.py            # NEW: the full e2e flow that today lives in scripts/sample_benchmark.py
                           #      (parse → construct → warmup → two-stage mine → timed contract run)
  circuits/
    bench.py               # bench_to_cnf as a LIBRARY (parse/unroll/Tseitin/simulate/pin, in-memory)
  cli.py                   # `gpap` entry point with subcommands (see §6)
  _version.py
docs/
  findings/                # all *_FINDINGS.md + an INSIGHTS.md index of settled results
  BENCHMARKING.md          # methodology: power-state check, kill-switch, seed/noise protocol
eval/                      # publication-facing, reproducible: benchmark driver + plots + colab
tests/                     # broadened (see §5)
benchmarks/circuits/bench/ # seq/ + comb/ .bench sources (tiny, committed) — CNFs are GENERATED
archive/                   # (optional, or delete outright — user's call per item, see §2)
```

Key structural moves, in order of value:

- **Move the pipeline out of the benchmark script.** `gpap_sample()` in
  `scripts/sample_benchmark.py` (~250 lines: single parse, warmup, two-stage mining with the
  recheck gate, contract-timed run, phase attribution) is library logic every consumer needs
  (API users, CLI, Colab driver). It becomes `gpap/sampling/pipeline.py::sample_file(...)` /
  `sample_clauses(...)` returning the same rich dict. The benchmark script shrinks to a
  table printer around it; `colab_sampling_benchmark.py` imports the same function.
- **Engine as a documented primitive with exactly two entry points** (this is what keeps it
  reusable for future apps — cube-and-conquer, UNSAT proving, inprocessing all consumed one
  or both):
  1. *Batched assumption checking*: `apply_assumptions_bulk(K×L) + propagate() → conflict/
     real_conflict/assignment` (the sum-trick BCP).
  2. *Fused trajectory sampling*: `run_trajectory(order, B, seed, …)` (whole decide+propagate
     loop in one launch).
  Write this contract down in `gpap/engine/__init__.py` docstring. The sampler must consume
  the engine only through these; no reaching into `_sat_count` internals from app code
  (today's soundness check does — give it an accessor).

---

## 2. Delete / archive / keep — explicit dispositions

"Archive" = move under `archive/` (or delete after extracting any insight into
`docs/findings/INSIGHTS.md` — confirm per item with the user at the phase boundary; deletion
is fine, git history keeps the code).

### scripts/ (current focus is circuit sampling — keep only what serves it)
| file | disposition | note |
|---|---|---|
| `sample_benchmark.py` | KEEP, shrink | becomes thin CLI/table around `sampling/pipeline.py` |
| `colab_sampling_benchmark.py`, `colab_sampling_plots.py` | KEEP → `eval/` | A100 driver |
| `bench_to_cnf.py` | KEEP → `gpap/circuits/bench.py` + thin CLI | already has batch mode |
| `profile_engine.py` | KEEP (trim) | engine-level profiling + the assumption generators tests/docs reference |
| `run_benchmark_suite.py`, `build_benchmark_suite.py` | archive | old BCP-throughput suites (pre-sampling era) |
| `colab_benchmark.py`, `colab_plots.py` | archive | old-engine cross-GPU flow, superseded by sampling driver |
| `clause_learn_solve.py` | archive | clause-learning app; insight in memory + findings |
| `els_*.py` (2), `inprocess_*.py` (6), `preprocess_solve.py` | archive | inprocessing experiments — concluded portfolio-only (ELS_INPROCESSING_FINDINGS.md) — EXCEPT: ELS itself stays in `gpap/sampling/preprocess.py` (it is a measured 1.48× sampling win) |

### gpap/solver/
| file | disposition | note |
|---|---|---|
| `sampler.py`, `miner.py`, `preprocess.py` | KEEP → `gpap/sampling/` | flagship |
| `bcp_engine.py` | KEEP, SLIM (see §3) | **do NOT delete wholesale** |
| `bcp_triton.py` + `bcp_triton_wide.py` | KEEP, UNIFY (see §4) | drop mono kernel + TritonBCPMono |
| `assumption_checker.py` | KEEP (small) → `gpap/engine/` | the batched-assumptions API face |
| `clause_learner.py` | archive with its script | |
| `inprocess.py`, `cadical_hook.py` | archive | inprocessing-era |
| `external_solver.py` | audit: keep only if tests/eval use PySAT verification through it, else fold into tests |

### benchmarks/
| item | disposition |
|---|---|
| `circuits/bench/{seq,comb}/` | KEEP — the canonical committed source; CNFs generated on demand |
| `suite*.json` (9 files), `cnf/` (old 20-CNF bundle), `colab_results/`, `rand_500v_2000c.cnf` | archive (old-engine era) |
| `sampling/`, `colab_sampling_results/` | KEEP → referenced by eval |
| `.experiments/` | gitignored already; leave |

### docs
- `ELS_INPROCESSING_FINDINGS.md`, `GPU_SAMPLING_FINDINGS.md` → `docs/findings/`.
- NEW `docs/findings/INSIGHTS.md`: one-paragraph summaries of every settled result (the §0.6
  list, orac/mono/lookahead/probe/memory model, the DEMOTIC/recurrent-paper comparison
  methodology with the local-CMSGen grounding caveat, the ~66%-distinct small-space caveat).
- `CLAUDE.md`: REWRITE to match the new layout (most of its K-memory-model detail concerns
  the old dense-engine path; keep the memory model where still true, the sampler lessons,
  and the run commands for the new CLI). Keep it short; deep material links to docs/.
- `README.md`: rewrite for the package (install, quickstart API + CLI, headline numbers with
  hardware caveats, citation stub).

---

## 3. The "old engine" — discard *paths*, keep the *oracle* (important nuance)

The user's instinct "the old engine can be discarded, the custom kernel is far better" is
right about the *hot paths* but the PyTorch `BCPEngine` cannot be deleted outright, because
it is currently:
1. the **A/B oracle** every kernel test asserts bit-identical fixpoints against (this is the
   entire correctness story — BCP confluence makes exact equality testable);
2. the **fallback** when CUDA/Triton is unavailable (packaging goal = runs on other
   platforms; CPU smoke tests need it);
3. the **flush-cache builder** (`reset_state(1) + propagate()` at K=1, once per formula);
4. the executor of **non-fused bandit configs** (the sk_/DEFAULT portfolio tensor path).

So: keep `BCPEngine`, but slim it to the two-entry-point contract. Audit-and-drop candidates
inside it (verify no test/app uses them after Phase-2 archiving): soft-stop machinery
(`soft_stop_frac`/`min_rounds` — sampler-side soft-stop is separate), `real_conflict`
bookkeeping if clause-learning is archived (check miner first — it uses conflict mining),
`timed_out` round-cap paths beyond the defensive cap, any K-memory helpers only the old
suite scripts used (`compute_k_safe` callers shrink after archiving). Also fold
`gpap/clsdata/` (2 files) into `gpap/engine/formula.py`.

---

## 4. Kernel unification (`gpap/engine/kernels.py`)

Today: 5 near-identical trajectory kernels across two files (unpacked, packed 16|16, packed
8|24 wide, mono8, + the v1 fixpoint kernel), where 16|16 and 8|24 differ by three constants
and mono is a measured regression.

- Merge the packed trajectory kernel into ONE `@triton.jit` with `SHIFT: tl.constexpr`
  (16 → 16|16, 8 → 8|24) — constexpr specialization compiles to exactly the same binaries,
  so this is a pure de-duplication; verify with the existing bit-identical model tests.
- **Make wide the default** (`use_kernel="auto"` picks per formula: 2N ≤ 65535 → 16|16;
  ≤ 0xFFFFFF and max_len ≤ 255 → 8|24; else unpacked). Kill the `"wide"` opt-in flag once
  auto covers it. While here, fix the known K_fused artifact: the packed branch deducts
  `eng_hold` for engine tensors that an all-fused portfolio never allocates (costs b12-c25
  ~9% K). Either compute K_fused lazily at first chunk (portfolio known) or deduct only when
  the portfolio contains non-fused configs.
- **Delete** `_bcp_trajectory_mono8_kernel` + `TritonBCPMono` + `tests/test_mono_kernel.py`
  (record in INSIGHTS.md: 0.93× vs wide, scan-at-unit tax > K gain; keep the wide tests).
- Keep the v1 fixpoint kernel (`apply_and_propagate`) — it is the batched-assumptions API's
  fast path and the natural entry for future apps.
- Fold `TritonBCPWide` into the base wrapper once the kernel is parameterized (one class,
  mode decided in `_build_flush_cache`).

---

## 5. Tests — broaden via generated circuit CNFs (also fixes portability)

Principle: tests must not depend on `/home/yshalmon/benchmarks/**` (user-local). The repo
bundles the small `.bench` sources; generate CNFs at test time.

- `tests/conftest.py`: session-scoped fixture that calls the (now library) circuit unroller
  to produce a parameterized matrix into `tmp_path_factory`, e.g.
  `{b03, b06, b10, s386, s526} × {c1, c5, c10}` (seconds to build, small enough for CI) and
  one `c25` for the deep-cascade path. Combinational: `{c432, c880} × pin_outputs=1`.
- Test axes over that matrix:
  - **Soundness**: every emitted model CPU-verifies against the original clauses (pre-simplify).
  - **Determinism**: same seed ⇒ identical model sets (kernel path).
  - **A/B**: oracle vs kernel fixpoint equality (existing contract tests, now parameterized).
  - **Reconstruction**: simplify/ELS on ⇒ models still verify on the ORIGINAL formula.
  - **Contract**: n_time delivered or `<`-marked with a firing valve.
  - **Quality smoke**: distinct fraction ≈ 1.0 on one non-tiny instance.
- Markers: `@pytest.mark.gpu` (skipped automatically off-CUDA — this makes a CPU-only smoke
  suite run on any platform), `@pytest.mark.slow` for c25 depth.
- Migrate the useful hand-written cases from `test_fast_bcp.py` / `test_sampler.py`; retire
  tests that only exercised archived apps (`test_inprocess.py` goes with inprocess).

---

## 6. Packaging & user API

- `pyproject.toml`: package `gpap` (confirm the public name with the user — it may collide on
  PyPI), `dependencies = [torch, numpy, python-sat]`, extras: `[cuda]` (triton pin),
  `[eval]` (matplotlib), `[dev]` (pytest, ruff). Set the `expandable_segments` env in
  `gpap/__init__.py` exactly as `sampler.py` does today (before torch CUDA init) and
  document why.
- Python API (thin, stable, documented in README):
  ```python
  import gpap
  models, info = gpap.sample("f.cnf", n=50_000, seed=5, device="cuda")     # pipeline.sample_file
  models, info = gpap.sample_clauses(nv, clauses, support=..., n=...)
  models, info = gpap.sample_circuit("b10.bench", cycles=25, n=..., pin_outputs=None)
  eng = gpap.engine.BatchedBCP(nv, clauses, device)                        # the primitive itself
  conflicts = eng.check_assumptions(assumption_sets)                       # (K,L) -> (K,) bool
  ```
  `info` = today's rich dict (rates, phases, mining stats, soundness). Library code uses
  `logging`, never `print` (prints move to the CLI layer).
- CLI (entry point `gpap = gpap.cli:main`):
  - `gpap sample <cnf|bench> [-n] [--time-limit] [--seed] [--json out.json]`
  - `gpap bench <glob> [--cmsgen PATH] [--n-cov] ...` (today's sample_benchmark)
  - `gpap circuit2cnf <bench|dir> --cycles T [--pin-outputs k]`
  - `gpap info` (device, triton availability, power-limit warning — surface the 40 W gotcha)
- Publication extras: `LICENSE` (user must choose), `CITATION.cff`, `eval/` scripts that
  regenerate the paper tables from committed result JSONs (the DEMOTIC/recurrent comparison
  with its footnotes lives here).

## 7. Things the user didn't list (recommended, cheap)

- **CI**: GitHub Actions running lint + CPU smoke suite on push (GPU suite stays manual).
- **ruff** (format+lint) config in pyproject; run once repo-wide as its own commit.
- **Tunables in one place**: `CIRCUIT_PORTFOLIO`/`DEFAULT_PORTFOLIO`, `mine_gate`, K
  calibration constants (`_CAL_MULT`, fused caps, safety fractions) → `gpap/tunables.py`,
  each with a one-line provenance comment (where measured, when).
- **BENCHMARKING.md**: the measurement protocol that keeps biting us — power-state check
  first, same-state A/B only, multi-seed shootout for arbitration (same-seed repeats measure
  timing noise only), like-for-like formulas (mined vs raw), kill-switch, `<` marker
  semantics, distinct-fraction reporting for unique-solutions/s claims.
- **Version the result JSON schema** (a `"schema": 1` field) so eval/plots stay stable.

## 8. Phase order (each ends: tests green + benchmark no-regression + proposed commit)

1. **Baseline capture** (JSONs + record clocks/power state) — no code changes.
2. **Archive sweep** (§2): move/delete scripts, apps, old suites; write
   `docs/findings/INSIGHTS.md`; retire their tests. (Pure deletion — benchmark unaffected;
   full pytest proves nothing kept depends on the archived code.)
3. **Package layout move** (§1): `engine/`, `sampling/`, `circuits/`; imports fixed;
   `pipeline.py` extracted from the benchmark script; scripts become thin. (No behavior
   change intended ⇒ same-seed identical models + rates within noise.)
4. **Kernel unification + auto-wide default + K_fused deduction fix** (§4). This is the only
   phase with intended perf change (b12-c25-class +, nothing −): full A/B suite, 0.9× rule.
5. **Engine slim** (§3) after apps are archived — audit-driven deletions only.
6. **Tests broadening** (§5) + CI.
7. **Packaging polish** (§6): pyproject, CLI, README/CLAUDE.md rewrite, LICENSE/CITATION.

Estimated shape: phases 2–3 are the bulk moves; 4 is the only risky one (guarded by the
bit-identical tests); everything after is additive.

## 9. Open questions to confirm with the user before/while executing

1. Archive directory vs outright deletion (git history keeps everything either way)?
2. Public package name (`gpap`? check PyPI) and license.
3. Does the sk_ (projected-sampling, non-circuit) portfolio stay a supported path in
   `sampling/` (recommended: yes, it is the CMSGen-comparison story) or archive with the rest?
4. Keep `notebooks/gpap_colab_benchmark.ipynb` (old engine) or only the sampling notebook?
5. Default `n_time`/`time-limit` for the CLI (`50k/60s` as today?).

# GPAP C++ core — implementation plan (option C)

**Goal.** Make the engine a real library: one implementation of the K-parallel packed BCP
primitive, written in C++, callable from C++ *and* from Python. Python keeps the whole
application layer (sampling, circuits, CLI, eval); it stops carrying a second engine.

**Decision record** (measured 2026-07-28, see `docs/findings/CPP_BACKEND_FINDINGS.md`):

| Question | Answer | Consequence for this plan |
|---|---|---|
| Is Python the bottleneck? | No — 22-41 µs/chunk (≤0.07%); 79-86% of a chunk is the fused kernel | The C++ core is for *linkability + memory ownership*, not for host speed |
| Must the kernels be rewritten in CUDA C++? | No — Triton AOT-compiles `kernels.py` unmodified; SASS is bit-identical to the JIT binary | **Kernels stay in Triton.** One kernel source, forever |
| Is there a performance reason? | **No — retracted 2026-07-28 after the port was measurable.** The shipped core TIES the torch engine (geomean 1.013× over b03/b04/b06/b10/b12, identical model sets). The earlier "1.2-1.35× from owning memory" was really a device-memory PLACEMENT effect: the same cubin on the same input runs 43-44 ms or 54-56 ms depending on where the (K,M) buffer lands, and allocating it first vs after several small buffers flips it *some* of the time | Do the port for the library, not for speed. Placement is tracked as a separate experiment (§9) that would benefit BOTH engines |
| Does the build need nvcc / CUDA toolkit? | No — triton bundles `ptxas` + `cuda.h`; runtime needs only `libcuda.so.1` | Build deps: python+triton, gcc, cmake. **No CUDA toolkit** |

**Non-goals.** Rewriting kernels in CUDA C++ (evaluated, not justified — see findings §4 option D).
Removing torch from the *application* layer. Changing any sampling semantics.

---

## 1. Architecture

```
                    ┌──────────────────────────────────────────────┐
  C/C++ consumer ──►│  libgpap.so   —  include/gpap/gpap.h (C ABI) │
                    │                                              │
  Python  ──ctypes─►│  formula (DIMACS/CSR/packed init)            │
   gpap.engine      │  engine state: packed, abits, trail, flags   │──► cubins ──► GPU
   (thin facade)    │      OWNED via cuMemAlloc  (the perf win)    │   (AOT from
                    │  launches + accessors + K calibration        │    kernels.py)
                    └──────────────────────────────────────────────┘
```

**Ownership rule.** Everything the kernels touch every chunk (`packed` (K,M), `abits` (K,NW),
`trail`, `active`, `conflict`, counters, occ CSR, packed-init/flush cache) is allocated and
owned by the core with the driver API. Nothing hot lives in torch's caching allocator.

**Kernel source of truth stays `gpap/engine/kernels.py` (now `cpp/kernels/bcp.py`)** — still
`@triton.jit`, still the file to edit. A build step compiles every specialization ahead of time
into cubins that get embedded in the library. Editing a kernel = rebuild, no other change.

### Layout

```
cpp/                         # AS SHIPPED
  include/gpap/gpap.h        # THE C ABI — the stable boundary
  include/gpap/gpap.hpp      # header-only C++ wrapper (RAII, exceptions, std::vector)
  src/formula.cpp            # DIMACS (+ `c ind`) + CSR occ + clause_sizes + init_lit_sum + packed
  src/engine.cpp             # state, reset/flush cache, launches, counters, mode/split dispatch
  src/readout.cpp            # selections, on-device soundness, bit gathers, assignment rows
  src/calibrate.cpp          # cuMemGetInfo + the K model (port of calibrate.py)
  src/runtime.cpp            # driver-API helpers, cubin registry, launches, device buffers
  src/generated/             # GENERATED cubin blobs + registry (gitignored)
  tools/build_kernels.py     # AOT compiler: kernels.py + helpers.py -> src/generated/
  kernels/helpers.py         # support kernels (broadcast, completion, gathers, reductions)
  examples/demo.c            # standalone C consumer     examples/sample.cpp  # C++ consumer
  CMakeLists.txt             # + install/export so downstream does find_package(gpap)
gpap/engine/
  _core.py                   # ctypes binding to libgpap (private)
  bcp_engine.py              # BCPEngine: same public API + the selection API, backed by _core
  formula.py                 # CnfFormula: facade over the core builder (numpy fallback off-GPU)
  _torch_engine.py           # the retired PyTorch/Triton-JIT engine — A/B baseline only
  kernels.py                 # unchanged (THE kernel source, AOT-compiled into the library)
  calibrate.py               # the K model (mirrored in C++; agreement is gated by a test)
scripts/build_core.sh        # one-command build   scripts/engine_ab.py  # core-vs-baseline A/B
```

### Why ctypes (not pybind11)

The C ABI has to exist anyway for C++ users. Binding it with `ctypes` means: no Python
extension to compile, no Python-version/ABI coupling (one `.so` serves every interpreter),
and no extra build dependency. Call overhead is ~1-5 µs against a 50-150 ms chunk — measured
irrelevant. A pybind11/nanobind ergonomic layer can be added later without changing the ABI.

---

## 2. The C ABI (`gpap.h`) — design sketch

> Shipped shape differs in places (options struct for `gpap_engine_create`, a third
> scratch selection slot, `gpap_selection_set`, `gpap_formula_support`). The header
> [`cpp/include/gpap/gpap.h`](../../cpp/include/gpap/gpap.h) is authoritative.

```c
typedef enum { GPAP_OK=0, GPAP_ERR_CUDA, GPAP_ERR_ENVELOPE, GPAP_ERR_IO, GPAP_ERR_ARG } gpap_status;
typedef enum { GPAP_MODE_AUTO=0, GPAP_MODE_SCALAR, GPAP_MODE_WAVE } gpap_mode;

typedef struct gpap_formula gpap_formula;
typedef struct gpap_engine  gpap_engine;

/* ---- formula: built once, host-side, then uploaded ---- */
gpap_status gpap_formula_from_dimacs(const char* path, gpap_formula** out);
gpap_status gpap_formula_from_clauses(int32_t num_vars, const int32_t* flat_lits,
                                      const int32_t* clause_lens, int32_t num_clauses,
                                      gpap_formula** out);          /* DIMACS literals */
typedef struct { int32_t num_vars, num_clauses, max_clause_len, shift;
                 int64_t total_lits; int32_t has_empty_clause; } gpap_formula_info;
void gpap_formula_info_get(const gpap_formula*, gpap_formula_info*);
void gpap_formula_free(gpap_formula*);
gpap_status gpap_packed_envelope(int32_t num_vars, int32_t max_clause_len, int32_t* shift_out);

/* ---- engine ---- */
gpap_status gpap_engine_create(const gpap_formula*, int device, gpap_mode, gpap_engine** out);
void        gpap_engine_free(gpap_engine*);
gpap_status gpap_engine_set_mode(gpap_engine*, gpap_mode);
gpap_status gpap_engine_reset(gpap_engine*, int32_t K);

/* entry point 1: batched assumption checking (K x L literal indices, -1 pad) */
gpap_status gpap_engine_apply(gpap_engine*, const int64_t* decisions_host, int32_t K, int32_t L);
/* entry point 2: fused trajectory sampling (caller supplies the decision order) */
gpap_status gpap_engine_trajectory(gpap_engine*, const int32_t* order_host, int32_t P,
                                   int32_t n_support, int32_t B, int64_t seed, int soft,
                                   int32_t* max_rounds_out);
/* incremental-units hook (miner) */
gpap_status gpap_engine_recache_flush(gpap_engine*);
int         gpap_engine_flush_conflict(const gpap_engine*);

/* ---- selections + readout (slot 0/1 hold row selections between calls) ---- */
gpap_status gpap_pair_mask(gpap_engine*, const int32_t* vars, int32_t n_vars, int32_t* mask_id);
gpap_status gpap_select_complete(gpap_engine*, int32_t mask_id, int32_t slot, int32_t* count);
gpap_status gpap_selection_rows(gpap_engine*, int32_t slot, int32_t* rows_out, int32_t max);
gpap_status gpap_check_sound(gpap_engine*, int32_t slot, int32_t max_rows, int32_t* n_unsound);
gpap_status gpap_var_bits(gpap_engine*, int32_t slot, const int32_t* cols, int32_t n_cols,
                          uint8_t* out_host);                      /* (count, n_cols) */
gpap_status gpap_assignment_rows(gpap_engine*, int32_t slot, int8_t* out_host);   /* (count, 2N) */
gpap_status gpap_flags(gpap_engine*, int8_t* active_out, int8_t* conflict_out);   /* (K,) each */
gpap_status gpap_flushed_assignment(gpap_engine*, int8_t* out_host);              /* (2N,) */

/* ---- work counters + memory ---- */
typedef struct { int64_t ops, props, decs; int64_t launches; } gpap_counters;
gpap_status gpap_counters_get(gpap_engine*, gpap_counters*);
int64_t     gpap_mem_available(int device, double fraction);
int32_t     gpap_compute_k_safe(int32_t M, int32_t N, int64_t bytes, int32_t cap,
                                int32_t max_clause_len, int64_t total_lits);
const char* gpap_last_error(void);
```

Design notes:
- **Order is caller-supplied**, exactly as `run_trajectory` is today ⇒ Python keeps using
  `torch.randperm` ⇒ results stay **bit-identical** to the current engine, which is what makes
  the golden-hash gate usable. A convenience `gpap_make_order()` for C++ users comes in M4.
- **Selections replace mask tensors.** Today the sampler ANDs `active` with `~any_free(...)` in
  torch and uses the mask as a row selector. In the core that becomes one kernel writing a
  row mask + count into a slot; readout functions take the slot. Row order stays ascending
  (host-side compaction of the mask), so emitted rows keep today's order.
- **Soundness moves in.** `gpap_check_sound` computes "every clause of every selected row has
  sat_count > 0" on device and returns a count — no `(rows, M)` transfer at all, retiring
  `_SOUND_SLICE_BYTES` and the 4096-row cap juggling.

---

## 3. Kernel build pipeline

`cpp/tools/build_kernels.py` imports the kernel modules and calls `triton.compile()` directly
(not `triton.tools.compile`'s CLI — that hardcodes the grid, ignores shared memory in its
wrapper, and mangles names). For each specialization it emits the cubin blob + metadata into
`src/kernels_generated.c`, keyed for runtime lookup.

Specializations to emit:

| kernel | SHIFT | WAVE | VSCAN | SOFT | count |
|---|---|---|---|---|---|
| `_bcp_fixpoint_kernel` | 16, 8 | 0, 1 | — | — | 4 |
| `_bcp_trajectory_kernel` | 16, 8 | 0, 1 | 0, 1 | 0, 1 | 16 |
| helpers (`cpp/kernels/helpers.py`) | 16, 8 where relevant | — | — | — | ~8 |

`W=32`/`num_warps=1` only (measured optimum; `GPAP_WAVE_W>32` was 2-5× worse). ~1 MB of cubins.
`seed` is always `i64` in the core (the JIT specializes small seeds to i32 — same arithmetic).

New helper kernels (Triton, so the toolchain stays nvcc-free):
`_bcast_rows` (reset broadcast), `_complete_mask` (active & no-free-var → mask+count),
`_sound_rows` (per-row min sat_count > 0), `_var_bits` (bit gather), `_unpack_rows`
(bitmap → int8 (n,2N)), `_sum_i32` (counter accumulation).

Arch: default = the local GPU's target; `GPAP_CUDA_ARCHS="80;86;89;90"` emits a multi-arch
registry. Triton can compile for an explicit target without a GPU present (CI-friendly).

---

## 4. What changes in the app layer

| Site | Today | After |
|---|---|---|
| `sampler._run_chunk` | returns `(fully_mask, sup_mask, rounds, 0.0)` torch bools | returns `(n_fully, n_sup, rounds, 0.0)` ints + slot handles |
| `sampler.sample` | `int(fully.sum().item())`, `sat_rows` slicing loop | uses the returned counts; `gpap_check_sound` |
| `sampler._project` | `var_true_bits` → torch ops on GPU | `var_bits` → numpy on host (same bytes crossing PCIe) |
| `sampler` collect path | `assignment_rows(fully)` torch → cpu | `assignment_rows(slot)` → numpy |
| `miner` | `eng.conflict.clone()`, `flushed_assignment` | `gpap_flags`, `gpap_flushed_assignment` (numpy) |
| `checker` | returns torch tensors | returns numpy (documented API change) |

Everything else (bandit, portfolios, pipeline, circuits, CLI, benchmark, eval) is untouched.

---

## 5. Gates — nothing merges without these

- **G1 semantic** — `tests/test_bcp_engine.py`, `test_fast_bcp.py`, `test_assumption_checker.py`,
  `test_wide_kernel.py` parametrised over both engines; the pure-Python `reference_bcp.py`
  oracle must agree exactly (conflict flag + assigned-literal set) for both.
- **G2 bit-identity** — fixed instance matrix (circuits + ELS + wide + units-flush), pinned
  K, `K_fused=K`, single config, seed 5 ⇒ **identical model counts and model-set hashes**
  between the torch engine and the core. This is the repo's standard no-change gate; it is
  achievable here precisely because the order is caller-supplied.
- **G3 memory** — `nvidia-smi` kill-switch run on b05-c50 / b12-c25 / b14-c25. The core's
  allocations are exact where torch's caching allocator carried slack, so `compute_k_safe`
  and `K_fused` must be re-validated, not assumed. **This box spills instead of OOMing.**
- **G4 performance** — interleaved same-process A/B (b03, b06, b12, b04) core vs torch engine.
  The bar is **no regression** (≥0.98×). MEASURED 2026-07-28: 1.01 / 0.99 / 1.01 / 1.03 / 1.02,
  geomean **1.013×**, identical model counts — the port is performance-neutral, as it should be
  (bit-identical SASS). Re-confirm on the A100 before the eval re-run.
- **G5 suite** — full `pytest` green; `gpap bench` smoke on a 3-instance glob.

Migration switch: `GPAP_ENGINE=torch|core` selects the implementation while both exist. Default
flips to `core` only after G1-G4. The torch engine is deleted in M4, not before.

---

## 6. Milestones — status (2026-07-28)

| | status | evidence |
|---|---|---|
| **M1** core runs with no Python | **done** | 29 AOT specializations (852 KiB); `gpap_demo` / `gpap_sample` link `libcuda` alone and sample both packed splits, soundness clean |
| **M2** Python on the core | **done** | `_core.py` (ctypes) + `gpap.hpp` (RAII C++); `BCPEngine`/`CnfFormula` are facades; G1 green |
| **M3** app layer | **done** | sampler/miner/checker on the selection API; per-file suite green |
| **M4** gates | **G2/G3/G4 done** | see below; A100 confirmation still pending |

- **G1 semantic** — `test_bcp_engine` / `test_fast_bcp` / `test_assumption_checker` /
  `test_wide_kernel` all pass against the core (64 tests), oracle-checked.
- **G2 bit-identity** — `tests/test_core_equivalence.py`, 23 tests: identical formula structures,
  conflict flags, assignments, implications, counters and **model sets row-for-row**, in scalar
  and wave, both splits, plus the degenerate shapes (M=0, N=0, empty order) that the port
  actually broke on. At the sampler level, `scripts/engine_ab.py` on 5 circuit unrolls:
  **identical model counts on every instance**.
- **G3 memory** — nvidia-smi kill-switch, `-n 3000 --mine off`: b05-c50 **6.36 GB**, b12-c50
  6.36 GB, b12-c25 6.34 GB, b04-c25 6.30 GB peak on the 12 GB card (kill line 11 GB). The core
  holds *less* than the torch path did (7.4 GB historically) because there is no caching-
  allocator slack — the K model stays unchanged and remains conservative.
- **G4 performance** — kernel level geomean **1.013×**; sampler level (pinned K=8192, B=8,
  seed 5, 4 chunks) **1.028×** geomean, range 0.91-1.25× — noise-dominated, no regression.
- **G5 suite** — full `pytest -m "not slow"`: **189 passed** (25:54) on the core. Plus e2e
  `gpap bench` (rare targets, 3 seeds, circuit portfolio) against the numbers stored in
  `.experiments/circuits_rare_3seed_v3.json`, which were measured on the PRE-PORT engine:

  | | baseline sps | core sps | ratio | baseline succ | core succ |
  |---|---|---|---|---|---|
  | b03 | 634,689 | 656,775 | 1.03× | 0.93 | 0.96 |
  | b04 | 46,827 | 46,934 | 1.00× | 0.57 | 0.57 |
  | b06 | 478,213 | 445,020 | 0.93× | 0.45 | 0.45 |
  | b08 | 15,543 | 16,469 | 1.06× | 0.05 | 0.05 |

  **geomean 1.006×**, success rates unchanged to two decimals on 3 of 4 (b03's is a rare-target
  draw). Cross-session, so power-state noise applies to the rates — the success rates are the
  tighter signal, and they say the sampler behaves identically.

**Deferred deliberately:** deleting `_torch_engine.py`. It is the only independent
implementation left, it is what `scripts/engine_ab.py` compares against, and it costs one file.
It goes when the eval has been re-run on the core.

> **STATUS 2026-08-08 — done.** The full eval has been re-run on the core (L4 + 4080), so
> `gpap/engine/_torch_engine.py` and `scripts/engine_ab.py` were deleted. The port gate
> `tests/test_core_equivalence.py` became `tests/test_core.py`: same degenerate-shape and
> readout regressions, now checked against the pure-Python oracles (`_init_formula_numpy`,
> `reference_bcp.py`) and against clause satisfaction, rather than against a second engine.
> Everything below this line is the historical plan and is left as written.

## 7. Milestones (original)

**M1 — the core runs (no Python).** Build pipeline + `gpap.h` + formula + engine state +
reset/flush + both kernel entry points + `examples/demo.c`. Acceptance: `demo` reproduces the
`bcp_engine.py` demo result, and a C harness reproduces b03/b06/b12 model counts exactly
(2337 / 19087 / 5635 — already measured with the prototype).

**M2 — Python on the core.** Helper kernels, accessors, counters, calibration, `_core.py`
ctypes binding, `BCPEngine`/`CnfFormula` facades. Acceptance: G1 passes with `GPAP_ENGINE=core`.

**M3 — app layer.** Adapt sampler/miner/checker to the selection API. Acceptance: G2 + G5.

**M4 — flip and clean.** G3 + G4, make `core` the default, delete the torch engine path and
`GPAP_ENGINE`, update `README`/`CLAUDE.md`/`docs/ENGINE.md`, CI builds the library, add
`gpap_make_order` + a documented C++ example. Then re-run the eval suite on the new engine and
compare against the Colab numbers gathered on the old one.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| K-calibration was tuned around torch's allocator slack | G3 kill-switch re-validation is mandatory, on the biggest instances |
| Perf win doesn't survive off this laptop | G4 on A100 before claiming it; the port stands on linkability alone regardless |
| Triton AOT tooling is thin | We call `triton.compile()` directly and generate our own registry/launcher — the CLI is not a dependency |
| Cubin arch coverage | Multi-arch registry + PTX fallback (M4) |
| Two engines during migration | Time-boxed to M1-M3, one env switch, deleted in M4 |
| Silent divergence in the formula builder | The core is the *only* builder; Python's `CnfFormula` becomes a facade over it (no second implementation) |

---

## 9. Open experiment — device-memory placement (~25% on the table, either engine)

Found while validating the port, NOT caused by it: on b03-c25 (wave, K=34,775) the *identical*
cubin on the *identical* input runs at **43-44 ms** or **54-56 ms** depending only on where the
2 GB `(K, M)` packed buffer lands in device memory. Allocating it before the small formula
buffers gave the fast placement consistently in a standalone C process, and intermittently in a
Python process; the shipped core (small buffers first) sits at the slow end, as does the torch
engine. Same K, same order, same seed, same model count (2337) throughout — it is not a
work-difference, a clock effect (device-side traffic before the launch changes nothing), a
footprint effect (+3 GB of padding changes nothing) or a measurement-order artifact (reversing
the arms preserves it).

Next step when the migration is done: give the core a single **state arena** — one allocation
covering packed + abits + trail + flags + counters, sub-allocated with a large alignment — and
A/B it against the current per-buffer allocations across instances and K. If it holds, it is a
~1.2× on the flagship kernel that the Python engine could never have reached, and the first real
performance argument for the core.

---

## 10. The C++ API surface — two tiers (2026-07-29)

`gpap.hpp` grew a **tier 1** so that using the library needs no knowledge of how the engine
works. The complexity that a C++ consumer previously had to absorb — batch sizing against free
VRAM, the wave/scalar dispatch threshold, selection slots, pair masks, the internal 2N literal
indexing, and above all the support-first decision order (getting it wrong silently produced
0 models, which is exactly what happened when `sample.cpp` was first written) — is now the
sampler's job, not the caller's.

| | tier 1 | tier 2 |
|---|---|---|
| types | `Sampler`, `SampleOptions`, `Solutions`, `AssumptionChecker`, `CheckResults`, `Value` | `Formula`, `Engine`, `Options`, `Mode` |
| literals | **DIMACS** (`+v` / `-v`, 1-based) — the numbers in the `.cnf` file | the ABI's 0-based variable indices / 2N literal indices |
| batch size | chosen from free VRAM (the Python sampler's kill-switch-validated model, ported) | caller passes K to `reset` |
| kernel variant | `Mode::Auto` resolves on the first `reset`, at 30x resident warps | caller sets it |
| decision order | support first, reshuffled per chunk, phase boundary respected | caller supplies `order` + `n_support` |
| soundness | on-device check per chunk; a bad row throws | caller calls `check_sound` |

Tier 1 is written entirely in terms of tier 2 — no private back door — so anything it does can
be taken over piecemeal (`sampler.engine()` hands back the live `Engine`).

Deliberately NOT ported to C++: the adaptive config portfolio (bandit), entailed-clause mining,
preprocessing/ELS, the uniformity readout and the circuit front-ends. Those stay in
`gpap.sampling` / `gpap.circuits`; the C++ side is the engine plus a straightforward loop over
it, and the comprehensive checking of that loop lives in `tests/test_cpp_api.py` (Python drives
the built example binaries and verifies their output against `reference_bcp.py`).

### G6 — the seq family re-run on the core (2026-07-29)

The whole `seq` family (ITC'99 b* + ISCAS-89 s*, 19 instances x {tseitin, abc:fast} x
{rare, random} x {25, 50} cycles) re-run GPAP-only on the C++ core and compared row-for-row
against `.experiments/full_eval/nvidia-geforce-rtx-4080-laptop-gpu/` (measured 2026-07-27 on the
torch engine). 92/92 rows paired, none skipped or errored.

| stage | rows | geomean core/baseline |
|---|---|---|
| seq-c25-rare | 23 | 0.955x |
| seq-c25-random | 24 | 0.942x |
| seq-c50-rare | 20 | 1.019x |
| seq-c50-random | 25 | 0.969x |
| **all** | **92** | **0.969x** (min 0.74x, max 1.59x) |

(Non-degenerate rows only: vacuous-pin instances b02/b05/b07/s526 and near-vacuous b01/b09 are
excluded from the geomeans, as are rows where either side found 0 models.)

**The engine PATH is identical on 91 of 92 rows** — same packed split, same propagate variant,
K within 2%. The single exception is b15-c25-random tseitin (K 833 -> 642), a free-VRAM drift on
an instance whose K is tiny to begin with. Seven rows differ in success rate; every one of them
is the adaptive bandit landing on a different config (`tuned` differs), not different engine
behaviour.

**The 0.969x is the measurement environment, not the port**, and the interleaved A/B proves it:
`scripts/engine_ab.py` on 6 instances, pinned K=16384/B=8/seed 5, three rounds run back to back
the same morning —

| round | b03 | b04 | b06 | b09 | b12 | s1196 | geomean |
|---|---|---|---|---|---|---|---|
| 1 | 1.06x | 1.02x | 1.31x | 0.99x | 1.07x | 1.05x | **1.079x** |
| 2 | 1.04x | 1.04x | 0.88x | 0.98x | 1.05x | 1.04x | **1.003x** |
| 3 | 1.06x | 1.04x | 0.97x | 1.02x | 1.05x | 1.06x | **1.033x** |

with **identical model counts on every instance in every round** (7240 / 5300 / 7470 / 54368 /
3 / 2245). Note b06 alone: 1.31x, 0.88x, 0.97x on *identical work* — that single cell spans the
whole cross-run offset, which is why per-row rates from different sessions cannot be compared
here (BENCHMARKING.md §1b: SM clock ranged 615-1710 MHz while >90% busy).

### G7 — the C++ tier-1 API (2026-07-29)

- **Soundness**, `tests/test_cpp_api.py` (4 tests, Python drives the built binaries): every
  emitted row verified against `reference_bcp.py` — full assignments satisfy every clause and are
  a subset of the brute-force model set; support projections propagate to a conflict-free
  fixpoint satisfying every clause; same seed reproduces, different seed does not; the
  `AssumptionChecker` demo matches the oracle's conflict flags and implied literals.
- **Parity with the Python sampler** (fixed 8 s wall per arm, interleaved): K within 0.4% and the
  same kernel variant on all 5 instances, success rates matching (0.116/0.120, 0.092/0.086,
  0.562/0.521, 0.050/0.046, 0.332/0.350), throughput geomean **1.003x**.
- **Memory**, nvidia-smi kill-switch: b05-c50 6.39 GB, b12-c50 6.39 GB, b12-c25 6.37 GB peak of a
  12 GB card (kill line 11 GB) — the ported `auto_k` lands where the Python model does.

---

## 11. Code review of the branch (2026-07-29) — findings and fixes

A full read of the diff (C core, ctypes binding, Python facades, examples, CI, build script).
Seven issues, all fixed on the branch; three were memory-safety defects.

| # | severity | issue | fix |
|---|---|---|---|
| 1 | **memory safety** | `gpap_var_bits` / `gpap_assignment_rows` / `gpap_selection_rows` wrote a caller buffer whose size was only IMPLIED by the current selection count. A caller whose count was one chunk stale overflowed the heap silently. | explicit row-capacity parameter; the core writes `min(selection, capacity)`. Regression test `test_readout_never_writes_past_the_caller_buffer`. |
| 2 | **memory safety** | `_core.Engine.flags()` / `num_propagated()` sized their buffers from the Python-tracked `_K`, which is 0 before the first `reset()` — but construction runs an internal `reset(1)` for the units-flush cache, so the core writes 1 element into a 0-element array. | size from the core's `K`. |
| 3 | **memory safety** | `gpap_formula_from_clauses` did not validate clause lengths; a negative one walked the read cursor backwards out of the caller's buffer. | reject negative lengths; also reject `total_lits > 2^31` (the occurrence CSR indexes with int32 and would have wrapped into silent corruption). |
| 4 | **correctness/API** | `vscan` violated the header's own "every field is 0 means the sensible default" contract: `gpap_engine_options o = {0}` DISABLED the warp-wide decision scan (worth 12-19% on wave instances). | 0 = default (on), 1 = on, -1 = off, across the C ABI, `gpap.hpp` and the ctypes binding. |
| 5 | **CI is broken** | the `build-core` job could never have passed. Its stub `libcuda` defined no symbols: `libgpap.so` links anyway (shared libraries tolerate undefined symbols) but the example EXECUTABLES do not. Reproduced locally, then fixed and re-verified end to end. | `cpp/tools/make_libcuda_stub.py` derives the symbol list from the sources (and emits the `_v2` aliases `cuda.h` macro-remaps to), so it cannot drift when a driver call is added. |
| 6 | robustness | `gpap_engine_trajectory_dev` dereferenced `std::max_element` on a possibly-empty vector (unreachable today only because construction leaves `K == 1`). | guard on `K > 0`. |
| 7 | leak hazard | `gpap_pair_mask_create` allocates per call with no way to release; the Python binding caches, the C++ `Engine::pair_mask()` does not, so a mask created per chunk leaks steadily. | documented at both layers ("create once per variable set, never in the chunk loop"). |

Also cleaned: `numpy` was imported function-locally in five places in `sampler.py` (two on the
per-chunk path) with no module-level import — hoisted.

Re-validated after the fixes: `tests/test_core_equivalence.py` (26), `test_cpp_api.py` (4),
`test_assumption_checker.py`, `test_bcp_engine.py` all green; all three examples run; the header
compiles clean under `-Wall -Wextra -Wshadow -Wconversion -Wpedantic`; and the clean-machine
build (stub driver, `--arch 80`, no GPU) produces the library and all three executables.

# UX simplification plan — "no user should ever have to type K"

**Status: IMPLEMENTED 2026-08-14** (written and approved the same day, during the lean-and-mean
pass). Items 1–4 and bug fixes 0.1–0.3 all landed; item 5 of the original proposal
(`sample_clauses`'s 30 parameters) was explicitly **out of scope** and remains so.

Two things were added beyond the written plan, both for consistency between the two faces:
the **C++ `AssumptionChecker` chunks too** (it had the same unbounded-K flaw, verified identical
to manual slicing on 200k sets), and `gpap.sample_circuit` now fills `info["wire_map"]` from the
labels it already had, so the README's claim about named columns is true on that path as well.

Verification results are in §6; the "after" measurements for item 4 are in Appendix C.

## The goal, in one line

> A user should never type `K`, and never learn it exists — unless they are benchmarking, and then
> it should be one obvious knob. Everything below follows from that.

---

# 0. What the investigation turned up (evidence, not opinion)

These were **measured on this box before writing the plan**. Three of them are pre-existing bugs
that the changes below either expose or must fix.

### 0.1 🐛 BUG — out-of-range assumption literals corrupt other rows (memory safety)

`gpap.engine.checker.AssumptionChecker` does **no literal validation**. `ext_lit_to_lit_index(99)`
on a 3-variable formula gives internal index 196; the kernel then writes to
`ab_row + (196 >> 5)` = **6 words past the end of this row's bitmap** (NW = 1 for 2N = 6).

Demonstrated — row 0 gets `[[99]]`, rows 1–7 get `[]`:

```
row0: assigned lits = []
row6: assigned lits = [4]      <-- silently corrupted by row 0
rows 1..7 unchanged?  False
```

The **C++** `AssumptionChecker::check` raises `"assumption literal out of range"`. Python does not.
This is reachable straight from the README example. **Must be fixed as part of item 4.**

Related, same root cause: DIMACS literal `0` maps to internal `-1`, which the kernel reads as
*padding*, so `[[0]]` is silently "no assumptions" instead of an error.

### 0.2 🐛 BUG — `out_cap > 0` on an UNSAT formula raises `KeyError: 'orig_nv'`

Reachable **today** via `gpap sample unsat.cnf --out f.bits`. Root cause:
`GPUSampler._unsat_sampler` sets `_recon = {"unsat": True}` with no `orig_nv` / `orig_support`,
and `_build_projection_plan` does:

```python
if r is not None and r.get("orig_support"):  cols = list(r["orig_support"])
elif r is not None:                          cols = list(range(r["orig_nv"]))   # KeyError
```

**This blocks item 1**, which routes every `gpap.sample()` call through `out_cap`.

### 0.3 🐛 BUG — `check_assumptions([])` raises `GpapError: reset: bad argument`

An empty list should be a no-op returning empty results, not a driver-level error.

### 0.4 Every existing `GPUSampler` caller already passes an explicit `k_cap`

Checked all of them (pipeline ×3, `eval/uniformity/barbarik_gpap.py`, `eval/uniformity/fuzz_uniformity.py`,
`eval/colab/colab_sampling_benchmark.py`, and every test). **Nobody relies on the `16384` default**, so
changing it to `"auto"` is a behavioural no-op for existing code. This is what makes item 2 safe.

### 0.5 Three different default batch caps

| entry point | default `k_cap` | VRAM-proportional? |
|---|---|---|
| `GPUSampler(formula, device)` | **16,384** | no |
| `pipeline.sample_clauses(...)` | `"auto"` → 49,152 here | **yes** |
| C++ `SampleOptions` | **49,152** hardcoded | no |

The C++ one is the very bug `default_k_cap` was introduced to fix (an 80 GB A100 gets the 12 GB
laptop's batch); the fix only ever landed on the Python side. **The C ABI exposes no k-cap helper
at all** — `gpap.hpp::auto_k` calls `gpap_compute_k_safe` + `gpap_mem_available` directly.

### 0.6 The headline call returns nothing

```python
models, info = gpap.sample("formula.cnf", n=1000)
len(models)        # 0
info["n_models"]   # 901,603
```

README example 1 works around it with `n_cov=5_000`; README example 2
(`gpap.sample_circuit(..., n=50_000)`) **forgets**, and silently returns `[]`.

### 0.7 `AssumptionChecker` cost model — what is unbudgeted

Measured on `10.sk_1_46.cnf` (N=1494, M=2215):

```
K=  1000:   46 ms   returns  11.9 MB
K= 20000:  721 ms   returns 237.1 MB      <-- sat_count is (K, M), built on EVERY call
```

`compute_k_safe` models **neither** of the two costs this path actually has:

| cost | size | modelled by `compute_k_safe`? |
|---|---|---|
| device row state | `K·(4M + 4·ceil(2N/32) + 4·(N+72))` | yes |
| **(K, L) decision tensor** | `8·K·L` bytes | **no** |
| **host tensors returned** | `K·(2N + 4M)` bytes | **no** |

At K=200,000 / L=200 that is 320 MB of unbudgeted device memory and **2.37 GB of host tensors per
call**. On this box an over-large K spills to host RAM instead of erroring, so the failure mode is
a silent 10× slowdown.

### 0.8 Consumers that must keep working

- `gpap/benchmark.py:180` reads `info["models"]` (full DIMACS models) to compute 2-wise coverage
  → **the `n_cov` path must survive item 1**, only the *default* changes.
- `tests/test_assumption_checker.py` reads `results['conflict']`, `['num_propagated']`,
  `['sat_count'].shape == (2, 3)`, `['assignment'].shape == (2, 6)` → **item 4 changes these
  defaults, so this test must be updated in the same commit.**
- `tests/test_k_cap.py` pins the k-cap contract, including `resolve_k_cap(None, dev) is None`
  ("no cap" is a real, distinct value from "auto"). Item 2 must not disturb it.
- `gpap/sampling/miner.py` has its own `k_cap=8192` + `safe_k()` VRAM clamp. **Leave it alone** —
  it is a different (probe-shaped) memory profile with its own validated margin.

---

# 1. `gpap.sample()` returns the samples

### Current
`gpap/__init__.py::sample` → `pipeline.sample_file(..., n_time=n)` → returns `info.pop("models")`,
which is populated **only** by the optional coverage pass (`n_cov > 0`, default 0).

### Target
Route the top-level API through the **bit-projection path** (`out_cap`) that the CLI's `--out`
already uses — ~10× cheaper than reconstructing full N-variable models, and it is the artifact
people actually want.

```python
samples, info = gpap.sample("formula.cnf", n=1000)
samples.shape      # (1000, 46)  numpy uint8, one bit per SUPPORT variable
info["cols"]       # the original variable id of each column
```

**Return-type change, agreed:** `list[list[int]]` (full DIMACS models) → numpy `(rows, |support|)`
uint8. Keep the old behaviour behind `full_models=True`.

### Edits
- `gpap/__init__.py::sample`, `sample_clauses`, `sample_circuit`: add
  `max_keep: int = 100_000` and `full_models: bool = False`; pass `out_cap=min(n, max_keep)`
  unless `full_models`, in which case pass `n_cov=n` as today. Return `info.pop("samples")`.
- Normalise the empty case: when `out_cap == 0` or nothing was found, return a `(0, S)` array,
  never `None` (pipeline currently sets `samples = cols = None` when `out_cap == 0`).
- **Fix 0.2 first** (`_build_projection_plan` on the UNSAT recon): make `_unsat_sampler` record
  `orig_nv` / `orig_support`, or have the plan handle `r["unsat"]` explicitly and return an
  `S = len(support)` all-zero plan. UNSAT must yield a `(0, S)` matrix, not an exception.
- Update the README quickstart — **both** examples, since example 2 is currently wrong.

### Corner cases (all probed; expected behaviour after the change)

| case | expected |
|---|---|
| UNSAT (`[[1],[-1]]`) | `(0, S)`, no exception ← **currently KeyError** |
| UNSAT after unit-prop (`[[1],[-1,2],[-2]]`) | same ← **currently KeyError** |
| no support declared | columns = all N variables *(verified OK)* |
| `out_cap` > solutions found | returns what exists *(verified OK: 1000 asked, 1000 rows)* |
| `n = 0` | `(0, S)` |
| `M = 0` (no clauses) | free hypercube *(verified OK)* |
| `simplify` / `els` reconstruction | forced vars constant, don't-cares randomised per class *(verified OK)* |
| circuit input | `info["wire_map"]` still names each column |

### Verification
`tests/test_output.py` already covers `sample_bits` soundness. Add: UNSAT → empty matrix; `n=0`;
`full_models=True` still returns DIMACS lists; every emitted row satisfies the formula.

---

# 2. One K policy — auto everywhere, still overridable

### Target
`auto` is the default at **every** entry point; an explicit int always wins.

### Edits
- `gpap/sampling/sampler.py::GPUSampler.__init__`: `k_cap: int | str | None = "auto"`, resolved
  via `resolve_k_cap(k_cap, device)` at the top of `__init__`.
  - **Keep the three-value contract intact**: `"auto"` → VRAM-proportional, `int` → that cap,
    `None` → **no cap** (analytic bound only). `tests/test_k_cap.py` pins this.
  - Safe because of §0.4 — no existing caller uses the default.
- `cpp/include/gpap/gpap.hpp`: `SampleOptions::k_cap = 0` meaning auto; `auto_k()` applies the
  VRAM-proportional rule.
- Add `int32_t gpap_default_k_cap(int device)` to the C ABI (`gpap.h` + a small
  `cpp/src/calibrate.cpp` implementation mirroring `K_CAP_REF · total_VRAM / K_CAP_VRAM_REF`,
  floored at `K_CAP_REF`), and have `gpap.hpp` call it. Do **not** re-derive the constants in the
  header — two copies of that rule is exactly what §0.5 is about.
  - `K_CAP_REF = 49152`, `K_CAP_VRAM_REF = 12878086144` (RTX 4080 Laptop total). Keys off
    **total** VRAM, never free — a free-based cap made K jitter run-to-run.
- `gpap/engine/_core.py`: bind the new symbol.

### Risk
This is a **K change** for anyone who used the defaults, and K changes are kill-switch territory
per the ground rules. Mitigations: (a) no in-repo caller uses the default (§0.4); (b) the resulting
K is the one the pipeline has always used and validated; (c) the C++ change only ever *raises* the
cap on cards larger than the reference, where the analytic bound still binds the real allocation.
Run the nvidia-smi kill-switch on one large-M instance anyway.

### Verification
- `tests/test_k_cap.py` must pass unchanged.
- Add: `GPUSampler(f, dev)` and `pipeline.sample_clauses(...)` agree on `batch_size` for the same
  formula; `k_cap=None` still means no cap; explicit ints pass through.
- C++: `gpap_sample` prints the same K as the Python path on the same formula.

---

# 3. Demote the memory model out of the public API

### Edits
Remove `compute_k_safe`, `gpu_mem_available`, `default_k_cap`, `resolve_k_cap` from
`gpap/engine/__init__.py`'s imports and `__all__`. They stay importable from
`gpap.engine.calibrate` (which is where benchmarks, `pipeline`, `sampler` and `tests/test_core.py`
already get them or can).

Public surface: **15 → 11 names**, all of which are things a user would actually call.

### Corner cases
- `tests/test_core.py::test_k_calibration_matches` imports `from gpap.engine.calibrate import
  compute_k_safe` — already the deep path, unaffected.
- `tests/test_k_cap.py` imports `default_k_cap` / `resolve_k_cap` — **check and repoint** to
  `gpap.engine.calibrate`.
- `gpap/benchmark.py`, `gpap/sampling/pipeline.py` — repoint any `from gpap.engine import ...`.
- Grep for `from gpap.engine import` across `eval/`, `scripts/`, `notebooks/` before cutting.

---

# 4. `AssumptionChecker` chunks itself — the user never sees K

**The headline item.** Batched assumption propagation is the project's primitive, and it is
currently the least finished path.

### Target

```python
ck = AssumptionChecker("formula.cnf")        # engine built ONCE, reused across calls
r  = ck.check_assumptions(sets)              # ANY number of sets; chunked internally
r["conflict"]                                # (n,) bool   — always
r["num_propagated"]                          # (n,) int32  — always (cheap)
r["assignment"]                              # (n, 2N) int8  — only if assignments=True
r["sat_count"]                               # (n, M) int32  — only if sat_count=True
```

### Design

1. **Build the engine once**, in `__init__` (or lazily on first use), and `reset(K)` per chunk.
   Today Python rebuilds `BCPEngine` on *every* call — re-uploading the occurrence CSR and
   re-running the units flush. The C++ face already does it right; match it.
2. **Chunk size** from `compute_k_safe(M, N, gpu_mem_available(device), cap=resolve_k_cap("auto"),
   max_clause_size=..., total_lits=...)`, then **subtract the two costs §0.7 shows are
   unmodelled**:
   - the `(K, L)` decision tensor: `8·K·L`
   - whatever the caller asked to be returned: `K·(2N if assignments else 0) + K·(4M if sat_count
     else 0)` — this one lands in **host** RAM, so bound it separately and document the ceiling.

   Concretely: `K_chunk = min(K_safe_adjusted, len(sets))`. `L` is the max set length **over the
   whole input**, or better **per chunk** (a single long set should not shrink every chunk).
   Sorting sets by length would minimise padding but **must not reorder the output** — if done,
   keep an index map and restore input order.
3. **Assemble results in input order.** Preallocate the output arrays and fill per chunk.
4. **Validate literals on the host, before the kernel** — `lit == 0` or `abs(lit) > num_vars`
   raises `ValueError`, matching the C++ message. This closes §0.1. Do it vectorised (numpy) so a
   million-set call does not pay a Python loop.
5. **Empty input** → return correctly-shaped empty arrays, no engine call. Closes §0.3.

### Backward compatibility — decide explicitly
`sat_count` and `assignment` become opt-in. `tests/test_assumption_checker.py:51-52` asserts both
shapes unconditionally, so **that test changes in the same commit**. Nothing else in the repo
reads either key (checked). The README example only prints `r["conflict"]`.

### Corner cases (all probed; expected behaviour after the change)

| case | today | expected after |
|---|---|---|
| `[]` (no sets) | `GpapError: reset: bad argument` | empty arrays, no launch |
| `[[]]` (one empty set) | OK | OK — no assumptions, row stays active |
| ragged sets | OK (padded to max L) | OK — pad per chunk |
| duplicate literal in a set | OK (kernel skips) | OK |
| `l` and `~l` in one set | OK (conflict) | OK |
| **literal out of range** | **silent OOB write into other rows** | **`ValueError`** |
| **literal `0`** | silently treated as padding | **`ValueError`** |
| formula whose own units conflict | every row conflicted | unchanged |
| `M = 0` | OK, `sat_count` is `(n, 0)` | unchanged |
| `N = 0` | — | must not divide by zero when sizing K |
| out-of-envelope formula | `ValueError` from `BCPEngine` | unchanged, raised at construction |
| one set with L > N literals | trail_cap grows (`ensure_state(e, K, L)`) | unchanged; L must feed the K budget |
| n_sets < K_chunk | one chunk | one chunk, `reset(len(sets))` not `reset(K_chunk)` |

### Verification
- Every existing `tests/test_assumption_checker.py` case still passes (with the two shape
  assertions updated to request the arrays explicitly).
- **New**: chunked vs single-chunk results are **identical** — run the same 50k sets with the
  auto chunk size and with `pin`ned tiny chunks; conflict flags, implications and assignments must
  match element for element, in input order.
- **New**: out-of-range literal and literal 0 raise `ValueError` (and a regression test that the
  §0.1 corruption is gone — row 6 stays empty).
- **New**: `check_assumptions([])` returns empty.
- **New**: a set count far larger than any single chunk (e.g. 5× K) completes and is correct.
- Oracle: compare conflict flags + implied sets against `tests/reference_bcp.py` on random CNFs.

---

# 5. Out of scope (explicitly)

- `pipeline.sample_clauses`'s 30 parameters — user said ignore for now.
- The sampler's own `K` / `K_fused` / `batch_size` / `pin_batch` internals — they stay; they are
  the measurement API and are already documented as such.
- `gpap/sampling/miner.py`'s separate `k_cap` / `safe_k` (§0.8).
- Anything touching the kernels or the sampling algorithm.

---

# 6. Post-implementation check-list — ALL RUN, results inline

1. ✅ `ruff check gpap tests scripts eval cpp` — clean.
2. ✅ Full suite `pytest -q`: **256 passed** (240 before + 16 new), re-run after a from-scratch
   `build_core.sh` and still 256.
3. ✅ Driverless CI path — 121 passed, 77 skipped, 58 deselected.
4. ✅ **No-change gate** — byte-identical to the baseline captured on the pre-change tree, across
   all six commits. Not one sampling number moved.
5. ✅ **Throughput A/B** vs the pre-change worktree: geomean **1.0124** (n=8, 0.902–1.078), model
   counts and batch sizes identical on every cell. The same-tree **control** read **0.9855**
   (0.935–1.074), so the measured difference is inside the instrument's own noise — no
   throughput change. (Running the control is what makes this statement possible; a bare 1.0124
   would not have been interpretable.)
6. ✅ CLI: `info`, `sample` (CNF, `.bench --cycles 4`, `--out`, `--stats`, `--uniformity exact`),
   `circuit2cnf` (file + dir), `bench` (glob, and a second run with `--n-cov --twise` to exercise
   the full-model coverage path §0.8 depends on). **`gpap sample unsat.cnf --out` now prints
   "no samples to write" instead of raising `KeyError`.**
7. ✅ Python API: all three top-level entry points, `AssumptionChecker`, raw `BCPEngine`.
8. ✅ C/C++: `scripts/build_core.sh` end to end (kernels + cmake), `gpap_demo`,
   `gpap_assumptions`, `gpap_sample`; `ldd` shows no unresolved symbols and `libcuda.so.1` is
   still the only CUDA dependency.
9. ✅ **K consistency**: `gpap_sample` and the Python sampler both report **184,027** on
   `10.sk_1_46.cnf`, and `_core.default_k_cap(0) == calibrate.default_k_cap(cuda) == 49,152`
   (now gated by `test_core_and_python_caps_agree`).
10. ✅ nvidia-smi kill-switch. On large M the ANALYTIC bound binds and the cap is irrelevant
    (b05-c50 K=4,542, b12-c50 K=4,162) — as predicted. Ran it where the cap actually binds too
    (small M, old 16,384 vs new auto): peak **6,284 MiB of 12,282 MiB**, `sound=True` throughout,
    no spill. Note the K rise can *lower* the effective batch on some instances (10.sk: 195,500 →
    183,440) because `eng_hold` scales with K — memory-safe either way.
11. ✅ `eval/` generators on the real L4 data: `paper_tables` (357 cells, 0 skipped/errors/unsound,
    repro 23/23 exact), `slide_tables` (agrees with paper_tables — epfl-rare 9.4×, zero-search
    exclusions intact), `soft_price_tables` (0.629× safe/soft), `eval_plots` (all figures +
    summary.md). `verify_artifact` reports one pre-existing FAIL on the softarm pairing stage
    ("no GPU recorded"); **confirmed identical on the pre-change tree**, so it is not ours.

## Commit order (each independently revertible)

```
1. fix 0.2 (UNSAT projection plan)            -- unblocks item 1, standalone bug fix
2. fix 0.1 + 0.3 (literal validation, empty)  -- standalone bug fixes, memory safety
3. item 3 (demote memory model)               -- pure API surface, no behaviour
4. item 2 (K auto everywhere, Python + C++)   -- K change, kill-switch
5. item 1 (sample() returns samples)          -- return-type change, README
6. item 4 (AssumptionChecker auto-chunk)      -- the big one, new tests
```

Items 0.1–0.3 are bug fixes worth landing **regardless** of whether the rest proceeds.

---

# Appendix A — the no-change gate, inline

Nothing here depends on a scratchpad (they are wiped between sessions). **Run this on the
CURRENT tree first and keep the JSON**, then re-run after each commit; the two must be identical.
`pin_batch` makes it immune to the item-2 default change, and the explicit `k_cap=4096` keeps it
valid either way.

```python
"""No-change gate: pinned K, ONE config, fixed seed -> model COUNT must be identical."""
import json, sys, torch
from gpap.engine.formula import CnfFormula
from gpap.sampling.sampler import GPUSampler, parse_independent_support
from gpap.sampling.pipeline import CIRCUIT_PORTFOLIO

dev = torch.device("cuda")
out = {}
for path in sys.argv[1:]:
    f = CnfFormula(dev); f.parse_cnf_file(path)
    for cfgs, tag in [(None, "single"), (list(CIRCUIT_PORTFOLIO), "bandit")]:
        smp = GPUSampler(f, dev, k_cap=4096, seed=5)
        smp.pin_batch(4096)                       # bound holds regardless of the k_cap default
        smp.support = parse_independent_support(path)
        r = smp.sample(20000, configs=cfgs, decisions_per_round=8, random_var=True,
                       max_chunks=8, collect=False, min_measure_chunks=2)
        out[f"{path.split('/')[-1]}:{tag}"] = [r["n_models"], r["attempts"], r["bcp_decisions"],
                                               r["bcp_implications"], r.get("dominant_config")]
print(json.dumps(out, indent=0, sort_keys=True))
```

```bash
python gate.py benchmarks/sampling/10.sk_1_46.cnf \
               benchmarks/sampling/27.sk_3_32.cnf \
               benchmarks/sampling/109.sk_4_36.cnf > gate_before.json
# ... make the change ...
python gate.py <same three> > gate_after.json && diff gate_before.json gate_after.json
```

If a "before" run was not captured, regenerate it from the pre-change commit:
`git worktree add /tmp/base <sha>` and run the gate there with
`GPAP_LIB=$(readlink -f build/libgpap.so)` so both arms share one library.

# Appendix B — the three bug reproducers (turn these into tests)

```python
# 0.1  out-of-range literal corrupts OTHER rows (memory safety).  Expect ValueError after the fix.
from gpap.engine.checker import AssumptionChecker
import numpy as np
ck = AssumptionChecker(); ck.formula.init_formula(3, [[1, 2], [-1, 3], [-2, -3]])
r = ck.check_assumptions([[99]] + [[] for _ in range(7)])   # 99 does not exist; 2N = 6, NW = 1
# TODAY: row 6 comes back with literal 4 set — 196 >> 5 = six words past row 0's bitmap.
assert np.nonzero(r["assignment"][6].cpu().numpy())[0].size == 0, "row 6 was corrupted"

# 0.2  out_cap on an UNSAT formula.  Expect a (0, S) matrix after the fix.
from gpap.sampling.pipeline import sample_clauses
import torch
sample_clauses(nv=1, clauses=[[1], [-1]], support=None, device=torch.device("cuda"),
               n_time=10, out_cap=10, mine=False)          # TODAY: KeyError: 'orig_nv'
# also reachable from the CLI:  gpap sample unsat.cnf -n 10 --out f.bits

# 0.3  empty input.  Expect empty arrays after the fix.
ck.check_assumptions([])                                    # TODAY: GpapError: reset: bad argument
```

# Appendix C — measurements to re-take after item 4

The point of item 4 is that these numbers get better. Measured on
`benchmarks/sampling/10.sk_1_46.cnf` (N=1494, M=2215):

| | before | after |
|---|---|---|
| `check_assumptions` at 1,000 sets | 46 ms, 11.9 MB returned | **2 ms, 0.0 MB** |
| `check_assumptions` at 20,000 sets | 721 ms, 237.1 MB returned | **19 ms, 0.1 MB** |
| 200,000 sets | — | 86 ms (5 chunks) |
| 1,000,000 sets | — | 670 ms (21 chunks) |
| engine construction per call | yes (re-uploads occ CSR + re-runs units flush) | **once, reused** |
| largest set count that works | whatever fits; no chunking, silent host-RAM spill | **any** |

The old cost is exactly what you now pay only when you ask for it: with
`assignments=True, sat_count=True` the same calls read 38 ms / 11.9 MB and 755 ms / 237.1 MB —
i.e. nothing got slower, the expensive part just stopped being mandatory.

Chunking is invisible: 50,000 sets run with the auto chunk size and with a forced 997-set chunk
are **identical element for element, in input order**, on every returned key (also gated by
`tests/test_assumption_checker.py::TestChunking`). The C++ face was given the same treatment and
matches manual 1,000-set slicing exactly over 200,000 sets.

A request whose RESULT cannot fit is now refused with a message naming the size and the way out,
rather than silently spilling to host RAM:

```
assignments for 4,000,000 assumption sets needs 11.1 GiB, but only 6.5 GiB is available on cuda.
Check the sets in batches, or drop the flag — 'conflict' and 'num_propagated' are ~5 bytes/set
and scale to any number of sets.
```

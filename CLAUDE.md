# GPAP - GPU Parallel Assumption Propagation

## Project Summary

GPU-accelerated BCP (Boolean Constraint Propagation) for SAT formulas using PyTorch. Checks multiple assumption sets simultaneously on GPU. Companion project to [GPSS](/home/yshalmon/git/GPSS) which does GPU local search on SAT using sparse clause matrices.

## Core Design

### The Key Trick: No Clause Scanning for BCP

Instead of a clause matrix (like GPSS), GPAP uses:

1. **Negated occurrence lists** (CSR format): `occ[l]` = clauses where `~l` appears. When literal `l` is set TRUE, `~l` is falsified in those clauses.
2. **Two per-clause counters** (batched over K assumption sets):
   - `sat_count (K, M)`: remaining non-falsified literals (init = clause_size). Decremented when a literal is falsified. dtype auto-selected: int8 if max clause size ≤ 127, int16 ≤ 32767, else int32.
   - `lit_sum (K, M)`: running sum of non-falsified literal indices (init = sum of all lit indices in clause). Subtracted when a literal is falsified. No overflow in practice (int32 is sufficient for N ≤ 100K with typical clause sizes).
3. **The sum trick**: when `sat_count == 1`, the remaining literal's index IS `lit_sum`. No clause scanning needed.
4. **Conflict**: `sat_count == 0` (all literals falsified).

### Literal Indexing (2N scheme)

- Positive literal `v` -> index `2*(v-1)`
- Negative literal `-v` -> index `2*(v-1) + 1`
- Negation: `lit_index ^ 1`
- Assignment: `(K, 2N)` tensor, values 0 (unassigned) / 1 (TRUE)

### K (Batch Size) Memory Model

K is computed from peak GPU memory across three sequential phases:

```
formula_bytes   = 8*M + 16*N + 8*total_lits_est      (one-time)
per_batch_bytes = (sat_b+4)*M + 2*N                   (static: sat_count+lit_sum+assignment)
                + max(8*N, 1000*total_lits//N) + 3    (peak phase: apply or propagate)
K_safe = gpu_mem_available(device) / per_batch_bytes  (post-load: free already excludes formula)
```

Phase peaks (take the max — phases are sequential, not concurrent):
- **apply_assumptions_bulk**: 4 simultaneous K×2N bool tensors → **+8N/batch**
- **_bulk_propagate_lits**: 7 int64 + 2 int32 temporaries all of size
  `total = K × P × avg_occ_length` per round (P ≈ 10 props/batch/round,
  avg_occ_length = total_lits/N) → **+1000 × total_lits/N per batch**
  (calibrated from goldb: 570 bytes/unit empirically; ×1.75 safety = 1000 for dense formulas)
- **reset_state**: no K×M intermediate — dtype conversion done on M-element vector before
  `expand()` (bcp_engine.py). Earlier `.expand().clone().to(dtype)` created full K×M int32 peak.

Which phase dominates:
- `apply_peak > prop_peak` when `N² > 90 × total_lits` (sparse/large-N, e.g. goldb N=10316)
- `prop_peak > apply_peak` when `N² < 90 × total_lits` (dense/small-N, e.g. sat24 N=450)

Oversizing K causes tensors to spill to shared system RAM via PCIe (~12 GB/s vs ~200 GB/s VRAM).
Diagnostic: reset time ~464 ms/chunk (spilled) vs ~27 ms/chunk (VRAM).
Safety fraction: 0.75 (not 0.9) — WDDM/WSL2 reserves VRAM the CUDA driver can't see.

Call `gpu_mem_available(device)` AFTER loading the formula, then pass `deduct_formula=False`:
  `K = compute_k_safe(M, N, gpu_mem_available(device), ..., deduct_formula=False)`
  `mem_get_info().free` already excludes the loaded formula, so no second deduction needed.
Pass actual `formula.clause_sizes.sum()` as `total_lits` for accurate prop-peak estimation.
Pass `cap=None` for clause learning (maximize K); `cap=65536` for benchmarks (consistent timing).

## Project Structure

```
gpap/
  clsdata/
    occ_list.py            # OccurrenceList: CSR format, negated convention
    cnf_formula.py         # CnfFormula: DIMACS parser + clause_sizes + init_lit_sum
  solver/
    bcp_engine.py          # BCPEngine: sat_count (int8/16)/lit_sum (int32) counters, BCP loop
    assumption_checker.py  # AssumptionChecker: high-level API
    external_solver.py     # PySAT/MinisatGH wrapper
    clause_learner.py      # extract_conflict_clauses(): conflict → learned clause
  utils/                   # sat_def.py, time_utils.py
tests/                     # pytest suite (20 tests)
scripts/
  profile_engine.py        # single-instance GPAP vs PySAT profiling + K sweep
  run_benchmark_suite.py   # run suite.json and optionally compare vs baseline
  build_benchmark_suite.py # scan CNF dirs, compute K/L params, write suite manifest
  clause_learn_solve.py    # GPAP clause learning → kissat augmented solve
  colab_benchmark.py       # cross-GPU driver: pairs anchor + random-L×soft-stop e2e grid + compare
  colab_plots.py           # plots for a colab_benchmark payload (throughput/speedup/soft-stop tradeoff)
notebooks/
  gpap_colab_benchmark.ipynb # Colab front-end: clone → benchmark this GPU → plots → rank GPUs
benchmarks/
  cnf/                     # 20 bundled family-diverse CNFs (incl. 6 medium M 10k–40k; used by colab_benchmark)
  colab_results/           # per-GPU result JSON + PNG plots (committed; read by compare_saved)
  suite.json               # 15-instance mixed suite (sat2006 + sat24)
  suite_sat24_small.json   # 108 sat24 instances with M ≤ 10,000
  .experiments/            # result JSON files (gitignored)
```

## Running Tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
```

## Key Scripts

### Profile a single instance
```bash
PYTHONPATH=. .venv/bin/python scripts/profile_engine.py \
    --cnf /path/to/formula.cnf --K 65536 --assumptions 2 --device cuda
```

### Run the benchmark suite
```bash
PYTHONPATH=. .venv/bin/python scripts/run_benchmark_suite.py \
    --suite benchmarks/suite_sat24_small.json \
    --output .experiments/results_$(date +%Y%m%d).json \
    --no-external --skip-k1
```

### Clause learning + kissat solve
```bash
# Exhaustive pairs (all ±v1,±v2 with v1<v2), K auto-sized from GPU memory
PYTHONPATH=. .venv/bin/python scripts/clause_learn_solve.py \
    --cnf /path/to/formula.cnf --mode pairs --auto-k --device cuda

# Random rounds
PYTHONPATH=. .venv/bin/python scripts/clause_learn_solve.py \
    --cnf /path/to/formula.cnf --mode random --K 65536 --L 2 --rounds 5
```

### Cross-GPU benchmark (Colab or local)
```bash
# Two e2e measurements on bundled benchmarks/cnf/, GPU vs sequential CPU PySAT:
#  1. Pairs anchor — exhaustive (±v1,±v2) sets, no cutoff: the deterministic headline sets/s
#     + a conflict-count sanity check (GPU real_conflict ≤ CPU all-conflicts).
#  2. Random-L × soft-stop grid — for each L, random L-literal sets streamed in target_chunks×K
#     batches, run with no cutoff (baseline) and several soft_stop_frac (min_rounds=5). Reports
#     sets/s and successful_sets/s (finished = ¬timed_out) so soft-stop's throughput gain shows.
# K auto-sizes to ALL the GPU's free VRAM (compute_k_safe cap=None); workload = target_chunks×K.
# Writes benchmarks/colab_results/<gpu>.json + PNG plots (scripts/colab_plots.py).
PYTHONPATH=. .venv/bin/python scripts/colab_benchmark.py --quick --device cuda
PYTHONPATH=. .venv/bin/python scripts/colab_benchmark.py --device cuda \
    --target-chunks 32 --l-values 2,5,10,20 --soft-stop-fracs 0.9,0.7,0.5,0.3,0.1
PYTHONPATH=. .venv/bin/python scripts/colab_benchmark.py --compare   # rank saved GPUs
```
On Colab, open `notebooks/gpap_colab_benchmark.ipynb` (GPU runtime): it clones the repo,
benchmarks the attached GPU, shows the plots, and `compare_saved()` ranks all committed results.
`target_chunks` must be large enough that the GPU is amortized over many chunks (quick=6, full=32);
K is uncapped so high-VRAM GPUs get more parallelism. The full grid runs minutes to ~1 h per GPU.

### Build a suite manifest
```bash
PYTHONPATH=. .venv/bin/python scripts/build_benchmark_suite.py \
    --dirs /path/to/sat_benchmarks \
    --max-m 10000 --no-limit \
    --output benchmarks/suite_sat24_small.json
```

## BCPEngine API

```python
engine = BCPEngine(formula)
engine.init_state(K)              # allocate (K,M) tensors
engine.reset_state(K)             # in-place reset (reuse allocations when K unchanged)
engine.apply_assumptions_bulk(t)  # t: (K,L) int64 tensor from make_assumption_tensor()
engine.propagate(
    max_rounds=None,              # hard BCP round cap
    soft_stop_frac=None,          # stop when round work < frac * peak_work
    min_rounds=0,                 # don't soft-stop before this many rounds
)
engine.conflict        # (K,) bool — all conflicts (including trivial)
engine.real_conflict   # (K,) bool — BCP-derived only (round_count > 1); use for clause learning
engine.timed_out       # (K,) bool — hit round cap before fixpoint
engine.assignment      # (K, 2N) int8
```

## Clause Learning Pipeline

For each conflicting assumption set `{a1,...,aL}` that required BCP (real_conflict=True),
the clause `(¬a1 ∨ ... ∨ ¬aL)` is entailed by the formula. These learned clauses are:
- **Not trivial**: `real_conflict` filters out round-1 conflicts where the assumption directly
  violated an existing clause (learned clause would be subsumed).
- **Binary** when L=2: pairs mode generates all C(N,2)×4 binary clauses in ~10s for small N.

Observed speedup on `mdp-28-14-sat` (N=792, M=4386): 474ms GPAP → 5,734 binary clauses
→ kissat 3.9× faster (42s plain → 10.7s augmented). More clauses ≠ better; sweet spot varies.

## Soft-Stop Heuristic

`propagate(soft_stop_frac=0.05, min_rounds=5)` stops BCP when a round's unique propagations
drop below 5% of the peak seen so far (after at least 5 rounds). Batches still in flight are
marked `timed_out`. Gives ~1.3–1.6× speedup on deep instances with <5pp conflict rate loss.

## Benchmarks

Results stored in `benchmarks/.experiments/`. Key baselines:
- `results_baseline_stable_l{2,20,50}.json` — stable L=2/20/50 runs for regression checks.

Two speedup metrics in suite results:
- `speedup_vs_pysat` — extrapolated PySAT(K) / GPAP(K): the real-world speedup headline
- `speedup_vs_k1` — (K × GPAP(K=1)) / GPAP(K): GPU parallelism efficiency

## Assumption Generation

```python
from scripts.profile_engine import (
    generate_random_assumptions,       # K random sets of L literals
    generate_all_pairs_chunks,         # exhaustive (±v1,±v2), v1<v2, chunked by K
    generate_all_triplets_chunks,      # exhaustive (±v1,±v2,±v3), v1<v2<v3, chunked
    count_all_pairs,                   # N*(N-1)*2
    count_all_triplets,                # N*(N-1)*(N-2)//6 * 8
)
```

Feasibility at K=175K (mdp, auto-k): pairs exhaustive in 8 chunks (~4s); triplets ~12,700 chunks (impractical for large N).

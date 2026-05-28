# GPAP - GPU Parallel Assumption Propagation

GPU-accelerated Boolean Constraint Propagation (BCP) for SAT formulas using PyTorch.  
Check **multiple assumption sets simultaneously** on GPU.

## Overview

GPAP provides infrastructure for propagating assumptions on CNF (SAT) formulas, where the core feature is fast parallel checking of K assumption sets on GPU.

**Key features:**
- **Occurrence lists** (CSR format) for O(1) lookup of affected clauses when a literal is set
- **Clause sat_count + lit_sum counters** for O(1) unit and conflict detection
- **The sum trick**: unit literal found via arithmetic — no clause scanning needed
- **Batched BCP** across K assumption sets sharing the same formula

## Quick Start

```python
from gpap.solver.assumption_checker import AssumptionChecker

checker = AssumptionChecker("formula.cnf")
results = checker.check_assumptions([
    [1, -2, 3],     # assumption set 1: x1=T, x2=F, x3=T
    [-1, 2, -3],    # assumption set 2: x1=F, x2=T, x3=F
    [1, 2, 3],      # assumption set 3: x1=T, x2=T, x3=T
])

print(results['conflict'])       # (K,) bool tensor
print(results['num_propagated']) # (K,) how many literals BCP propagated
print(results['assignment'])     # (K, 2*N) final assignment state
```

## Run on Colab (benchmark any GPU)

Open [`notebooks/gpap_colab_benchmark.ipynb`](notebooks/gpap_colab_benchmark.ipynb) in Google
Colab, pick a GPU runtime (`Runtime ▸ Change runtime type ▸ GPU`), and run all cells. It clones
this repo, benchmarks the attached GPU on the bundled CNFs in `benchmarks/cnf/`, and prints the
headline: **how fast GPAP propagates assumption sets (sets/s) vs a sequential CPU PySAT baseline.**
Batch size `K` auto-sizes to the GPU's VRAM, so the same notebook works on a T4 or an A100.

Results are saved per-GPU to `benchmarks/colab_results/<gpu>.json`; the last cell ranks every
saved GPU so you can say "GPU A is N× faster than GPU B." The same driver runs headless:

```bash
PYTHONPATH=. python scripts/colab_benchmark.py --quick     # benchmark this machine's GPU
PYTHONPATH=. python scripts/colab_benchmark.py --compare   # rank saved per-GPU results
```

## Architecture

```
CNF file
  |
  v
CnfFormula  ──>  OccurrenceList (CSR: literal -> negated clauses)
  |                + clause_sizes, init_lit_sum
  v
BCPEngine   ──>  sat_count (K, M)  +  lit_sum (K, M)
  |                + assignment (K, 2N)
  v
Results: conflict, assignment, num_propagated per batch
```

### Data Flow

1. **Parse**: DIMACS CNF file -> occurrence lists + per-clause lit_sum
2. **Initialize**: K copies of clause counters (sat_count = clause_sizes, lit_sum = init_lit_sum)
3. **Apply assumptions**: For each assumed literal `l`, decrement sat_count and subtract `~l` from lit_sum for all clauses in `occ[l]`
4. **BCP loop**: Detect unit clauses (sat_count == 1), read forced literal directly from lit_sum, propagate, repeat until fixpoint or conflict

### The Sum Trick

Each clause maintains `lit_sum` = sum of non-falsified literal indices. When a literal is falsified, its index is subtracted. When `sat_count` reaches 1, `lit_sum` **is** the remaining literal's index.

**Zero clause scanning. Zero gather. Pure arithmetic.**

### Literal Indexing (2N scheme)

- N variables -> 2N literal indices
- Positive literal `v` -> index `2*(v-1)`
- Negative literal `-v` -> index `2*(v-1) + 1`
- Negation: `lit_index ^ 1`
- `occ[l]` stores clauses where `~l` appears (negated occurrence)

## Project Structure

```
gpap/
  gpap/
    clsdata/           # CNF formula + occurrence lists
      cnf_formula.py   # DIMACS parser, clause data
      occ_list.py      # CSR occurrence list
    solver/            # BCP engine + checker API
      bcp_engine.py    # Core BCP loop
      assumption_checker.py  # High-level API
      external_solver.py     # PySAT validation
    analyzer/          # Analysis tools
    utils/             # Type definitions, timing
    vis/               # Visualization
  tests/               # Test suite
  scripts/             # Examples, benchmarks
  benchmarks/          # Benchmark data
```

## Requirements

- Python 3.10+
- PyTorch
- python-sat (for external solver validation)

## Relation to GPSS

GPAP is a companion project to [GPSS](https://github.com/yshalmon/GPSS) (GPU SAT Solver).

| | GPSS | GPAP |
|---|---|---|
| **Approach** | Clause matrix multiplication | Occurrence lists + counters |
| **Use case** | Local search / genetic SAT solving | BCP / assumption propagation |
| **Formula repr** | Sparse COO tensor (clauses x vars) | CSR occurrence lists (lits -> clauses) |
| **Evaluation** | `score = clause_size + cls_matrix * assg` | `sat_count` counter per clause |
| **Unit detection** | `score == 1` | `sat_count == 1`, literal = `lit_sum` |

## License

MIT License - Yogev Shalmon

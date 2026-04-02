# GPAP - GPU Parallel Assumption Propagation

## Project Summary

GPU-accelerated BCP (Boolean Constraint Propagation) for SAT formulas using PyTorch. Checks multiple assumption sets simultaneously on GPU. Companion project to [GPSS](/home/yshalmon/git/GPSS) which does GPU local search on SAT using sparse clause matrices.

## Core Design

### The Key Trick: No Clause Scanning for BCP

Instead of a clause matrix (like GPSS), GPAP uses:

1. **Negated occurrence lists** (CSR format): `occ[l]` = clauses where `~l` appears. When literal `l` is set TRUE, `~l` is falsified in those clauses.
2. **Two per-clause counters** (batched over K assumption sets):
   - `sat_count (K, M)`: remaining non-falsified literals (init = clause_size). Decremented when a literal is falsified.
   - `lit_sum (K, M)`: running sum of non-falsified literal indices (init = sum of all lit indices in clause). Subtracted when a literal is falsified.
3. **The sum trick**: when `sat_count == 1`, the remaining literal's index IS `lit_sum`. No clause scanning needed.
4. **Conflict**: `sat_count == 0` (all literals falsified).

### Literal Indexing (2N scheme)

- Positive literal `v` -> index `2*(v-1)`
- Negative literal `-v` -> index `2*(v-1) + 1`
- Negation: `lit_index ^ 1`
- Assignment: `(K, 2N)` tensor, values 0 (unassigned) / 1 (TRUE)

## Project Structure

```
gpap/
  clsdata/                 # clause data structures (analogous to gpss/clsmat/)
    occ_list.py            # OccurrenceList: CSR format, negated convention
    cnf_formula.py         # CnfFormula: DIMACS parser + clause_sizes + init_lit_sum
  solver/                  # solver/checker (analogous to gpss/solver/)
    bcp_engine.py          # BCPEngine: sat_count/lit_sum counters, BCP loop
    assumption_checker.py  # AssumptionChecker: high-level API
    external_solver.py     # PySAT/MinisatGH wrapper (adapted from GPSS)
  analyzer/                # analysis tools (placeholder)
  utils/                   # sat_def.py (type defs), time_utils.py
  vis/                     # visualization (placeholder)
tests/                     # pytest suite (20 tests)
scripts/                   # examples, benchmarks (placeholder)
benchmarks/                # benchmark data (placeholder)
```

## Running Tests

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```

Uses the GPSS venv at `/home/yshalmon/git/GPSS/.venv/` which has torch and pytest.

## What's Reused from GPSS

- `sat_def.py`: type definitions (ext_lit, ext_cls, ext_assg)
- `time_utils.py`: timing utility
- `external_solver.py`: PySAT wrapper (adapted)
- CNF parser pattern from `gpss/clsmat/cls_matrix_base.py`
- `.gitignore`, project layout conventions

## Status

- Core infrastructure complete: occurrence lists, CNF parsing, BCP engine, assumption checker API
- 20 tests passing (formula, BCP, end-to-end)
- Placeholder modules: analyzer, vis, scripts, benchmarks
- Next steps: GPU performance benchmarks, bulk scatter-based BCP for large K, analyzer tooling

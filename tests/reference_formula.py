"""Pure-numpy reference build of the engine's formula structures — the ORACLE for the core.

`cpp/src/formula.cpp` is THE builder the library ships (one implementation, shared by the Python
and C++ faces). This module is its independent re-derivation, kept where reference
implementations live in this repo (`reference_bcp.py` is the same idea for propagation):
`tests/test_core.py` builds every structure both ways and compares entry for entry, and the
host-only suite uses it to cover DIMACS parsing on machines with no libgpap at all.

It is deliberately torch-free and device-free: the structures under test are plain arrays, and
an oracle should carry no machinery of its own.

What a formula compiles to (identical to the core, see docs/ENGINE.md):

  * occ CSR (`occ_offsets`, `occ_clauses`): for each literal l, the clauses where ~l appears —
    setting l TRUE falsifies ~l in exactly those clauses;
  * `clause_sizes`: literals per clause (the packed word's initial sat_count);
  * `init_lit_sum`: per-clause SUM of literal indices mod 2^32 (the packed word's initial sum
    field; the engine masks it down to its split's width, and mod-2^32-then-mod-2^F == mod-2^F).

Literal indexing is the 2N scheme: +v -> 2(v-1), -v -> 2(v-1)+1, negation = index ^ 1.
An EMPTY clause cannot be represented in the counters, so it is dropped from the structures and
recorded in `has_empty_clause` (the engine turns that flag into an all-rows conflict).
"""
import numpy as np
from itertools import chain


def flat_lit_arrays(clauses: list) -> tuple:
    """(lens, lit_idx, cls_idx) int64 numpy arrays for a clause list — one shared flatten
    feeding the occ CSR, clause_sizes and init_lit_sum."""
    M = len(clauses)
    lens = np.fromiter(map(len, clauses), dtype=np.int64, count=M)
    total = int(lens.sum()) if M else 0
    if total:
        flat = np.fromiter(chain.from_iterable(clauses), dtype=np.int64, count=total)
    else:
        flat = np.empty(0, dtype=np.int64)
    lit_idx = 2 * (np.abs(flat) - 1) + (flat < 0)      # ext_lit_to_lit_index, vectorised
    cls_idx = np.repeat(np.arange(M, dtype=np.int64), lens)
    return lens, lit_idx, cls_idx


class ReferenceFormula:
    """The numpy build of one formula's engine structures. Compare against `_core.Formula`."""

    def __init__(self, num_vars: int, clauses: list):
        self.has_empty_clause = any(len(c) == 0 for c in clauses)
        if self.has_empty_clause:
            clauses = [c for c in clauses if c]
        self.num_vars = int(num_vars)
        self.num_clauses = len(clauses)

        lens, lit_idx, cls_idx = flat_lit_arrays(clauses)
        # occ CSR over the NEGATED literal (lit_idx ^ 1). The stable argsort keeps each
        # literal's clause indices in ascending clause order — the layout the core produces.
        num_lits = 2 * self.num_vars
        neg = lit_idx ^ 1
        order = np.argsort(neg, kind="stable")
        offsets = np.zeros(num_lits + 1, dtype=np.int64)
        if len(neg):
            np.cumsum(np.bincount(neg, minlength=num_lits), out=offsets[1:])
        self.occ_offsets = offsets.astype(np.int32)
        self.occ_clauses = cls_idx[order].astype(np.int32)

        # Per-clause SUM of literal indices, exact mod 2^32: int64 add, then REINTERPRET the low
        # 32 bits (a cast could saturate; a view is exact).
        M = self.num_clauses
        if M:
            bounds = np.zeros(M, dtype=np.int64)
            np.cumsum(lens[:-1], out=bounds[1:])
            sums = (np.add.reduceat(lit_idx, bounds).astype(np.uint64)
                    & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
        else:
            sums = np.zeros(0, dtype=np.int32)
        self.clause_sizes = lens.astype(np.int32)
        self.init_lit_sum = sums

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_clauses(cls, num_vars: int, clauses: list) -> "ReferenceFormula":
        return cls(num_vars, clauses)

    @classmethod
    def from_file(cls, cnf_file: str) -> "ReferenceFormula":
        """Reference DIMACS parser. Token-stream (clauses are 0-terminated, NOT line-based):
        multiple clauses per line and clauses spanning lines are legal, as is the SATLIB '%'
        end marker. N = max(header N, highest variable used) so trailing declared variables
        (e.g. support vars appearing only in `c ind` lines) keep their slots."""
        header_n = 0
        body: list[str] = []
        with open(cnf_file) as f:
            for line in f:
                s = line.strip()
                if not s or s[0] == "c":
                    continue
                if s[0] == "%":                # SATLIB end marker: rest of file is padding
                    break
                if s[0] == "p":
                    parts = s.split()
                    if len(parts) >= 3 and parts[2].lstrip("-").isdigit():
                        header_n = int(parts[2])
                    continue
                body.append(s)
        arr = np.array(" ".join(body).split(), dtype=np.int64) if body \
            else np.empty(0, dtype=np.int64)
        ends = np.nonzero(arr == 0)[0]
        starts = np.concatenate([[0], ends[:-1] + 1]) if len(ends) else np.empty(0, np.int64)
        clauses = [arr[a:b].tolist() for a, b in zip(starts, ends)]
        if len(arr) and (len(ends) == 0 or ends[-1] != len(arr) - 1):
            tail = arr[(ends[-1] + 1) if len(ends) else 0:]        # unterminated final clause
            clauses.append(tail.tolist())
        max_var = int(np.abs(arr).max()) if len(arr) else 0
        return cls(max(header_n, max_var), clauses)

    # ------------------------------------------------------------------ lookup

    def occ(self, lit_index: int) -> np.ndarray:
        """Clause indices for one literal (the clauses where ~lit appears)."""
        return self.occ_clauses[self.occ_offsets[lit_index]:self.occ_offsets[lit_index + 1]]

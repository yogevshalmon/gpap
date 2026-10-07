"""Pure-numpy reference build of the engine's formula structures, the oracle for
cpp/src/formula.cpp. tests/engine/test_formula.py builds every structure both ways and compares
them; the host-only tests also use it to cover DIMACS parsing without libgpap.

The structures (docs/ENGINE.md). Only clauses of length >= 3 ("long") get a packed word.
  * occ CSR (`occ_offsets`, `occ_mid`, `occ_clauses`): for literal l, the clauses holding ~l.
    Binary entries come first, up to `occ_mid[l]`, with the other literal's index as payload;
    then long entries, with the long-clause index (0..num_long-1). Unit clauses have no occ
    entry (the construction flush asserts them). Each part is clause-ascending.
  * `clause_sizes`: literal count of every kept clause, in input order.
  * `init_lit_sum`: per-clause sum of literal indices mod 2^32 (the engine masks it to its split).
  * `bin_pairs` (sorted pairs, input order) and `unit_lits`.

Literal indices: +v -> 2(v-1), -v -> 2(v-1)+1, negation is index ^ 1. An empty clause is dropped
and recorded in `has_empty_clause` (the engine makes every row conflict).
"""
import numpy as np
from itertools import chain

#: Sign bit the core sets on binary occ payloads (GPAP_OCC_BIN_TAG in cpp/include/gpap/gpap.h),
#: so a kernel can classify an entry from the value it already loaded.
BIN_TAG = -0x80000000


def flat_lit_arrays(clauses: list) -> tuple:
    """(lens, lit_idx) int64 arrays for a clause list: the one literal encoding that the occ
    build, clause_sizes and init_lit_sum all read. Clause c owns
    lit_idx[bounds[c] : bounds[c] + lens[c]] where bounds = exclusive cumsum(lens)."""
    M = len(clauses)
    lens = np.fromiter(map(len, clauses), dtype=np.int64, count=M)
    total = int(lens.sum()) if M else 0
    if total:
        flat = np.fromiter(chain.from_iterable(clauses), dtype=np.int64, count=total)
    else:
        flat = np.empty(0, dtype=np.int64)
    lit_idx = 2 * (np.abs(flat) - 1) + (flat < 0)      # ext_lit_to_lit_index, vectorised
    return lens, lit_idx


class ReferenceFormula:
    """The numpy build of one formula's engine structures. Compare against `_core.Formula`."""

    def __init__(self, num_vars: int, clauses: list):
        self.has_empty_clause = any(len(c) == 0 for c in clauses)
        if self.has_empty_clause:
            clauses = [c for c in clauses if c]
        self.num_vars = int(num_vars)
        self.num_clauses = len(clauses)

        lens, lit_idx = flat_lit_arrays(clauses)
        bounds = np.zeros(self.num_clauses + 1, dtype=np.int64)
        np.cumsum(lens, out=bounds[1:])
        # Occ build over the negated literal, in plain loops (an oracle favours clarity). Input
        # order makes each part clause-ascending, as the core's counting sort does.
        num_lits = 2 * self.num_vars
        bins: list = [[] for _ in range(num_lits)]
        longs: list = [[] for _ in range(num_lits)]
        self.bin_pairs: list = []
        self.unit_lits: list = []
        long_id = 0
        for c in range(self.num_clauses):
            # Slice the shared encoding rather than re-deriving it, so occ and init_lit_sum
            # cannot disagree with each other.
            li = [int(x) for x in lit_idx[bounds[c]:bounds[c + 1]]]
            if len(li) == 1:
                self.unit_lits.append(li[0])
            elif len(li) == 2:
                a, b = li
                bins[a ^ 1].append(b | BIN_TAG)      # sign-bit tag, as the core stores it
                bins[b ^ 1].append(a | BIN_TAG)
                self.bin_pairs.append((min(a, b), max(a, b)))
            else:
                for x in li:
                    longs[x ^ 1].append(long_id)
                long_id += 1
        self.num_long = long_id
        self.num_bin = len(self.bin_pairs)
        offsets = np.zeros(num_lits + 1, dtype=np.int64)
        mid = np.zeros(num_lits, dtype=np.int64)
        entries: list = []
        for l in range(num_lits):
            offsets[l + 1] = offsets[l] + len(bins[l]) + len(longs[l])
            mid[l] = offsets[l] + len(bins[l])
            entries.extend(bins[l])
            entries.extend(longs[l])
        self.occ_offsets = offsets.astype(np.int32)
        self.occ_mid = mid.astype(np.int32)
        self.occ_clauses = (np.asarray(entries, dtype=np.int32) if entries
                            else np.empty(0, dtype=np.int32))

        # Per-clause sum of literal indices mod 2^32: int64 add, then view the low 32 bits (a
        # cast could saturate; a view is exact).
        M = self.num_clauses
        if M:
            sums = (np.add.reduceat(lit_idx, bounds[:-1]).astype(np.uint64)
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
        """Reference DIMACS parser. Clauses are 0-terminated tokens, not lines, and the SATLIB
        '%' end marker is accepted. N = max(header N, highest variable used), so declared
        variables that appear in no clause keep their slots."""
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
        """One literal's full occ segment: binary payloads first, then long clause ids."""
        return self.occ_clauses[self.occ_offsets[lit_index]:self.occ_offsets[lit_index + 1]]

    def occ_bin(self, lit_index: int) -> np.ndarray:
        """The binary prefix as plain literal indices (tag removed); setting lit TRUE forces
        each of them."""
        seg = self.occ_clauses[self.occ_offsets[lit_index]:self.occ_mid[lit_index]]
        return (seg & ~np.int32(BIN_TAG)).astype(np.int32)

    def occ_long(self, lit_index: int) -> np.ndarray:
        """The long suffix: length>=3 clause ids (0..num_long-1) where ~lit appears."""
        return self.occ_clauses[self.occ_mid[lit_index]:self.occ_offsets[lit_index + 1]]

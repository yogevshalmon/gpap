"""Pure-numpy reference build of the engine's formula structures — the ORACLE for the core.

`cpp/src/formula.cpp` is THE builder the library ships (one implementation, shared by the Python
and C++ faces). This module is its independent re-derivation, kept where reference
implementations live in this repo (`reference_bcp.py` is the same idea for propagation):
`tests/engine/test_core.py` builds every structure both ways and compares entry for entry, and the
host-only suite uses it to cover DIMACS parsing on machines with no libgpap at all.

It is deliberately torch-free and device-free: the structures under test are plain arrays, and
an oracle should carry no machinery of its own.

What a formula compiles to (identical to the core, see docs/ENGINE.md). Clauses are
SEGREGATED BY LENGTH — only length>=3 ("long") clauses carry a packed word:

  * occ CSR (`occ_offsets`, `occ_mid`, `occ_clauses`): for each literal l, the entries where ~l
    appears — setting l TRUE falsifies ~l in exactly those clauses. Within a segment, BINARY
    entries come first (payload = the OTHER literal's index; the whole clause is an implication
    edge with no per-row state) up to `occ_mid[l]`, then LONG entries (payload = the length>=3
    clause's re-index, 0..num_long-1). Unit clauses have NO occ entry: the construction flush
    asserts them into the baseline bitmap. Each kind stays clause-ascending.
  * `clause_sizes`: literals per clause, ALL kept clauses in input order;
  * `init_lit_sum`: per-clause SUM of literal indices mod 2^32 (the packed word's initial sum
    field; the engine masks it down to its split's width, and mod-2^32-then-mod-2^F == mod-2^F);
  * `bin_pairs` (sorted pairs, input order) and `unit_lits`, the per-kind side lists.

Literal indexing is the 2N scheme: +v -> 2(v-1), -v -> 2(v-1)+1, negation = index ^ 1.
An EMPTY clause cannot be represented in the counters, so it is dropped from the structures and
recorded in `has_empty_clause` (the engine turns that flag into an all-rows conflict).
"""
import numpy as np
from itertools import chain

#: Sign bit the core sets on BINARY occ payloads (GPAP_OCC_BIN_TAG in cpp/include/gpap/gpap.h),
#: so a kernel can classify an entry from the value it already loaded.
BIN_TAG = -0x80000000


def flat_lit_arrays(clauses: list) -> tuple:
    """(lens, lit_idx) int64 numpy arrays for a clause list — ONE flatten and ONE encoding of the
    2N literal scheme, feeding the occ build, clause_sizes and init_lit_sum alike. Clause c owns
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
        # Segregated occ build over the NEGATED literal, in explicit loops: this is an ORACLE,
        # clarity beats speed. Iterating clauses in input order makes each kind's sub-range
        # clause-ascending — the layout the core's counting sort produces.
        num_lits = 2 * self.num_vars
        bins: list = [[] for _ in range(num_lits)]
        longs: list = [[] for _ in range(num_lits)]
        self.bin_pairs: list = []
        self.unit_lits: list = []
        long_id = 0
        for c in range(self.num_clauses):
            # SLICE the one vectorised encoding — never re-derive it, or the oracle can end up
            # self-inconsistent (occ built one way, init_lit_sum the other) and still "pass".
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

        # Per-clause SUM of literal indices, exact mod 2^32: int64 add, then REINTERPRET the low
        # 32 bits (a cast could saturate; a view is exact).
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
        """One literal's full occ segment: binary payloads first, then long clause ids."""
        return self.occ_clauses[self.occ_offsets[lit_index]:self.occ_offsets[lit_index + 1]]

    def occ_bin(self, lit_index: int) -> np.ndarray:
        """The binary prefix: OTHER-literal payloads, TAG MASKED OFF (setting lit TRUE forces
        each of them)."""
        seg = self.occ_clauses[self.occ_offsets[lit_index]:self.occ_mid[lit_index]]
        return (seg & ~np.int32(BIN_TAG)).astype(np.int32)

    def occ_long(self, lit_index: int) -> np.ndarray:
        """The long suffix: length>=3 clause ids (0..num_long-1) where ~lit appears."""
        return self.occ_clauses[self.occ_mid[lit_index]:self.occ_offsets[lit_index + 1]]

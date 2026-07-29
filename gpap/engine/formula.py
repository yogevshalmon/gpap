"""CNF formula representation for the GPAP engine: literal indexing, CSR occurrence lists,
and the per-clause counters (clause_sizes / init_lit_sum) the packed sum-trick BCP consumes.

Literal indexing (the 2N scheme):
  positive literal  v -> index 2*(v-1)
  negative literal -v -> index 2*(v-1) + 1
  negation: lit_index ^ 1

The remaining-literal trick is a MODULAR SUM: a clause tracks the sum of its non-falsified
literal indices mod 2^F (F = the packed word's high-field width); when exactly one literal
survives, the field holds that literal exactly (every removal is exact mod 2^F, and a single
literal index < 2^F). The sum form is what makes the warp-parallel kernel possible: ONE
atomic_add applies count-=1 and sum-=neg together (the count field never borrows — count >= 1
at every decrement — and the sum field wraps off the top of the word), and the atomic's
RETURN VALUE alone yields exact conflict/unit detection plus the forced literal.
"""
import torch
import numpy as np
from itertools import chain

# External (DIMACS) SAT type aliases
ext_lit = int                      # signed DIMACS literal
ext_cls = list[ext_lit]            # one clause
ext_assg = list[ext_lit]           # one assignment / assumption set
ext_assumptions = list[ext_assg]   # multiple assumption sets


def flat_lit_arrays(clauses: list) -> tuple:
    """(lens, lit_idx, cls_idx) int64 numpy arrays for a clause list — the shared vectorised
    path for CSR-occ / clause-size / lit-xor construction. The per-literal Python loops this
    replaces cost seconds per call on 1M-clause circuit unrolls (and construction runs several
    times per instance: sampler, miner prober, post-mining rebuild)."""
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


def ext_lit_to_lit_index(ext_lit: int) -> int:
    """Convert DIMACS literal to internal literal index.
    Positive literal v -> 2*(v-1)
    Negative literal -v -> 2*(v-1) + 1
    """
    var = abs(ext_lit)
    return 2 * (var - 1) + (0 if ext_lit > 0 else 1)


def lit_index_to_var(lit_index: int) -> int:
    """Literal index to 0-based variable index."""
    return lit_index // 2


def lit_index_to_ext_lit(lit_index: int) -> int:
    """Inverse of ext_lit_to_lit_index: internal index -> DIMACS literal.
    Even index -> +var, odd index -> -var (1-based variable).
    """
    var = lit_index // 2 + 1
    return var if lit_index % 2 == 0 else -var


def lit_index_negation(lit_index: int) -> int:
    """Negate a literal index: flip the last bit."""
    return lit_index ^ 1


class OccurrenceList:
    """GPU-resident occurrence lists for all 2*N literals in CSR format, int32 throughout
    (what the Triton kernels consume directly — no per-engine int64->int32 copies).

    For each literal l, occ[l] stores the clause indices where ~l (the negation) appears.
    This way, when literal l is set TRUE, occ[l] gives us the clauses where ~l is falsified.
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._occ_offsets = None   # (2*N + 1,) int32
        self._occ_clauses = None   # (total_occurrences,) int32
        self._num_vars = 0
        self._num_clauses = 0

    @staticmethod
    def from_clauses(num_vars: int, clauses: list[ext_cls], device: torch.device) -> 'OccurrenceList':
        """Build occurrence lists from a list of clauses.

        For each literal l in a clause, we add the clause index to occ[~l]
        (the negation's occurrence list). This way occ[l] = clauses where ~l appears.
        """
        _, lit_idx, cls_idx = flat_lit_arrays(clauses)
        return OccurrenceList.from_flat(num_vars, len(clauses), lit_idx ^ 1, cls_idx, device)

    @staticmethod
    def from_arrays(num_vars: int, num_clauses: int, offsets, clauses,
                    device: torch.device) -> 'OccurrenceList':
        """Wrap a CSR the core already built (offsets, clause indices) as device tensors."""
        occ = OccurrenceList(device)
        occ._num_vars = num_vars
        occ._num_clauses = num_clauses
        occ._occ_offsets = torch.from_numpy(np.ascontiguousarray(offsets, np.int32)).to(device)
        occ._occ_clauses = torch.from_numpy(np.ascontiguousarray(clauses, np.int32)).to(device)
        return occ

    @staticmethod
    def from_flat(num_vars: int, num_clauses: int, neg_lit_idx, cls_idx,
                  device: torch.device) -> 'OccurrenceList':
        """CSR build from pre-flattened arrays (see flat_lit_arrays). The stable argsort keeps
        each literal's clause indices in ascending clause order — identical layout to the old
        per-clause append loop."""
        occ = OccurrenceList(device)
        occ._num_vars = num_vars
        occ._num_clauses = num_clauses
        num_lits = 2 * num_vars
        if len(neg_lit_idx) == 0:
            occ._occ_offsets = torch.zeros(num_lits + 1, dtype=torch.int32, device=device)
            occ._occ_clauses = torch.empty(0, dtype=torch.int32, device=device)
            return occ
        order = np.argsort(neg_lit_idx, kind="stable")
        offsets = np.zeros(num_lits + 1, dtype=np.int64)
        np.cumsum(np.bincount(neg_lit_idx, minlength=num_lits), out=offsets[1:])
        occ._occ_offsets = torch.from_numpy(offsets.astype(np.int32)).to(device)
        occ._occ_clauses = torch.from_numpy(cls_idx[order].astype(np.int32)).to(device)
        return occ

    def get_clause_indices(self, lit_index: int) -> torch.Tensor:
        """Clause indices for one literal (clauses where ~lit appears). Debug/test helper —
        syncs twice; never call on a hot path."""
        start = int(self._occ_offsets[lit_index].item())
        end = int(self._occ_offsets[lit_index + 1].item())
        return self._occ_clauses[start:end]

    @property
    def num_lits(self) -> int:
        return 2 * self._num_vars

    @property
    def occ_offsets(self) -> torch.Tensor:
        return self._occ_offsets

    @property
    def occ_clauses(self) -> torch.Tensor:
        return self._occ_clauses


class CnfFormula:
    """CNF formula for the engine: occurrence lists and per-clause counters.

    The structures are BUILT BY THE C++ CORE (`cpp/src/formula.cpp`) whenever libgpap is
    available, so the engine's Python and C++ faces can never disagree about what a formula is.
    A pure-numpy builder is kept as the fallback for hosts without the library (the portable
    smoke suite runs there); `tests/test_core_equivalence.py` asserts the two agree exactly.

    No clause->literal structure is stored. Instead we maintain:
    - Occurrence lists (CSR, int32): for each literal l, clauses where ~l appears
    - clause_sizes: number of literals per clause
    - init_lit_sum: initial sum of literal indices per clause, mod 2^32 (the engine masks it
      to its packed field width; see the module docstring for why SUM, not XOR)

    An EMPTY clause makes the formula trivially UNSAT; it cannot be represented in the
    counters, so it is dropped from the structures and recorded in `has_empty_clause`
    (the engine turns that flag into an all-rows conflict).

    `clause_sizes` / `init_lit_sum` / `occ_list` are materialized as torch tensors ON DEMAND —
    the core keeps its own device copies, so a normal sampling run never pays for them. Prefer
    `max_clause_len` / `total_lits` (free, from the core) over reductions over `clause_sizes`.
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._occ_list: OccurrenceList = None
        self._clause_sizes: torch.Tensor = None   # (M,) int32, materialized on demand
        self._init_lit_sum: torch.Tensor = None   # (M,) int32 (sum mod 2^32, exact bit pattern)
        self._num_vars = 0
        self._num_clauses = 0
        self._has_empty_clause = False
        self._core = None                         # gpap.engine._core.Formula when available
        self._max_clause_len = 0
        self._total_lits = 0

    # ------------------------------------------------------------------ core handle

    @property
    def core(self):
        """The `_core.Formula` handle the engine consumes. Raises if libgpap is unavailable."""
        if self._core is None:
            from gpap.engine._core import GpapError
            raise GpapError(0, "this CnfFormula was built without libgpap",
                            "rebuild the core (see cpp/README.md) or use GPAP_ENGINE=torch")
        return self._core

    @property
    def has_core(self) -> bool:
        return self._core is not None

    def _adopt_core(self, cf) -> None:
        """Take the shape from a core-built formula (the structures stay on the C++ side)."""
        self._core = cf
        self._num_vars = int(cf.info.num_vars)
        self._num_clauses = int(cf.info.num_clauses)
        self._has_empty_clause = bool(cf.info.has_empty_clause)
        self._max_clause_len = int(cf.info.max_clause_len)
        self._total_lits = int(cf.info.total_lits)
        self._occ_list = None
        self._clause_sizes = None
        self._init_lit_sum = None

    # ------------------------------------------------------------------ shape

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def num_clauses(self) -> int:
        return self._num_clauses

    @property
    def max_clause_len(self) -> int:
        """Longest clause — bounds the packed word's count field (free; no reduction)."""
        if not self._max_clause_len and self._clause_sizes is not None and self._num_clauses:
            self._max_clause_len = int(self._clause_sizes.max().item())
        return self._max_clause_len

    @property
    def total_lits(self) -> int:
        """Total literal occurrences (free; no reduction)."""
        if not self._total_lits and self._clause_sizes is not None:
            self._total_lits = int(self._clause_sizes.sum().item())
        return self._total_lits

    @property
    def occ_list(self) -> OccurrenceList:
        if self._occ_list is None and self._core is not None:
            off, cls_ = self._core.occ()
            self._occ_list = OccurrenceList.from_arrays(self._num_vars, self._num_clauses,
                                                        off, cls_, self._device)
        return self._occ_list

    @property
    def clause_sizes(self) -> torch.Tensor:
        if self._clause_sizes is None and self._core is not None:
            self._materialize_clause_arrays()
        return self._clause_sizes

    @property
    def init_lit_sum(self) -> torch.Tensor:
        if self._init_lit_sum is None and self._core is not None:
            self._materialize_clause_arrays()
        return self._init_lit_sum

    def _materialize_clause_arrays(self) -> None:
        sizes, lsum = self._core.clause_arrays()
        self._clause_sizes = torch.from_numpy(sizes).to(self._device)
        self._init_lit_sum = torch.from_numpy(lsum).to(self._device)

    @property
    def has_empty_clause(self) -> bool:
        """True iff the input contained an empty clause (formula trivially UNSAT)."""
        return self._has_empty_clause

    def init_formula(self, num_vars: int, clauses: list[ext_cls]):
        """Build from a list of clauses (external DIMACS literals).

        Goes through the C++ core when libgpap is available — the same builder C++ consumers
        use — and falls back to the vectorised numpy path otherwise."""
        from gpap.engine import _core
        if _core.available():
            self._adopt_core(_core.Formula.from_clauses(num_vars, clauses))
            return
        self._init_formula_numpy(num_vars, clauses)

    def _init_formula_numpy(self, num_vars: int, clauses: list[ext_cls]):
        """Fallback builder for hosts without libgpap. Vectorised: one shared flatten
        (flat_lit_arrays) feeds the occ CSR, clause_sizes and init_lit_sum — the old
        per-literal Python loops cost seconds per build on 1M-clause circuit unrolls, and
        construction runs several times per instance (sampler, miner prober, post-mining
        rebuild). Kept honest by tests/test_core_equivalence.py."""
        self._core = None
        self._max_clause_len = 0
        self._total_lits = 0
        self._has_empty_clause = any(len(c) == 0 for c in clauses)
        if self._has_empty_clause:
            clauses = [c for c in clauses if c]
        self._num_vars = num_vars
        self._num_clauses = len(clauses)

        lens, lit_idx, cls_idx = flat_lit_arrays(clauses)
        self._occ_list = OccurrenceList.from_flat(num_vars, len(clauses), lit_idx ^ 1, cls_idx,
                                                  self._device)
        # per-clause SUM of literal indices, exact mod 2^32 (int64 add then reinterpret the low
        # 32 bits — an integer CAST could saturate, a view is exact; the engine masks further
        # down to its packed field width, and mod-2^32-then-mod-2^F == mod-2^F)
        M = len(clauses)
        if M:
            bounds = np.zeros(M, dtype=np.int64)
            np.cumsum(lens[:-1], out=bounds[1:])
            sums = (np.add.reduceat(lit_idx, bounds).astype(np.uint64)
                    & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
        else:
            sums = np.zeros(0, dtype=np.int32)
        self._clause_sizes = torch.from_numpy(lens.astype(np.int32)).to(self._device)
        self._init_lit_sum = torch.from_numpy(sums).to(self._device)

    def parse_cnf_file(self, cnf_file: str):
        """Parse a DIMACS CNF file and build the formula (core parser when available).

        Token-stream parsing (clauses are 0-terminated, NOT line-based): multiple clauses on
        one line and clauses spanning lines are handled per the DIMACS standard, as is the
        SATLIB '%' end marker. N = max(header N, highest variable used) so trailing declared
        variables (e.g. support vars appearing only in `c ind` lines) keep their slots."""
        from gpap.engine import _core
        if _core.available():
            self._adopt_core(_core.Formula.from_file(cnf_file))
            return
        self._parse_cnf_file_numpy(cnf_file)

    def _parse_cnf_file_numpy(self, cnf_file: str):
        """Fallback DIMACS parser (see _init_formula_numpy)."""
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
        try:
            arr = np.array(" ".join(body).split(), dtype=np.int64) if body \
                else np.empty(0, dtype=np.int64)
        except ValueError as e:
            raise ValueError(f"non-integer token in DIMACS clause body of {cnf_file}: {e}")
        ends = np.nonzero(arr == 0)[0]
        starts = np.concatenate([[0], ends[:-1] + 1]) if len(ends) else np.empty(0, np.int64)
        clauses = [arr[a:b].tolist() for a, b in zip(starts, ends)]
        if len(arr) and (len(ends) == 0 or ends[-1] != len(arr) - 1):
            tail = arr[(ends[-1] + 1) if len(ends) else 0:]        # unterminated final clause
            clauses.append(tail.tolist())
        max_var = int(np.abs(arr).max()) if len(arr) else 0
        self._init_formula_numpy(max(header_n, max_var), clauses)

    def print_stats(self):
        print(f"Variables: {self._num_vars}")
        print(f"Clauses: {self._num_clauses}")
        print(f"Total literal occurrences: {self._occ_list.occ_clauses.numel()}")
        print(f"Clause sizes: min={self._clause_sizes.min().item()}, max={self._clause_sizes.max().item()}, avg={self._clause_sizes.float().mean().item():.2f}")

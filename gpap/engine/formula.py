"""CNF formula representation for the GPAP engine: literal indexing, CSR occurrence lists,
and the per-clause counters (clause_sizes / init_lit_sum) the sum-trick BCP engine consumes.

Literal indexing (the 2N scheme):
  positive literal  v -> index 2*(v-1)
  negative literal -v -> index 2*(v-1) + 1
  negation: lit_index ^ 1
"""
import os

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
    path for CSR-occ / clause-size / lit-sum construction. The per-literal Python loops this
    replaces cost seconds per call on 1M-clause circuit unrolls (and construction runs several
    times per instance: sampler, miner prober, post-mining rebuild)."""
    M = len(clauses)
    lens = np.fromiter(map(len, clauses), dtype=np.int64, count=M)
    if M and lens.sum():
        flat = np.array(list(chain.from_iterable(clauses)), dtype=np.int64)
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
    """GPU-resident occurrence lists for all 2*N literals in CSR format.

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
            occ._occ_offsets = torch.zeros(num_lits + 1, dtype=torch.int64, device=device)
            occ._occ_clauses = torch.empty(0, dtype=torch.int64, device=device)
            return occ
        order = np.argsort(neg_lit_idx, kind="stable")
        offsets = np.zeros(num_lits + 1, dtype=np.int64)
        np.cumsum(np.bincount(neg_lit_idx, minlength=num_lits), out=offsets[1:])
        occ._occ_offsets = torch.from_numpy(offsets).to(device)
        occ._occ_clauses = torch.from_numpy(cls_idx[order]).to(device)
        return occ

    def get_clause_indices(self, lit_index: int) -> torch.Tensor:
        """Return clause indices for a single literal (clauses where ~lit appears)."""
        start = self._occ_offsets[lit_index].item()
        end = self._occ_offsets[lit_index + 1].item()
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
    """CNF formula representation with occurrence lists and per-clause lit_sum.

    No clause->literal structure is stored. Instead we maintain:
    - Occurrence lists (CSR): for each literal l, clauses where ~l appears
    - clause_sizes: number of literals per clause
    - init_lit_sum: initial sum of literal indices per clause (for the sum trick)
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._occ_list: OccurrenceList = None
        self._clause_sizes: torch.Tensor = None   # (M,) int32
        self._init_lit_sum: torch.Tensor = None    # (M,) int32
        self._num_vars = 0
        self._num_clauses = 0

    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def num_clauses(self) -> int:
        return self._num_clauses

    @property
    def occ_list(self) -> OccurrenceList:
        return self._occ_list

    @property
    def clause_sizes(self) -> torch.Tensor:
        return self._clause_sizes

    @property
    def init_lit_sum(self) -> torch.Tensor:
        return self._init_lit_sum

    def init_formula(self, num_vars: int, clauses: list[ext_cls]):
        """Initialize from a list of clauses (external literal format). Vectorised: one shared
        flatten (flat_lit_arrays) feeds the occ CSR, clause_sizes and init_lit_sum — the old
        per-literal Python loops cost seconds per build on 1M-clause circuit unrolls, and
        construction runs several times per instance (sampler, miner prober, post-mining
        rebuild)."""
        self._num_vars = num_vars
        self._num_clauses = len(clauses)

        lens, lit_idx, cls_idx = flat_lit_arrays(clauses)
        self._occ_list = OccurrenceList.from_flat(num_vars, len(clauses), lit_idx ^ 1, cls_idx,
                                                  self._device)
        # per-clause sum of literal indices (bincount weights are float64 — exact below 2^53)
        if len(clauses):
            lit_sums = np.bincount(cls_idx, weights=lit_idx,
                                   minlength=len(clauses)).astype(np.int32)
        else:
            lit_sums = np.zeros(0, dtype=np.int32)
        self._clause_sizes = torch.from_numpy(lens.astype(np.int32)).to(self._device)
        self._init_lit_sum = torch.from_numpy(lit_sums).to(self._device)

    def parse_cnf_file(self, cnf_file: str):
        """Parse a DIMACS CNF file and build the formula."""
        from tqdm import tqdm

        max_var_num = 0
        clauses = []

        with open(cnf_file) as file:
            lines = [line.rstrip() for line in file]

        # quiet the bar for piped/background runs (GPAP_QUIET=1 or non-tty stdout); show interactively
        _quiet = bool(os.environ.get("GPAP_QUIET")) or not os.isatty(1)
        for line in tqdm(lines, desc="Parsing CNF", disable=_quiet):
            if len(line) == 0 or line.isspace():
                continue
            if line.startswith('c'):
                continue
            if line.startswith('p'):
                continue
            clause = []
            try:
                lits = line.split()
                for lit_str in lits:
                    lit_int = int(lit_str)
                    max_var_num = max(abs(lit_int), max_var_num)
                    if lit_int != 0:
                        clause.append(lit_int)
            except Exception:
                raise Exception("Error parsing line", line)
            if clause:
                clauses.append(clause)

        self.init_formula(max_var_num, clauses)

    def print_stats(self):
        print(f"Variables: {self._num_vars}")
        print(f"Clauses: {self._num_clauses}")
        print(f"Total literal occurrences: {self._occ_list.occ_clauses.numel()}")
        print(f"Clause sizes: min={self._clause_sizes.min().item()}, max={self._clause_sizes.max().item()}, avg={self._clause_sizes.float().mean().item():.2f}")

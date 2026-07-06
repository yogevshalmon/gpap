import torch
import numpy as np
from itertools import chain

from gpap.utils.sat_def import ext_cls


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


if __name__ == "__main__":
    device = torch.device("cpu")
    # clauses: (x1 OR x2), (x1 OR ~x2 OR x3), (~x1 OR ~x2), (~x3)
    clauses = [[-1, 2], [1, -2, 3], [-1, -2], [-3]]
    occ = OccurrenceList.from_clauses(3, clauses, device)

    print("Occurrence list (occ[l] = clauses containing ~l):")
    for l in range(occ.num_lits):
        var = lit_index_to_var(l) + 1
        pol = "pos" if l % 2 == 0 else "neg"
        cls_ids = occ.get_clause_indices(l).tolist()
        print(f"  lit {l} (var {var} {pol}): clauses {cls_ids}")

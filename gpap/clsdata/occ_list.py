import torch
from collections import defaultdict

from gpap.utils.sat_def import ext_cls


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
        occ = OccurrenceList(device)
        occ._num_vars = num_vars
        occ._num_clauses = len(clauses)

        num_lits = 2 * num_vars
        # collect clause indices per literal (negated)
        lit_to_clauses = defaultdict(list)

        for cls_idx, clause in enumerate(clauses):
            for ext_lit in clause:
                lit_idx = ext_lit_to_lit_index(ext_lit)
                # store in the negation's list: occ[~lit] gets this clause
                neg_lit_idx = lit_index_negation(lit_idx)
                lit_to_clauses[neg_lit_idx].append(cls_idx)

        # build CSR arrays
        offsets = [0]
        all_clauses = []
        for l in range(num_lits):
            cls_list = lit_to_clauses.get(l, [])
            all_clauses.extend(cls_list)
            offsets.append(len(all_clauses))

        occ._occ_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        occ._occ_clauses = torch.tensor(all_clauses, dtype=torch.int64, device=device) if all_clauses else torch.empty(0, dtype=torch.int64, device=device)

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

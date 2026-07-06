import os

import torch
from tqdm import tqdm

import numpy as np

from gpap.utils.sat_def import ext_lit, ext_cls
from gpap.clsdata.occ_list import OccurrenceList, ext_lit_to_lit_index, flat_lit_arrays


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
            except:
                raise Exception("Error parsing line", line)
            if clause:
                clauses.append(clause)

        self.init_formula(max_var_num, clauses)

    def print_stats(self):
        print(f"Variables: {self._num_vars}")
        print(f"Clauses: {self._num_clauses}")
        print(f"Total literal occurrences: {self._occ_list.occ_clauses.numel()}")
        print(f"Clause sizes: min={self._clause_sizes.min().item()}, max={self._clause_sizes.max().item()}, avg={self._clause_sizes.float().mean().item():.2f}")


if __name__ == "__main__":
    device = torch.device("cpu")
    formula = CnfFormula(device)

    # test with inline clauses: (~x1 OR x2), (x1 OR ~x2 OR x3), (~x1 OR ~x2), (~x3)
    clauses = [[-1, 2], [1, -2, 3], [-1, -2], [-3]]
    formula.init_formula(3, clauses)
    formula.print_stats()

    print("clause_sizes:", formula.clause_sizes)
    print("init_lit_sum:", formula.init_lit_sum)
    # clause 0: lits ~x1(idx=1) + x2(idx=2) = 3
    # clause 1: lits x1(idx=0) + ~x2(idx=3) + x3(idx=4) = 7
    # clause 2: lits ~x1(idx=1) + ~x2(idx=3) = 4
    # clause 3: lits ~x3(idx=5) = 5

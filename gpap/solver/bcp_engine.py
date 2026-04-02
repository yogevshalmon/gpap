import torch

from gpap.clsdata.cnf_formula import CnfFormula
from gpap.clsdata.occ_list import lit_index_negation


class BCPEngine:
    """Boolean Constraint Propagation engine using occurrence lists and the sum trick.

    For K parallel assumption sets and M clauses, maintains:
    - sat_count (K, M): remaining non-falsified literals per clause (init = clause_sizes)
    - lit_sum (K, M): running sum of non-falsified literal indices (init = init_lit_sum)
    - assignment (K, 2*N): 0 = unassigned, 1 = TRUE

    When literal l is set TRUE:
    - ~l is falsified in all clauses in occ[l]
    - sat_count[:, occ[l]] -= 1
    - lit_sum[:, occ[l]] -= negated_lit_index (i.e. l ^ 1)

    Conflict: sat_count == 0
    Unit: sat_count == 1, remaining literal = lit_sum value
    """

    def __init__(self, formula: CnfFormula):
        self._formula = formula
        self._device = formula._device
        self._num_vars = formula.num_vars
        self._num_clauses = formula.num_clauses

    def init_state(self, num_batches: int):
        """Initialize BCP state for K parallel assumption sets."""
        K = num_batches
        M = self._num_clauses
        N = self._num_vars

        # clause counters: (K, M)
        self._sat_count = self._formula.clause_sizes.unsqueeze(0).expand(K, -1).clone().to(torch.int32)
        self._lit_sum = self._formula.init_lit_sum.unsqueeze(0).expand(K, -1).clone().to(torch.int32)

        # assignment: (K, 2*N), 0=unassigned, 1=TRUE
        self._assignment = torch.zeros((K, 2 * N), dtype=torch.int8, device=self._device)

        # track which batches are still active (no conflict)
        self._active = torch.ones(K, dtype=torch.bool, device=self._device)

        # track conflict result per batch
        self._conflict = torch.zeros(K, dtype=torch.bool, device=self._device)

        # track number of propagated literals per batch
        self._num_propagated = torch.zeros(K, dtype=torch.int32, device=self._device)

    def _apply_literal(self, lit_index: int, batch_mask: torch.Tensor):
        """Set literal lit_index TRUE for batches where batch_mask is True.
        Update sat_count and lit_sum for affected clauses.
        """
        occ = self._formula.occ_list
        # clauses where ~lit_index appears (these are the clauses affected)
        affected_clauses = occ.get_clause_indices(lit_index)

        if affected_clauses.numel() == 0:
            return

        neg_lit = lit_index_negation(lit_index)

        # update for selected batches
        # sat_count[batch_mask, affected_clauses] -= 1
        # lit_sum[batch_mask, affected_clauses] -= neg_lit
        batch_indices = torch.where(batch_mask)[0]
        if batch_indices.numel() == 0:
            return

        # use advanced indexing: create grid of (batch, clause) pairs
        # batch_indices: (B,), affected_clauses: (C,)
        self._sat_count[batch_indices.unsqueeze(1), affected_clauses.unsqueeze(0)] -= 1
        self._lit_sum[batch_indices.unsqueeze(1), affected_clauses.unsqueeze(0)] -= neg_lit

        # mark literal as TRUE in assignment
        self._assignment[batch_indices, lit_index] = 1

    def apply_assumptions(self, assumptions_lit_indices: list[list[int]]):
        """Apply assumption sets. assumptions_lit_indices[k] is a list of literal indices for batch k.

        Each literal is set TRUE. Its negation is implicitly FALSE.
        """
        K = self._sat_count.size(0)

        for k in range(K):
            if not self._active[k]:
                continue
            batch_mask = torch.zeros(K, dtype=torch.bool, device=self._device)
            batch_mask[k] = True

            for lit_idx in assumptions_lit_indices[k]:
                # check for contradiction: negation already assigned
                neg_lit = lit_index_negation(lit_idx)
                if self._assignment[k, neg_lit] == 1:
                    self._conflict[k] = True
                    self._active[k] = False
                    break
                # skip if already assigned
                if self._assignment[k, lit_idx] == 1:
                    continue
                self._apply_literal(lit_idx, batch_mask)

    def apply_assumptions_bulk(self, assumptions_lit_indices: list[list[int]]):
        """Apply assumptions in bulk using scatter operations for better GPU performance.

        All assumptions for all batches are applied at once.
        """
        K = self._sat_count.size(0)
        occ = self._formula.occ_list

        for k in range(K):
            if not self._active[k]:
                continue
            for lit_idx in assumptions_lit_indices[k]:
                neg_lit = lit_index_negation(lit_idx)
                if self._assignment[k, neg_lit] == 1:
                    self._conflict[k] = True
                    self._active[k] = False
                    break
                if self._assignment[k, lit_idx] == 1:
                    continue

                # get affected clauses
                affected_clauses = occ.get_clause_indices(lit_idx)
                if affected_clauses.numel() > 0:
                    self._sat_count[k, affected_clauses] -= 1
                    self._lit_sum[k, affected_clauses] -= neg_lit

                self._assignment[k, lit_idx] = 1

    def propagate(self) -> torch.Tensor:
        """Run BCP to fixpoint for all active batches.

        Returns:
            conflict: (K,) boolean tensor, True if batch has a conflict
        """
        K = self._sat_count.size(0)
        occ = self._formula.occ_list

        while True:
            # detect conflicts: sat_count == 0 in any active batch
            conflict_mask = (self._sat_count == 0) & self._active.unsqueeze(1)
            batch_has_conflict = conflict_mask.any(dim=1)
            new_conflicts = batch_has_conflict & self._active
            self._conflict |= new_conflicts
            self._active &= ~batch_has_conflict

            # detect unit clauses: sat_count == 1 in active batches
            unit_mask = (self._sat_count == 1) & self._active.unsqueeze(1)

            if not unit_mask.any():
                break

            # get (batch, clause) pairs for unit clauses
            unit_pairs = torch.nonzero(unit_mask)  # (num_unit, 2)

            # the forced literal index for each unit clause is lit_sum[k, c]
            forced_lits = self._lit_sum[unit_pairs[:, 0], unit_pairs[:, 1]]  # (num_unit,)

            # filter: skip if forced literal already TRUE (clause was satisfied)
            # and detect contradictions (negation already TRUE)
            any_propagated = False

            for i in range(unit_pairs.size(0)):
                k = unit_pairs[i, 0].item()
                lit_idx = forced_lits[i].item()

                if not self._active[k]:
                    continue

                # already assigned TRUE -> clause was satisfied, skip
                if self._assignment[k, lit_idx] == 1:
                    continue

                neg_lit = lit_index_negation(lit_idx)
                # negation already TRUE -> contradiction
                if self._assignment[k, neg_lit] == 1:
                    self._conflict[k] = True
                    self._active[k] = False
                    continue

                # propagate: set this literal TRUE
                affected_clauses = occ.get_clause_indices(lit_idx)
                if affected_clauses.numel() > 0:
                    self._sat_count[k, affected_clauses] -= 1
                    self._lit_sum[k, affected_clauses] -= neg_lit

                self._assignment[k, lit_idx] = 1
                self._num_propagated[k] += 1
                any_propagated = True

            if not any_propagated:
                break

        return self._conflict

    @property
    def conflict(self) -> torch.Tensor:
        return self._conflict

    @property
    def assignment(self) -> torch.Tensor:
        return self._assignment

    @property
    def active(self) -> torch.Tensor:
        return self._active

    @property
    def num_propagated(self) -> torch.Tensor:
        return self._num_propagated

    @property
    def sat_count(self) -> torch.Tensor:
        return self._sat_count

    @property
    def lit_sum(self) -> torch.Tensor:
        return self._lit_sum


if __name__ == "__main__":
    from gpap.clsdata.occ_list import ext_lit_to_lit_index

    device = torch.device("cpu")
    formula = CnfFormula(device)
    # (x1 OR x2), (~x1 OR x3), (~x2 OR ~x3), (x1 OR ~x2)
    clauses = [[1, 2], [-1, 3], [-2, -3], [1, -2]]
    formula.init_formula(3, clauses)

    engine = BCPEngine(formula)
    engine.init_state(num_batches=2)

    # batch 0: assume x1=TRUE (lit_idx=0), batch 1: assume x1=FALSE (lit_idx=1)
    assumptions = [
        [ext_lit_to_lit_index(1)],   # x1 TRUE
        [ext_lit_to_lit_index(-1)],  # x1 FALSE (~x1 TRUE)
    ]
    engine.apply_assumptions_bulk(assumptions)
    conflict = engine.propagate()

    print("Conflicts:", conflict)
    print("Assignment batch 0:", engine.assignment[0])
    print("Assignment batch 1:", engine.assignment[1])
    print("Propagated:", engine.num_propagated)

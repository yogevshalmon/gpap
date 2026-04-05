import torch

from gpap.clsdata.cnf_formula import CnfFormula
from gpap.clsdata.occ_list import lit_index_negation


def make_assumption_tensor(
    assumptions_lit_indices: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """Convert list[list[int]] to a (K, L) int64 padded tensor for BCPEngine.apply_assumptions_bulk.

    L = max assumption count across all batches; empty slots filled with -1.
    Build this once before any timed / hot path.
    """
    K = len(assumptions_lit_indices)
    L = max((len(lits) for lits in assumptions_lit_indices), default=0)
    t = torch.full((K, L), -1, dtype=torch.long)
    for k, lits in enumerate(assumptions_lit_indices):
        if lits:
            t[k, :len(lits)] = torch.tensor(lits, dtype=torch.long)
    return t.to(device)


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
        self._bcp_rounds = 0

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

        # pre-allocated workspace for propagation round (avoids per-iteration allocation)
        self._tmp_prop = torch.zeros(K, 2 * N, dtype=torch.bool, device=self._device)

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

    def _bulk_propagate_lits(self, batch_ids: torch.Tensor, lit_ids: torch.Tensor) -> None:
        """Bulk-propagate literals: update sat_count, lit_sum, and assignment for all (batch, lit) pairs.

        batch_ids: (P,) int64 — unique (batch, lit) pairs, all active, none already assigned
        lit_ids:   (P,) int64
        """
        occ = self._formula.occ_list
        M = self._num_clauses
        P = batch_ids.size(0)
        if P == 0:
            return

        # CSR lookup for all P literals at once — .long() required (occ_offsets is int32)
        starts  = occ.occ_offsets[lit_ids].long()      # (P,)
        ends    = occ.occ_offsets[lit_ids + 1].long()  # (P,)
        lengths = ends - starts                         # (P,) may include zeros

        total = lengths.sum().item()
        if total > 0:
            # Expand batch_ids to one entry per occ hit
            exp_batches = batch_ids.repeat_interleave(lengths)  # (total,)

            # Segment-offset trick: local_off[j] = position of j within its group
            cum = torch.zeros(P, dtype=torch.long, device=self._device)
            if P > 1:
                cum[1:] = lengths[:-1].cumsum(0)
            flat_range  = torch.arange(total, dtype=torch.long, device=self._device)
            local_off   = flat_range - cum.repeat_interleave(lengths)
            csi         = starts.repeat_interleave(lengths) + local_off
            exp_clauses = occ.occ_clauses[csi].long()   # (total,) clause indices

            # Flat 1D index into (K, M)
            flat_idx = exp_batches * M + exp_clauses    # (total,) int64

            # Scatter-subtract 1 from sat_count (int32)
            neg_ones = torch.full((total,), -1, dtype=torch.int32, device=self._device)
            self._sat_count.view(-1).scatter_add_(0, flat_idx, neg_ones)

            # Scatter-subtract neg_lit from lit_sum (int32)
            neg_lits     = (lit_ids ^ 1).to(torch.int32)               # (P,) XOR flips bit 0
            exp_neg_lits = (-neg_lits).repeat_interleave(lengths)       # (total,) int32
            self._lit_sum.view(-1).scatter_add_(0, flat_idx, exp_neg_lits)

        # Mark literals as TRUE
        self._assignment[batch_ids, lit_ids] = 1

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

    def apply_assumptions_bulk(self, assumptions: torch.Tensor):
        """Apply assumptions in bulk using scatter operations for better GPU performance.

        Args:
            assumptions: (K, L) int64 tensor. Each row is one batch's literals;
                         -1 = padding. Build with make_assumption_tensor() outside
                         any timed / hot loop.
        """
        K = self._sat_count.size(0)
        N = self._num_vars

        assumptions = assumptions.to(self._device)
        if assumptions.numel() == 0:
            return

        # Vectorised build — no Python loops, one scatter
        valid     = assumptions >= 0                                               # (K, L)
        if not valid.any():
            return
        batch_ids = torch.arange(K, device=self._device).unsqueeze(1) \
                        .expand_as(assumptions)[valid]                             # (P,)
        lit_ids   = assumptions[valid]                                             # (P,)

        # Build (K, 2N) assumption matrix — scatter is idempotent for duplicate (batch, lit) pairs
        assum_mat = torch.zeros(K, 2 * N, dtype=torch.bool, device=self._device)
        assum_mat[batch_ids, lit_ids] = True

        # Intra-batch contradiction: same batch assumes both l and ~l for some variable
        # Even cols = positive polarity, odd cols = negative polarity
        intra_conflict = (assum_mat[:, 0::2] & assum_mat[:, 1::2]).any(dim=1) & self._active

        # Contradiction against existing assignment: assume l but ~l already TRUE
        assum_neg = torch.zeros_like(assum_mat)
        assum_neg[:, 0::2] = assum_mat[:, 1::2]
        assum_neg[:, 1::2] = assum_mat[:, 0::2]
        existing_conflict = (assum_neg & self._assignment.bool()).any(dim=1) & self._active

        new_conflicts = intra_conflict | existing_conflict
        self._conflict |= new_conflicts
        self._active   &= ~new_conflicts

        # Get unique (batch, lit) pairs for still-active batches (assum_mat.nonzero() deduplicates)
        pairs = assum_mat.nonzero()           # (total_unique, 2) int64
        valid = self._active[pairs[:, 0]]
        pairs = pairs[valid]

        if pairs.size(0) > 0:
            self._bulk_propagate_lits(pairs[:, 0], pairs[:, 1])

    def propagate(self) -> torch.Tensor:
        """Run BCP to fixpoint for all active batches.

        Returns:
            conflict: (K,) boolean tensor, True if batch has a conflict
        """
        K = self._sat_count.size(0)
        N = self._num_vars

        while True:
            self._bcp_rounds += 1
            # (A) Conflict detection — vectorized
            conflict_mask = (self._sat_count == 0) & self._active.unsqueeze(1)
            batch_has_conflict = conflict_mask.any(dim=1)
            self._conflict |= batch_has_conflict & self._active
            self._active &= ~batch_has_conflict

            # (B) Unit detection — vectorized
            unit_mask = (self._sat_count == 1) & self._active.unsqueeze(1)
            if not unit_mask.any():
                break

            unit_pairs  = torch.nonzero(unit_mask)                            # (U, 2)
            batch_ids   = unit_pairs[:, 0]                                    # (U,) int64
            forced_lits = self._lit_sum[batch_ids, unit_pairs[:, 1]].long()  # (U,) int64

            # (C) Vectorized filter
            neg_forced       = forced_lits ^ 1
            is_active        = self._active[batch_ids]
            is_new           = self._assignment[batch_ids, forced_lits] == 0
            is_contradiction = self._assignment[batch_ids, neg_forced]  == 1

            # (D) Mark batches where negation already in assignment as conflicted
            contra_mask = is_active & is_contradiction
            if contra_mask.any():
                self._conflict[batch_ids[contra_mask]] = True
                self._active[batch_ids[contra_mask]]   = False

            # (E) Candidates: active, not yet assigned, no existing contradiction
            to_prop = is_active & is_new & ~is_contradiction
            if not to_prop.any():
                break
            prop_batches = batch_ids[to_prop]
            prop_lits    = forced_lits[to_prop]

            # (F+G) Build prop bitmap — idempotent, deduplicates duplicates from step E.
            # Then detect intra-round contradictions (both polarities forced in one batch).
            self._tmp_prop.fill_(False)
            self._tmp_prop[prop_batches, prop_lits] = True
            intra_conflict = (self._tmp_prop[:, 0::2] & self._tmp_prop[:, 1::2]).any(dim=1) & self._active
            if intra_conflict.any():
                conflict_idxs = torch.where(intra_conflict)[0]
                self._conflict[conflict_idxs] = True
                self._active[conflict_idxs]   = False
                self._tmp_prop[conflict_idxs] = False  # clear conflicted rows

            # Extract deduplicated (batch, lit) pairs from bitmap
            prop_pairs = self._tmp_prop.nonzero()  # (P', 2), inherently unique
            if prop_pairs.numel() == 0:
                break
            prop_batches = prop_pairs[:, 0]
            prop_lits    = prop_pairs[:, 1]

            # (H) Bulk update sat_count, lit_sum, assignment
            self._bulk_propagate_lits(prop_batches, prop_lits)

            # (I) Per-batch propagation count
            ones = torch.ones(prop_batches.size(0), dtype=torch.int32, device=self._device)
            self._num_propagated.scatter_add_(0, prop_batches, ones)

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

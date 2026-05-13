import time

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

    Propagation uses a frontier of (batch, clause) pairs whose sat_count was just
    decremented, avoiding a full (K, M) scan each round.
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

        # track which batches hit the round limit before reaching fixpoint/conflict
        self._timed_out = torch.zeros(K, dtype=torch.bool, device=self._device)

        # Frontier: (batch, clause) pairs whose sat_count was just decremented.
        # propagate() reads frontier_batches/clauses and replaces them each round.
        # Seeded here with clauses already at unit/conflict in the original formula.
        init_unit_cls = (self._formula.clause_sizes <= 1).nonzero(as_tuple=False).view(-1)
        if init_unit_cls.numel() > 0:
            self._frontier_batches = (
                torch.arange(K, device=self._device)
                .unsqueeze(1).expand(-1, init_unit_cls.numel()).reshape(-1)
            )
            self._frontier_clauses = init_unit_cls.unsqueeze(0).expand(K, -1).reshape(-1)
        else:
            self._frontier_batches = torch.empty(0, dtype=torch.long, device=self._device)
            self._frontier_clauses = torch.empty(0, dtype=torch.long, device=self._device)

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

    def _bulk_propagate_lits(
        self, batch_ids: torch.Tensor, lit_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bulk-propagate literals: update sat_count, lit_sum, and assignment.

        Returns (frontier_batches, frontier_clauses): the (batch, clause) pairs whose
        sat_count was just decremented. Pass these as the next round's frontier so
        propagate() only checks cells that actually changed.

        batch_ids: (P,) int64 — unique (batch, lit) pairs, all active, none already assigned
        lit_ids:   (P,) int64
        """
        occ = self._formula.occ_list
        M = self._num_clauses
        P = batch_ids.size(0)
        _empty = torch.empty(0, dtype=torch.long, device=self._device)
        if P == 0:
            return _empty, _empty

        # CSR lookup for all P literals at once
        starts  = occ.occ_offsets[lit_ids]          # (P,)
        ends    = occ.occ_offsets[lit_ids + 1]       # (P,)
        lengths = ends - starts                       # (P,) — may include zeros
        total   = lengths.sum().item()               # CPU sync: necessary scalar transfer

        if total > 0:
            # Expand batch_ids to one entry per occ hit
            exp_batches = batch_ids.repeat_interleave(lengths)  # (total,)

            # Segment-offset trick: local_off[j] = position of j within its segment
            cum = torch.zeros(P, dtype=torch.long, device=self._device)
            if P > 1:
                cum[1:] = lengths[:-1].cumsum(0)
            flat_range  = torch.arange(total, dtype=torch.long, device=self._device)
            local_off   = flat_range - cum.repeat_interleave(lengths)
            csi         = starts.repeat_interleave(lengths) + local_off
            exp_clauses = occ.occ_clauses[csi]           # (total,) clause indices

            # Flat 1D index into (K, M) for scatter operations
            flat_idx = exp_batches * M + exp_clauses     # (total,)

            # Decrement sat_count for all affected (batch, clause) pairs
            neg_ones = torch.full((total,), -1, dtype=torch.int32, device=self._device)
            self._sat_count.view(-1).scatter_add_(0, flat_idx, neg_ones)

            # Subtract neg_lit from lit_sum for all affected pairs
            neg_lits     = (lit_ids ^ 1).to(torch.int32)           # (P,) XOR flips bit 0
            exp_neg_lits = (-neg_lits).repeat_interleave(lengths)   # (total,)
            self._lit_sum.view(-1).scatter_add_(0, flat_idx, exp_neg_lits)

            self._assignment[batch_ids, lit_ids] = 1
            return exp_batches, exp_clauses

        self._assignment[batch_ids, lit_ids] = 1
        return _empty, _empty

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

    @torch.inference_mode()
    def apply_assumptions_bulk(self, assumptions: torch.Tensor):
        """Apply assumptions in bulk using scatter operations for better GPU performance.

        Populates the frontier with affected (batch, clause) pairs so propagate()
        only checks cells that were modified.

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
        intra_conflict = (assum_mat[:, 0::2] & assum_mat[:, 1::2]).any(dim=1) & self._active

        # Contradiction against existing assignment: assume l but ~l already TRUE
        assum_neg = torch.zeros_like(assum_mat)
        assum_neg[:, 0::2] = assum_mat[:, 1::2]
        assum_neg[:, 1::2] = assum_mat[:, 0::2]
        existing_conflict = (assum_neg & self._assignment.bool()).any(dim=1) & self._active

        new_conflicts = intra_conflict | existing_conflict
        self._conflict |= new_conflicts
        self._active   &= ~new_conflicts

        # Get unique (batch, lit) pairs for still-active batches
        pairs = assum_mat.nonzero()           # (total_unique, 2) int64
        valid = self._active[pairs[:, 0]]
        pairs = pairs[valid]

        if pairs.size(0) > 0:
            new_b, new_c = self._bulk_propagate_lits(pairs[:, 0], pairs[:, 1])
            if new_b.numel() > 0:
                self._frontier_batches = torch.cat([self._frontier_batches, new_b])
                self._frontier_clauses = torch.cat([self._frontier_clauses, new_c])

    @torch.inference_mode()
    def propagate(
        self,
        max_rounds: int | None = None,
        soft_stop_frac: float | None = None,
        min_rounds: int = 0,
    ) -> torch.Tensor:
        """Run BCP to fixpoint (or stop early) for all active batches.

        Uses a frontier of (batch, clause) pairs updated each round — only checks
        cells whose sat_count changed, avoiding a full (K, M) scan.

        Args:
            max_rounds: hard cap; stop after this many rounds even if not at fixpoint.
                        Batches still in the frontier are marked in self._timed_out.
                        None (default) means no hard cap.
            soft_stop_frac: after round_count >= min_rounds, stop when a round's
                            useful work (unique propagations) drops below
                            soft_stop_frac × peak observed work. None (default)
                            disables the heuristic. Batches still in the frontier
                            when triggered are marked in self._timed_out.
            min_rounds: don't allow soft_stop_frac to trigger before this round.

        Returns:
            conflict: (K,) boolean tensor, True if batch has a conflict
        """
        N = self._num_vars

        frontier_b = self._frontier_batches
        frontier_c = self._frontier_clauses
        round_count = 0
        peak_work   = 0

        while frontier_b.numel() > 0:
            self._bcp_rounds += 1
            round_count += 1

            if max_rounds is not None and round_count > max_rounds:
                # Mark batches still active in this frontier as timed out
                in_flight = torch.unique(frontier_b)
                still_active = in_flight[self._active[in_flight]]
                if still_active.numel() > 0:
                    self._timed_out[still_active] = True
                break

            # --- Filter frontier to active batches ---
            active_mask = self._active[frontier_b]
            if not active_mask.any():
                break
            fb = frontier_b[active_mask]
            fc = frontier_c[active_mask]

            # --- Gather sat_count at frontier positions ---
            sc_vals = self._sat_count[fb, fc]

            # --- Conflicts: sat_count == 0 ---
            is_conflict = sc_vals == 0
            if is_conflict.any():
                self._conflict[fb[is_conflict]] = True
                self._active[fb[is_conflict]]   = False

            # --- Units: sat_count == 1 for still-active batches ---
            is_unit = (sc_vals == 1) & self._active[fb]
            if not is_unit.any():
                break

            unit_b      = fb[is_unit]
            unit_c      = fc[is_unit]
            forced_lits = self._lit_sum[unit_b, unit_c].long()

            # --- Vectorized filter (B/C/D) ---
            neg_forced       = forced_lits ^ 1
            is_active        = self._active[unit_b]
            is_new           = self._assignment[unit_b, forced_lits] == 0
            is_contradiction = self._assignment[unit_b, neg_forced]  == 1

            contra_mask = is_active & is_contradiction
            if contra_mask.any():
                self._conflict[unit_b[contra_mask]] = True
                self._active[unit_b[contra_mask]]   = False

            to_prop = is_active & is_new & ~is_contradiction
            if not to_prop.any():
                break

            prop_batches = unit_b[to_prop]
            prop_lits    = forced_lits[to_prop]

            # --- Sparse dedup + intra-round contradiction check (E) ---
            # Encode (batch, lit) as a flat scalar; torch.unique deduplicates cheaply
            two_N = 2 * N
            flat_unique = torch.unique(prop_batches * two_N + prop_lits)  # sorted, deduped
            prop_batches = flat_unique.div(two_N, rounding_mode='floor')
            prop_lits    = flat_unique % two_N

            # Intra-round conflict: same batch has both l and ~l queued
            flat_neg = prop_batches * two_N + (prop_lits ^ 1)
            intra_mask = torch.isin(flat_neg, flat_unique)
            if intra_mask.any():
                bad = torch.unique(prop_batches[intra_mask])
                bad = bad[self._active[bad]]
                if bad.numel() > 0:
                    self._conflict[bad] = True
                    self._active[bad]   = False
                    keep = ~torch.isin(prop_batches, bad)
                    prop_batches = prop_batches[keep]
                    prop_lits    = prop_lits[keep]

            if prop_batches.numel() == 0:
                break

            # --- Propagate; next frontier = clauses whose sat_count just changed (F) ---
            this_round_work = prop_batches.numel()  # free: tensor metadata, no CUDA sync
            frontier_b, frontier_c = self._bulk_propagate_lits(prop_batches, prop_lits)

            # --- Per-batch propagation count (G) ---
            ones = torch.ones(prop_batches.size(0), dtype=torch.int32, device=self._device)
            self._num_propagated.scatter_add_(0, prop_batches, ones)

            # --- Soft cutoff: stop when per-round work decays below soft_stop_frac × peak (H) ---
            if this_round_work > peak_work:
                peak_work = this_round_work
            if (soft_stop_frac is not None
                    and round_count >= min_rounds
                    and peak_work > 0
                    and this_round_work < soft_stop_frac * peak_work):
                if frontier_b.numel() > 0:
                    in_flight = torch.unique(frontier_b)
                    still_active = in_flight[self._active[in_flight]]
                    if still_active.numel() > 0:
                        self._timed_out[still_active] = True
                break

        return self._conflict

    @torch.inference_mode()
    def propagate_profiled(self) -> tuple[torch.Tensor, dict]:
        """Like propagate() but with per-step, per-round timing.

        Inlines _bulk_propagate_lits to expose sub-step costs.
        Use only for profiling — sync overhead inflates wall time.

        Returns:
            conflict: (K,) boolean tensor
            summary:  dict[step_name -> {total_ms, mean_ms, n}]
        """
        N = self._num_vars
        device = self._device
        is_cuda = device.type == "cuda"
        occ = self._formula.occ_list
        M = self._num_clauses

        def ts() -> float:
            if is_cuda:
                torch.cuda.synchronize()
            return time.perf_counter()

        steps: dict[str, list[float]] = {
            "frontier_filter": [],  # active-mask filter on frontier
            "sc_gather":       [],  # sat_count[fb, fc] gather
            "conflict":        [],  # sat==0 detection + mark
            "unit_filter":     [],  # B/C/D: forced-lit gather + active/new/contra filter
            "sparse_dedup":    [],  # E: unique encode + intra-conflict (sparse, no bitmap)
            "csr_lookup":      [],  # CSR occ_offsets slice + lengths
            "expand":          [],  # repeat_interleave + segment-offset + flat_idx
            "scatter_sat":     [],  # scatter_add_ on sat_count
            "scatter_lit":     [],  # scatter_add_ on lit_sum
            "assign":          [],  # assignment write + num_propagated scatter_add
        }

        frontier_b = self._frontier_batches
        frontier_c = self._frontier_clauses

        while frontier_b.numel() > 0:
            self._bcp_rounds += 1

            t0 = ts()
            active_mask = self._active[frontier_b]
            any_active = active_mask.any().item()
            steps["frontier_filter"].append(ts() - t0)
            if not any_active:
                break
            fb = frontier_b[active_mask]
            fc = frontier_c[active_mask]

            t0 = ts()
            sc_vals = self._sat_count[fb, fc]
            steps["sc_gather"].append(ts() - t0)

            t0 = ts()
            is_conflict = sc_vals == 0
            if is_conflict.any():
                self._conflict[fb[is_conflict]] = True
                self._active[fb[is_conflict]]   = False
            steps["conflict"].append(ts() - t0)

            t0 = ts()
            is_unit = (sc_vals == 1) & self._active[fb]
            if not is_unit.any():
                steps["unit_filter"].append(ts() - t0)
                break
            unit_b      = fb[is_unit]
            unit_c      = fc[is_unit]
            forced_lits = self._lit_sum[unit_b, unit_c].long()
            neg_forced       = forced_lits ^ 1
            is_active        = self._active[unit_b]
            is_new           = self._assignment[unit_b, forced_lits] == 0
            is_contradiction = self._assignment[unit_b, neg_forced]  == 1
            contra_mask = is_active & is_contradiction
            if contra_mask.any():
                self._conflict[unit_b[contra_mask]] = True
                self._active[unit_b[contra_mask]]   = False
            to_prop = is_active & is_new & ~is_contradiction
            steps["unit_filter"].append(ts() - t0)
            if not to_prop.any():
                break

            prop_batches = unit_b[to_prop]
            prop_lits    = forced_lits[to_prop]

            t0 = ts()
            two_N = 2 * N
            flat_unique = torch.unique(prop_batches * two_N + prop_lits)
            prop_batches = flat_unique.div(two_N, rounding_mode='floor')
            prop_lits    = flat_unique % two_N
            flat_neg = prop_batches * two_N + (prop_lits ^ 1)
            intra_mask = torch.isin(flat_neg, flat_unique)
            if intra_mask.any():
                bad = torch.unique(prop_batches[intra_mask])
                bad = bad[self._active[bad]]
                if bad.numel() > 0:
                    self._conflict[bad] = True
                    self._active[bad]   = False
                    keep = ~torch.isin(prop_batches, bad)
                    prop_batches = prop_batches[keep]
                    prop_lits    = prop_lits[keep]
            steps["sparse_dedup"].append(ts() - t0)
            if prop_batches.numel() == 0:
                break

            P = prop_batches.size(0)

            # Inlined _bulk_propagate_lits with sub-step timing
            t0 = ts()
            starts  = occ.occ_offsets[prop_lits]
            ends    = occ.occ_offsets[prop_lits + 1]
            lengths = ends - starts
            total   = lengths.sum().item()          # CPU sync: necessary scalar transfer
            steps["csr_lookup"].append(ts() - t0)

            if total > 0:
                t0 = ts()
                exp_batches = prop_batches.repeat_interleave(lengths)
                cum = torch.zeros(P, dtype=torch.long, device=device)
                if P > 1:
                    cum[1:] = lengths[:-1].cumsum(0)
                flat_range  = torch.arange(total, dtype=torch.long, device=device)
                local_off   = flat_range - cum.repeat_interleave(lengths)
                csi         = starts.repeat_interleave(lengths) + local_off
                exp_clauses = occ.occ_clauses[csi]
                flat_idx    = exp_batches * M + exp_clauses
                steps["expand"].append(ts() - t0)

                t0 = ts()
                neg_ones = torch.full((total,), -1, dtype=torch.int32, device=device)
                self._sat_count.view(-1).scatter_add_(0, flat_idx, neg_ones)
                steps["scatter_sat"].append(ts() - t0)

                t0 = ts()
                neg_lits     = (prop_lits ^ 1).to(torch.int32)
                exp_neg_lits = (-neg_lits).repeat_interleave(lengths)
                self._lit_sum.view(-1).scatter_add_(0, flat_idx, exp_neg_lits)
                steps["scatter_lit"].append(ts() - t0)

                frontier_b, frontier_c = exp_batches, exp_clauses
            else:
                steps["expand"].append(0.0)
                steps["scatter_sat"].append(0.0)
                steps["scatter_lit"].append(0.0)
                _empty = torch.empty(0, dtype=torch.long, device=device)
                frontier_b, frontier_c = _empty, _empty

            t0 = ts()
            self._assignment[prop_batches, prop_lits] = 1
            ones = torch.ones(P, dtype=torch.int32, device=device)
            self._num_propagated.scatter_add_(0, prop_batches, ones)
            steps["assign"].append(ts() - t0)

        def summarize(times: list[float]) -> dict:
            if not times:
                return {"total_ms": 0.0, "mean_ms": 0.0, "n": 0}
            ms = [t * 1e3 for t in times]
            return {"total_ms": sum(ms), "mean_ms": sum(ms) / len(ms), "n": len(ms)}

        return self._conflict, {k: summarize(v) for k, v in steps.items()}

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
    def timed_out(self) -> torch.Tensor:
        """(K,) bool — True if batch hit the round limit before reaching fixpoint/conflict."""
        return self._timed_out

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

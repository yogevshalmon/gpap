"""Conflict clause extraction from GPAP BCP results."""
import torch

from gpap.clsdata.occ_list import lit_index_to_ext_lit


def extract_conflict_clauses(
    assumptions_dimacs: list[list[int]],
    conflict: "list[bool] | torch.Tensor",
) -> list[list[int]]:
    """For each conflicting assumption set, return its negation as a clause.

    If {a1, ..., aL} causes a BCP conflict then (¬a1 ∨ ... ∨ ¬aL) is entailed
    by the original formula and can be safely added as a learned clause.

    Args:
        assumptions_dimacs: K assumption sets as DIMACS literals (±integers)
        conflict: (K,) bool list or tensor

    Returns:
        List of learned clauses, each a list of DIMACS literals.
    """
    if isinstance(conflict, torch.Tensor):
        conflict = conflict.tolist()
    return [
        [-lit for lit in assumptions_dimacs[k]]
        for k, is_conflict in enumerate(conflict)
        if is_conflict
    ]


def extract_conflict_clauses_tensor(
    assumptions_lit_idx: torch.Tensor,
    real_conflict: torch.Tensor,
) -> list[list[int]]:
    """Learned clauses from conflicting assumption rows, without a K-sized Python pass.

    Selects the conflicting rows on-GPU, moves only that (usually small) subset to the
    CPU, and negates each assumption literal: a learned clause holds the negation of
    every assumed literal, so clause literal = lit_index_to_ext_lit(i ^ 1).

    Args:
        assumptions_lit_idx: (K, L) int64 literal indices; -1 entries are padding.
        real_conflict:       (K,) bool — use engine.real_conflict (round_count > 1).

    Returns:
        List of learned clauses, each a list of DIMACS literals.
    """
    if real_conflict.dtype != torch.bool:
        real_conflict = real_conflict.bool()
    rows = assumptions_lit_idx[real_conflict]
    if rows.numel() == 0:
        return []
    clauses: list[list[int]] = []
    for row in rows.cpu().tolist():
        clause = [lit_index_to_ext_lit(i ^ 1) for i in row if i >= 0]
        if clause:
            clauses.append(clause)
    return clauses

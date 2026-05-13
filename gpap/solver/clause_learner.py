"""Conflict clause extraction from GPAP BCP results."""
import torch


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

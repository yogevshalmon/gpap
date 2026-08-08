import torch

from gpap.engine.formula import CnfFormula, ext_assumptions, ext_lit_to_lit_index
from gpap.engine.bcp_engine import BCPEngine, make_assumption_tensor


class AssumptionChecker:
    """High-level API for checking multiple assumption sets against a CNF formula.
    Requires CUDA + triton (the engine is kernel-only).

    Usage:
        checker = AssumptionChecker("formula.cnf")
        results = checker.check_assumptions([[1, -2, 3], [-1, 2]])
        print(results['conflict'])  # tensor([False, True])
    """

    def __init__(self, cnf_file: "str | None" = None, device: "torch.device | None" = None):
        if device is None:
            device = torch.device("cuda")
        self._device = device
        self._formula = CnfFormula(device)
        if cnf_file is not None:
            self._formula.parse_cnf_file(cnf_file)

    @property
    def formula(self) -> CnfFormula:
        return self._formula

    @property
    def device(self) -> torch.device:
        return self._device

    def check_assumptions(self, assumptions: ext_assumptions) -> dict:
        """Check K assumption sets in parallel (one fixpoint kernel launch).

        Args:
            assumptions: list of K assumption sets, each a list of external literals (DIMACS
                         format). Positive literal = variable TRUE, negative = variable FALSE.

        Returns:
            dict with:
                'conflict': (K,) bool tensor, True if assumption set leads to conflict
                'assignment': (K, 2*N) int8 tensor, final assignment state
                'num_propagated': (K,) int32 tensor, implications derived by BCP
                'sat_count': (K, M) int32 tensor, final clause sat counts
        """
        K = len(assumptions)

        # convert external literals to literal indices, then to padded tensor (setup cost)
        assumptions_lit_indices = [
            [ext_lit_to_lit_index(lit) for lit in assg] for assg in assumptions
        ]
        assum_tensor = make_assumption_tensor(assumptions_lit_indices, self._device)

        engine = BCPEngine(self._formula)
        engine.reset(K)
        engine.apply_and_propagate(assum_tensor)

        return {
            'conflict': engine.conflict,
            'assignment': engine.assignment,
            'num_propagated': engine.num_propagated,
            'sat_count': engine.sat_rows(torch.arange(K, device=self._device)),
        }

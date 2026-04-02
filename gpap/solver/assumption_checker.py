import torch

from gpap.clsdata.cnf_formula import CnfFormula
from gpap.clsdata.occ_list import ext_lit_to_lit_index, lit_index_to_var
from gpap.solver.bcp_engine import BCPEngine
from gpap.utils.sat_def import ext_assumptions


class AssumptionChecker:
    """High-level API for checking multiple assumption sets against a CNF formula.

    Usage:
        checker = AssumptionChecker("formula.cnf")
        results = checker.check_assumptions([[1, -2, 3], [-1, 2]])
        print(results['conflict'])  # tensor([False, True])
    """

    def __init__(self, cnf_file: str = None, device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def check_assumptions(self, assumptions: ext_assumptions, propagate: bool = True) -> dict:
        """Check K assumption sets in parallel.

        Args:
            assumptions: list of K assumption sets, each a list of external literals (DIMACS format).
                         Positive literal = variable TRUE, negative = variable FALSE.
            propagate: if True, run BCP after applying assumptions.

        Returns:
            dict with:
                'conflict': (K,) bool tensor, True if assumption set leads to conflict
                'assignment': (K, 2*N) int8 tensor, final assignment state
                'num_propagated': (K,) int32 tensor, number of literals propagated by BCP
                'sat_count': (K, M) int32 tensor, final clause sat counts
        """
        K = len(assumptions)

        # convert external literals to literal indices
        assumptions_lit_indices = []
        for assg in assumptions:
            lit_indices = [ext_lit_to_lit_index(lit) for lit in assg]
            assumptions_lit_indices.append(lit_indices)

        # create BCP engine and init state
        engine = BCPEngine(self._formula)
        engine.init_state(num_batches=K)

        # apply assumptions
        engine.apply_assumptions_bulk(assumptions_lit_indices)

        # run BCP if requested
        if propagate:
            engine.propagate()

        return {
            'conflict': engine.conflict,
            'assignment': engine.assignment,
            'num_propagated': engine.num_propagated,
            'sat_count': engine.sat_count,
        }

    def check_assumptions_tensor(self, assumptions: torch.Tensor, propagate: bool = True) -> dict:
        """Check assumptions given as a tensor.

        Args:
            assumptions: (K, N) tensor with values in {-1, 0, 1} (GPSS-compatible format).
                         -1 = variable FALSE, 0 = unassigned, 1 = variable TRUE.
            propagate: if True, run BCP after applying assumptions.

        Returns: same as check_assumptions.
        """
        K, N = assumptions.shape
        assert N == self._formula.num_vars, f"Expected {self._formula.num_vars} vars, got {N}"

        assumptions_lit_indices = []
        for k in range(K):
            lit_indices = []
            for v in range(N):
                val = assumptions[k, v].item()
                if val == 1:
                    lit_indices.append(2 * v)       # positive literal
                elif val == -1:
                    lit_indices.append(2 * v + 1)   # negative literal
            assumptions_lit_indices.append(lit_indices)

        engine = BCPEngine(self._formula)
        engine.init_state(num_batches=K)
        engine.apply_assumptions_bulk(assumptions_lit_indices)

        if propagate:
            engine.propagate()

        return {
            'conflict': engine.conflict,
            'assignment': engine.assignment,
            'num_propagated': engine.num_propagated,
            'sat_count': engine.sat_count,
        }


if __name__ == "__main__":
    device = torch.device("cpu")
    checker = AssumptionChecker(device=device)

    # manually set formula: (x1 OR x2), (~x1 OR x3), (~x2 OR ~x3)
    clauses = [[1, 2], [-1, 3], [-2, -3]]
    checker.formula.init_formula(3, clauses)

    results = checker.check_assumptions([
        [1, 2],    # assume x1=T, x2=T
        [-1, -2],  # assume x1=F, x2=F -> both lits false in clause 1 -> conflict
        [1],       # assume x1=T -> BCP should propagate
    ])

    print("Conflicts:", results['conflict'])
    print("Num propagated:", results['num_propagated'])
    for k in range(3):
        print(f"Batch {k} assignment:", results['assignment'][k])

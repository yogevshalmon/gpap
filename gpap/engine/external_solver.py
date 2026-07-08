import torch

from pysat.formula import CNF
from pysat.solvers import MinisatGH
from threading import Timer


class ExternalSolver():

    def __init__(self, cnf_file):
        self._solver = MinisatGH
        self.cnf = CNF(from_file=cnf_file)
        self._number_of_vars = self.cnf.nv

    def convert_to_internal_assump(self, assumps: torch.Tensor) -> list:
        assert self._number_of_vars == assumps.size()[0]
        assum_ind = torch.arange(1, self._number_of_vars + 1, device=assumps.device)
        internal_assumptions = (assum_ind * assumps).to(torch.int).tolist()
        internal_assumptions = [val for val in internal_assumptions if val != 0]
        assert len(internal_assumptions) > 0
        print("assumptions percentage:", len(internal_assumptions) / self._number_of_vars)
        return internal_assumptions

    def interrupt_solver(self, solver):
        solver.interrupt()
        self.is_timeout = True

    def solve_cnf(self, timeout, assumps: torch.Tensor = None, solve_under_assump=False):
        """Solve the CNF with optional assumptions.

        Returns: (model, stats)
        """
        stats = {}
        model = None
        self.is_timeout = False

        print("Start solving cnf with external solver")
        with MinisatGH(bootstrap_with=self.cnf, use_timer=True) as solver:

            internal_assump = []
            if assumps is not None:
                internal_assump = self.convert_to_internal_assump(assumps)
                solver.set_phases(literals=internal_assump)

            timer = Timer(timeout, self.interrupt_solver, [solver])
            timer.start()
            if solve_under_assump:
                res = solver.solve_limited(expect_interrupt=True, assumptions=internal_assump)
            else:
                res = solver.solve_limited(expect_interrupt=True)
            timer.cancel()

            stats = solver.accum_stats()
            stats.update({
                "time": solver.time_accum(),
                "result": "model" if res else "timeout" if self.is_timeout else "unsat"
            })

            model = solver.get_model()
            if model and assumps is not None:
                num_of_non_assump_vars = self._number_of_vars - len(internal_assump)
                model_tensor = torch.tensor(model, device=assumps.device)
                model_tensor = model_tensor.sign()
                print("number of vars:", self._number_of_vars)
                print("number of assumptions vars:", len(internal_assump))
                assumptions_model_diff = torch.sum(torch.abs((assumps - model_tensor).sign())).item() - num_of_non_assump_vars
                print("sum of diff between assumps and model:", assumptions_model_diff)
                print("Percentage of success of assumptions:", (len(internal_assump) - assumptions_model_diff) / len(internal_assump))

        return model, stats

    def propagate_assumptions(self, assumptions: list[int]) -> tuple:
        """Run unit propagation under assumptions using PySAT.

        Args:
            assumptions: list of DIMACS literals to assume

        Returns:
            (is_conflict, propagated_lits, solve_time_s): conflict flag, propagated
            literals, and internal solver time in seconds (excludes bootstrap overhead).
        """
        with MinisatGH(bootstrap_with=self.cnf, use_timer=True) as solver:
            st, propagated = solver.propagate(assumptions=assumptions)
            return (not st, propagated, solver.time())

    def propagate_many(self, assumptions_list: list[list[int]]) -> tuple:
        """Run unit propagation for multiple assumption sets using a single solver instance.

        The formula is bootstrapped once, avoiding repeated CNF loading overhead.

        Args:
            assumptions_list: list of K assumption sets (each a list of DIMACS literals)

        Returns:
            (conflicts, propagated_list, total_solve_time_s): list of conflict bools,
            list of propagated-literal lists, and total internal solver time in seconds.
        """
        conflicts = []
        propagated_list = []
        total_time_s = 0.0
        with MinisatGH(bootstrap_with=self.cnf, use_timer=True) as solver:
            for assumptions in assumptions_list:
                st, propagated = solver.propagate(assumptions=assumptions)
                conflicts.append(not st)
                propagated_list.append(propagated)
                total_time_s += solver.time()
        return conflicts, propagated_list, total_time_s

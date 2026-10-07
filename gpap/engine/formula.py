"""CNF formula representation for the GPAP engine: literal indexing and the formula facade.

Literal indexing (the 2N scheme): +v -> 2*(v-1), -v -> 2*(v-1)+1, negation = lit_index ^ 1.

A formula compiles (in the C++ core, `cpp/src/formula.cpp` — THE one builder both faces share) to
an occ CSR of int32 holding, for each literal l, the clauses where ~l appears (setting l TRUE
falsifies ~l in exactly those, which is what makes BCP scan-free); `clause_sizes`, the packed
word's initial sat_count; and `init_lit_sum`, the per-clause SUM of literal indices mod 2^32.

The remaining-literal trick is that MODULAR SUM: when exactly one literal survives, the field holds
it exactly, since every removal is exact mod 2^F and a single index is < 2^F. That form is what
makes the warp-parallel kernel possible — ONE atomic_add applies count-=1 and sum-=neg together
(the count never borrows, the sum wraps off the top), and its RETURN VALUE alone yields
conflict/unit detection plus the forced literal. The independent re-derivation these structures
are checked against is `tests/reference/formula.py`.
"""
import torch

ext_lit = int                      # signed DIMACS literal
ext_cls = list[ext_lit]            # one clause
ext_assg = list[ext_lit]           # one assignment / assumption set
ext_assumptions = list[ext_assg]   # multiple assumption sets


def ext_lit_to_lit_index(ext_lit: int) -> int:
    """DIMACS literal -> internal literal index (+v -> 2*(v-1), -v -> 2*(v-1)+1)."""
    var = abs(ext_lit)
    return 2 * (var - 1) + (0 if ext_lit > 0 else 1)


def lit_index_to_ext_lit(lit_index: int) -> int:
    """Inverse of ext_lit_to_lit_index: even index -> +var, odd -> -var (1-based)."""
    var = lit_index // 2 + 1
    return var if lit_index % 2 == 0 else -var


def lit_index_negation(lit_index: int) -> int:
    return lit_index ^ 1


def read_dimacs(cnf_file: str) -> tuple:
    """Parse a DIMACS file with THE parser (the core's) into `(num_vars, clauses, support)`:
    clauses as list[list[DIMACS literal]], support as a 0-based ascending variable list or None.
    The list-of-lists entry point for consumers that transform clauses on the host. Three
    normalizations: within-clause literal order comes back ascending by literal index; an input
    empty clause, which the builder drops and flags, is re-appended ONCE so the trivially-UNSAT
    formula stays visible; and `num_vars` honours the HEADER when it declares more variables than
    the clauses use, so `c ind`-only support variables keep their slots."""
    from gpap.engine import _core
    f = _core.Formula.from_file(cnf_file)
    clauses = f.clauses()
    if f.info.has_empty_clause:
        clauses.append([])
    sup = f.support()
    return int(f.info.num_vars), clauses, (sup.tolist() if sup.size else None)


class CnfFormula:
    """A CNF formula, compiled for the engine — a facade over the core's builder.

    Construction (`init_formula` from a clause list, `parse_cnf_file` from DIMACS) goes through
    libgpap: the occ CSR and the per-clause counters are built once on the C++ side and stay
    there. This class holds the SHAPE (`num_vars`, `num_clauses`, `max_clause_len`, `total_lits`,
    `has_empty_clause`) and the `core` handle an engine consumes.

    An EMPTY clause makes the formula trivially UNSAT and cannot be represented in the counters,
    so the builder drops it and records `has_empty_clause`, which the engine turns into an
    all-rows conflict. `clause_sizes` / `init_lit_sum` materialize as torch tensors ON DEMAND;
    prefer the free `max_clause_len` / `total_lits` over reductions over them.
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._clause_sizes: torch.Tensor = None   # (M,) int32, materialized on demand
        self._init_lit_sum: torch.Tensor = None   # (M,) int32 (sum mod 2^32, exact bit pattern)
        self._num_vars = 0
        self._num_clauses = 0
        self._has_empty_clause = False
        self._core = None
        self._max_clause_len = 0
        self._total_lits = 0


    def init_formula(self, num_vars: int, clauses: list[ext_cls]) -> None:
        from gpap.engine import _core
        self._adopt_core(_core.Formula.from_clauses(num_vars, clauses))

    def parse_cnf_file(self, cnf_file: str) -> None:
        """Parse a DIMACS CNF file and build the formula. Token-stream parsing (clauses are
        0-terminated, not line-based), so multiple clauses per line and clauses spanning lines are
        handled, as is the SATLIB '%' end marker; N = max(header N, highest variable used)."""
        from gpap.engine import _core
        self._adopt_core(_core.Formula.from_file(cnf_file))

    def _adopt_core(self, cf) -> None:
        self._core = cf
        self._num_vars = int(cf.info.num_vars)
        self._num_clauses = int(cf.info.num_clauses)
        self._has_empty_clause = bool(cf.info.has_empty_clause)
        self._max_clause_len = int(cf.info.max_clause_len)
        self._total_lits = int(cf.info.total_lits)
        self._clause_sizes = None
        self._init_lit_sum = None


    @property
    def core(self):
        if self._core is None:
            from gpap.engine._core import GpapError
            raise GpapError(0, "this CnfFormula holds no formula",
                            "build one with init_formula(...) or parse_cnf_file(...)")
        return self._core


    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def num_clauses(self) -> int:
        return self._num_clauses

    @property
    def max_clause_len(self) -> int:
        return self._max_clause_len

    @property
    def total_lits(self) -> int:
        return self._total_lits

    @property
    def has_empty_clause(self) -> bool:
        return self._has_empty_clause

    @property
    def support(self) -> "list[int] | None":
        """Independent support declared by the input's `c ind` / `cr` header lines, as 0-based
        ascending variable indices, or None when none was declared."""
        if self._core is None:
            return None
        s = self._core.support()
        return s.tolist() if s.size else None

    # ------------------------------------------------------------------ inspection (on demand)

    @property
    def clause_sizes(self) -> torch.Tensor:
        if self._clause_sizes is None and self._core is not None:
            self._materialize_clause_arrays()
        return self._clause_sizes

    @property
    def init_lit_sum(self) -> torch.Tensor:
        if self._init_lit_sum is None and self._core is not None:
            self._materialize_clause_arrays()
        return self._init_lit_sum

    def _materialize_clause_arrays(self) -> None:
        sizes, lsum = self._core.clause_arrays()
        self._clause_sizes = torch.from_numpy(sizes).to(self._device)
        self._init_lit_sum = torch.from_numpy(lsum).to(self._device)

"""CNF formula representation for the GPAP engine: literal indexing and the formula facade.

Literal indexing (the 2N scheme):
  positive literal  v -> index 2*(v-1)
  negative literal -v -> index 2*(v-1) + 1
  negation: lit_index ^ 1

What a formula compiles to (built by the C++ core, `cpp/src/formula.cpp` — THE one builder the
Python and C++ faces share):
  * occ CSR (int32): for each literal l, the clauses where ~l appears — setting l TRUE
    falsifies ~l in exactly those clauses, which is what makes BCP scan-free;
  * clause_sizes: literals per clause (the packed word's initial sat_count);
  * init_lit_sum: per-clause SUM of literal indices mod 2^32 (the engine masks it to its packed
    field width; mod-2^32-then-mod-2^F == mod-2^F).

The remaining-literal trick is that MODULAR SUM: when exactly one literal survives, the field
holds it exactly (every removal is exact mod 2^F, and a single literal index < 2^F). The sum
form is what makes the warp-parallel kernel possible: ONE atomic_add applies count-=1 and
sum-=neg together (the count field never borrows — count >= 1 at every decrement — and the sum
wraps off the top of the word), and the atomic's RETURN VALUE alone yields exact conflict/unit
detection plus the forced literal.

The independent re-derivation of these structures — the oracle `tests/test_core.py` compares the
core against, and the parser coverage for hosts with no libgpap — lives with the other reference
implementations, in `tests/reference_formula.py`.
"""
import torch

# External (DIMACS) SAT type aliases
ext_lit = int                      # signed DIMACS literal
ext_cls = list[ext_lit]            # one clause
ext_assg = list[ext_lit]           # one assignment / assumption set
ext_assumptions = list[ext_assg]   # multiple assumption sets


def ext_lit_to_lit_index(ext_lit: int) -> int:
    """Convert DIMACS literal to internal literal index.
    Positive literal v -> 2*(v-1)
    Negative literal -v -> 2*(v-1) + 1
    """
    var = abs(ext_lit)
    return 2 * (var - 1) + (0 if ext_lit > 0 else 1)


def lit_index_to_ext_lit(lit_index: int) -> int:
    """Inverse of ext_lit_to_lit_index: internal index -> DIMACS literal.
    Even index -> +var, odd index -> -var (1-based variable).
    """
    var = lit_index // 2 + 1
    return var if lit_index % 2 == 0 else -var


def lit_index_negation(lit_index: int) -> int:
    """Negate a literal index: flip the last bit."""
    return lit_index ^ 1


def read_dimacs(cnf_file: str) -> tuple:
    """Parse a DIMACS file with THE parser (the core's) into plain Python structures:
    `(num_vars, clauses, support)` — clauses as list[list[DIMACS literal]], support as a
    0-based ascending variable list or None when the file declares none (`c ind` / `cr` lines).

    This is the list-of-lists entry point for consumers that transform clauses on the host
    (simplification, mining, re-emission); it is why the library needs no third-party DIMACS
    parser. Three normalizations to know about: within-clause literal order comes back
    ascending by literal index (the builder retains no input order, and nothing downstream is
    order-sensitive); an empty clause in the input — which the builder drops from the
    structures and flags — is re-appended here, ONCE, so the trivially-UNSAT formula stays
    visible to formula-level consumers; and `num_vars` honours the HEADER when it declares
    more variables than the clauses use (the parser's documented rule, so `c ind`-only support
    variables keep their slots — a parser that recounts from the clauses would silently drop
    them). Requires a built libgpap (raises with build instructions)."""
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
    libgpap: the occ CSR and the per-clause counters are built once, on the C++ side, and stay
    there — the engine's kernels read the core's own device copies, so nothing is duplicated
    into Python. What this class holds is the SHAPE (`num_vars`, `num_clauses`,
    `max_clause_len`, `total_lits`, `has_empty_clause`) and the `core` handle an engine
    consumes. Requires a built libgpap; constructing one without it raises with the build
    instructions.

    An EMPTY clause makes the formula trivially UNSAT; it cannot be represented in the counters,
    so the builder drops it from the structures and records it in `has_empty_clause` (the engine
    turns that flag into an all-rows conflict).

    `clause_sizes` / `init_lit_sum` are materialized as torch tensors ON DEMAND for inspection
    and tests — a normal sampling run never pays for them. Prefer `max_clause_len` /
    `total_lits` (free, from the core) over reductions over `clause_sizes`.
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._clause_sizes: torch.Tensor = None   # (M,) int32, materialized on demand
        self._init_lit_sum: torch.Tensor = None   # (M,) int32 (sum mod 2^32, exact bit pattern)
        self._num_vars = 0
        self._num_clauses = 0
        self._has_empty_clause = False
        self._core = None                         # gpap.engine._core.Formula once built
        self._max_clause_len = 0
        self._total_lits = 0

    # ------------------------------------------------------------------ construction

    def init_formula(self, num_vars: int, clauses: list[ext_cls]) -> None:
        """Build from a list of clauses (external DIMACS literals)."""
        from gpap.engine import _core
        self._adopt_core(_core.Formula.from_clauses(num_vars, clauses))

    def parse_cnf_file(self, cnf_file: str) -> None:
        """Parse a DIMACS CNF file and build the formula.

        Token-stream parsing (clauses are 0-terminated, NOT line-based): multiple clauses on
        one line and clauses spanning lines are handled per the DIMACS standard, as is the
        SATLIB '%' end marker. N = max(header N, highest variable used) so trailing declared
        variables (e.g. support vars appearing only in `c ind` lines) keep their slots."""
        from gpap.engine import _core
        self._adopt_core(_core.Formula.from_file(cnf_file))

    def _adopt_core(self, cf) -> None:
        """Take the shape from a core-built formula (the structures stay on the C++ side)."""
        self._core = cf
        self._num_vars = int(cf.info.num_vars)
        self._num_clauses = int(cf.info.num_clauses)
        self._has_empty_clause = bool(cf.info.has_empty_clause)
        self._max_clause_len = int(cf.info.max_clause_len)
        self._total_lits = int(cf.info.total_lits)
        self._clause_sizes = None
        self._init_lit_sum = None

    # ------------------------------------------------------------------ core handle

    @property
    def core(self):
        """The `_core.Formula` handle the engine consumes. Raises if nothing was built yet."""
        if self._core is None:
            from gpap.engine._core import GpapError
            raise GpapError(0, "this CnfFormula holds no formula",
                            "build one with init_formula(...) or parse_cnf_file(...)")
        return self._core

    # ------------------------------------------------------------------ shape

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
        """Longest clause — bounds the packed word's count field (free; no reduction)."""
        return self._max_clause_len

    @property
    def total_lits(self) -> int:
        """Total literal occurrences (free; no reduction)."""
        return self._total_lits

    @property
    def has_empty_clause(self) -> bool:
        """True iff the input contained an empty clause (formula trivially UNSAT)."""
        return self._has_empty_clause

    @property
    def support(self) -> "list[int] | None":
        """Independent support declared by the input's `c ind` / `cr` header lines, as 0-based
        ascending variable indices — or None when none was declared (or nothing is built yet).
        Parsed by the core, so a file parsed once is never re-scanned for its header."""
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

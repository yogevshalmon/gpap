"""AssumptionChecker — the batched-assumption entry point, with no batch size in sight: entry
point 1 (K independent assumption sets, one BCP fixpoint each, one launch), fronted by an API that
asks for nothing but the sets. The chunk loop lives in the CORE (`gpap_check_sets`,
cpp/src/check.cpp), so validation, chunk sizing and padding are defined once; this module marshals
— flatten, raise a ValueError before any GPU work, return torch tensors in INPUT ORDER.
"""
import numpy as np
import torch

from gpap.engine.formula import CnfFormula, ext_assumptions
from gpap.engine.bcp_engine import BCPEngine
from gpap.engine.calibrate import gpu_mem_available


def _flatten(assumptions: ext_assumptions, num_vars: int):
    """(flat DIMACS literals, per-set lengths, (n+1,) offsets) — VALIDATED, vectorised. An
    out-of-range literal becomes an index past the end of its row's assignment bitmap, and the
    kernel's `atomic_or` then writes into a LATER row's words, undetectably; literal 0 maps to
    index -1, which the kernel reads as PADDING, so `[[0]]` would quietly mean "no assumptions".
    Validating HERE keeps the documented ValueError contract, and the flat form is the ABI's."""
    lens = np.fromiter((len(s) for s in assumptions), dtype=np.int64, count=len(assumptions))
    flat = np.fromiter((lit for s in assumptions for lit in s), dtype=np.int64,
                       count=int(lens.sum()))
    if flat.size:
        bad = (flat == 0) | (np.abs(flat) > num_vars)
        if bad.any():
            raise ValueError(f"assumption literal out of range: {int(flat[bad][0])} "
                             f"(formula has {num_vars} variables; literals are DIMACS, "
                             f"1..{num_vars} or -1..-{num_vars}, never 0)")
    off = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    return flat.astype(np.int32), lens, off


class AssumptionChecker:
    """Check assumption sets against a CNF formula, K of them per GPU launch (CUDA + libgpap; there
    is no CPU path).

        checker = AssumptionChecker("formula.cnf")                # or .from_clauses(nv, clauses)
        r = checker.check_assumptions([[1, -2, 3], [-1, 2]])
        r["conflict"]            # (n,) bool — did this set falsify a clause?

    The formula may be replaced after construction; the engine is built on first use and reused.
    """

    def __init__(self, cnf_file: "str | None" = None, device: "torch.device | None" = None):
        if device is None:
            device = torch.device("cuda")
        self._device = device
        self._formula = CnfFormula(device)
        self._engine: BCPEngine | None = None
        self._engine_key = None
        if cnf_file is not None:
            self._formula.parse_cnf_file(cnf_file)

    @classmethod
    def from_clauses(cls, num_vars: int, clauses: list,
                     device: "torch.device | None" = None) -> "AssumptionChecker":
        """Build straight from a clause list (DIMACS literals), as the C++ face does."""
        ck = cls(device=device)
        ck._formula.init_formula(int(num_vars), clauses)
        return ck

    @property
    def formula(self) -> CnfFormula:
        return self._formula

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def engine(self) -> BCPEngine:
        """The underlying engine, built once and reused (rebuilding per call would re-upload the
        occurrence CSR and re-run the unit flush), keyed on the formula's core handle."""
        key = id(self._formula.core)
        if self._engine is None or self._engine_key != key:
            self._engine = BCPEngine(self._formula)
            self._engine_key = key
        return self._engine

    def chunk_size(self, n_sets: int, max_lits: int = 1) -> int:
        """Assumption sets one launch can hold — the K the caller never has to name. A binding of
        the core's `gpap_check_chunk_size`: per-row state plus the (K, L) int64 decision tensor
        only this entry point allocates, read against LIVE free memory and bounded by the
        sampler's cap (an over-large chunk spills to host RAM silently)."""
        return self.engine.check_chunk_size(n_sets, max_lits)

    def check_assumptions(self, assumptions: ext_assumptions, *, assignments: bool = False,
                          sat_count: bool = False) -> dict:
        """Propagate every assumption set to a BCP fixpoint, independently. `assumptions` is any
        number of sets, each a list of DIMACS literals (+v = TRUE, -v = FALSE), split into
        GPU-sized chunks by the core; results come back in INPUT ORDER. Returns a dict, always
        with `conflict` (n,) bool — the set and its implications falsify a clause — and
        `num_propagated` (n,) int32; plus, on request, `assignment` (n, 2N) int8 and `sat_count`
        (n, M) int32, the (n, ·)-shaped fields that make a big call expensive.

        Which fields are SPECIFIED: `conflict` is exact for every set, and so is everything else
        for a set that does NOT conflict, since BCP fixpoints are confluent. For a set that DOES
        conflict, propagation stopped as soon as some clause emptied and which got there first
        depends on the order, so the other fields are a snapshot — they move with anything that
        changes propagation order, the chunk count included. Raises ValueError on a literal
        outside 1..N / -1..-N, 0 included.
        """
        n = len(assumptions)
        M, nv = self._formula.num_clauses, self._formula.num_vars
        dev = self._device
        # Refuse a result that cannot fit BEFORE any GPU work: chunking cannot shrink the
        # caller's own (n, ·) request.
        result_bytes = (2 * nv if assignments else 0) + (4 * M if sat_count else 0)
        if n and result_bytes:
            self._check_result_fits(n, result_bytes, assignments, sat_count)
        if n == 0:                     # nothing to check: no engine, no launch, right shapes
            out = {"conflict": torch.zeros(0, dtype=torch.bool, device=dev),
                   "num_propagated": torch.zeros(0, dtype=torch.int32, device=dev)}
            if assignments:
                out["assignment"] = torch.zeros((0, 2 * nv), dtype=torch.int8, device=dev)
            if sat_count:
                out["sat_count"] = torch.zeros((0, M), dtype=torch.int32, device=dev)
            return out

        flat, lens, off = _flatten(assumptions, nv)
        # Allocate the big DEVICE results BEFORE sizing the chunks, so `chunk_size`'s live
        # free-memory read excludes them; otherwise both budgets claim the same VRAM.
        dev_assign = (torch.zeros((n, 2 * nv), dtype=torch.int8, device=dev)
                      if assignments else None)
        dev_sat = torch.zeros((n, M), dtype=torch.int32, device=dev) if sat_count else None
        # Chunk size from the GLOBAL longest set: a chunk's matrix is sized by its own longest,
        # so budgeting for the worst only over-reserves. Passed explicitly so the core runs
        # exactly the chunks this face reports.
        step = self.chunk_size(n, int(lens.max()))
        conflict, prop, assign, sat = self.engine.check_sets(
            flat, off, chunk=step, assignments=assignments, sat_count=sat_count)
        out = {"conflict": torch.from_numpy(conflict).to(dev),
               "num_propagated": torch.from_numpy(prop).to(dev)}
        if assignments:
            dev_assign.copy_(torch.from_numpy(assign))
            out["assignment"] = dev_assign
        if sat_count:
            dev_sat.copy_(torch.from_numpy(sat))
            out["sat_count"] = dev_sat
        return out

    def _check_result_fits(self, n: int, row_bytes: int, assignments: bool, sat_count: bool):
        """Refuse a request whose RESULT cannot fit on the device, before spending GPU time:
        chunking bounds the work but not the (n, 2N) / (n, M) arrays the caller asked for."""
        want = n * row_bytes
        budget = gpu_mem_available(self._device)
        if want <= budget:
            return
        asked = " + ".join(k for k, on in (("assignments", assignments),
                                           ("sat_count", sat_count)) if on)
        raise MemoryError(
            f"{asked} for {n:,} assumption sets needs {want / 2**30:.1f} GiB, but only "
            f"{budget / 2**30:.1f} GiB is available on {self._device}. Check the sets in "
            f"batches, or drop the flag — 'conflict' and 'num_propagated' are ~5 bytes/set and "
            f"scale to any number of sets.")

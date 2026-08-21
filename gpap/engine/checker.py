"""AssumptionChecker — the batched-assumption entry point, with no batch size in sight.

This is the engine's whole reason for existing (entry point 1: K independent assumption sets,
one BCP fixpoint each, one launch), so the API that fronts it should ask for nothing but the
assumption sets. Hand it any number of them; they are sized to the GPU, run, and returned in
INPUT ORDER. Nothing here mentions K.

THE CHUNK LOOP LIVES IN THE CORE (`gpap_check_sets`, cpp/src/check.cpp), exactly as the probe
sweep does: validation, chunk sizing and padding are defined once, so this face and the C++ one
(`gpap::AssumptionChecker`) cannot drift apart. What this module does is marshal — flatten the
list-of-lists, give the caller a proper ValueError before any GPU work, and shape the results
as torch tensors on the formula's device.
"""
import numpy as np
import torch

from gpap.engine.formula import CnfFormula, ext_assumptions
from gpap.engine.bcp_engine import BCPEngine
from gpap.engine.calibrate import gpu_mem_available


def _flatten(assumptions: ext_assumptions, num_vars: int):
    """(flat DIMACS literals, per-set lengths, (n+1,) offsets) — VALIDATED, vectorised.

    Validation is not optional, and the core validates again for callers that reach it directly.
    An out-of-range literal becomes an internal index past the end of its row's assignment
    bitmap, and the kernel's `atomic_or` then writes into a LATER ROW's words: on a 3-variable
    formula (2N=6, one 32-bit word per row) literal 99 lands six words downstream, so row 0's
    bad assumption silently assigns a literal in row 6. Nothing downstream can detect that, so
    it must be caught before the launch. Literal 0 is rejected for the reason it terminates a
    clause in DIMACS: it maps to index -1, which the kernel reads as PADDING, so `[[0]]` would
    quietly mean "no assumptions". Validating HERE keeps the documented ValueError contract;
    the core's own check returns a GpapError instead. The message matches the C++ face.

    Flat + offsets rather than a list of lists: a million-set call must not pay a Python loop
    per set — and the flat form is exactly what the core's ABI takes.
    """
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
    """Check assumption sets against a CNF formula, K of them per GPU launch.

    Requires CUDA + a built libgpap (there is no CPU execution path).

        checker = AssumptionChecker("formula.cnf")                # from DIMACS
        checker = AssumptionChecker.from_clauses(nv, clauses)     # from a clause list
        r = checker.check_assumptions([[1, -2, 3], [-1, 2]])
        r["conflict"]            # (n,) bool — did this set falsify a clause?

    The formula may also be replaced after construction (`checker.formula.init_formula(...)`);
    the engine is built on first use and reused across calls.
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
        """Build straight from a clause list (DIMACS literals) — the same two constructors as
        the C++ face (`Formula::from_dimacs` / `Formula::from_clauses`)."""
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
        """The underlying engine, built once and reused.

        Rebuilding it per call — which this class used to do — re-uploads the occurrence CSR and
        re-runs the formula's unit flush every time, which is the dominant cost of a small call.
        Keyed on the formula's core handle so replacing the formula after a check is still
        honoured (`checker.formula.init_formula(...)` returns a new handle)."""
        key = id(self._formula.core)
        if self._engine is None or self._engine_key != key:
            self._engine = BCPEngine(self._formula)
            self._engine_key = key
        return self._engine

    def chunk_size(self, n_sets: int, max_lits: int = 1) -> int:
        """Assumption sets one launch can hold — the K the caller never has to name.

        A binding of the core's `gpap_check_chunk_size` (cpp/src/check.cpp): the engine's exact
        per-row state plus the (K, L) int64 decision tensor this entry point allocates and the
        sampler's path never does. Read against LIVE free memory, bounded by the same cap the
        sampler uses, never larger than the input. This box SPILLS to host RAM rather than
        failing, so an over-large chunk shows up as a silent ~10x slowdown, not an error —
        hence the conservative model."""
        return self.engine.check_chunk_size(n_sets, max_lits)

    def check_assumptions(self, assumptions: ext_assumptions, *, assignments: bool = False,
                          sat_count: bool = False) -> dict:
        """Propagate every assumption set to a BCP fixpoint, independently.

        `assumptions`: any number of sets, each a list of DIMACS literals (+v = TRUE, -v =
        FALSE). They are split into GPU-sized chunks by the core; results come back in INPUT
        ORDER, so a caller cannot tell how many launches it took.

        Returns a dict, always with:
            'conflict'        (n,) bool  — the set (with its implications) falsifies a clause
            'num_propagated'  (n,) int32 — implications BCP derived beyond the assumptions
        and, when asked for (both are (n, ·)-shaped, so they are what makes a big call expensive):
            'assignment'      (n, 2N) int8  — 1 where that literal is TRUE   [assignments=True]
            'sat_count'       (n, M) int32  — surviving literals per clause  [sat_count=True]

        WHICH FIELDS ARE SPECIFIED. `conflict` is exact for every set. For a set that does NOT
        conflict, so is everything else: BCP fixpoints are confluent, so the surviving row has ONE
        answer no matter how the propagation was ordered. For a set that DOES conflict, propagation
        stopped as soon as some clause emptied — and which clause got there first depends on the
        order — so `num_propagated`, `assignment` and `sat_count` on those rows are a snapshot of
        where BCP happened to stop, not a defined value. They can differ between two runs of the
        same call if anything moves the propagation order, including the number of chunks the call
        was split into (chunk size feeds the wave/scalar dispatch). Do not compare them across runs;
        `conflict` is the answer a conflicted row has.

        Raises ValueError on a literal outside 1..N / -1..-N (including 0) — see `_flatten`.
        """
        n = len(assumptions)
        M, nv = self._formula.num_clauses, self._formula.num_vars
        dev = self._device
        # Refuse a result that cannot fit BEFORE any GPU work; the big arrays are the caller's
        # own (n, ·) request and chunking cannot shrink them.
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
        # free-memory read already excludes them — otherwise the two budgets double-count the
        # same VRAM (measured: on a 60k x 60k formula the engine state and the result each
        # claimed 80% of free memory, 12.9 GiB peak against 10 GiB free — a silent host-RAM
        # spill on this hardware). The core fills HOST buffers; the copy at the end lands in
        # these pre-reserved tensors.
        dev_assign = (torch.zeros((n, 2 * nv), dtype=torch.int8, device=dev)
                      if assignments else None)
        dev_sat = torch.zeros((n, M), dtype=torch.int32, device=dev) if sat_count else None
        # Chunk size from the GLOBAL longest set: a chunk's decision matrix is sized by its own
        # longest set, so budgeting for the worst one can only over-reserve, never under-. Passed
        # explicitly so the core runs exactly the chunks this face reports (and so tests can pin
        # a tiny chunk to exercise the loop).
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
        """Refuse a request whose RESULT cannot fit on the device, before spending GPU time.

        Chunking bounds the work, but it cannot shrink what was asked for: `assignments` is
        (n, 2N) and `sat_count` is (n, M) for the caller's own n, and this face holds them on
        the device (plus a same-sized host staging buffer the core fills — host RAM, which
        merely swaps). This box degrades to host RAM instead of erroring, so
        without this the failure mode is a run that appears to work and is an order of magnitude
        slower — say so instead, and name the way out."""
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

"""AssumptionChecker — the batched-assumption entry point, with no batch size in sight.

This is the engine's whole reason for existing (entry point 1: K independent assumption sets,
one BCP fixpoint each, one launch), so the API that fronts it should ask for nothing but the
assumption sets. Hand it any number of them; it sizes the chunks to the GPU and the formula,
runs them, and returns results in INPUT ORDER. Nothing here mentions K.
"""
import numpy as np
import torch

from gpap.engine.formula import CnfFormula, ext_assumptions
from gpap.engine.bcp_engine import BCPEngine
from gpap.engine.calibrate import default_k_cap, gpu_mem_available, per_batch_mem_bytes


def _flatten(assumptions: ext_assumptions, num_vars: int):
    """(literal indices, per-set lengths, offsets) for the whole input — VALIDATED, vectorised.

    Validation is not optional. An out-of-range literal becomes an internal index past the end of
    its row's assignment bitmap, and the kernel's `atomic_or` then writes into a LATER ROW's
    words: on a 3-variable formula (2N=6, one 32-bit word per row) literal 99 lands six words
    downstream, so row 0's bad assumption silently assigns a literal in row 6. Nothing downstream
    can detect that, so it must be caught before the launch. Literal 0 is rejected for the reason
    it terminates a clause in DIMACS: it maps to index -1, which the kernel reads as PADDING, so
    `[[0]]` would quietly mean "no assumptions". The message matches the C++ face
    (`gpap::AssumptionChecker::check`).

    Flat + offsets rather than a list of lists: a million-set call must not pay a Python loop
    per set, and each chunk's padded matrix is then a single numpy scatter.
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
    idx = 2 * (np.abs(flat) - 1) + (flat < 0)
    off = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    return idx, lens, off


class AssumptionChecker:
    """Check assumption sets against a CNF formula, K of them per GPU launch.

    Requires CUDA + a built libgpap (there is no CPU execution path).

        checker = AssumptionChecker("formula.cnf")
        r = checker.check_assumptions([[1, -2, 3], [-1, 2]])
        r["conflict"]            # (n,) bool — did this set falsify a clause?

    The formula may also be supplied after construction (`checker.formula.init_formula(...)`);
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

        Two terms, and the second is the one the shared analytic model does NOT cover:

          * `per_batch_mem_bytes` — the engine's own per-row device state (the conservative,
            kill-switch-validated model shared with the sampler);
          * `8 * max_lits` — the (K, L) int64 decision tensor, which is device memory this entry
            point allocates and the sampler's path never does. At 200 literals per set that is
            1.6 kB/row, comparable to the engine state itself on a small formula.

        Read against LIVE free memory, so calling it after the result arrays are allocated
        correctly leaves room for them. Bounded by `default_k_cap` for the same reason the sampler
        is, and never larger than the input. This box SPILLS to host RAM rather than failing, so
        an over-large chunk shows up as a silent ~10x slowdown, not an error — hence the
        conservative model."""
        dev_per_row = per_batch_mem_bytes(
            self._formula.num_clauses, self._formula.num_vars,
            self._formula.max_clause_len or 127,
            self._formula.total_lits or None) + 8 * max(max_lits, 1)
        k = min(int(gpu_mem_available(self._device) // dev_per_row),
                default_k_cap(self._device))
        return max(1, min(k, n_sets))

    def check_assumptions(self, assumptions: ext_assumptions, *, assignments: bool = False,
                          sat_count: bool = False) -> dict:
        """Propagate every assumption set to a BCP fixpoint, independently.

        `assumptions`: any number of sets, each a list of DIMACS literals (+v = TRUE, -v =
        FALSE). They are split into GPU-sized chunks internally; results come back in INPUT
        ORDER, so a caller cannot tell how many launches it took.

        Returns a dict, always with:
            'conflict'        (n,) bool  — the set (with its implications) falsifies a clause
            'num_propagated'  (n,) int32 — implications BCP derived beyond the assumptions
        and, when asked for (both are (n, ·)-shaped, so they are what makes a big call expensive):
            'assignment'      (n, 2N) int8  — 1 where that literal is TRUE   [assignments=True]
            'sat_count'       (n, M) int32  — surviving literals per clause  [sat_count=True]

        Raises ValueError on a literal outside 1..N / -1..-N (including 0) — see `_flatten`.
        """
        n = len(assumptions)
        M, nv = self._formula.num_clauses, self._formula.num_vars
        dev = self._device
        # Results are allocated in FULL up front, then filled per chunk, which is what makes the
        # chunking invisible: row i of every array is set i, whatever launch produced it.
        # Allocating them BEFORE sizing the chunks also means `chunk_size`'s free-memory read
        # already excludes them — the two budgets cannot double-count.
        row_bytes = (2 * nv if assignments else 0) + (4 * M if sat_count else 0)
        if n and row_bytes:
            self._check_result_fits(n, row_bytes, assignments, sat_count)
        out = {"conflict": torch.zeros(n, dtype=torch.bool, device=dev),
               "num_propagated": torch.zeros(n, dtype=torch.int32, device=dev)}
        if assignments:
            out["assignment"] = torch.zeros((n, 2 * nv), dtype=torch.int8, device=dev)
        if sat_count:
            out["sat_count"] = torch.zeros((n, M), dtype=torch.int32, device=dev)
        if n == 0:                     # nothing to check: no engine, no launch, right shapes
            return out

        idx, lens, off = _flatten(assumptions, nv)
        eng = self.engine
        # Chunk size from the GLOBAL longest set: a chunk's decision tensor is sized by its own
        # longest set, so budgeting for the worst one can only over-reserve, never under-.
        step = self.chunk_size(n, int(lens.max()))
        for a in range(0, n, step):
            b = min(a + step, n)
            eng.reset(b - a)                       # the LAST chunk is short: reset to its size
            eng.apply_and_propagate(_pad(idx, lens, off, a, b))
            out["conflict"][a:b] = eng.conflict
            out["num_propagated"][a:b] = eng.num_propagated
            if assignments:
                out["assignment"][a:b] = eng.assignment
            if sat_count:
                out["sat_count"][a:b] = eng.sat_rows(np.arange(b - a, dtype=np.int32))
        return out

    def _check_result_fits(self, n: int, row_bytes: int, assignments: bool, sat_count: bool):
        """Refuse a request whose RESULT cannot fit, before spending any GPU time on it.

        Chunking bounds the work, but it cannot shrink what was asked for: `assignments` is
        (n, 2N) and `sat_count` is (n, M) for the caller's own n. This box degrades to host RAM
        instead of erroring, so without this the failure mode is a run that appears to work and is
        an order of magnitude slower — say so instead, and name the way out."""
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


def _pad(idx: np.ndarray, lens: np.ndarray, off: np.ndarray, a: int, b: int) -> np.ndarray:
    """Rows a..b of the flattened literals as a (b-a, L) int64 matrix, -1 padded.

    One scatter, no Python loop: `rows` repeats each row index by its length and `cols` counts
    within each row (a global arange minus each element's row start)."""
    ln = lens[a:b]
    L = max(int(ln.max()), 1)
    pad = np.full((b - a, L), -1, dtype=np.int64)
    if ln.sum():
        rows = np.repeat(np.arange(b - a, dtype=np.int64), ln)
        starts = np.repeat(off[a:b] - off[a], ln)
        cols = np.arange(off[b] - off[a], dtype=np.int64) - starts
        pad[rows, cols] = idx[off[a]:off[b]]
    return pad

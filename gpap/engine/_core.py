"""ctypes binding to libgpap — the C++ engine core.

Why ctypes and not a compiled extension: the C ABI (`cpp/include/gpap/gpap.h`) has to exist
anyway for C++ consumers, and binding it this way means there is no Python extension to build,
no interpreter/ABI coupling (one `.so` serves every Python), and no extra build dependency.
Call overhead is ~1-5 us against a 50-150 ms sampling chunk — measured irrelevant.

Library lookup order:
    $GPAP_LIB (explicit path)  ->  <repo>/build/  ->  gpap/lib/libgpap.so (installed wheel)
Build it with:
    python cpp/tools/build_kernels.py && cmake -S cpp -B build && cmake --build build -j
"""
from __future__ import annotations

import ctypes
import os
from ctypes import (POINTER, byref, c_char_p, c_int, c_int8, c_int32, c_int64, c_size_t,
                    c_void_p)
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ status / errors

GPAP_OK = 0
_STATUS = {0: "OK", 1: "bad argument", 2: "CUDA error", 3: "formula outside packed envelopes",
           4: "I/O error", 5: "missing kernel binary", 6: "device allocation failed"}


class GpapError(RuntimeError):
    """A libgpap call failed; carries the core's thread-local error message."""

    def __init__(self, status: int, what: str, detail: str = ""):
        self.status = status
        super().__init__(f"{what}: {_STATUS.get(status, status)}" + (f" — {detail}" if detail else ""))


class EnvelopeError(GpapError, ValueError):
    """The formula is outside both packed splits (no unpacked fallback exists).

    Also a ValueError: the engine's documented contract is that constructing it on an
    out-of-envelope formula raises one, and suite drivers catch it to skip the instance."""


# ------------------------------------------------------------------ library loading

_LIB = None


def _candidates() -> list[Path]:
    """$GPAP_LIB, then a repo-local build tree, then the copy inside the package.

    The build tree comes FIRST on purpose: in a development checkout it is the fresh one, and
    a stale `gpap/lib/libgpap.so` left by an earlier `build_core.sh --install` would otherwise
    silently shadow a rebuild. An installed wheel has no build tree, so it falls through."""
    env = os.environ.get("GPAP_LIB")
    if env and not Path(env).exists():
        # Explicit beats implicit: silently falling through to some OTHER library because the
        # requested path has a typo is how you benchmark a build you did not mean to.
        raise GpapError(0, "GPAP_LIB is set but does not exist", env)
    out = [Path(env)] if env else []
    here = Path(__file__).resolve()
    repo = here.parents[2]
    out += [repo / "build" / "libgpap.so", repo / "cpp" / "build" / "libgpap.so"]
    out.append(here.parents[1] / "lib" / "libgpap.so")          # installed alongside the package
    return out


def load() -> ctypes.CDLL:
    """Load libgpap once. Raises with build instructions if it is not there."""
    global _LIB
    if _LIB is not None:
        return _LIB
    tried = []
    for p in _candidates():
        tried.append(str(p))
        if p.exists():
            _LIB = ctypes.CDLL(str(p))
            _declare(_LIB)
            return _LIB
    raise GpapError(0, "libgpap not found", "looked in: " + ", ".join(tried) +
                    "\nBuild it with:\n"
                    "  python cpp/tools/build_kernels.py\n"
                    "  cmake -S cpp -B build && cmake --build build -j")


def available() -> bool:
    """True iff libgpap can be loaded (used by tests / engine dispatch)."""
    try:
        load()
        return True
    except Exception:       # noqa: BLE001 — any load failure just means "unavailable"
        return False


class FormulaInfo(ctypes.Structure):
    _fields_ = [("num_vars", c_int32), ("num_clauses", c_int32), ("max_clause_len", c_int32),
                ("shift", c_int32), ("has_empty_clause", c_int32), ("total_lits", c_int64)]


class EngineOptions(ctypes.Structure):
    """Mirror of `gpap_engine_options`. Every field is "0 means the sensible default"
    — including vscan, where -1 (not 0) disables the warp-wide decision scan."""
    _fields_ = [("device", c_int), ("mode", c_int), ("force_shift", c_int32), ("vscan", c_int32)]


class ProbeOptions(ctypes.Structure):
    """Mirror of `gpap_probe_options`. 0/negative means "no limit" everywhere except
    `distinct_vars`, whose useful default is 1 (drop tuples sharing a variable). `max_tuples` is a
    WORK BUDGET — the sweep probes that many rather than refusing (see `refuse_over_max`)."""
    _fields_ = [("seconds", ctypes.c_double), ("max_tuples", c_int64), ("batch", c_int32),
                ("distinct_vars", c_int32), ("refuse_over_max", c_int32)]


class ProbeStats(ctypes.Structure):
    """Mirror of `gpap_probe_stats`. `exhausted` is the field that turns a negative result into a
    proof: it means every candidate was probed."""
    _fields_ = [("total", c_int64), ("probed", c_int64), ("conflicts", c_int64),
                ("exhausted", c_int32), ("stop", c_int32), ("seconds", ctypes.c_double)]


#: `gpap_probe_stop` — why a sweep ended.
PROBE_STOP = {0: "", 1: "max_tuples", 2: "deadline", 3: "unsat"}


class Counters(ctypes.Structure):
    _fields_ = [("ops", c_int64), ("props", c_int64), ("decs", c_int64), ("launches", c_int64)]


_I32P, _I64P, _I8P = POINTER(c_int32), POINTER(c_int64), POINTER(c_int8)


def _declare(lib: ctypes.CDLL) -> None:
    sig = [
        ("gpap_formula_from_dimacs", [c_char_p, POINTER(c_void_p)], c_int),
        ("gpap_formula_from_clauses", [c_int32, _I32P, _I32P, c_int32, POINTER(c_void_p)], c_int),
        ("gpap_formula_info_get", [c_void_p, POINTER(FormulaInfo)], None),
        ("gpap_formula_free", [c_void_p], None),
        ("gpap_formula_occ", [c_void_p, _I32P, _I32P, _I32P, _I64P], c_int),
        ("gpap_formula_clause_arrays", [c_void_p, _I32P, _I32P], c_int),
        ("gpap_formula_clauses", [c_void_p, _I32P, _I64P], c_int),
        ("gpap_formula_support", [c_void_p, _I32P, _I32P], c_int),
        ("gpap_packed_envelope", [c_int32, c_int32, _I32P], c_int),
        ("gpap_engine_create", [c_void_p, POINTER(EngineOptions), POINTER(c_void_p)], c_int),
        ("gpap_engine_free", [c_void_p], None),
        ("gpap_engine_set_mode", [c_void_p, c_int], c_int),
        ("gpap_engine_mode", [c_void_p], c_int),
        ("gpap_engine_shift", [c_void_p], c_int32),
        ("gpap_engine_k", [c_void_p], c_int32),
        ("gpap_engine_num_vars", [c_void_p], c_int32),
        ("gpap_engine_num_clauses", [c_void_p], c_int32),
        ("gpap_engine_reset", [c_void_p, c_int32], c_int),
        ("gpap_engine_apply", [c_void_p, _I64P, c_int32], c_int),
        ("gpap_engine_apply_dev", [c_void_p, c_size_t, c_int32], c_int),
        ("gpap_engine_apply_clean", [c_void_p, _I64P, c_int32], c_int),
        ("gpap_engine_reset_stats", [c_void_p, POINTER(c_int64), POINTER(c_int64)], None),
        ("gpap_engine_trajectory", [c_void_p, _I32P, c_int32, c_int32, c_int32, c_int64, c_int,
                                    _I32P], c_int),
        ("gpap_engine_trajectory_dev", [c_void_p, c_size_t, c_int32, c_int32, c_int32, c_int64,
                                        c_int, _I32P], c_int),
        ("gpap_engine_recache_flush", [c_void_p], c_int),
        ("gpap_engine_flush_conflict", [c_void_p], c_int),
        ("gpap_engine_flushed_assignment", [c_void_p, _I8P], c_int),
        ("gpap_check_sets", [c_void_p, _I32P, _I64P, c_int64, c_int32, _I8P, _I32P, _I8P,
                             _I32P], c_int),
        ("gpap_check_chunk_size", [c_void_p, c_int64, c_int64], c_int64),
        ("gpap_pair_mask_create", [c_void_p, _I32P, c_int32, _I32P], c_int),
        ("gpap_select_complete", [c_void_p, c_int32, c_int32, _I32P], c_int),
        ("gpap_selection_rows", [c_void_p, c_int32, _I32P, c_int32], c_int),
        ("gpap_selection_set", [c_void_p, c_int32, _I32P, c_int32], c_int),
        ("gpap_check_sound", [c_void_p, c_int32, c_int32, _I32P], c_int),
        ("gpap_var_bits", [c_void_p, c_int32, _I32P, c_int32, _I8P, c_int32], c_int),
        ("gpap_assignment_rows", [c_void_p, c_int32, _I8P, c_int32], c_int),
        ("gpap_flags", [c_void_p, _I8P, _I8P], c_int),
        ("gpap_num_propagated", [c_void_p, _I32P], c_int),
        ("gpap_sat_rows", [c_void_p, _I32P, c_int32, _I32P], c_int),
        ("gpap_packed_rows", [c_void_p, _I32P, c_int32, _I32P], c_int),
        ("gpap_counters_get", [c_void_p, POINTER(Counters)], c_int),
        ("gpap_trail_cap", [c_int32, c_int32], c_int32),
        ("gpap_row_bytes", [c_int32, c_int32, c_int32], c_int64),
        ("gpap_compute_k_safe", [c_int32, c_int32, c_int64, c_int32, c_int32], c_int32),
        ("gpap_default_k_cap", [c_int], c_int32),
        ("gpap_mem_available", [c_int, ctypes.c_double], c_int64),
        ("gpap_resident_warps", [c_int], c_int32),
        ("gpap_dispatch_wave", [c_int32, c_int32, c_int], c_int32),
        ("gpap_probe_combinations",
         [c_void_p, _I32P, c_int32, c_int32, POINTER(ProbeOptions), POINTER(c_void_p)], c_int),
        ("gpap_probe_tuples",
         [c_void_p, _I32P, c_int64, c_int32, POINTER(ProbeOptions), POINTER(c_void_p)], c_int),
        ("gpap_probe_count", [c_void_p, _I32P, c_int32, c_int32, c_int32], c_int64),
        ("gpap_probe_free", [c_void_p], None),
        ("gpap_probe_stats_get", [c_void_p, POINTER(ProbeStats)], None),
        ("gpap_probe_conflicts", [c_void_p, _I32P, c_int64], c_int),
        ("gpap_free_literals", [c_void_p, _I32P, _I32P], c_int),
        ("gpap_assume_units", [c_void_p, _I32P, c_int32], c_int),
        ("gpap_last_error", [], c_char_p),
        ("gpap_version", [], c_char_p),
    ]
    for name, argtypes, restype in sig:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype


def check(status: int, what: str) -> None:
    if status == GPAP_OK:
        return
    detail = load().gpap_last_error().decode(errors="replace")
    raise (EnvelopeError if status == 3 else GpapError)(status, what, detail)


# ------------------------------------------------------------------ numpy helpers

def _p(arr: np.ndarray, ct):
    return arr.ctypes.data_as(POINTER(ct))


def i32(arr) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.int32)


def version() -> str:
    return load().gpap_version().decode()


def trail_cap(num_vars: int, decisions_per_round: int = 8) -> int:
    """Trail slots one row can need — bounded by the VARIABLE count (see gpap.h)."""
    return int(load().gpap_trail_cap(int(num_vars), int(decisions_per_round)))


def row_bytes(M: int, N: int, trail_cap_: int = 0) -> int:
    """Device bytes ONE row of engine state costs — the exact allocation list."""
    return int(load().gpap_row_bytes(int(M), int(N), int(trail_cap_)))


def compute_k_safe(M: int, N: int, gpu_mem_bytes: float, cap: "int | None" = None,
                   decisions_per_round: int = 8) -> int:
    """Largest K fitting `gpu_mem_bytes`. THE sizing call — the model lives in
    cpp/src/calibrate.cpp and this is a binding, not a second implementation."""
    return int(load().gpap_compute_k_safe(int(M), int(N), int(gpu_mem_bytes),
                                          int(cap or 0), int(decisions_per_round)))


def mem_available(device: int = 0, fraction: float = 0.80) -> int:
    """Bytes available for new allocations (free VRAM minus a driver reserve, times fraction)."""
    return int(load().gpap_mem_available(int(device), float(fraction)))


def resident_warps(device: int = 0) -> int:
    return int(load().gpap_resident_warps(int(device)))


def dispatch_wave(num_clauses: int, batch: int, device: int = 0) -> bool:
    """THE wave/scalar dispatch rule, from the core — see cpp/src/calibrate.cpp for why the
    deciding variable is the clause count and not the batch size."""
    return bool(load().gpap_dispatch_wave(int(num_clauses), int(batch), int(device)))


def default_k_cap(device: int = 0) -> int:
    """The core's VRAM-proportional batch cap for `device` — the same rule (and the same two
    constants) that `gpap.engine.calibrate.default_k_cap` applies from Python. Bound here so the
    two faces can be asserted equal rather than assumed equal (tests/test_k_cap.py)."""
    return int(load().gpap_default_k_cap(int(device)))


# ---------------------------------------------------------------------------- probing
# The sweep itself (enumeration, chunking, budget) runs in the core; these only marshal. A
# `ProbeSweep` owns a core handle, so it is a context manager — leaking one leaks its conflict
# buffer until the process exits.


class ProbeSweep:
    """One sweep's result, straight from the core. Free it (or use `with`)."""

    __slots__ = ("_h", "_k", "stats")

    def __init__(self, handle, k: int):
        self._h, self._k = handle, int(k)
        self.stats = ProbeStats()
        load().gpap_probe_stats_get(self._h, byref(self.stats))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.free()
        return False

    def conflicts(self):
        """(n, k) int32 of DIMACS literals: the tuples whose assumption drove BCP to a conflict."""
        n = int(self.stats.conflicts)
        out = np.empty((n, self._k), dtype=np.int32)
        if n:
            check(load().gpap_probe_conflicts(self._h, out.ctypes.data_as(_I32P), n),
                  "gpap_probe_conflicts")
        return out

    def free(self) -> None:
        if getattr(self, "_h", None):
            load().gpap_probe_free(self._h)
            self._h = None

    def __del__(self):
        self.free()


def _probe_options(seconds=None, max_tuples=None, batch=0, distinct_vars=True,
                   refuse_over_max=False):
    return ProbeOptions(float(seconds or 0.0), int(max_tuples or 0), int(batch or 0),
                        1 if distinct_vars else 0, 1 if refuse_over_max else 0)


def _pool_arg(pool) -> tuple:
    """(ptr, n) for a literal pool; None means "all free literals" (a NULL pool in the ABI)."""
    if pool is None:
        return None, 0
    arr = np.ascontiguousarray(pool, dtype=np.int32).reshape(-1)
    return arr.ctypes.data_as(_I32P), int(arr.size)


def packed_envelope(num_vars: int, max_clause_len: int) -> "tuple[int | None, str]":
    """(SHIFT, "") when the formula shape fits a packed split, else (None, reason)."""
    lib = load()
    shift = c_int32(0)
    st = lib.gpap_packed_envelope(num_vars, max_clause_len, byref(shift))
    if st == GPAP_OK:
        return int(shift.value), ""
    return None, lib.gpap_last_error().decode(errors="replace")


# ------------------------------------------------------------------ handles

class Formula:
    """Owns a `gpap_formula*`. Built once, then handed to one or more engines."""

    def __init__(self, handle: c_void_p):
        self._h = handle
        self.info = FormulaInfo()
        load().gpap_formula_info_get(self._h, byref(self.info))

    @classmethod
    def from_file(cls, path: str) -> "Formula":
        lib = load()
        h = c_void_p()
        check(lib.gpap_formula_from_dimacs(str(path).encode(), byref(h)), f"parse {path}")
        return cls(h)

    @classmethod
    def from_clauses(cls, num_vars: int, clauses) -> "Formula":
        lib = load()
        lens = np.fromiter((len(c) for c in clauses), dtype=np.int32, count=len(clauses))
        total = int(lens.sum())
        flat = np.empty(total, dtype=np.int32)
        i = 0
        for c in clauses:
            n = len(c)
            if n:
                flat[i:i + n] = c
                i += n
        h = c_void_p()
        check(lib.gpap_formula_from_clauses(num_vars, _p(flat, c_int32), _p(lens, c_int32),
                                            len(clauses), byref(h)), "build formula")
        return cls(h)

    @property
    def handle(self):
        return self._h

    def occ(self) -> "tuple[np.ndarray, np.ndarray]":
        """(offsets, clauses) of the negated-occurrence CSR — inspection / tests."""
        lib = load()
        n_off, n_cls = c_int32(0), c_int64(0)
        check(lib.gpap_formula_occ(self._h, None, None, byref(n_off), byref(n_cls)), "occ sizes")
        off = np.empty(n_off.value, dtype=np.int32)
        cls_ = np.empty(n_cls.value, dtype=np.int32)
        check(lib.gpap_formula_occ(self._h, _p(off, c_int32), _p(cls_, c_int32), None, None),
              "occ read")
        return off, cls_

    def support(self) -> np.ndarray:
        """Independent support from the file's `c ind` lines (0-based, ascending); empty when
        the file declares none."""
        lib = load()
        n = c_int32(0)
        check(lib.gpap_formula_support(self._h, None, byref(n)), "support size")
        out = np.empty(n.value, dtype=np.int32)
        if n.value:
            check(lib.gpap_formula_support(self._h, _p(out, c_int32), None), "support")
        return out

    def clauses(self) -> list:
        """The clauses back out of the built formula, as list[list[DIMACS literal]].

        Rebuilt from the occ CSR by the core (`gpap_formula_clauses`), so within-clause order is
        ascending by literal index, not input order — no consumer of a clause is
        order-sensitive. A dropped empty clause is NOT re-appended here; check
        `info.has_empty_clause` (read_dimacs does)."""
        total = int(self.info.total_lits)
        flat = np.empty(total, dtype=np.int32)
        if total:
            check(load().gpap_formula_clauses(self._h, _p(flat, c_int32), None),
                  "gpap_formula_clauses")
        sizes = np.empty(self.info.num_clauses, dtype=np.int32)
        check(load().gpap_formula_clause_arrays(self._h, _p(sizes, c_int32), None),
              "clause sizes")
        bounds = np.zeros(len(sizes) + 1, dtype=np.int64)
        np.cumsum(sizes, out=bounds[1:])
        fl = flat.tolist()
        return [fl[a:b] for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist())]

    def clause_arrays(self) -> "tuple[np.ndarray, np.ndarray]":
        """(clause_sizes, lit_sum) — inspection / tests."""
        m = self.info.num_clauses
        sizes = np.empty(m, dtype=np.int32)
        lsum = np.empty(m, dtype=np.int32)
        check(load().gpap_formula_clause_arrays(self._h, _p(sizes, c_int32), _p(lsum, c_int32)),
              "clause arrays")
        return sizes, lsum

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                load().gpap_formula_free(self._h)
                self._h = None
        except Exception:       # noqa: BLE001 — interpreter shutdown
            pass


class Engine:
    """Owns a `gpap_engine*`: the packed row state, the launches and the readout.

    This is the BASE Python wrapper over the ABI — every readout returns NUMPY, because the
    state lives in library-owned device memory, not in torch tensors (that ownership is the
    point). `gpap.engine.bcp_engine.BCPEngine` adapts this class to torch for the sampling
    stack; `gpap.engine.probe.LiteralProbe` consumes it directly."""

    MODE_SCALAR, MODE_WAVE, MODE_AUTO = 0, 1, 2
    _MODES = {"scalar": MODE_SCALAR, "wave": MODE_WAVE, "auto": MODE_AUTO}

    def __init__(self, formula: Formula, device: int = 0, mode: "str | None" = None,
                 force_shift: int = 0, vscan: int = 0):
        """`mode=None` resolves to $GPAP_KERNEL when set, else "auto" — HERE, once, so the env
        override reaches every face built on this class rather than only the ones that remember
        to read it. "auto" lets the CORE pick the kernel on every reset(K); the resolved request
        is kept in `requested_mode`.

        Pin "scalar"/"wave" only to A/B them: the two share the same state and reach the same
        fixpoints, so the choice can change speed and nothing else."""
        lib = load()
        self.requested_mode = mode or os.environ.get("GPAP_KERNEL") or "auto"
        self._f = formula                       # keep the formula alive
        h = c_void_p()
        opts = EngineOptions(device, self._mode_code(self.requested_mode), int(force_shift),
                             int(vscan))
        check(lib.gpap_engine_create(formula.handle, byref(opts), byref(h)), "create engine")
        self._h = h
        self._K = 0
        self._masks: dict = {}

    # -------------------------------------------------- properties
    @property
    def num_vars(self) -> int:
        return int(self._f.info.num_vars)

    @property
    def num_clauses(self) -> int:
        return int(self._f.info.num_clauses)

    @property
    def shift(self) -> int:
        return int(load().gpap_engine_shift(self._h))

    @property
    def K(self) -> int:
        return int(load().gpap_engine_k(self._h))

    @property
    def mode(self) -> str:
        return "wave" if load().gpap_engine_mode(self._h) == self.MODE_WAVE else "scalar"

    @classmethod
    def _mode_code(cls, mode: str) -> int:
        try:
            return cls._MODES[mode]
        except KeyError:
            raise ValueError(f"unknown kernel mode {mode!r} (scalar|wave|auto)") from None

    def set_mode(self, mode: str) -> None:
        """Pin a kernel, or "auto" to hand the choice back to the core's per-reset dispatch."""
        check(load().gpap_engine_set_mode(self._h, self._mode_code(mode)), "set mode")

    @property
    def flush_conflict(self) -> bool:
        return bool(load().gpap_engine_flush_conflict(self._h))

    # -------------------------------------------------- entry points
    def reset(self, K: int) -> None:
        check(load().gpap_engine_reset(self._h, int(K)), "reset")
        self._K = int(K)

    def apply(self, decisions: np.ndarray) -> None:
        """(K, L) int64 internal literal indices, -1 = padding. ONE fixpoint launch."""
        d = np.ascontiguousarray(decisions, dtype=np.int64)
        if d.ndim == 1:
            d = d.reshape(1, -1)
        check(load().gpap_engine_apply(self._h, _p(d, c_int64), d.shape[1]), "apply")

    def apply_dev(self, ptr: int, L: int) -> None:
        check(load().gpap_engine_apply_dev(self._h, ptr, L), "apply(dev)")

    def apply_clean(self, decisions: np.ndarray) -> None:
        """apply(), CLEAN mode: the same fixpoint — identical flags and counters — but the
        launch restores its own packed rows to the post-flush baseline before exiting, so the
        NEXT reset() skips the K x M broadcast (reset-lite). Use ONLY when nothing reads
        post-apply state: the assignment/sat accessors would return the BASELINE, not the
        fixpoint. $GPAP_CLEAN=0 (read at engine create) turns this back into apply()."""
        d = np.ascontiguousarray(decisions, dtype=np.int64)
        if d.ndim == 1:
            d = d.reshape(1, -1)
        check(load().gpap_engine_apply_clean(self._h, _p(d, c_int64), d.shape[1]),
              "apply(clean)")

    def trajectory(self, order, n_support: int, B: int, seed: int, soft: bool = False) -> int:
        """Whole randomized decide+propagate trajectories, ONE launch. Returns max rounds."""
        o = i32(order)
        rounds = c_int32(0)
        check(load().gpap_engine_trajectory(self._h, _p(o, c_int32), o.size, int(n_support),
                                            int(B), int(seed), 1 if soft else 0, byref(rounds)),
              "trajectory")
        return int(rounds.value)

    def trajectory_dev(self, ptr: int, P: int, n_support: int, B: int, seed: int,
                       soft: bool = False) -> int:
        """Same, with `order` already on the device (e.g. a torch tensor's data_ptr) — keeps a
        torch-built decision order zero-copy, which is what preserves bit-identical results
        against the historical engine."""
        rounds = c_int32(0)
        check(load().gpap_engine_trajectory_dev(self._h, ptr, int(P), int(n_support), int(B),
                                                int(seed), 1 if soft else 0, byref(rounds)),
              "trajectory(dev)")
        return int(rounds.value)

    def recache_flush(self) -> None:
        check(load().gpap_engine_recache_flush(self._h), "recache flush")

    # -------------------------------------------------- batched checking (entry point 1)
    def check_sets(self, lits, offsets, *, chunk: int = 0, assignments: bool = False,
                   sat_count: bool = False) -> tuple:
        """Propagate every assumption set to its BCP fixpoint — THE chunk loop, run by the CORE
        (`gpap_check_sets`, cpp/src/check.cpp), so every face shares one validation, one chunk
        model and one padding rule.

        `lits` is the sets FLATTENED as DIMACS literals; `offsets` is (n+1,) int64 with set i =
        lits[offsets[i]:offsets[i+1]]. `chunk` <= 0 sizes launches with `check_chunk_size`.
        Returns numpy `(conflict (n,) bool, propagated (n,) int32, assignment (n, 2N) int8 or
        None, sat (n, M) int32 or None)` — the last two only when asked for, since they are
        what makes a big call expensive."""
        lits = np.ascontiguousarray(lits, dtype=np.int32).reshape(-1)
        off = np.ascontiguousarray(offsets, dtype=np.int64).reshape(-1)
        n = int(off.size) - 1
        # The core validates offsets' SHAPE (monotone, starting at 0) but cannot know the
        # literal array's length — that bound is this binding's to enforce, or a short `lits`
        # is read past its end and the call silently returns answers built from stray memory.
        if n > 0 and int(off[-1]) > lits.size:
            raise ValueError(f"offsets[-1]={int(off[-1])} exceeds len(lits)={lits.size} — the "
                             f"flat literal array must cover every set")
        conflict = np.zeros(n, dtype=np.int8)
        prop = np.zeros(n, dtype=np.int32)
        assign = np.zeros((n, 2 * self.num_vars), dtype=np.int8) if assignments else None
        sat = np.zeros((n, self.num_clauses), dtype=np.int32) if sat_count else None
        if n > 0:
            check(load().gpap_check_sets(
                self._h, _p(lits, c_int32), _p(off, c_int64), n, int(chunk),
                _p(conflict, c_int8), _p(prop, c_int32),
                _p(assign, c_int8) if assign is not None else None,
                _p(sat, c_int32) if sat is not None else None), "gpap_check_sets")
        return conflict.astype(bool), prop, assign, sat

    def check_chunk_size(self, n_sets: int, max_lits: int = 1) -> int:
        """Assumption sets one launch holds — `gpap_check_chunk_size`, the core's one model
        (the engine's exact per-row state plus the (chunk, L) decision matrix this entry point
        allocates and the sampler's path never does)."""
        return int(load().gpap_check_chunk_size(self._h, int(n_sets), int(max_lits)))

    # -------------------------------------------------- probing (entry point 1b)
    # The sweep runs in the CORE: enumeration, chunking and the budget all live in
    # cpp/src/probe.cpp, so a Python caller and a C++ caller get the same candidate order, the
    # same batch model and the same "exhausted" semantics. Nothing here re-derives any of it.

    def probe_combinations(self, pool, k: int, **kw) -> ProbeSweep:
        """Probe every k-subset of `pool` (DIMACS literals); `pool=None` = every free literal."""
        ptr, n = _pool_arg(pool)
        h = c_void_p()
        check(load().gpap_probe_combinations(self._h, ptr, n, int(k),
                                             byref(_probe_options(**kw)), byref(h)),
              "gpap_probe_combinations")
        return ProbeSweep(h, k)

    def probe_tuples(self, tuples, **kw) -> ProbeSweep:
        """Probe an explicit (n, k) array of DIMACS literals."""
        arr = np.ascontiguousarray(tuples, dtype=np.int32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n, k = arr.shape
        h = c_void_p()
        check(load().gpap_probe_tuples(self._h, arr.ctypes.data_as(_I32P), int(n), int(k),
                                       byref(_probe_options(**kw)), byref(h)),
              "gpap_probe_tuples")
        return ProbeSweep(h, k)

    def probe_count(self, pool, k: int, distinct_vars: bool = True) -> int:
        """Exact k-subset count of `pool` (None = all free literals), WITHOUT generating it.

        The number to check a ceiling against before enumerating. One definition, in the core —
        a second copy of this arithmetic in Python is exactly the drift the sizing collapse
        removed."""
        ptr, n = _pool_arg(pool)
        return int(load().gpap_probe_count(self._h, ptr, n, int(k), 1 if distinct_vars else 0))

    def free_literals(self) -> np.ndarray:
        """DIMACS literals the unit flush left undetermined, ascending by variable — what is
        worth probing."""
        n = c_int32(0)
        check(load().gpap_free_literals(self._h, None, byref(n)), "gpap_free_literals")
        out = np.empty(int(n.value), dtype=np.int32)
        if n.value:
            check(load().gpap_free_literals(self._h, out.ctypes.data_as(_I32P), byref(n)),
                  "gpap_free_literals")
        return out

    def assume_units(self, lits) -> None:
        """Fold PROVEN units into the cached flush state in place (see the ABI note: this
        assumes entailment rather than checking it)."""
        arr = np.ascontiguousarray(lits, dtype=np.int32).reshape(-1)
        if arr.size == 0:
            return
        check(load().gpap_assume_units(self._h, arr.ctypes.data_as(_I32P), int(arr.size)),
              "gpap_assume_units")

    # -------------------------------------------------- selections + readout
    def pair_mask(self, vars_=None) -> int:
        """Register a variable set for the completion test; returns a mask id (cached)."""
        key = None if vars_ is None else tuple(np.asarray(vars_, dtype=np.int64).tolist())
        if key in self._masks:
            return self._masks[key]
        mid = c_int32(0)
        if vars_ is None:
            check(load().gpap_pair_mask_create(self._h, None, 0, byref(mid)), "pair mask")
        else:
            v = i32(vars_)
            check(load().gpap_pair_mask_create(self._h, _p(v, c_int32), v.size, byref(mid)),
                  "pair mask")
        self._masks[key] = int(mid.value)
        return int(mid.value)

    def select_complete(self, mask_id: int, slot: int = 0) -> int:
        """Rows that are active AND have no free variable under `mask_id`; returns the count."""
        n = c_int32(0)
        check(load().gpap_select_complete(self._h, int(mask_id), int(slot), byref(n)), "select")
        return int(n.value)

    def selection_rows(self, slot: int, count: int) -> np.ndarray:
        rows = np.empty(count, dtype=np.int32)
        if count:
            check(load().gpap_selection_rows(self._h, int(slot), _p(rows, c_int32), count),
                  "rows")
        return rows

    SCRATCH_SLOT = 2        # reserved for ad-hoc row readout; 0/1 belong to the caller

    def selection_set(self, slot: int, rows) -> int:
        """Put an EXPLICIT ascending row list in a slot, so the readout below can serve
        arbitrary rows (tests, the checker) and not only completed ones."""
        r = i32(rows)
        check(load().gpap_selection_set(self._h, int(slot), _p(r, c_int32), r.size),
              "selection set")
        return int(r.size)

    def rows_assignment(self, rows) -> np.ndarray:
        """(len(rows), 2N) int8 assignments for arbitrary rows."""
        n = self.selection_set(self.SCRATCH_SLOT, rows)
        return self.assignment_rows(self.SCRATCH_SLOT, n)

    def check_sound(self, slot: int = 0, max_rows: int = 4096) -> int:
        """How many selected rows have a clause with sat_count == 0 (must be 0)."""
        n = c_int32(0)
        check(load().gpap_check_sound(self._h, int(slot), int(max_rows), byref(n)), "soundness")
        return int(n.value)

    def var_bits(self, slot: int, cols, count: int) -> np.ndarray:
        """(count, len(cols)) uint8 — 1 iff variable cols[j] is TRUE in the i-th selected row."""
        c = i32(cols)
        out = np.empty((count, c.size), dtype=np.int8)
        if count:
            check(load().gpap_var_bits(self._h, int(slot), _p(c, c_int32), c.size,
                                       _p(out, c_int8), count), "var bits")
        return out.view(np.uint8)

    def assignment_rows(self, slot: int, count: int) -> np.ndarray:
        """(count, 2N) int8 assignment rows for a selection."""
        out = np.empty((count, 2 * self.num_vars), dtype=np.int8)
        if count:
            check(load().gpap_assignment_rows(self._h, int(slot), _p(out, c_int8), count),
                  "assignments")
        return out

    def flags(self) -> "tuple[np.ndarray, np.ndarray]":
        """(active, conflict) as (K,) bool arrays.

        Sized from the CORE's K, not the last reset() seen through this object: construction
        runs an internal reset(1) to build the units-flush cache, so the core's K is 1 before
        the caller resets and a locally-tracked 0 would under-allocate."""
        k = self.K
        a = np.empty(k, dtype=np.int8)
        c = np.empty(k, dtype=np.int8)
        check(load().gpap_flags(self._h, _p(a, c_int8), _p(c, c_int8)), "flags")
        return a.astype(bool), c.astype(bool)

    def num_propagated(self) -> np.ndarray:
        out = np.empty(self.K, dtype=np.int32)
        check(load().gpap_num_propagated(self._h, _p(out, c_int32)), "implications")
        return out

    def sat_rows(self, rows) -> np.ndarray:
        r = i32(rows)
        out = np.empty((r.size, self.num_clauses), dtype=np.int32)
        check(load().gpap_sat_rows(self._h, _p(r, c_int32), r.size, _p(out, c_int32)), "sat rows")
        return out

    def packed_rows(self, rows) -> np.ndarray:
        """RAW packed words (sat_count | sum << SHIFT) for the given rows — sat_rows masks the
        sum field off, which would hide sum corruption from the CLEAN-mode identity gate."""
        r = i32(rows)
        out = np.empty((r.size, self.num_clauses), dtype=np.int32)
        check(load().gpap_packed_rows(self._h, _p(r, c_int32), r.size, _p(out, c_int32)),
              "packed rows")
        return out

    def reset_stats(self) -> "tuple[int, int]":
        """(full broadcasts, lite resets) this engine has run — the lite count is the proof
        CLEAN mode actually removed broadcasts (a silent fallback would measure 1.00x)."""
        full = c_int64(0)
        lite = c_int64(0)
        load().gpap_engine_reset_stats(self._h, byref(full), byref(lite))
        return int(full.value), int(lite.value)

    def flushed_assignment(self) -> np.ndarray:
        out = np.empty(2 * self.num_vars, dtype=np.int8)
        check(load().gpap_engine_flushed_assignment(self._h, _p(out, c_int8)), "flushed assignment")
        return out

    # -------------------------------------------------- counters
    def counters(self) -> Counters:
        c = Counters()
        check(load().gpap_counters_get(self._h, byref(c)), "counters")
        return c

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                load().gpap_engine_free(self._h)
                self._h = None
        except Exception:       # noqa: BLE001 — interpreter shutdown
            pass

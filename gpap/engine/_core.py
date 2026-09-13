"""ctypes binding to libgpap — the C++ engine core.

ctypes rather than a compiled extension: the C ABI (`cpp/include/gpap/gpap.h`) exists anyway for
C++ consumers, so there is no extension to build and no interpreter/ABI coupling (one `.so` serves
every Python), and call overhead is negligible against a sampling chunk. Lookup order: $GPAP_LIB
-> $GPAP_BUILD_DIR -> <repo>/build/ -> <repo>/cpp/build/ -> gpap/lib/libgpap.so; build with
`python cpp/tools/build_kernels.py && cmake
-S cpp -B build && cmake --build build -j`.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import (POINTER, byref, c_char_p, c_int, c_int8, c_int32, c_int64, c_size_t,
                    c_void_p)
from pathlib import Path

import numpy as np


GPAP_OK = 0
_STATUS = {0: "OK", 1: "bad argument", 2: "CUDA error", 3: "formula outside packed envelopes",
           4: "I/O error", 5: "missing kernel binary", 6: "allocation failed"}


class GpapError(RuntimeError):
    """A libgpap call failed; carries the core's thread-local error message."""

    def __init__(self, status: int, what: str, detail: str = ""):
        self.status = status
        super().__init__(f"{what}: {_STATUS.get(status, status)}" + (f" — {detail}" if detail else ""))


class EnvelopeError(GpapError, ValueError):
    """The formula is outside both packed splits; also a ValueError, which suite drivers catch."""


_LIB = None


def _candidates() -> list[Path]:
    """$GPAP_LIB, then $GPAP_BUILD_DIR, then a repo-local build tree, then the package copy —
    build trees FIRST, so a fresh build wins over an older installed copy.

    `$GPAP_BUILD_DIR` is here because `scripts/build_core.sh` honours it: told to build into a
    fast scratch filesystem and not given `--install`, the library existed and nothing could find
    it. The repo root is counted from this file rather than taken from `gpap.paths.repo_root()`
    on purpose — that walks up for a `pyproject.toml`, which under a venv inside someone's project
    would find THEIR project and make `<their project>/build/libgpap.so` a candidate."""
    env = os.environ.get("GPAP_LIB")
    if env and not Path(env).exists():
        # Explicit beats implicit: falling through on a typo benchmarks the wrong build.
        raise GpapError(0, "GPAP_LIB is set but does not exist", env)
    out = [Path(env)] if env else []
    build = os.environ.get("GPAP_BUILD_DIR")
    if build:
        # Same rule as $GPAP_LIB above: an explicit build tree with no library in it is a typo,
        # and falling through to `build/` would load the tree the caller was steering away from.
        if not (Path(build) / "libgpap.so").exists():
            raise GpapError(-1, "GPAP_BUILD_DIR is set but holds no libgpap.so", build)
        out.append(Path(build) / "libgpap.so")
    here = Path(__file__).resolve()
    repo = here.parents[2]
    out += [repo / "build" / "libgpap.so", repo / "cpp" / "build" / "libgpap.so"]
    out.append(here.parents[1] / "lib" / "libgpap.so")
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
    try:
        load()
        return True
    except Exception:       # noqa: BLE001 — any load failure just means "unavailable"
        return False


class FormulaInfo(ctypes.Structure):
    _fields_ = [("num_vars", c_int32), ("num_clauses", c_int32), ("max_clause_len", c_int32),
                ("shift", c_int32), ("has_empty_clause", c_int32), ("total_lits", c_int64),
                ("num_bin", c_int32), ("num_long", c_int32)]


class EngineOptions(ctypes.Structure):
    """Mirror of `gpap_engine_options`; every field's 0 is its default (vscan disables at -1)."""
    _fields_ = [("device", c_int), ("mode", c_int), ("force_shift", c_int32), ("vscan", c_int32)]


class ProbeOptions(ctypes.Structure):
    """Mirror of `gpap_probe_options`; every 0 is a default (hence `same_var`), and `max_tuples`
    is a WORK BUDGET unless `refuse_over_max`."""
    _fields_ = [("seconds", ctypes.c_double), ("max_tuples", c_int64), ("batch", c_int32),
                ("same_var", c_int32), ("refuse_over_max", c_int32)]


class ProbeStats(ctypes.Structure):
    """Mirror of `gpap_probe_stats`; `exhausted` turns a negative result into a proof."""
    _fields_ = [("total", c_int64), ("probed", c_int64), ("conflicts", c_int64),
                ("exhausted", c_int32), ("stop", c_int32), ("seconds", ctypes.c_double)]


#: `gpap_probe_stop` — why a sweep ended.
PROBE_STOP = {0: "", 1: "max_tuples", 2: "deadline", 3: "unsat"}


class Counters(ctypes.Structure):
    _fields_ = [("ops", c_int64), ("props", c_int64), ("decs", c_int64), ("launches", c_int64)]


class SampleConfig(ctypes.Structure):
    _fields_ = [("random_order", c_int32), ("batch", c_int32), ("soft", c_int32)]


class SamplerOptions(ctypes.Structure):
    """Mirror of `gpap_sampler_options`, gpap_sampler_run's PER-RUN knobs (seed and k_cap live on
    the handle); every zero is a default, `retry_with_lookahead` included."""
    _fields_ = [("max_chunks", c_int64), ("keep_cap", c_int64),
                ("time_limit_s", ctypes.c_double), ("min_measure_chunks", c_int32),
                ("grace_chunks", c_int32), ("verify", c_int32),
                ("retry_with_lookahead", c_int32), ("support_only", c_int32)]


class ConfigStats(ctypes.Structure):
    _fields_ = [("models", c_int64), ("attempts", c_int64), ("chunks", c_int64),
                ("kept_rows", c_int64), ("decs", c_int64), ("props", c_int64),
                ("wall_s", ctypes.c_double)]


class SampleStats(ctypes.Structure):
    """Mirror of `gpap_sample_stats`; `sampling_s` is THE clock (delivery bills to `deliver_s`,
    the soundness self-test to `verify_s`)."""
    _fields_ = [("n_models", c_int64), ("attempts", c_int64), ("chunks", c_int64),
                ("n_support_complete", c_int64), ("sound_rows_checked", c_int64),
                ("unsound_rows", c_int64),
                ("best_config", c_int32), ("dominant_config", c_int32), ("stop", c_int32),
                ("timed_out", c_int32), ("retried", c_int32), ("lookahead_used", c_int32),
                ("rounds_max", c_int32),
                ("rounds_mean", ctypes.c_double), ("sampling_s", ctypes.c_double),
                ("deliver_s", ctypes.c_double), ("verify_s", ctypes.c_double)]


#: `gpap_sample_stop` — why a run ended. "unsat" is a PROOF; "zero_yield" only an absence of it.
SAMPLE_STOP = {0: "target", 1: "time", 2: "chunks", 3: "zero_yield", 4: "unsat"}


_I32P, _I64P, _I8P = POINTER(c_int32), POINTER(c_int64), POINTER(c_int8)


def _declare(lib: ctypes.CDLL) -> None:
    sig = [
        ("gpap_formula_from_dimacs", [c_char_p, POINTER(c_void_p)], c_int),
        ("gpap_formula_from_clauses", [c_int32, _I32P, _I32P, c_int32, POINTER(c_void_p)], c_int),
        ("gpap_formula_info_get", [c_void_p, POINTER(FormulaInfo)], None),
        ("gpap_formula_free", [c_void_p], None),
        ("gpap_formula_occ", [c_void_p, _I32P, _I32P, _I32P, _I32P, _I64P], c_int),
        ("gpap_formula_clause_arrays", [c_void_p, _I32P, _I32P], c_int),
        ("gpap_formula_clauses", [c_void_p, _I32P, _I64P], c_int),
        ("gpap_formula_support", [c_void_p, _I32P, _I32P], c_int),
        ("gpap_packed_envelope", [c_int32, c_int32, _I32P], c_int),
        ("gpap_engine_create", [c_void_p, POINTER(EngineOptions), POINTER(c_void_p)], c_int),
        ("gpap_engine_free", [c_void_p], None),
        ("gpap_engine_set_mode", [c_void_p, c_int], c_int),
        ("gpap_engine_mode", [c_void_p], c_int),
        ("gpap_engine_auto", [c_void_p], c_int32),
        ("gpap_engine_shift", [c_void_p], c_int32),
        ("gpap_engine_k", [c_void_p], c_int32),
        ("gpap_engine_num_vars", [c_void_p], c_int32),
        ("gpap_engine_num_clauses", [c_void_p], c_int32),
        ("gpap_engine_num_long", [c_void_p], c_int32),
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
        ("gpap_flags", [c_void_p, _I8P, _I8P, c_int32], c_int),
        ("gpap_num_propagated", [c_void_p, _I32P, c_int32], c_int),
        ("gpap_sat_rows", [c_void_p, _I32P, c_int32, _I32P], c_int),
        ("gpap_packed_rows", [c_void_p, _I32P, c_int32, _I32P], c_int),
        ("gpap_counters_get", [c_void_p, POINTER(Counters)], c_int),
        ("gpap_trail_cap", [c_int32], c_int32),
        ("gpap_row_bytes", [c_int32, c_int32, c_int32], c_int64),
        ("gpap_compute_k_safe", [c_int32, c_int32, c_int64, c_int32], c_int32),
        ("gpap_default_k_cap", [c_int], c_int32),
        ("gpap_mem_available", [c_int, ctypes.c_double], c_int64),
        ("gpap_resident_warps", [c_int], c_int32),
        ("gpap_dispatch_wave", [c_int32, c_int32, c_int], c_int32),
        ("gpap_probe_combinations",
         [c_void_p, _I32P, c_int32, c_int32, POINTER(ProbeOptions), POINTER(c_void_p)], c_int),
        ("gpap_probe_tuples",
         [c_void_p, _I32P, c_int64, c_int32, POINTER(ProbeOptions), POINTER(c_void_p)], c_int),
        ("gpap_probe_count",
         [c_void_p, _I32P, c_int32, c_int32, POINTER(ProbeOptions)], c_int64),
        ("gpap_probe_free", [c_void_p], None),
        ("gpap_probe_stats_get", [c_void_p, POINTER(ProbeStats)], None),
        ("gpap_probe_conflicts", [c_void_p, _I32P, c_int64], c_int),
        ("gpap_free_literals", [c_void_p, _I32P, _I32P], c_int),
        ("gpap_assume_units", [c_void_p, _I32P, c_int32], c_int),
        ("gpap_sampler_create", [c_void_p, c_int64, c_int32, POINTER(c_void_p)], c_int),
        ("gpap_sampler_free", [c_void_p], None),
        ("gpap_sampler_k", [c_void_p], c_int32),
        ("gpap_sampler_set_k", [c_void_p, c_int32], c_int),
        ("gpap_sampler_reseed", [c_void_p, c_int64], c_int),
        ("gpap_sampler_set_support", [c_void_p, _I32P, c_int32], c_int),
        ("gpap_sampler_set_columns", [c_void_p, _I32P, c_int32], c_int),
        ("gpap_sampler_run",
         [c_void_p, c_int64, POINTER(SampleConfig), c_int32, POINTER(SamplerOptions),
          POINTER(SampleStats)], c_int),
        ("gpap_sampler_config_stats", [c_void_p, c_int32, POINTER(ConfigStats)], c_int),
        ("gpap_sampler_bits", [c_void_p, c_int32, POINTER(ctypes.c_uint8), c_int64], c_int),
        ("gpap_last_error", [], c_char_p),
        ("gpap_version", [], c_char_p),
        ("gpap_build_id", [], c_char_p),
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


def _p(arr: np.ndarray, ct):
    return arr.ctypes.data_as(POINTER(ct))


def i32(arr) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.int32)


def version() -> str:
    return load().gpap_version().decode()


def build_id() -> str:
    """Identity of this BUILD: "<version>+<libgpap byte fingerprint>"; key caches on it, never
    on version()."""
    return load().gpap_build_id().decode()


def trail_cap(num_vars: int) -> int:
    """Trail slots one row needs — a function of the VARIABLE count alone (see gpap.h)."""
    return int(load().gpap_trail_cap(int(num_vars)))


def row_bytes(num_long: int, N: int, trail_cap_: int = 0) -> int:
    """Device bytes ONE row of engine state costs — the exact allocation list."""
    return int(load().gpap_row_bytes(int(num_long), int(N), int(trail_cap_)))


def compute_k_safe(num_long: int, N: int, gpu_mem_bytes: float,
                   cap: "int | None" = None) -> int:
    """Largest K fitting `gpu_mem_bytes` — a binding to the one model in cpp/src/calibrate.cpp."""
    return int(load().gpap_compute_k_safe(int(num_long), int(N), int(gpu_mem_bytes),
                                          int(cap or 0)))


def mem_available(device: int = 0, fraction: float = 0.80) -> int:
    """Bytes available for new allocations (free VRAM minus a driver reserve, times fraction)."""
    return int(load().gpap_mem_available(int(device), float(fraction)))


def resident_warps(device: int = 0) -> int:
    return int(load().gpap_resident_warps(int(device)))


def dispatch_wave(num_clauses: int, batch: int, device: int = 0) -> bool:
    """THE wave/scalar dispatch rule; it keys off the clause count, not the batch size."""
    return bool(load().gpap_dispatch_wave(int(num_clauses), int(batch), int(device)))


def default_k_cap(device: int = 0) -> int:
    """The core's default batch cap — one constant on every device (`device` is ignored)."""
    return int(load().gpap_default_k_cap(int(device)))


# The sweep (enumeration, chunking, budget) runs in the core; these only marshal. A `ProbeSweep`
# owns a core handle, so it is a context manager: leaking one leaks its buffer.


class ProbeSweep:

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
                        0 if distinct_vars else 1, 1 if refuse_over_max else 0)


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


class Formula:
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

    def occ(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """(offsets, mid, entries) of the negated-occurrence CSR. Entries of occ[l] below mid[l]
        are BINARY (payload = the other literal's 2N index), the rest LONG (the length>=3 clause
        index); units have none."""
        lib = load()
        n_off, n_cls = c_int32(0), c_int64(0)
        check(lib.gpap_formula_occ(self._h, None, None, None, byref(n_off), byref(n_cls)),
              "occ sizes")
        off = np.empty(n_off.value, dtype=np.int32)
        mid = np.empty(max(n_off.value - 1, 0), dtype=np.int32)
        cls_ = np.empty(n_cls.value, dtype=np.int32)
        check(lib.gpap_formula_occ(self._h, _p(off, c_int32), _p(mid, c_int32),
                                   _p(cls_, c_int32), None, None), "occ read")
        return off, mid, cls_

    def support(self) -> np.ndarray:
        """Independent support from `c ind` lines (0-based, ascending); empty when there is none."""
        lib = load()
        n = c_int32(0)
        check(lib.gpap_formula_support(self._h, None, byref(n)), "support size")
        out = np.empty(n.value, dtype=np.int32)
        if n.value:
            check(lib.gpap_formula_support(self._h, _p(out, c_int32), None), "support")
        return out

    def clauses(self) -> list:
        """The clauses back out of the built formula, list[list[DIMACS literal]]. Rebuilt from the
        occ CSR, so within-clause order is ascending by literal index and a dropped empty clause
        is NOT re-appended (check `info.has_empty_clause`)."""
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
    """Owns a `gpap_engine*`: the packed row state, the launches and the readout. The BASE Python
    wrapper over the ABI, so every readout returns NUMPY — the state lives in library-owned device
    memory. `bcp_engine.BCPEngine` adapts it to torch; `probe.LiteralProbe` uses it directly."""

    # Mirrors gpap_mode in gpap.h — AUTO is 0, so a zeroed options struct is the default.
    MODE_AUTO, MODE_SCALAR, MODE_WAVE = 0, 1, 2
    _MODES = {"auto": MODE_AUTO, "scalar": MODE_SCALAR, "wave": MODE_WAVE}

    def __init__(self, formula: Formula, device: int = 0, mode: "str | None" = None,
                 force_shift: int = 0, vscan: int = 0):
        """`mode=None` means "auto": the CORE picks the kernel on every reset(K) and honours
        $GPAP_KERNEL=scalar|wave there, so the override reaches every consumer. `requested_mode`
        is what the caller asked for, `mode` what the core resolved; pin one only to A/B them."""
        lib = load()
        self.requested_mode = mode or "auto"
        self._f = formula                       # keep the formula alive
        h = c_void_p()
        opts = EngineOptions(device, self._mode_code(self.requested_mode), int(force_shift),
                             int(vscan))
        check(lib.gpap_engine_create(formula.handle, byref(opts), byref(h)), "create engine")
        self._h = h
        self._K = 0
        self._masks: dict = {}

    @property
    def num_vars(self) -> int:
        return int(self._f.info.num_vars)

    @property
    def num_clauses(self) -> int:
        return int(self._f.info.num_clauses)

    @property
    def num_long(self) -> int:
        """Packed-word (length >= 3) clauses: the per-row state width, packed_rows' M."""
        return int(self._f.info.num_long)

    @property
    def shift(self) -> int:
        return int(load().gpap_engine_shift(self._h))

    @property
    def K(self) -> int:
        return int(load().gpap_engine_k(self._h))

    @property
    def mode(self) -> str:
        return "wave" if load().gpap_engine_mode(self._h) == self.MODE_WAVE else "scalar"

    @property
    def auto_dispatch(self) -> bool:
        """Is the CORE still choosing the kernel per reset(K)? Asked of the core, so a local
        set_mode() cannot override a $GPAP_KERNEL pin."""
        return bool(load().gpap_engine_auto(self._h))

    @classmethod
    def _mode_code(cls, mode: str) -> int:
        try:
            return cls._MODES[mode]
        except KeyError:
            raise ValueError(f"unknown kernel mode {mode!r} (scalar|wave|auto)") from None

    def set_mode(self, mode: str) -> None:
        check(load().gpap_engine_set_mode(self._h, self._mode_code(mode)), "set mode")

    @property
    def flush_conflict(self) -> bool:
        return bool(load().gpap_engine_flush_conflict(self._h))

    def reset(self, K: int) -> None:
        check(load().gpap_engine_reset(self._h, int(K)), "reset")
        self._K = int(K)

    def _decisions(self, decisions: np.ndarray, what: str) -> np.ndarray:
        """(K, L) int64, contiguous, with the ROW COUNT checked against the engine's K. The check
        is memory safety: the C side stages K*L int64s from this pointer, so a short array is read
        past its end and the garbage becomes literal indices — one past 2N writes into a later
        row."""
        d = np.ascontiguousarray(decisions, dtype=np.int64)
        if d.ndim == 1:
            d = d.reshape(1, -1)
        if d.ndim != 2:
            raise ValueError(f"{what}: decisions must be (K, L), got shape {d.shape}")
        k = self.K
        if d.shape[0] != k:
            raise ValueError(
                f"{what}: decisions has {d.shape[0]} rows but the engine is sized for K={k} — "
                f"pass one row per assumption set, or reset({d.shape[0]}) first")
        return d

    def apply(self, decisions: np.ndarray) -> None:
        """(K, L) int64 internal literal indices, -1 = padding. ONE fixpoint launch."""
        d = self._decisions(decisions, "apply")
        check(load().gpap_engine_apply(self._h, _p(d, c_int64), d.shape[1]), "apply")

    def apply_dev(self, ptr: int, L: int) -> None:
        check(load().gpap_engine_apply_dev(self._h, ptr, L), "apply(dev)")

    def apply_clean(self, decisions: np.ndarray) -> None:
        """apply(), CLEAN mode: same (K, L) contract, same fixpoint, identical flags and counters,
        but the launch restores its own PACKED rows to the post-flush baseline before exiting, so
        the next reset() skips the K x M broadcast. Use ONLY when nothing reads post-launch state:
        flags and counters stay valid, but sat/soundness readout then sees the BASELINE, assignment
        readout the FIXPOINT, and recache_flush is REFUSED. $GPAP_CLEAN=0 restores apply()
        (docs/findings/CLEAN_MODE_FINDINGS.md)."""
        d = self._decisions(decisions, "apply_clean")
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
        """Same, with `order` already on the device (e.g. a torch tensor's data_ptr), zero-copy."""
        rounds = c_int32(0)
        check(load().gpap_engine_trajectory_dev(self._h, ptr, int(P), int(n_support), int(B),
                                                int(seed), 1 if soft else 0, byref(rounds)),
              "trajectory(dev)")
        return int(rounds.value)

    def recache_flush(self) -> None:
        check(load().gpap_engine_recache_flush(self._h), "recache flush")

    def check_sets(self, lits, offsets, *, chunk: int = 0, assignments: bool = False,
                   sat_count: bool = False) -> tuple:
        """Propagate every assumption set to its BCP fixpoint; the chunk loop runs in the CORE
        (`gpap_check_sets`, cpp/src/check.cpp), so every face shares one validation, chunk model
        and padding rule. `lits` is the sets FLATTENED as DIMACS literals, `offsets` is (n+1,)
        int64 with set i = lits[offsets[i]:offsets[i+1]], `chunk` <= 0 sizes launches with
        `check_chunk_size`. Returns `(conflict (n,), propagated (n,), assignment (n, 2N) int8 or
        None, sat (n, M) int32 or None)`, the last two only when asked for."""
        lits = np.ascontiguousarray(lits, dtype=np.int32).reshape(-1)
        off = np.ascontiguousarray(offsets, dtype=np.int64).reshape(-1)
        n = int(off.size) - 1
        # The core validates offsets' SHAPE but not `lits`' length: that bound is this binding's.
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
        """Assumption sets one launch holds — `gpap_check_chunk_size`, the core's one model."""
        return int(load().gpap_check_chunk_size(self._h, int(n_sets), int(max_lits)))

    # Probing (entry point 1b): enumeration, chunking and the budget live in cpp/src/probe.cpp,
    # so both faces get the same candidate order and "exhausted" semantics.

    def probe_combinations(self, pool, k: int, **kw) -> ProbeSweep:
        """Probe every k-subset of `pool` (DIMACS literals); `pool=None` = every free literal."""
        ptr, n = _pool_arg(pool)
        h = c_void_p()
        check(load().gpap_probe_combinations(self._h, ptr, n, int(k),
                                             byref(_probe_options(**kw)), byref(h)),
              "gpap_probe_combinations")
        return ProbeSweep(h, k)

    def probe_tuples(self, tuples, **kw) -> ProbeSweep:
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
        """Exact k-subset count of `pool` (None = all free literals) WITHOUT generating it."""
        ptr, n = _pool_arg(pool)
        opts = _probe_options(distinct_vars=distinct_vars)
        return int(load().gpap_probe_count(self._h, ptr, n, int(k), ctypes.byref(opts)))

    def free_literals(self) -> np.ndarray:
        """DIMACS literals the unit flush left undetermined, ascending by variable."""
        n = c_int32(0)
        check(load().gpap_free_literals(self._h, None, byref(n)), "gpap_free_literals")
        out = np.empty(int(n.value), dtype=np.int32)
        if n.value:
            check(load().gpap_free_literals(self._h, out.ctypes.data_as(_I32P), byref(n)),
                  "gpap_free_literals")
        return out

    def assume_units(self, lits) -> None:
        """Fold PROVEN units into the cached flush state in place; entailment is assumed."""
        arr = np.ascontiguousarray(lits, dtype=np.int32).reshape(-1)
        if arr.size == 0:
            return
        check(load().gpap_assume_units(self._h, arr.ctypes.data_as(_I32P), int(arr.size)),
              "gpap_assume_units")

    def pair_mask(self, vars_=None) -> int:
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
        """Put an EXPLICIT ascending row list in a slot, so the readout can serve arbitrary rows."""
        r = i32(rows)
        check(load().gpap_selection_set(self._h, int(slot), _p(r, c_int32), r.size),
              "selection set")
        return int(r.size)

    def rows_assignment(self, rows) -> np.ndarray:
        n = self.selection_set(self.SCRATCH_SLOT, rows)
        return self.assignment_rows(self.SCRATCH_SLOT, n)

    def check_sound(self, slot: int = 0, max_rows: int = 0) -> int:
        """Selected rows with a clause at sat_count == 0 (must be 0); `bound` 0 checks all, a
        positive one only that PREFIX."""
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
        out = np.empty((count, 2 * self.num_vars), dtype=np.int8)
        if count:
            check(load().gpap_assignment_rows(self._h, int(slot), _p(out, c_int8), count),
                  "assignments")
        return out

    def flags(self) -> "tuple[np.ndarray, np.ndarray]":
        """(active, conflict) as (K,) bool arrays, sized from the CORE's K (construction runs an
        internal reset(1), so a locally tracked value under-allocates)."""
        k = self.K
        a = np.empty(k, dtype=np.int8)
        c = np.empty(k, dtype=np.int8)
        check(load().gpap_flags(self._h, _p(a, c_int8), _p(c, c_int8), k), "flags")
        return a.astype(bool), c.astype(bool)

    def num_propagated(self) -> np.ndarray:
        out = np.empty(self.K, dtype=np.int32)
        check(load().gpap_num_propagated(self._h, _p(out, c_int32), out.size),
              "implications")
        return out

    def sat_rows(self, rows) -> np.ndarray:
        r = i32(rows)
        out = np.empty((r.size, self.num_clauses), dtype=np.int32)
        check(load().gpap_sat_rows(self._h, _p(r, c_int32), r.size, _p(out, c_int32)), "sat rows")
        return out

    def packed_rows(self, rows) -> np.ndarray:
        """RAW packed words (sat_count | sum << SHIFT), (n, num_long); only length>=3 clauses have
        words. sat_rows masks the sum field off, hiding sum corruption from the CLEAN gate."""
        r = i32(rows)
        out = np.empty((r.size, self.num_long), dtype=np.int32)
        check(load().gpap_packed_rows(self._h, _p(r, c_int32), r.size, _p(out, c_int32)),
              "packed rows")
        return out

    def reset_stats(self) -> "tuple[int, int]":
        """(full broadcasts, lite resets) — the lite count proves CLEAN mode removed broadcasts."""
        full = c_int64(0)
        lite = c_int64(0)
        load().gpap_engine_reset_stats(self._h, byref(full), byref(lite))
        return int(full.value), int(lite.value)

    def flushed_assignment(self) -> np.ndarray:
        out = np.empty(2 * self.num_vars, dtype=np.int8)
        check(load().gpap_engine_flushed_assignment(self._h, _p(out, c_int8)), "flushed assignment")
        return out

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


class Sampler:
    """Owns a `gpap_sampler*` over an Engine: the sampling chunk loop + adaptive config bandit
    (cpp/src/sample.cpp). Marshalling only — the loop, the bandit, the stop valves and the
    delivery/verify accounting live in the core, shared with the C++ face. The engine must outlive
    the sampler and nothing else may drive it meanwhile; the reference held here keeps it alive
    but does NOT order the destructions, which is why `gpap_sampler_free` touches only itself."""

    def __init__(self, engine: Engine, *, seed: int = 0, k_cap: int = 0):
        lib = load()
        self._e = engine                        # the lifetime guarantee (see class docstring)
        h = c_void_p()
        check(lib.gpap_sampler_create(engine._h, int(seed), int(k_cap), byref(h)),
              "create sampler")
        self._h = h
        self._n_cols = engine.num_vars          # kept-row width for the NEXT run
        self._run_cols = self._n_cols           # ...and the width the LAST run gathered under
        self._last: "SampleStats | None" = None

    @property
    def k(self) -> int:
        return int(load().gpap_sampler_k(self._h))

    def set_k(self, k: int) -> None:
        """Pin the batch (the pin_batch primitive); k <= 0 restores the calibrated value."""
        check(load().gpap_sampler_set_k(self._h, int(k)), "sampler set_k")

    def reseed(self, seed: int) -> None:
        check(load().gpap_sampler_reseed(self._h, int(seed)), "sampler reseed")

    def set_support(self, vars_0based) -> None:
        """Phase-1 variables (0-based). None/empty = no support: one phase over every var."""
        if vars_0based is None or len(vars_0based) == 0:
            check(load().gpap_sampler_set_support(self._h, None, 0), "sampler set_support")
            return
        v = i32(vars_0based)
        check(load().gpap_sampler_set_support(self._h, _p(v, c_int32), len(v)),
              "sampler set_support")

    def set_columns(self, cols_0based) -> None:
        """Kept-row columns (0-based, output order). None/empty = every variable."""
        if cols_0based is None or len(cols_0based) == 0:
            check(load().gpap_sampler_set_columns(self._h, None, 0), "sampler set_columns")
            self._n_cols = self._e.num_vars
            return
        c = i32(cols_0based)
        check(load().gpap_sampler_set_columns(self._h, _p(c, c_int32), len(c)),
              "sampler set_columns")
        self._n_cols = len(c)

    def run(self, n_target: int, configs, *, time_limit_s: float = 0.0, max_chunks: int = 0,
            min_measure_chunks: int = 0, grace_chunks: int = 0, keep_cap: int = 0,
            verify: bool = False, retry_with_lookahead: bool = False,
            support_only: bool = False) -> SampleStats:
        """Run the loop. `configs` = [(random_order, batch, soft), ...] is the PORTFOLIO, one entry
        pinning the algorithm (no bandit); the RNG continues across runs (reseed restarts it).
        Returns the core's stats verbatim."""
        arr = (SampleConfig * len(configs))(
            *[SampleConfig(int(bool(c[0])), int(c[1]),
                           int(bool(c[2])) if len(c) > 2 else 0) for c in configs])
        o = SamplerOptions(int(max_chunks), int(keep_cap), float(time_limit_s),
                           int(min_measure_chunks), int(grace_chunks), int(verify),
                           int(retry_with_lookahead), int(support_only))
        st = SampleStats()
        check(load().gpap_sampler_run(self._h, int(n_target), arr, len(configs), byref(o),
                                      byref(st)), "sampler run")
        # Snapshot the column width as the CORE does (gpap_sampler.run_cols_n): it reads kept rows
        # back at `rows * run_cols_n` bytes, so the live mirror would overrun after a narrow.
        self._run_cols = self._n_cols
        self._last = st
        return st

    def config_stats(self, i: int) -> ConfigStats:
        out = ConfigStats()
        check(load().gpap_sampler_config_stats(self._h, int(i), byref(out)),
              "sampler config_stats")
        return out

    def bits(self, i: int = -1, rows: "int | None" = None) -> np.ndarray:
        if rows is None:
            idx = int(self._last.best_config) if (i < 0 and self._last is not None) else max(i, 0)
            rows = int(self.config_stats(idx).kept_rows)
        # zeros, not empty: the core writes min(capacity, kept) rows, so an over-large request
        # would read uninitialized memory in the tail.
        out = np.zeros((int(rows), self._run_cols), dtype=np.uint8)
        if rows:
            check(load().gpap_sampler_bits(self._h, int(i), _p(out, ctypes.c_uint8), int(rows)),
                  "sampler bits")
        return out

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                load().gpap_sampler_free(self._h)
                self._h = None
        except Exception:       # noqa: BLE001 — interpreter shutdown
            pass

"""Batch sizing and kernel dispatch — a BINDING, not a second implementation.

The model lives in `cpp/src/calibrate.cpp` and this module calls it through the ABI. That is
deliberate and it is the lesson of this file's own history: the arithmetic used to be written
out here as well, the two copies drifted, and the symptom of a drifted memory model is not an
error but a silent spill to host RAM (this hardware degrades rather than failing) or a batch a
third the size it should be. If you need to change how K is computed, change the C++ and both
faces move together.

What stays Python: the `"auto"` sentinel and the torch-device plumbing, because those are
Python-API conveniences with no C++ counterpart.

    compute_k_safe / row_bytes / trail_cap   -> cpp/src/calibrate.cpp   (the memory model)
    default_k_cap / resolve_k_cap            -> the batch ceiling
    dispatch_mode                            -> scalar or wave for a formula + batch
"""
import torch

from gpap.engine import _core

# THE batch cap, spelled out here only so `default_k_cap` can answer for a non-CUDA device
# without touching the driver; the number that matters is the one `gpap_default_k_cap` returns,
# and tests assert the two agree. It is device-INDEPENDENT (see cpp/src/calibrate.cpp): the
# same batch on every GPU, so `--k-cap auto` means the same thing on every machine and a
# cross-GPU comparison needs no pinned override.
K_CAP = 262_144


def trail_cap(num_vars: int) -> int:
    """Trail slots one row needs — a function of the VARIABLE count ALONE.

    Not of how many literals are assumed or decided at once: every trail push is deduplicated
    against the assignment bitmap, so a variable contributes at most one entry before the row
    conflicts. There is consequently no ceiling on the decision batch B anywhere in the engine —
    deciding the whole support in one round is a supported configuration, and the one the
    "B >= |support| is exactly uniform" property uses. Proof: cpp/src/calibrate.cpp::gpap_trail_cap."""
    return _core.trail_cap(num_vars)


def row_bytes(num_long: int, N: int) -> int:
    """Device bytes ONE row of engine state costs — the exact allocation list, not an estimate.

    ⚠️ `num_long` is the STATE clause count (`BCPEngine.num_long` / `FormulaInfo.num_long`), NOT
    num_clauses: only length>=3 clauses carry a packed word, since binaries are occ implication
    edges and units live in the flush bitmap. Passing the total is the natural misreading and it
    is SILENT — no error, just 1.78-2.36x fewer rows on circuit and sk_ CNFs.

    A caller that allocates something else per row (a (K, L) decision matrix, say) should add its
    own term to this rather than inventing a whole model."""
    return _core.row_bytes(num_long, N, _core.trail_cap(N))


def gpu_mem_available(device: torch.device, fraction: float = 0.80) -> int:
    """Bytes available for new allocations on `device` (free VRAM minus a driver reserve).

    `fraction` is the ONLY headroom in the sizing path now that the row model is exact: it covers
    readout transients and whatever else shares the card."""
    if device.type != "cuda":
        return int(8e9)                      # conservative fallback for a CPU device
    return _core.mem_available(device.index or 0, fraction)


def compute_k_safe(num_long: int, N: int, gpu_mem_bytes: float,
                   cap: "int | None" = None) -> int:
    """Largest K fitting `gpu_mem_bytes`. `cap` None/0 means uncapped.

    `num_long`, not num_clauses — see row_bytes above for what the misreading costs."""
    return _core.compute_k_safe(num_long, N, gpu_mem_bytes, cap)


def default_k_cap(device: torch.device) -> int:
    """The library's default batch cap — the same number on every device."""
    if device.type != "cuda":
        return K_CAP
    return _core.default_k_cap(device.index or 0)


def resolve_k_cap(k_cap: "int | str | None", device: torch.device) -> "int | None":
    """Resolve the "auto" sentinel to `default_k_cap(device)`; pass ints and None through.

    Three values, and all three are meaningful: "auto" is the library's cap, an int is that cap
    verbatim (what an A/B against a recorded sweep needs), and None means no cap at all — the
    memory model alone decides."""
    return default_k_cap(device) if k_cap == "auto" else k_cap


def resident_warps(device: torch.device) -> int:
    """Warps this GPU can keep resident at once — SMs x (max threads per SM / 32)."""
    if device.type != "cuda":
        return 0
    return _core.resident_warps(device.index or 0)


def dispatch_mode(num_clauses: int, batch: int, device: torch.device) -> str:
    """"wave" or "scalar" for a chunk of `batch` rows of an `num_clauses`-clause formula.

    WAVE hands one row a whole warp, so it pays off once a row HAS work to spread across 32 lanes —
    which is a property of the clause count, not of the batch. The rule and the measurements behind
    it live in the C++ core (`gpap_dispatch_wave`), and are read from there rather than mirrored.

    Both variants share the same state and reach the same fixpoints (BCP is confluent), so this is
    a pure launch-flag choice that can never change an answer.

    NOTE the signature: this used to take `(batch, device)` only, and keying on the batch was a
    PROXY for the clause count that a batch-sizing change invalidated — see the
    history note in cpp/src/calibrate.cpp before considering a batch-only rule again."""
    return "wave" if _core.dispatch_wave(num_clauses, batch, device.index or 0) else "scalar"

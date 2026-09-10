"""Batch sizing and kernel dispatch: a binding, not a second implementation.

The model lives in `cpp/src/calibrate.cpp` and this module calls it through the ABI. Do not
write the arithmetic out here as well: a drifted memory model does not raise, it spills to host
RAM or silently sizes the batch a third too small. Change the C++ and both faces move together.
Python keeps only the `"auto"` sentinel and the torch-device plumbing.

    compute_k_safe / row_bytes / trail_cap   -> cpp/src/calibrate.cpp   (the memory model)
    default_k_cap / resolve_k_cap            -> the batch ceiling
    dispatch_mode                            -> scalar or wave for a formula + batch
"""
import torch

from gpap.engine import _core

# The batch cap, spelled out here only so `default_k_cap` can answer for a non-CUDA device
# without the driver; `gpap_default_k_cap` is authoritative and tests assert the two agree. It is
# device-independent, so `--k-cap auto` means the same thing on every machine.
K_CAP = 262_144


def trail_cap(num_vars: int) -> int:
    """Trail slots one row needs: a function of the variable count alone, not of how many
    literals are assumed or decided at once (every push is deduplicated against the assignment
    bitmap). So there is no ceiling on the decision batch B: deciding the whole support in one
    round is supported, and is what "B >= |support| is exactly uniform" uses. Proof:
    cpp/src/calibrate.cpp::gpap_trail_cap."""
    return _core.trail_cap(num_vars)


def row_bytes(num_long: int, N: int) -> int:
    """Device bytes one row of engine state costs: the exact allocation list, not an estimate.

    `num_long` is the state clause count (`BCPEngine.num_long`), not `num_clauses` — only
    length>=3 clauses carry a packed word. Passing the total is the natural misreading and it is
    silent, costing a large fraction of the rows. A caller that allocates something else per row
    should add its own term rather than inventing a model."""
    return _core.row_bytes(num_long, N, _core.trail_cap(N))


def gpu_mem_available(device: torch.device, fraction: float = 0.80) -> int:
    """Bytes available for new allocations on `device` (free VRAM minus a driver reserve).

    `fraction` is the only headroom in the sizing path, covering readout transients and whatever
    else shares the card. `GPAP_VRAM_BUDGET_MIB` caps the answer (see `vram_budget_cap`) on BOTH
    branches: the core applies it inside `gpap_mem_available` for a real device, and it is
    applied here for the CPU placeholder, which never reaches the core. A ceiling that holds
    only on the branch a given machine happens to take is not a ceiling."""
    if device.type != "cuda":
        placeholder = int(8e9)               # conservative fallback for a CPU device
        cap = vram_budget_cap()
        return min(placeholder, cap) if cap else placeholder
    return _core.mem_available(device.index or 0, fraction)


#: The `GPAP_VRAM_BUDGET_MIB` grammar, mirroring `env_budget_cap` in cpp/src/calibrate.cpp
#: character for character: optional blanks, an optional '+', digits, nothing else. It is narrow
#: on purpose. C's strtoll takes the longest numeric PREFIX, so a shared value like "1e3" parsed
#: as 1 there and as nothing in Python — the engine capped to 1 MiB while the rival ran uncapped,
#: which is the two-sides-capped-apart failure the cap exists to prevent. Outside the grammar,
#: and past what the shift can hold, the answer is NO CAP rather than a small one.
_CAP_GRAMMAR = __import__("re").compile(r"^[ \t]*\+?([0-9]+)[ \t]*$")
_CAP_MAX_MIB = ((1 << 63) - 1) >> 20


def vram_budget_cap_mib() -> int:
    """The ceiling in MiB, or 0 for "no cap". Gated against the C reader by tests/test_k_cap.py:
    a divergence here caps one side of a comparison and not the other."""
    import os
    m = _CAP_GRAMMAR.match(os.environ.get("GPAP_VRAM_BUDGET_MIB") or "")
    if not m:
        return 0
    mib = int(m.group(1))
    return mib if 0 < mib <= _CAP_MAX_MIB else 0


def vram_budget_cap() -> int:
    """The `GPAP_VRAM_BUDGET_MIB` ceiling in bytes, or 0 when unset — read here only to REPORT
    it (`gpap info`, a result row's provenance). The clamp itself is applied once, in
    `gpap_mem_available`; do not re-apply it in a caller."""
    return vram_budget_cap_mib() << 20


def compute_k_safe(num_long: int, N: int, gpu_mem_bytes: float,
                   cap: "int | None" = None) -> int:
    """Largest K fitting `gpu_mem_bytes`; `cap` None/0 means uncapped. `num_long`, not
    `num_clauses` — see row_bytes for what the misreading costs."""
    return _core.compute_k_safe(num_long, N, gpu_mem_bytes, cap)


def default_k_cap(device: torch.device) -> int:
    """The library's default batch cap — the same number on every device."""
    if device.type != "cuda":
        return K_CAP
    return _core.default_k_cap(device.index or 0)


def resolve_k_cap(k_cap: "int | str | None", device: torch.device) -> "int | None":
    """Resolve the "auto" sentinel to `default_k_cap(device)`; pass ints and None through. All
    three are meaningful: the library's cap, that cap verbatim (what an A/B against a recorded
    sweep needs), and no cap at all."""
    return default_k_cap(device) if k_cap == "auto" else k_cap


def resident_warps(device: torch.device) -> int:
    """Warps this GPU can keep resident at once — SMs x (max threads per SM / 32)."""
    if device.type != "cuda":
        return 0
    return _core.resident_warps(device.index or 0)


def dispatch_mode(num_clauses: int, batch: int, device: torch.device) -> str:
    """"wave" or "scalar" for a chunk of `batch` rows of a `num_clauses`-clause formula.

    WAVE hands one row a whole warp, so it pays off once a row has work to spread across the
    lanes, which is a property of the clause count and not of the batch. Do not key this on the
    batch alone: that is a proxy a batch-sizing change invalidates. The rule lives in
    `gpap_dispatch_wave` and is read from there, never mirrored. Both variants reach the same
    fixpoints (BCP is confluent), so this is a pure launch-flag choice."""
    return "wave" if _core.dispatch_wave(num_clauses, batch, device.index or 0) else "scalar"

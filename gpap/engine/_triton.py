"""Is a working Triton JIT available on this host?

This is a BUILD-CAPABILITY probe, not a runtime requirement. The engine runs from the cubins
already embedded in libgpap (see cpp/kernels/), so a production host needs `libcuda.so.1` and
nothing else — use `gpap.engine.core_available()` to ask whether the engine can run.

What this answers is the different question "could I rebuild the kernels here?", which is what
`gpap info` reports and what `scripts/build_core.sh` needs. It is deliberately NOT the gate for
the GPU test suite: those tests exercise the C core, so they gate on `core_available()`.
"""

_avail_cache: "bool | None" = None


def kernel_available() -> bool:
    """True iff triton is importable AND a trivial kernel compiles+runs on this GPU (cached)."""
    global _avail_cache
    if _avail_cache is not None:
        return _avail_cache
    ok = False
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001 — any import failure just means "unavailable"
        _avail_cache = False
        return False

    @triton.jit
    def _probe_kernel(x_ptr):
        tl.store(x_ptr + tl.program_id(0), 1)

    if torch.cuda.is_available():
        try:
            z = torch.zeros(4, dtype=torch.int32, device=torch.device("cuda"))
            _probe_kernel[(4,)](z)
            torch.cuda.synchronize()
            ok = bool((z == 1).all().item())
        except Exception:  # noqa: BLE001 — any compile/launch failure = unavailable
            ok = False
    _avail_cache = ok
    return ok

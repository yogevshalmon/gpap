"""How much device memory GPAP may hold and must leave free: the memory limit and the memory
reserve (docs/ENGINE.md §7).

The rule, the ledger and the one parser of a limit live in the core (cpp/src/memory.cpp); this
is the Python face over them. Importing it loads nothing: the core is opened on first use.
"""
import os


def set_mem_limit(spec, *, device=None) -> None:
    """Cap the device memory GPAP's engines hold, summed over engines.

    `spec` is a size with its unit — "8G", "512M", "1.5G", "8GiB", "8GB" (binary units: 1G is
    1024**3 bytes) — a share of the device such as "75%", "none", or bytes as an int.

    With `device=None` the limit is the process default, `GPAP_MEM_LIMIT`: every device, and
    every child process, follows it; `spec=None` removes it. With a device number it applies to
    that device of the current backend only and overrides the default there; `spec=None` hands
    the device back to the default.

    The limit is a ceiling. On CUDA GPAP also stays inside free memory; on OpenCL, which cannot
    measure free memory, the limit takes the place of the default share of the total. A batch
    that fits no row under it raises `MemoryError` (`gpap.engine._core.GpapMemoryError`).
    Raises ValueError, before anything changes, for a malformed spec.
    """
    from gpap.engine import _core
    if device is None:
        if spec is None:
            os.environ.pop("GPAP_MEM_LIMIT", None)
            return
        text = _core._spec(spec).decode()
        _core.parse_mem_limit(text)                 # raises before anything changes
        os.environ["GPAP_MEM_LIMIT"] = text
        return
    _core.set_mem_limit(spec, int(device))


def set_mem_reserve(spec, *, device=None) -> None:
    """Set the device memory GPAP leaves free beside everything it holds, for what it cannot
    count: other processes and the desktop, your own tensors on the device.

    `spec` has `set_mem_limit`'s grammar ("1G", "512M", "5%" of the device, bytes as an int), or
    "none" for no reserve. Without one the library's default applies (`memory()["reserve"]`).
    The default batch is sized from free memory less the reserve, and on CUDA an allocation that
    would leave less than half of it free raises `MemoryError` instead of spilling to host
    memory, which WSL2 would do silently.

    With `device=None` it is the process default, `GPAP_MEM_RESERVE`, which child processes
    inherit; `spec=None` removes it. With a device number it applies to that device of the
    current backend only; `spec=None` hands the device back to the default. Raises ValueError,
    before anything changes, for a malformed spec.
    """
    from gpap.engine import _core
    if device is None:
        if spec is None:
            os.environ.pop("GPAP_MEM_RESERVE", None)
            return
        text = _core._spec(spec).decode()
        _core.parse_mem_limit(text)                 # one grammar: raises before anything changes
        os.environ["GPAP_MEM_RESERVE"] = text
        return
    _core.set_mem_reserve(spec, int(device))


def memory(device: int = 0) -> dict:
    """GPAP's view of one device's memory now, on the current backend:

    - `limit`: the limit in force, in bytes (0 = none), and `limit_source`: "GPAP_MEM_LIMIT",
      "set_mem_limit" or None;
    - `reserve`: what GPAP leaves free there, in bytes;
    - `held`: what GPAP's engines hold there, counted buffer by buffer, and `peak`: the most they
      held since `reset_peak`;
    - `available`: what a new engine's rows would be sized against.
    """
    from gpap.engine import _core
    return {"limit": _core.mem_limit(device), "limit_source": _core.mem_limit_source(device),
            "reserve": _core.mem_reserve(device),
            "held": _core.mem_held(device), "peak": _core.mem_peak(device),
            "available": _core.mem_available(device)}


def reset_peak(device: int = 0) -> None:
    """Start `memory(device)["peak"]` again from what GPAP holds now."""
    from gpap.engine import _core
    _core.mem_reset_peak(device)

"""Generate a link-only stub of libcuda, for building libgpap where no NVIDIA driver exists.

CI runners have no driver, but `scripts/build_core.sh` must still be provable on a clean machine
— that is the whole "no CUDA toolkit, no nvcc, driver-only" claim in the README. A shared library
tolerates undefined symbols, so `libgpap.so` itself links against anything; the EXAMPLES do not,
and that is what fails without this.

The symbol list is DERIVED from the sources rather than hard-coded, so adding a driver call to
the core cannot silently break the CI build. It doubles as an executable inventory of exactly how
much CUDA the core is coupled to (see docs/findings/PORTABILITY.md).

    python cpp/tools/make_libcuda_stub.py /tmp/libcuda.so [--print]

The stub is for LINKING ONLY. Every entry point returns CUDA_ERROR_NOT_INITIALIZED (3), so if a
binary built against it is ever actually run, it fails loudly instead of pretending to work.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CPP = Path(__file__).resolve().parent.parent
SYMBOL = re.compile(r"\bcu[A-Z][A-Za-z0-9_]*")


def driver_symbols() -> list[str]:
    """Every `cu*` driver entry point referenced by the core sources.

    The PUBLIC headers are scanned too, not just src/. `include/gpap/gpap.hpp` is header-only and
    is compiled into every C++ consumer — including the examples this stub exists to let CI link
    — so a driver call added there would break the driverless build while the scan said nothing,
    which is exactly the failure deriving the list is meant to prevent.

    A `cu*` name appearing only in a COMMENT is counted. That is the same conservative direction
    the `_v2`/`_v3` aliasing below takes: an unused symbol in a link stub costs nothing, a
    missing one breaks the build."""
    found: set[str] = set()
    sources = (sorted((CPP / "src").glob("*.cpp")) + sorted((CPP / "src").glob("*.hpp"))
               + sorted((CPP / "include" / "gpap").glob("*.h*")))
    for path in sources:
        found.update(SYMBOL.findall(path.read_text()))
    return sorted(found)


def linked_symbols(names: list[str]) -> list[str]:
    """The names the LINKER will actually look for.

    `cuda.h` macro-remaps a number of entry points to ABI-versioned symbols
    (`#define cuMemAlloc cuMemAlloc_v2`), so a source scan sees `cuMemAlloc` while the object
    file references `cuMemAlloc_v2`. Which ones are remapped is a driver-header detail we do not
    want to hard-code and track, so emit every variant — an unused symbol in a link stub costs
    nothing, a missing one breaks the build."""
    out: list[str] = []
    for n in names:
        out += [n, f"{n}_v2", f"{n}_v3"]
    return out


def emit_c(symbols: list[str]) -> str:
    body = "\n".join(f"int {s}() {{ return 3; }}" for s in symbols)
    return ("/* GENERATED link-only libcuda stub — see cpp/tools/make_libcuda_stub.py.\n"
            " * NOT a CUDA implementation: every entry point returns CUDA_ERROR_NOT_INITIALIZED,\n"
            " * so anything built against this fails loudly rather than pretending to work. */\n"
            + body + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", type=Path, help="path of the libcuda.so to write")
    ap.add_argument("--print", action="store_true", help="print the symbol list and exit")
    args = ap.parse_args()

    symbols = driver_symbols()
    if args.print or not args.out:
        print(f"{len(symbols)} CUDA driver entry points used by the core:")
        for s in symbols:
            print("  " + s)
        return 0

    emitted = linked_symbols(symbols)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "libcuda_stub.c"
        src.write_text(emit_c(emitted))
        rc = subprocess.run(["gcc", "-shared", "-fPIC", "-o", str(args.out), str(src)])
        if rc.returncode != 0:
            return rc.returncode
    print(f"wrote {args.out}: {len(symbols)} driver entry points "
          f"({len(emitted)} symbols incl. ABI-versioned aliases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

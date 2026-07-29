#!/usr/bin/env bash
# Build libgpap — the GPAP engine core — in one command.
#
#   scripts/build_core.sh              # compile kernels for THIS GPU, then the library
#   scripts/build_core.sh --arch 80    # cross-compile the kernels for sm_80 (no GPU needed)
#   scripts/build_core.sh --install    # also copy the .so into gpap/lib/ so Python finds it
#                                      #   without the repo build directory
#
# Requirements: a C++17 compiler, cmake, the NVIDIA driver (libcuda), and python+triton
# (build time only — it compiles the kernels ahead of time). NO CUDA toolkit, no nvcc.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${GPAP_BUILD_DIR:-$REPO/build}"
PYTHON="${PYTHON:-}"
ARCH=""
INSTALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH="$2"; shift 2 ;;
        --install) INSTALL=1; shift ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# Prefer the repo's virtualenv, then whatever python has triton.
if [[ -z "$PYTHON" ]]; then
    for cand in "$REPO/.venv/bin/python" python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import triton" >/dev/null 2>&1; then
            PYTHON="$cand"; break
        fi
    done
fi
if [[ -z "$PYTHON" ]]; then
    echo "error: no python with triton found (needed to compile the kernels)." >&2
    echo "       pip install triton, or set PYTHON=/path/to/python" >&2
    exit 1
fi

echo "==> compiling kernels ahead of time ($($PYTHON -c 'import triton;print("triton "+triton.__version__)'))"
"$PYTHON" "$REPO/cpp/tools/build_kernels.py" ${ARCH:+--arch "$ARCH"}

echo "==> configuring"
cmake -S "$REPO/cpp" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null

echo "==> building"
cmake --build "$BUILD" -j "$(nproc 2>/dev/null || echo 4)"

if [[ "$INSTALL" == "1" ]]; then
    mkdir -p "$REPO/gpap/lib"
    cp -a "$BUILD"/libgpap.so* "$REPO/gpap/lib/"
    echo "==> installed -> gpap/lib/libgpap.so"
fi

echo
echo "libgpap: $BUILD/libgpap.so"
echo "try it:  $BUILD/gpap_demo             # C, the raw ABI"
echo "         $BUILD/gpap_sample f.cnf     # C++, gpap::Sampler"
echo "         $BUILD/gpap_assumptions      # C++, gpap::AssumptionChecker"
echo "         $PYTHON -c 'from gpap.engine import core_available; print(core_available())'"

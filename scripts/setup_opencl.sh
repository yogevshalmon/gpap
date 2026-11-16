#!/usr/bin/env bash
# Install an OpenCL runtime for the OpenCL backend (GPAP_BACKEND=opencl), without root.
#
#   scripts/setup_opencl.sh cpu          # PoCL's CPU driver, from PyPI: checks the kernels'
#                                        #   correctness on any machine (what CI runs)
#   scripts/setup_opencl.sh cuda         # PoCL with its CUDA driver, from conda-forge: OpenCL
#                                        #   on an NVIDIA GPU whose driver exposes none (WSL2)
#   scripts/setup_opencl.sh env [MODE]   # print the `export` lines for an installed runtime
#                                        #   (MODE = cpu | cuda; default: cuda if installed)
#   scripts/setup_opencl.sh oclgrind CMD # run CMD on the Oclgrind simulator (installed by the
#                                        #   cuda mode): reports out-of-bounds accesses and
#                                        #   data races in the kernels. Slow; small inputs.
#
# Use the result in a shell with:   source <(scripts/setup_opencl.sh env)
#
# Not needed where the system already has OpenCL for the GPU: on native Linux the NVIDIA driver
# ships it, and the system ICD loader (libOpenCL.so.1) is all libgpap looks for. Under WSL2 the
# driver exposes CUDA only, which is what the `cuda` mode works around.
#
# Everything lands under GPAP's disposable-machinery root (`python3 -m gpap.paths home`);
# deleting it loses only time. docs/OPENCL.md is the guide.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="$(cd "$REPO" && python3 -m gpap.paths home)"
VENV="$HOME_DIR/opencl-cpu"          # mode cpu: a virtualenv holding PoCL and an ICD loader
PREFIX="$HOME_DIR/opencl-cuda"       # mode cuda: a conda environment
MAMBA="$HOME_DIR/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME_DIR/mamba"     # its package cache, under the same root
# The versions the kernels were checked with (docs/findings/OPENCL_FINDINGS.md). A new PoCL is
# a new compiler for bcp.cl: change these deliberately, and re-run the backend's tests.
CPU_PACKAGES="pocl-binary-distribution==3.0 pyopencl==2026.1.5"
CUDA_PACKAGES="pocl=7.2 pocl-cuda=7.2 ocl-icd=2.3.4 clinfo=3.0.21.02.21 oclgrind=21.10"
MAMBA_URL="https://github.com/mamba-org/micromamba-releases/releases/download/2.10.0-0/micromamba-linux-64"
MAMBA_SHA256="701f39c06f8e6a670a0e186c8479b9e5c0502a2f0cda6cef65d83bdd2875e8bd"
# Two things PoCL's CUDA driver needs that the conda packages do not arrange:
VENDORS="$PREFIX/etc/OpenCL/gpap-vendors"                  # an ICD directory with PoCL ALONE
LIBDEVICE="$PREFIX/nvvm/libdevice/libdevice.10.bc"         # NVIDIA's libdevice bitcode

# A python that has triton, as scripts/build_core.sh finds one: its wheel ships libdevice.
triton_python() {
    local cand
    for cand in "${PYTHON:-}" "$REPO/.venv/bin/python" python3 python; do
        if [[ -n "$cand" ]] && command -v "$cand" >/dev/null 2>&1 \
                && "$cand" -c "import triton" >/dev/null 2>&1; then
            echo "$cand"; return 0
        fi
    done
    return 1
}

# Local file work only, safe to repeat.
finish_cuda() {
    # 1. PoCL alone in the loader. Another driver in the same process that carries its own
    #    LLVM (Oclgrind, installed beside it, does) corrupts PoCL's compiler: it aborts on
    #    every kernel. Oclgrind is reached through the `oclgrind` mode instead.
    mkdir -p "$VENDORS"
    cp "$PREFIX/etc/OpenCL/vendors/pocl.icd" "$VENDORS/pocl.icd"
    # 2. libdevice. It ships with the CUDA toolkit, which GPAP does not need; the triton wheel
    #    carries the same file.
    if [[ ! -f "$LIBDEVICE" ]]; then
        local py src
        py="$(triton_python)" || {
            echo "error: no python with triton found (PoCL's CUDA driver needs triton's" >&2
            echo "       libdevice.10.bc). pip install triton, or set PYTHON=/path/to/python" >&2
            exit 1
        }
        src="$("$py" -c 'import os, triton; print(os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "lib", "libdevice.10.bc"))')"
        mkdir -p "$(dirname "$LIBDEVICE")"
        cp "$src" "$LIBDEVICE"
    fi
}

print_env() {
    # libgpap opens $GPAP_OPENCL_LIB (else the system loader); the loader finds drivers through
    # OCL_ICD_FILENAMES (one library) or OCL_ICD_VENDORS (a directory of .icd files).
    local mode="${1:-}"
    if [[ -z "$mode" ]]; then
        if [[ -f "$VENDORS/pocl.icd" ]]; then mode=cuda; else mode=cpu; fi
    fi
    echo "export POCL_CACHE_DIR=\"$HOME_DIR/pocl-cache\""   # its compiled kernels, here too
    if [[ "$mode" == "cuda" ]]; then
        if [[ ! -f "$VENDORS/pocl.icd" || ! -f "$LIBDEVICE" ]]; then
            echo "the cuda runtime is not installed: run scripts/setup_opencl.sh cuda" >&2
            exit 1
        fi
        echo "export GPAP_OPENCL_LIB=\"$PREFIX/lib/libOpenCL.so.1\""
        echo "export OCL_ICD_VENDORS=\"$VENDORS\""
        echo "export POCL_CUDA_TOOLKIT_PATH=\"$PREFIX\""
        # PoCL's CUDA driver loads libcuda itself; under WSL2 it is not on the default path.
        echo "export LD_LIBRARY_PATH=\"$PREFIX/lib:/usr/lib/wsl/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
        return
    fi
    local libs loader pocl
    libs="$(echo "$VENV"/lib/python*/site-packages/pyopencl/.libs)"
    loader="$(echo "$libs"/libOpenCL-*.so*)"
    pocl="$(echo "$libs"/libpocl-*.so*)"
    if [[ ! -f "$loader" || ! -f "$pocl" ]]; then
        echo "the cpu runtime is not installed: run scripts/setup_opencl.sh cpu" >&2
        exit 1
    fi
    echo "export GPAP_OPENCL_LIB=\"$loader\""
    echo "export OCL_ICD_FILENAMES=\"$pocl\""
    # PoCL 3.0's default work-group method (its loop vectorizer) miscompiles the WAVE kernels
    # for an AVX-512 processor: they write their local arrays through a bad pointer and the
    # process dies. Work-item replication does not (docs/findings/OPENCL_FINDINGS.md §15).
    echo "export POCL_WORK_GROUP_METHOD=\"repl\""
}

case "${1:-}" in
    cpu)
        # pyopencl is installed for the ICD loader its wheel bundles, not for Python use.
        python3 -m venv "$VENV"
        # shellcheck disable=SC2086
        "$VENV/bin/pip" install --quiet $CPU_PACKAGES
        ;;
    cuda)
        if [[ ! -f "$PREFIX/etc/OpenCL/vendors/pocl.icd" ]]; then
            mkdir -p "$HOME_DIR/bin"
            if [[ ! -x "$MAMBA" ]]; then
                echo "==> fetching micromamba"
                curl -fsSL -o "$MAMBA.part" "$MAMBA_URL"
                echo "$MAMBA_SHA256  $MAMBA.part" | sha256sum -c --quiet
                chmod +x "$MAMBA.part"
                mv "$MAMBA.part" "$MAMBA"
            fi
            echo "==> installing PoCL (CUDA and CPU drivers), the ICD loader, clinfo, Oclgrind"
            # shellcheck disable=SC2086
            "$MAMBA" create -y -p "$PREFIX" -c conda-forge $CUDA_PACKAGES
        fi
        finish_cuda
        ;;
    env)
        print_env "${2:-}"
        exit 0
        ;;
    oclgrind)
        shift
        if [[ ! -x "$PREFIX/bin/oclgrind" || $# -eq 0 ]]; then
            echo "usage: scripts/setup_opencl.sh oclgrind COMMAND...  (after the cuda mode)" >&2
            exit 2
        fi
        # Oclgrind is its own OpenCL library, not a driver behind the loader. No
        # --uninitialized: that plugin of Oclgrind 21.10 aborts on LLVM's `freeze`.
        # --global-mem-size: its default device (128 MiB) is below the engine's memory reserve,
        # so every batch the core sizes would be one row and no launch would hold two rows; 2 GiB
        # (4 GiB and above wrap to 0 in 21.10).
        # Its reports go to a file, not to stderr, where a test runner would swallow them; a
        # report fails the run whatever CMD returned.
        TMP_DIR="$(cd "$REPO" && python3 -m gpap.paths tmp)"
        mkdir -p "$TMP_DIR"
        LOG="$TMP_DIR/oclgrind-$$.log"
        rm -f "$LOG"
        status=0
        GPAP_BACKEND=opencl GPAP_OPENCL_LIB="$PREFIX/lib/liboclgrind-rt.so" \
            LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$PREFIX/bin/oclgrind" --data-races --global-mem-size 2147483648 --log "$LOG" "$@" \
            || status=$?
        if [[ -s "$LOG" ]]; then
            echo "Oclgrind REPORTED ($(grep -c 'Kernel:' "$LOG") kernel events); the first lines:" >&2
            head -n 20 "$LOG" >&2
            echo "full log: $LOG" >&2
            exit 1
        fi
        rm -f "$LOG"
        echo "Oclgrind: no report"
        exit "$status"
        ;;
    *)
        sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        exit 2
        ;;
esac

echo
echo "OpenCL runtime installed. Use it in this shell with:"
echo "    source <(scripts/setup_opencl.sh env $1)"
echo "then check that libgpap sees a device:"
echo "    GPAP_BACKEND=opencl gpap info"

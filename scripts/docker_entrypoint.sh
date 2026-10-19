#!/usr/bin/env bash
# The tool image's entry point (Dockerfile). It makes sure a libgpap compiled for the visible GPU
# exists, then runs the command:
#
#   docker run --gpus all gpap sample f.cnf    # not a command on PATH: runs `gpap sample f.cnf`
#   docker run --gpus all -it gpap bash        # a command on PATH runs as itself
#
# WHY IT COMPILES. The kernels are compiled ahead of time for one GPU architecture and the engine
# refuses any other (cpp/src/runtime.cpp), so no image can carry one library for every card. The
# first run on a GPU cross-compiles for that GPU's architecture (`build_core.sh --arch`, which
# needs no driver to compile) into $GPAP_CACHE/<build key>/sm_<cc>, in under a minute, and
# exports GPAP_BUILD_DIR, the directory the loader reads. A volume at $GPAP_CACHE keeps it. The
# build key hashes the engine's sources and the Triton version when the image was built, so a
# newer image never loads a library an older one compiled into the same volume.
#
# GPAP_SKIP_BUILD=1 skips the compile, to get a shell in an image whose compile fails.
#
# Sourced, this file only defines its functions: the artifact image's entry point uses them for
# the steps that run on the GPU.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GPAP_CACHE:?the image sets GPAP_CACHE}"
KEY_FILE=/opt/gpap-image/build-key

# The visible GPU's compute capability as two digits (89), or nothing without one. It asks torch,
# which honours CUDA_VISIBLE_DEVICES as the engine does, so it names the device the engine uses.
visible_cc() {
    python3 -c '
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}{minor}")' 2>/dev/null
}

# Prints the compute capability of a GPU the engine can run on; otherwise says why, and fails.
usable_cc() {
    local cc
    cc="$(visible_cc)"
    if [ -z "$cc" ]; then
        # The toolkit mounts nvidia-smi with the GPU, so its answer separates "no GPU was given
        # to the container" from "a GPU is here and CUDA cannot use it", which is a driver too
        # old for the image's CUDA.
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
            echo "gpap: a GPU is visible but CUDA cannot use it; is the driver older than CUDA" \
                 "12.8 needs? (docs/DOCKER.md, When something goes wrong)" >&2
        else
            echo "gpap: no NVIDIA GPU in this container; commands that run the engine need --gpus all" >&2
        fi
        return 1
    fi
    # The requirement docs/DOCKER.md states: Ampere or newer.
    if [ "$cc" -lt 80 ]; then
        echo "gpap: this GPU is sm_$cc; GPAP needs compute capability 8.0 or newer" >&2
        return 1
    fi
    echo "$cc"
}

ensure_core() {
    local cc key dir
    cc="$(usable_cc)" || return 0           # said why; a command that needs no GPU still runs
    key="$(cat "$KEY_FILE")" || return 1
    dir="$GPAP_CACHE/$key/sm_$cc"
    # One volume may serve root and other users (`--user`), so what the compile creates there,
    # Triton's cache included, is writable by all; the command itself runs under umask 022.
    umask 000
    mkdir -p "$dir" || return 1
    exec 9>"$dir.lock" && flock 9           # two containers on one volume compile once
    if [ ! -f "$dir/.complete" ]; then
        echo "gpap: compiling the engine for this GPU (sm_$cc), under a minute" >&2
        rm -f "$dir"/libgpap.so*            # an interrupted link must not pass for a library
        if ! GPAP_BUILD_DIR="$dir" bash "$REPO/scripts/build_core.sh" --arch "$cc" \
                >"$dir/build.log" 2>&1; then
            echo "gpap: the compile failed; the end of $dir/build.log:" >&2
            tail -20 "$dir/build.log" >&2
            echo "gpap: GPAP_SKIP_BUILD=1 starts the container without compiling" >&2
            exec 9>&-
            return 1
        fi
        touch "$dir/.complete"
    fi
    exec 9>&-
    export GPAP_BUILD_DIR="$dir"
}

[ "${BASH_SOURCE[0]}" = "$0" ] || return 0

if [ "${GPAP_SKIP_BUILD:-0}" != 1 ]; then
    ensure_core || exit 1
fi
umask 022
# A program, not a shell builtin: `help` or `test` would be exec'd and fail instead of reaching gpap.
if [ $# -gt 0 ] && type -P -- "$1" >/dev/null 2>&1; then
    exec "$@"
fi
exec gpap "$@"

#!/usr/bin/env bash
# The OpenCL backend on a Colab GPU, against the GPU vendor's own OpenCL driver: what the device
# is, the test suites on both backends, then the CUDA-vs-OpenCL A/B on probing work
# (eval/profiling/backend_ab.py). GPAP only, no rival. Every part writes a file and is skipped
# when its file is already there, and the A/B resumes row by row, so the run survives
# colab_supervisor.py's VM churn.
#
#   bash eval/colab/colab_opencl_l4.sh                       # everything (hours on an L4)
#   SKIP_TESTS=1 ONLY="seq/s386 seq/b14" bash eval/colab/colab_opencl_l4.sh
#   ARMS="cuda-wave-clean opencl-wave-clean" WIDTH=3 BATCH=0 bash eval/colab/colab_opencl_l4.sh
#
# The VM needs OpenCL beside CUDA: `apt-get install ocl-icd-libopencl1 clinfo`, and, on an image
# whose /etc/OpenCL/vendors is empty, the NVIDIA driver registered by hand:
#   echo "$(ldconfig -p | awk '/libnvidia-opencl\.so\.1 /{print $NF; exit}')" \
#       > /etc/OpenCL/vendors/nvidia.icd
# Part 1 records what `clinfo -l` then lists.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # this file lives in eval/colab/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
PY="${PYTHON:-python3}"
EXPERIMENTS="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m eval.paths experiments)"
: "${EXPERIMENTS:?cannot resolve — python3 must be able to import the repo (gpap/paths.py, eval/paths.py)}"
OUT="${OUT:-$EXPERIMENTS/opencl_l4}"
mkdir -p "$OUT"

echo "########## OPENCL BACKEND  $(date -Is)"

# ---- 1. the device, as each layer sees it
if [ ! -s "$OUT/device.txt" ]; then
    {
        echo "== nvidia-smi";  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        echo "== clinfo -l";   clinfo -l
        echo "== OpenCL facts (clinfo)"
        clinfo | grep -E "Platform Name|Device Name|Device Version|Device OpenCL C Version|Max work group size|Max memory allocation|Global memory size|Device Extensions" | sort -u
        echo "== gpap info, CUDA backend";   "$PY" -m gpap.cli info
        echo "== gpap info, OpenCL backend"; GPAP_BACKEND=opencl "$PY" -m gpap.cli info
    } > "$OUT/device.tmp" 2>&1
    mv "$OUT/device.tmp" "$OUT/device.txt"
fi

# ---- 2. the test suites, once: the backend's own tests, then everything on each backend
if [ "${SKIP_TESTS:-0}" != "1" ] && [ ! -s "$OUT/correctness.json" ]; then
    "$PY" -m pytest -p no:cacheprovider tests/engine/test_opencl_backend.py -q -rs \
        > "$OUT/tests_opencl_backend.txt" 2>&1;  rc_own=$?
    "$PY" -m pytest -p no:cacheprovider -q > "$OUT/tests_cuda.txt" 2>&1;  rc_cuda=$?
    GPAP_BACKEND=opencl "$PY" -m pytest -p no:cacheprovider -q > "$OUT/tests_opencl.txt" 2>&1;  rc_ocl=$?
    "$PY" - "$OUT" "$rc_own" "$rc_cuda" "$rc_ocl" <<'PYSUM'
import json, sys
out, codes = sys.argv[1], [int(x) for x in sys.argv[2:]]
names = ("tests_opencl_backend", "tests_cuda", "tests_opencl")
res = {}
for name, rc in zip(names, codes):
    lines = [ln.rstrip() for ln in open(f"{out}/{name}.txt") if ln.strip()]
    res[name] = {"exit": rc, "last_line": lines[-1] if lines else "",
                 "failed": [ln.split(" - ")[0] for ln in lines if ln.startswith(("FAILED", "ERROR"))]}
json.dump(res, open(f"{out}/correctness.json", "w"), indent=1)
print(json.dumps(res, indent=1))
PYSUM
fi

# ---- 3. the A/B. ARMS names the arms (default: all eight); WIDTH the literals per tuple;
#         BATCH one batch for every arm, or 0 for each arm's own.
only=()
[ -n "${ONLY:-}" ] && only=(--only ${ONLY})
arms=()
[ -n "${ARMS:-}" ] && arms=(--arms ${ARMS})
"$PY" eval/profiling/backend_ab.py --out "$OUT/backend_ab.json" --resume \
    --cycles "${CYCLES:-10}" --target "${TARGET:-rare}" --batch "${BATCH:-65536}" \
    --width "${WIDTH:-2}" --reps "${REPS:-3}" --target-s "${TARGET_S:-1.0}" \
    --max-chunks "${MAX_CHUNKS:-256}" --max-clauses "${MAX_CLAUSES:-2000000}" \
    "${arms[@]}" "${only[@]}" \
    || echo "[warn] backend_ab.py returned non-zero"
echo "OPENCL_L4_COMPLETE"

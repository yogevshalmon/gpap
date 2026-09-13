#!/usr/bin/env bash
# WAVE vs SCALAR propagate-kernel A/B on a Colab L4 — GPAP only (eval/kernel_ab.py): every
# benchmark family, both circuit encoders, T=25 rare targets, pinned b8 config, mining off,
# validation-then-timing per cell. One resumable JSON, so it survives colab_supervisor.py's
# VM churn (rows are keyed by (family, instance, encoder) and written after every cell).
#
#   bash eval/colab/colab_kernel_ab_l4.sh                  # full matrix (~5 h on an L4)
#   CYCLES=25 ENCODERS=tseitin bash eval/colab/colab_kernel_ab_l4.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # this file lives in eval/colab/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
PY="${PYTHON:-python3}"
EXPERIMENTS="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m eval.paths experiments)"
: "${EXPERIMENTS:?cannot resolve — python3 must be able to import the repo (gpap/paths.py, eval/paths.py)}"
OUT="${OUT:-$EXPERIMENTS/kernel_ab}"
mkdir -p "$OUT"

echo "########## KERNEL A/B (wave vs scalar)  $(date -Is)"
$PY eval/kernel_ab.py --out "$OUT/kernel_ab.json" --resume \
    --cycles "${CYCLES:-25}" --encoders ${ENCODERS:-tseitin abc:fast} --target "${TARGET:-rare}" \
    --portfolio "${PORTFOLIO:-b8}" --time-limit "${TIME_LIMIT:-20}" --reps "${REPS:-2}" \
    || echo "[warn] kernel_ab.py returned non-zero"
echo "KERNEL_AB_COMPLETE"

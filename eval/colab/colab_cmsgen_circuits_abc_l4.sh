#!/usr/bin/env bash
# GPAP vs CMSGen — the CIRCUIT families only (seq / comb / epfl / hwmcc; no sk), ABC ENCODING
# ONLY (`abc:fast`). The same stages, budgets and portfolio as colab_cmsgen_l4.sh, so rows are
# protocol-comparable to the full-matrix campaigns; the two deliberate restrictions are the
# family list and ENCODERS. Resumable under eval/colab/colab_supervisor.py exactly like the
# full driver: every stage JSON is written after each row and --resume skips finished cells.
#
# The CMSGen preflight is a HARD fail on purpose (review of 2026-09-02): a stage resumed
# without the binary would append armless rows to an armed file — `gpap bench --resume` now
# refuses that mix, so a soft warn here would burn a VM producing nothing.
#
#   bash eval/colab/colab_cmsgen_circuits_abc_l4.sh
#   TARGETS=rare CYCLES=25 bash eval/colab/colab_cmsgen_circuits_abc_l4.sh   # headline slice
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # this file lives in eval/colab/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PYTHON="${PYTHON:-python3}"

export ENCODERS="${ENCODERS:-abc:fast}"
export TARGETS="${TARGETS:-rare random}"
export CYCLES="${CYCLES:-25 50}"
export N_TIME="${N_TIME:-50000}"
export TIME_LIMIT="${TIME_LIMIT:-60}"
# Its own OUT: this matrix is a subset of the full one, and sharing a directory with a
# full-matrix run would let --resume interleave rows measured under different ENCODERS.
export OUT="${OUT:-$REPO_ROOT/.experiments/full_eval/nvidia-l4-abc}"
export CMSGEN="${CMSGEN:-$REPO_ROOT/.gpap/bin/cmsgen}"

if [ ! -x "$CMSGEN" ]; then
    echo "[FATAL] no cmsgen at $CMSGEN — this sweep IS the CMSGen comparison." >&2
    exit 1
fi
if ! "$CMSGEN" --help 2>&1 | grep -q -- --nowrite; then
    echo "[FATAL] cmsgen lacks scripts/cmsgen_gpap_eval.patch: it would be timed WITH sample" >&2
    echo "        emission and every ratio would be inflated." >&2
    exit 1
fi
if ! "$CMSGEN" --help 2>&1 | grep -q -- --maxtime; then
    echo "[FATAL] cmsgen predates --maxtime — a killed run reports a FLOORED count." >&2
    echo "        Rebuild: scripts/build_cmsgen.sh" >&2
    exit 1
fi

# seq first: the flagship family, and the run may be cut short (rare targets already lead
# inside each stage because TARGETS lists them first).
for stage in seq comb epfl hwmcc; do
    echo; echo "########## STAGE $stage  $(date -Is)"
    bash eval/run_full_eval.sh "$stage" || echo "[warn] stage $stage returned non-zero"
    echo "STAGE_${stage}_DONE"
done

echo "CMSGEN_CIRCUITS_ABC_COMPLETE"

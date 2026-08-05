#!/usr/bin/env bash
# GPAP vs CMSGen across every benchmark family, on a Colab L4.
#
# This is `run_full_eval.sh`'s five benchmark stages (no `rivals` — that is a separate sweep) in
# ONE resumable command, ordered so the headline lands first. Every stage already writes its JSON
# after each row and skips finished cells on re-run, which is what makes it safe under
# eval/colab_supervisor.py: a Colab L4 assignment is reclaimed after ~1 h, so this script will be
# started several times on several VMs and must pick up where the last one stopped.
#
# ORDER MATTERS because the run may be cut short: `rare` targets at T=25 are the headline regime
# (at `random` pins CPU rejection sampling beats every CNF sampler, ours included), so the stages
# run rare-first and the cheap families go first within that.
#
#   bash eval/colab_cmsgen_l4.sh              # full matrix
#   TARGETS=rare CYCLES=25 bash eval/...      # the headline slice only
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PYTHON="${PYTHON:-python3}"

export TARGETS="${TARGETS:-rare random}"
export CYCLES="${CYCLES:-25 50}"
export N_TIME="${N_TIME:-50000}"
export TIME_LIMIT="${TIME_LIMIT:-60}"
export CMSGEN="${CMSGEN:-/root/.cache/gpap/bin/cmsgen}"

if [ ! -x "$CMSGEN" ]; then
    echo "[FATAL] no cmsgen at $CMSGEN — this sweep IS the CMSGen comparison." >&2
    exit 1
fi
if ! "$CMSGEN" --help 2>&1 | grep -q -- --nowrite; then
    echo "[FATAL] cmsgen lacks scripts/cmsgen_gpap_eval.patch: it would be timed WITH sample" >&2
    echo "        emission (~88% of its wall on 27.sk_3_32) and every ratio would be inflated." >&2
    exit 1
fi

for stage in sk comb seq epfl hwmcc; do
    echo; echo "########## STAGE $stage  $(date -Is)"
    bash eval/run_full_eval.sh "$stage" || echo "[warn] stage $stage returned non-zero"
    echo "STAGE_${stage}_DONE"
done

echo "CMSGEN_SWEEP_COMPLETE"

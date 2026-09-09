#!/usr/bin/env bash
# GPAP-ONLY re-measurement of the CIRCUIT families on a Colab L4 — no CMSGen, no rivals.
#
# Why this campaign exists: the sampling loop moved into libgpap after the last full L4
# campaign, so every recorded GPAP circuit number predates the engine that ships. This re-runs
# only the GPAP arm of the circuit matrix under the identical protocol, so new-vs-old GPAP is a
# clean engine A/B (eval/recheck_compare.py) and the tables can be regenerated pairing it with
# the carried CMSGen and rival arms, whose numbers did not change. Seed-era note
# (docs/BENCHMARKING.md §3): older counts are not same-seed-comparable, so the compare gates on
# the deterministic fields and success rate, not counts.
#
# Protocol pinned to the 20260827 campaign: tseitin+abc:fast, rare+random, T=25+50,
# n_time 50k, 60 s budget, VERIFY_SOUND=1 (run_full_eval.sh's campaign default), seed 5.
# TARGET-MAJOR order, rare first: a rented VM dies at ~1 h and `rare` is the regime we publish.
#
#   bash eval/colab/colab_gpap_circuits_l4.sh            # full circuit matrix (~3-4 h GPAP-only)
#   TARGETS=rare CYCLES=25 bash eval/colab/colab_gpap_circuits_l4.sh   # the headline slice
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # this file lives in eval/colab/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PYTHON="${PYTHON:-python3}"

export CYCLES="${CYCLES:-25 50}"
export N_TIME="${N_TIME:-50000}"
export TIME_LIMIT="${TIME_LIMIT:-60}"
# GPAP-only, deliberately: pointing CMSGEN at a path that cannot exist makes run_full_eval.sh
# warn and omit the arm on every stage. This campaign is tabled with the earlier CMSGen rows.
export CMSGEN=/nonexistent-gpap-only
# A fixed OUT, not the GPU slug: the supervisor mirrors these filenames, and a slug surprise on
# a new VM image must not silently retarget the sweep.
export OUT="${OUT:-$REPO_ROOT/.experiments/full_eval/nvidia-l4-gpaponly}"

"$PYTHON" -m gpap.cli info 2>/dev/null | head -8 || true

TARGETS_LIST="${TARGETS:-rare random}"
for t in $TARGETS_LIST; do
    for stage in seq comb epfl hwmcc; do
        echo; echo "########## STAGE $stage TARGET $t  $(date -Is)"
        TARGETS="$t" bash eval/run_full_eval.sh "$stage" \
            || echo "[warn] stage $stage/$t returned non-zero"
        echo "STAGE_${stage}_${t}_DONE"
    done
done

echo "GPAP_CIRCUITS_COMPLETE"

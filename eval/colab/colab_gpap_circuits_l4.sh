#!/usr/bin/env bash
# GPAP-ONLY re-measurement of the CIRCUIT families on a Colab L4 — no CMSGen, no rivals.
#
# Why this campaign exists: the sampling loop moved into libgpap (2026-09-01, `gpap_sampler_*`,
# seed era re-baselined) after the last full L4 campaign (l4_campaign_20260827, repo c33ed7b),
# so every recorded GPAP circuit number predates the engine that now ships. This sweep re-runs
# ONLY the GPAP arm of the circuit matrix under the identical protocol, so that:
#   * new-vs-old GPAP is a clean engine A/B (eval/recheck_compare.py against the 20260827
#     campaign's gpap_vs_cmsgen_l4 — deterministic fields must match, rates are the result);
#   * the tables can be regenerated pairing this GPAP data with the CARRIED CMSGen and rival
#     arms (their numbers did not change; re-running them re-measures nothing new).
# ⚠️ Seed-era note (docs/BENCHMARKING.md §3): counts recorded before 2026-09-01 are not
# same-seed-comparable to these rows; the compare gates on M/support/split/mode + succ, not counts.
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
# GPAP-ONLY, deliberately: pointing CMSGEN at a path that cannot exist makes run_full_eval.sh
# print its "[warn] no cmsgen" line and omit the --cmsgen arm on every stage. The CMSGen and
# rival numbers this campaign will be tabled WITH are the 20260827 campaign's.
export CMSGEN=/nonexistent-gpap-only
# A fixed OUT (not the GPU slug): the supervisor mirrors these filenames, and a slug surprise
# on a new VM image must not silently retarget the sweep.
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

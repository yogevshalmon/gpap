#!/usr/bin/env bash
# The uniformity half of the experiment section: is what we emit actually uniform, what does the
# `soft` config do to that, and what does excluding `soft` cost in throughput.
#
# Two experiments in one script because they answer two halves of one question: `soft` was
# dropped from the default circuit portfolio because Barbarik rejects it, which is defensible
# only if we can state the price.
#
#   1. fuzz_uniformity.py — GPAP (several configs including soft), the shipped pipelines and
#      CMSGen, on instances whose entire solution set is enumerated, so the target distribution
#      is known and chi-square has real power. Both controls are included: without them the
#      instrument is uncalibrated and "no skew detected" means nothing.
#   2. profile_soft_portfolio.py — the throughput cost of dropping `soft`, both arms in one
#      process on the same encoded CNF with the arm order alternated, because the SM clock
#      swings mid-run and any other method measures the clock.
#
# Both write JSON; eval/artifact/paper_tables.py turns them into the paper's uniformity tables.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # this file lives in eval/colab/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
PY="${PYTHON:-python3}"

EXPERIMENTS="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m eval.paths experiments)"
: "${EXPERIMENTS:?cannot resolve — python3 must be able to import the repo (eval/paths.py)}"
OUT="${OUT:-$EXPERIMENTS/uniformity}"
mkdir -p "$OUT"
# Ask, never recompute: gpap/paths.py is the one definition (see its docstring).
GPAP_HOME="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m gpap.paths home)"
: "${GPAP_HOME:?cannot resolve — python3 must be able to import the repo (gpap/paths.py, eval/paths.py)}"
CMSGEN="${CMSGEN:-$GPAP_HOME/bin/cmsgen}"
CMS_ARGS=()
[ -x "$CMSGEN" ] && CMS_ARGS=(--cmsgen "$CMSGEN") \
    || echo "[warn] no cmsgen — uniformity runs GPAP-only (no head-to-head column)" >&2

COUNT="${COUNT:-72}"          # instances; 3 seeds each, so ~216 sampler-runs per config
SEEDS="${SEEDS:-3}"

echo "########## UNIFORMITY FUZZ (exact oracle)  $(date -Is)"
# configs: b1/b8 are the shipped ones, `soft` is the opt-in under test, and b16/b32/b64 walk the
# B ladder, which becomes exactly uniform once B >= |support| (rejection sampling).
# pipelines: what a user gets from `gpap sample`, with and without --allow-soft.
$PY eval/uniformity/fuzz_uniformity.py \
    --count "$COUNT" --kinds cnf circuit --seeds "$SEEDS" \
    --configs b1 b8 soft b16 b32 b64 \
    --pipelines default circuit circuit+soft \
    --controls 0.0 0.05 0.10 \
    ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
    --out "$OUT/uniformity_campaign.json" 2>&1 | tail -40 \
    || echo "[warn] fuzz_uniformity returned non-zero (unsound emission or a FAILED negative control)"
echo "UNIFORMITY_FUZZ_DONE"

echo "########## SOFT PORTFOLIO PRICE  $(date -Is)"
$PY eval/profiling/profile_soft_portfolio.py \
    --glob 'benchmarks/circuits/bench/seq/*.bench' \
    --encoder abc --abc-opt fast --cycles 25 --target rare \
    --reps 2 --out "$OUT/soft_profile.json" 2>&1 | tail -40 \
    || echo "[warn] profile_soft_portfolio returned non-zero"
echo "SOFT_PROFILE_DONE"

echo "UNIFORMITY_SUITE_COMPLETE"

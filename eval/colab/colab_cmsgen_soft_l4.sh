#!/usr/bin/env bash
# The UNIFORMITY-PRICE arm: the entire GPAP-vs-CMSGen matrix re-run with `--allow-soft`.
#
# `soft` is Barbarik-rejected on nearly every circuit instance, so it is out of the default
# portfolio and opt-in: every published number already runs without it. This sweep states the
# price — how much throughput, and how many instances, the uniformity guarantee costs in the
# head-to-head we publish. (`eval/profile_soft_portfolio.py` answers the narrower GPAP-vs-GPAP
# version.) Same protocol, budgets, encoders, targets, instances and code path, one bit changed.
# Pair the two output directories per (stage, instance, encoder) with:
#
#     eval/artifact/soft_price_tables.py --safe <no-soft dir> --soft <this dir>
#
# Rare-first, like the safe arm: a rented VM dies within the hour, so the stages most likely to
# reach publication run first.
#
#   bash eval/colab/colab_cmsgen_soft_l4.sh                 # full matrix
#   TARGETS=rare CYCLES=25 bash eval/colab/colab_cmsgen_soft_l4.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # this file lives in eval/colab/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PYTHON="${PYTHON:-python3}"

export TARGETS="${TARGETS:-rare random}"
export CYCLES="${CYCLES:-25 50}"
export N_TIME="${N_TIME:-50000}"
export TIME_LIMIT="${TIME_LIMIT:-60}"
GPAP_HOME="${GPAP_HOME:-$REPO_ROOT/.gpap}"      # tool root: one spelling, run_full_eval.sh's
export CMSGEN="${CMSGEN:-$GPAP_HOME/bin/cmsgen}"

# THE one changed bit, and a separate output tree so the two arms cannot overwrite each other.
export EXTRA_ARGS="${EXTRA_ARGS:---allow-soft}"
export OUT="${OUT:-$REPO_ROOT/.experiments/full_eval/nvidia-l4-soft}"

if [ ! -x "$CMSGEN" ]; then
    echo "[FATAL] no cmsgen at $CMSGEN — this sweep IS the CMSGen comparison." >&2
    exit 1
fi
if ! "$CMSGEN" --help 2>&1 | grep -q -- --nowrite; then
    echo "[FATAL] cmsgen lacks scripts/cmsgen_gpap_eval.patch: it would be timed WITH sample" >&2
    echo "        emission (~88% of its wall on 27.sk_3_32) and every ratio would be inflated." >&2
    exit 1
fi

echo "UNIFORMITY-PRICE ARM: EXTRA_ARGS='$EXTRA_ARGS' -> $OUT"

# Target-major order: every family at `rare` first, then every family at `random`. Stage-major
# order would spend the first hours on random pins before touching any family's rare stage, and
# a rented VM can take the budget away at any point. `rare` is the regime we publish, and `soft`
# never won the bandit at random pins, so a truncated run still answers the question.
for t in rare random; do
    for stage in seq comb epfl sk hwmcc; do
        echo; echo "########## STAGE $stage TARGET $t  $(date -Is)"
        TARGETS="$t" bash eval/run_full_eval.sh "$stage" \
            || echo "[warn] stage $stage/$t returned non-zero"
        echo "STAGE_${stage}_${t}_DONE"
    done
done

echo "CMSGEN_SOFT_SWEEP_COMPLETE"

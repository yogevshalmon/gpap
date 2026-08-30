#!/usr/bin/env bash
# The UNIFORMITY-PRICE arm: the entire GPAP-vs-CMSGen matrix re-run with `--allow-soft`.
#
# What this measures, and why it needs a whole second sweep.
#
# `soft` (soft global lookahead) is Barbarik-REJECTED on 64 of 65 circuit instances, so it was
# removed from CIRCUIT_PORTFOLIO and is now opt-in. The shipped sampler — and every
# number in `eval/colab/colab_cmsgen_l4.sh`'s output — therefore already runs WITHOUT it (`allow_soft:
# false` is recorded in each of those JSONs' `args`). What we could not previously state is the
# PRICE: how much throughput, and how many instances, the uniformity guarantee costs us in the
# head-to-head we actually publish.
#
# `eval/profile_soft_portfolio.py` answers a narrower version of that question (GPAP vs GPAP, 19
# sequential circuits, 0.42x geomean). It cannot answer this one, because "we gave up 2.4x" and
# "we gave up our lead over CMSGen" are different claims, and only the second one matters to a
# reader of the comparison table.
#
# So: same protocol, same budgets, same encoders, same targets, same instances, same code path
# (`EXTRA_ARGS` is injected into run_full_eval.sh's single `bench` function) — one bit changed.
# Pair the two output directories per (stage, instance, encoder) with:
#
#     eval/artifact/soft_price_tables.py --safe <no-soft dir> --soft <this dir>
#
# Ordered rare-first for the same reason as the safe arm: a rented VM dies at ~1 h, so the stages
# most likely to survive to publication must run first.
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
export CMSGEN="${CMSGEN:-${GPAP_HOME:-/root/.gpap}/bin/cmsgen}"

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

# TARGET-MAJOR order: every family at `rare` first, then every family at `random`.
#
# The stage-major order (all of seq, then all of comb, ...) spends the first ~2 h of a ~5 h sweep
# on sequential RANDOM pins before touching any other family's rare stage — and a rented VM can
# take the budget away at any point. `rare` is the regime we publish, and measured here `soft`
# never once won the bandit at random pins (0/8 cells), so the random half is also the half that
# carries least information about what `soft` is worth. Rare-first means a truncated run still
# answers the question.
for t in rare random; do
    for stage in seq comb epfl sk hwmcc; do
        echo; echo "########## STAGE $stage TARGET $t  $(date -Is)"
        TARGETS="$t" bash eval/run_full_eval.sh "$stage" \
            || echo "[warn] stage $stage/$t returned non-zero"
        echo "STAGE_${stage}_${t}_DONE"
    done
done

echo "CMSGEN_SOFT_SWEEP_COMPLETE"

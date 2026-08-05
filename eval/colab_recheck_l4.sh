#!/usr/bin/env bash
# CONSISTENCY RE-CHECK: re-measure instances we already have numbers for, on a FRESH L4 VM.
#
# The point is not more coverage — it is asking whether the numbers already collected are
# reproducible on a different machine of the same type, after ~8 h of sweeps across many VMs.
# Two things it can catch that a single sweep cannot:
#   * a result that depended on one VM's state (a bad build, a warm cache, a noisy neighbour)
#   * a silent protocol drift between the code that produced the old rows and today's tree
#
# It writes to its OWN directory, never the sweep's, so a re-check can never be mistaken for
# (or resumed as) the original measurement. `eval/recheck_compare.py` does the diff.
#
# Everything below is pinned to match the original sweep exactly: same seed, same budgets, same
# encoders, same portfolio. The ONE thing deliberately left free is the GPU itself.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
PY="${PYTHON:-python3}"

OUT="${OUT:-$REPO_ROOT/.experiments/recheck}"
SCRATCH="$OUT/cnf"
mkdir -p "$OUT" "$SCRATCH"
CMSGEN="${CMSGEN:-/root/.cache/gpap/bin/cmsgen}"

CMS_ARGS=()
if [ -x "$CMSGEN" ] && "$CMSGEN" --help 2>&1 | grep -q -- --nowrite; then
    CMS_ARGS=(--cmsgen "$CMSGEN")
else
    echo "[warn] no patched cmsgen — GPAP-side re-check only" >&2
fi

# A deliberately MIXED sample: the flagship sequential family, the combinational family, the
# projected-CNF family, and EPFL (our weakest, where succ==0 cells live). Small and large M.
SEQ="${SEQ:-b03 b04 b12 s1196}"
COMB="${COMB:-c432 c1355 c3540}"
EPFL="${EPFL:-ctrl dec i2c adder}"
SK_GLOB="${SK_GLOB:-benchmarks/sampling/27.sk_3_32.cnf}"

# Stage the chosen instances as symlinks and glob THAT directory. `--glob` goes to Python's
# glob.glob, which has no brace/extglob syntax, so a shell pattern would silently match nothing.
STAGE="$OUT/stage"
rm -rf "$STAGE"; mkdir -p "$STAGE/seq" "$STAGE/comb" "$STAGE/epfl"
for i in $SEQ;  do ln -sf "$REPO_ROOT/benchmarks/circuits/bench/seq/$i.bench"  "$STAGE/seq/";  done
for i in $COMB; do ln -sf "$REPO_ROOT/benchmarks/circuits/bench/comb/$i.bench" "$STAGE/comb/"; done
for i in $EPFL; do ln -sf "$REPO_ROOT/benchmarks/circuits/epfl/$i.aig"         "$STAGE/epfl/"; done
seq_glob="$STAGE/seq/*.bench"
comb_glob="$STAGE/comb/*.bench"
epfl_glob="$STAGE/epfl/*.aig"
echo "staged: $(ls "$STAGE"/seq "$STAGE"/comb "$STAGE"/epfl | tr '\n' ' ')"

run() {   # run <name> <glob> <extra args...>
    local name="$1" glob="$2"; shift 2
    echo; echo "########## RECHECK $name  $(date -Is)"
    $PY -m gpap.benchmark --glob "$glob" \
        --n-time 50000 --time-limit 60 --n-cov 0 --scratch "$SCRATCH" \
        --output "$OUT/$name.json" --resume --tag "recheck-$name" \
        ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} "$@" 2>&1 | tail -30
}

run seq-c25-rare  "$seq_glob"  --encoders tseitin abc:fast --cycles 25 --target rare --portfolio circuit
run comb-rare     "$comb_glob" --encoders tseitin abc:fast --cycles 1 --pin-outputs 1 --target rare --portfolio circuit
run epfl-rare     "$epfl_glob" --encoders tseitin abc:fast --cycles 1 --target rare --portfolio circuit
run sk            "$SK_GLOB"   --portfolio default

echo "RECHECK_COMPLETE"

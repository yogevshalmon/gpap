#!/usr/bin/env bash
# GPAP full experimental sweep — built to run in a TERMINAL (Colab's notebook UI disconnects long
# cells; a detached shell process survives). Every stage is independently invocable and RESUMABLE:
# results are appended to a per-stage JSON after every row, and re-running the stage skips what is
# already in it. One instance blowing up never loses the rest.
#
#   # once per machine
#   eval/rivals_setup.sh
#
#   # then, one family at a time (each prints a live table; add & + nohup to detach)
#   eval/run_full_eval.sh seq
#   nohup eval/run_full_eval.sh all > full_eval.log 2>&1 &   ;  tail -f full_eval.log
#
# Stages
#   check    environment/readiness table (no runs)
#   smoke    every stage, 1-2 instances each, tiny budgets (~minutes) — "does this machine run
#            the whole pipeline?" Run it FIRST on a new box; the numbers are meaningless.
#   seq      ITC'99 b* + ISCAS-89 s* sequential unrolls  — the flagship CircuitSAT family
#   comb     ISCAS-85 c* combinational (T=1, single pinned output = the DEMOTIC-style setup)
#   sk       benchmarks/sampling/*.cnf — projected CNF sampling vs CMSGen (no encoder axis)
#   epfl     EPFL combinational AIGs (T=1, all outputs pinned)
#   hwmcc    HWMCC'17 single-property AIGs (T=CYCLES, the witness-sampling instances)
#   rivals   DEMOTIC / Recurrent-CircuitSAT / HTS head-to-heads on the same GPU
#   plots    figures + summary tables from whatever JSONs exist
#   all      every stage above, in order
#
# Every circuit stage runs the full matrix: encoders x targets x unroll depths, i.e.
#   ENCODERS="tseitin abc:fast"   TARGETS="rare random"   CYCLES="25 50"
# (one `gpap bench` invocation per target x depth; both encoders run inside it, back to back, so
# the A/B pair is measured in the same process on the same GPU state.) Override any of them:
#   CYCLES=25 TARGETS=rare eval/run_full_eval.sh seq
#   N_TIME=100000 TIME_LIMIT=120 eval/run_full_eval.sh sk
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export PYTHONPATH="${PYTHONPATH:-.}"

GPAP_HOME="${GPAP_HOME:-$HOME/.cache/gpap}"
CMSGEN="${CMSGEN:-$GPAP_HOME/bin/cmsgen}"
GPU_SLUG="$($PY -c "
import re, torch
n = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'cpu'
print(re.sub(r'[^a-z0-9]+', '-', n.lower()).strip('-'))" 2>/dev/null || echo unknown-gpu)"
OUT_EXPLICIT="${OUT:-}"          # remember whether the caller chose the directory (smoke needs it)
OUT="${OUT:-$REPO_ROOT/.experiments/full_eval/$GPU_SLUG}"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

ENCODERS="${ENCODERS:-tseitin abc:fast}"
TARGETS="${TARGETS:-rare random}"
CYCLES="${CYCLES:-25 50}"
N_TIME="${N_TIME:-50000}"
TIME_LIMIT="${TIME_LIMIT:-60}"
N_COV="${N_COV:-0}"                 # >0 adds the quality pass (2-wise + distinct); costs a 2nd pass
SEEDS="${SEEDS:-}"                  # e.g. "5 6 7" -> median-of-3 (docs/BENCHMARKING.md §3)
MAX_CLAUSES="${MAX_CLAUSES:-4000000}"
SCRATCH="${SCRATCH:-$OUT/cnf}"
STAGE="${1:-all}"
mkdir -p "$SCRATCH"

# Per-family instance globs (overridable; `smoke` trims them to a couple of fast instances).
GLOB_SEQ="${GLOB_SEQ:-benchmarks/circuits/bench/seq/*.bench}"
GLOB_COMB="${GLOB_COMB:-benchmarks/circuits/bench/comb/*.bench}"
GLOB_SK="${GLOB_SK:-benchmarks/sampling/*.cnf}"
GLOB_EPFL="${GLOB_EPFL:-benchmarks/circuits/epfl/*.aig}"
GLOB_HWMCC="${GLOB_HWMCC:-benchmarks/circuits/hwmcc17/*.aig}"

if [ "$STAGE" = "smoke" ]; then
    # A few minutes that touch EVERY stage: does this machine run the whole pipeline at all?
    # Deliberately not a measurement — budgets are far too small for any number to mean anything.
    N_TIME=2000; TIME_LIMIT=5; N_COV=0; CYCLES="5"; TARGETS="rare"; SEEDS=""
    GLOB_SEQ="benchmarks/circuits/bench/seq/b0[36].bench"
    GLOB_COMB="benchmarks/circuits/bench/comb/c432.bench"
    GLOB_SK="benchmarks/sampling/27.sk_3_32.cnf"
    GLOB_EPFL="benchmarks/circuits/epfl/ctrl.aig"
    GLOB_HWMCC="benchmarks/circuits/hwmcc17/texasifetch1p8.aig"
    RIVAL_SEQ="${RIVAL_SEQ:-b03}"; RIVAL_COMB="${RIVAL_COMB:-c432}"
    RIVAL_SEEDS="${RIVAL_SEEDS:-5}"; RIVAL_T="${RIVAL_T:-5}"; RIVAL_STEPS="${RIVAL_STEPS:-1}"
    # MUST land somewhere else: several stage files share a name with the real run (sk.json,
    # comb-rare.json, ...) and --resume would then treat 5-second smoke rows as finished work.
    [ -n "$OUT_EXPLICIT" ] || OUT="$REPO_ROOT/.experiments/full_eval/${GPU_SLUG}-smoke"
    LOGS="$OUT/logs"; SCRATCH="$OUT/cnf"; mkdir -p "$OUT" "$LOGS" "$SCRATCH"
    echo "SMOKE run (tiny budgets, 1-2 instances per family) -> $OUT"
fi

CMS_ARGS=()
if [ -x "$CMSGEN" ]; then CMS_ARGS=(--cmsgen "$CMSGEN"); else
    echo "[warn] no cmsgen at $CMSGEN — running GPAP-only (scripts/build_cmsgen.sh)" >&2
fi
SEED_ARGS=(); [ -n "$SEEDS" ] && SEED_ARGS=(--seeds $SEEDS)

banner() { echo; echo "=============== $* ==============="; date -Is; }

# PREFLIGHT. The engine is a compiled library now and has no Python fallback, so a missing
# libgpap does not fail the sweep — it fails every ROW of it, quietly, for hours. Check once.
# (`check` is exempt: reporting what is missing is exactly its job.)
if [ "$STAGE" != "check" ]; then
    if ! "$PY" -c "import sys; from gpap.engine import core_available; sys.exit(0 if core_available() else 1)" 2>/dev/null; then
        echo "[FATAL] libgpap (the engine) is not available — every GPAP row would error." >&2
        echo "        Build it:  scripts/build_core.sh" >&2
        echo "        Details:   eval/rivals_setup.sh check" >&2
        exit 1
    fi
fi

# bench <name> <glob> [extra gpap-bench args...]
bench() {
    local name="$1" pattern="$2"; shift 2
    local json="$OUT/$name.json" log="$LOGS/$name.log"
    banner "$name  ->  $json"
    # ${arr[@]+"${arr[@]}"} — an empty array under `set -u` is an error on older bash otherwise
    "$PY" -m gpap.benchmark --glob "$pattern" \
        --n-time "$N_TIME" --time-limit "$TIME_LIMIT" --n-cov "$N_COV" \
        --max-clauses "$MAX_CLAUSES" --scratch "$SCRATCH" \
        --output "$json" --resume --tag "$name" \
        ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} "$@" 2>&1 | tee -a "$log"
}

stage_check() { bash "$REPO_ROOT/eval/rivals_setup.sh" check; }

# Circuit stages: one bench run per (target, depth); both encoders inside each run.
stage_seq() {
    for t in $TARGETS; do for c in $CYCLES; do
        bench "seq-c${c}-${t}" "$GLOB_SEQ" \
            --encoders $ENCODERS --cycles "$c" --target "$t" --sim-baseline --portfolio circuit
    done; done
}

stage_comb() {
    # ISCAS-85: T=1 and ONE pinned output — pinning all outputs of a combinational block
    # over-constrains it to a near-point solution set (benchmarks/circuits/README.md), and the
    # single pin is what DEMOTIC's own setup does.
    for t in $TARGETS; do
        bench "comb-${t}" "$GLOB_COMB" \
            --encoders $ENCODERS --cycles 1 --pin-outputs 1 --target "$t" \
            --sim-baseline --portfolio circuit
    done
}

stage_sk() {
    # Projected CNF sampling (`c ind`): no encoder axis — these are already CNF. This is the
    # reference point vs CMSGen, so it uses the sk_ portfolio.
    bench "sk" "$GLOB_SK" --portfolio default --cmsgen-both
}

stage_epfl() {
    for t in $TARGETS; do
        bench "epfl-${t}" "$GLOB_EPFL" \
            --encoders $ENCODERS --cycles 1 --target "$t" --sim-baseline --portfolio circuit
    done
}

stage_hwmcc() {
    for t in $TARGETS; do for c in $CYCLES; do
        bench "hwmcc-c${c}-${t}" "$GLOB_HWMCC" \
            --encoders $ENCODERS --cycles "$c" --target "$t" --sim-baseline --portfolio circuit
    done; done
}

stage_rivals() {
    local seq_insts="${RIVAL_SEQ:-b03 b04 b10 b11 b12 b13 s386 s832 s1196}"
    local comb_insts="${RIVAL_COMB:-c432 c499 c880 c1355 c1908 c2670 c3540 c5315 c6288 c7552}"
    # ONE seed for the GPAP side here (the bench stages are where multi-seed medians matter).
    # MEASURED: the rivals stage cost ~10 min/row, of which the rival subprocess was ~2.7 min —
    # the rest was GPAP re-sampling at 3 seeds. These ratios span 2-5 orders of magnitude, so
    # seed noise (+-40% worst case, docs/BENCHMARKING.md §3) cannot flip any of them.
    local rseeds="${RIVAL_SEEDS:-5}"
    local T="${RIVAL_T:-25}"
    # Circuit VARIANTS: the rivals consume .bench and unroll it themselves, so when GPAP is
    # measured on an ABC-optimized circuit they are handed the same optimized circuit (written
    # back as a primitive-gate .bench and equivalence-checked) — never raw-vs-optimized.
    local cvars="${RIVAL_VARIANTS:-raw abc}"
    local rsteps="${RIVAL_STEPS:-10}"
    # Rival subprocess cap: 300 s = 5x the 60 s sampling budget GPAP gets, because the rival's
    # wall includes the PyTorch model build that its OWN metric excludes. NOT 60 s: measured on
    # this suite, 18/18 Recurrent-CircuitSAT runs exceed 60 s (median 165 s), so that cap would
    # have produced zero valid measurements for the strongest rival. Timed out -> error row.
    local rto="${RIVAL_TIMEOUT:-300}"
    if [ -d "$GPAP_HOME/src/Demotic/src" ]; then
        banner "rivals: DEMOTIC (their ISCAS-85 pins)"
        "$PY" eval/rivals_demotic.py --mode native $comb_insts --seeds $rseeds --steps "$rsteps" --timeout "$rto" \
            --circuit-variants $cvars \
            --output "$OUT/rivals_demotic_native.json" 2>&1 | tee -a "$LOGS/rivals.log"
        banner "rivals: DEMOTIC (our rare fixed-T instances, as miters)"
        "$PY" eval/rivals_demotic.py --mode miter $seq_insts -T "$T" --target rare \
            --seeds $rseeds --steps "$rsteps" --timeout "$rto" --circuit-variants $cvars \
            --output "$OUT/rivals_demotic_miter.json" 2>&1 | tee -a "$LOGS/rivals.log"
    else
        echo "[skip] DEMOTIC not installed (eval/rivals_setup.sh)"
    fi
    if [ -d "$GPAP_HOME/src/Recurrent-CircuitSAT-Sampler/src" ]; then
        banner "rivals: Recurrent-CircuitSAT (their target table, then our rare pins)"
        "$PY" eval/rivals_recurrent.py $seq_insts -T "$T" --seeds $rseeds --steps "$rsteps" \
            --timeout "$rto" \
            --circuit-variants $cvars \
            --output "$OUT/rivals_recurrent_table.json" 2>&1 | tee -a "$LOGS/rivals.log"
        "$PY" eval/rivals_recurrent.py $seq_insts -T "$T" --rare --seeds $rseeds --steps "$rsteps" \
            --timeout "$rto" \
            --circuit-variants $cvars \
            --output "$OUT/rivals_recurrent_rare.json" 2>&1 | tee -a "$LOGS/rivals.log"
    else
        echo "[skip] Recurrent-CircuitSAT not installed (eval/rivals_setup.sh)"
    fi
    if [ -d "$GPAP_HOME/src/High-Throughput-SAT-Sampler/src" ]; then
        # HTS eats CNF, so build exactly the instances it should see (the bench stages delete
        # their intermediates — a full circuit matrix is tens of GB).
        local hts_dir="$OUT/hts_cnf"
        mkdir -p "$hts_dir"
        banner "rivals: HTS (same CNFs, same GPU)"
        # HTS is a CNF consumer, so its 'circuit variant' is the ENCODER: it gets both our
        # Tseitin CNF and the ABC-encoded one, the same two GPAP is measured on.
        for inst in $seq_insts; do
            local src="benchmarks/circuits/bench/seq/$inst.bench"
            [ -f "$src" ] || continue
            for enc in tseitin abc; do
                case " $cvars " in *" ${enc/tseitin/raw} "*|*" $enc "*) ;; *) continue ;; esac
                local cnf="$hts_dir/$inst-c$T-$enc.cnf"
                [ -f "$cnf" ] || "$PY" -m gpap.cli circuit2cnf "$src" "$cnf" \
                    --cycles "$T" --target rare --encoder "$enc" --abc-opt fast \
                    >/dev/null 2>&1 || echo "[warn] could not build $enc CNF for $inst"
            done
        done
        "$PY" eval/rivals_hts.py "$hts_dir/*.cnf" --timeout "$rto" \
            --output "$OUT/rivals_hts.json" 2>&1 | tee -a "$LOGS/rivals.log"
    else
        echo "[skip] HTS not installed (eval/rivals_setup.sh)"
    fi
}

stage_plots() {
    banner "plots -> $OUT/plots"
    "$PY" eval/eval_plots.py "$OUT" --out-dir "$OUT/plots" 2>&1 | tee -a "$LOGS/plots.log"
}

case "$STAGE" in
    check)  stage_check ;;
    smoke)  stage_check; stage_seq; stage_comb; stage_sk; stage_epfl; stage_hwmcc
            stage_rivals; stage_plots ;;
    seq)    stage_seq ;;
    comb)   stage_comb ;;
    sk)     stage_sk ;;
    epfl)   stage_epfl ;;
    hwmcc)  stage_hwmcc ;;
    rivals) stage_rivals ;;
    plots)  stage_plots ;;
    all)    stage_check; stage_seq; stage_comb; stage_sk; stage_epfl; stage_hwmcc
            stage_rivals; stage_plots ;;
    *)      sed -n '2,40p' "$0"; exit 1 ;;
esac

banner "stage '$STAGE' done — results in $OUT"
ls -la "$OUT"/*.json 2>/dev/null | tail -20

#!/usr/bin/env bash
# GPAP full experimental sweep, built to run in a terminal (Colab's notebook UI disconnects long
# cells). Every stage is independently invocable and resumable: rows are appended to a per-stage
# JSON after each one, and re-running a stage skips what is already there.
#
#   eval/rivals/rivals_setup.sh                              # once per machine
#   eval/run_full_eval.sh seq
#   nohup eval/run_full_eval.sh all > full_eval.log 2>&1 &
#
# The paper's experiment is ONE stage, `campaign`: the rivals' own published sets, each rival
# under its own pins and our rare pin, then GPAP and CMSGen on that very CNF — three tools, one
# constraint, one row. The protocol is docs/EVAL_RUNBOOK.md §0 (v1.0, frozen 2026-09-06).
#
# Stages
#   check    environment/readiness table (no runs)
#   smoke    every stage, 1-2 instances, tiny budgets — "does this machine run the pipeline?"
#            Run it first on a new box; the numbers are meaningless.
#   campaign THE PAPER RUN: `rivals` then `plots`
#   rivals   DEMOTIC / Recurrent-CircuitSAT head-to-heads, each on its own paper's instances, one
#            run per instance, GPAP + CMSGen arms on the same CNF (HTS is off: no circuit in it)
#   plots    figures + summary tables from whatever JSONs exist
#   --- supplementary family sweeps (`gpap bench` on instances pinned OUR way, not the paper's
#       instance set; `plots` summarises them, the paper's tables do not use them) ---
#   seq      ITC'99 b* + ISCAS-89 s* sequential unrolls
#   comb     ISCAS-85 c* combinational (T=1, single pinned output)
#   sk       benchmarks/sampling/*.cnf — projected CNF sampling vs CMSGen (no encoder axis)
#   epfl     EPFL combinational AIGs (T=1, all outputs pinned)
#   hwmcc    HWMCC'17 single-property AIGs (T=CYCLES, witness sampling)
#   all      every stage above, in order
#
# The supplementary circuit stages run encoders x targets x depths, one `gpap bench` per
# (target, depth) with every listed encoder inside it. Override any of:
#   CYCLES=25 TARGETS=rare eval/run_full_eval.sh seq
#   N_TIME=100000 TIME_LIMIT=120 eval/run_full_eval.sh sk
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export PYTHONPATH="${PYTHONPATH:-.}"

# Ask, never recompute: gpap/paths.py is the one definition (see its docstring).
GPAP_HOME="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m gpap.paths home)"
: "${GPAP_HOME:?cannot resolve — python3 must be able to import the repo (gpap/paths.py, eval/paths.py)}"
CMSGEN="${CMSGEN:-$GPAP_HOME/bin/cmsgen}"
GPU_SLUG="$($PY -c "
import re, torch
n = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'cpu'
print(re.sub(r'[^a-z0-9]+', '-', n.lower()).strip('-'))" 2>/dev/null || echo unknown-gpu)"
OUT_EXPLICIT="${OUT:-}"          # remember whether the caller chose the directory (smoke needs it)
EXPERIMENTS="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m eval.paths experiments)"
: "${EXPERIMENTS:?cannot resolve — python3 must be able to import the repo (gpap/paths.py, eval/paths.py)}"
OUT="${OUT:-$EXPERIMENTS/full_eval/$GPU_SLUG}"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

# One encoder: the published tables are abc:fast, and a Tseitin arm doubles every circuit
# stage's wall for rows nothing publishes. ENCODERS="tseitin abc:fast" restores the in-process A/B.
ENCODERS="${ENCODERS:-abc:fast}"
TARGETS="${TARGETS:-rare random}"
CYCLES="${CYCLES:-25 50}"
N_TIME="${N_TIME:-50000}"
TIME_LIMIT="${TIME_LIMIT:-60}"
N_COV="${N_COV:-0}"                 # >0 adds the quality pass (2-wise + distinct); costs a 2nd pass
# The on-GPU re-check that every emitted model satisfies every clause: off everywhere else (a
# self-test, not part of a throughput number), on here because a campaign's rows get published.
# Its wall is `verify_sound_s`, excluded from every rate. VERIFY_SOUND=0 skips it.
VERIFY_SOUND="${VERIFY_SOUND:-1}"
SEEDS="${SEEDS:-}"                  # e.g. "5 6 7" -> median-of-3 (docs/BENCHMARKING.md §3)
MAX_CLAUSES="${MAX_CLAUSES:-4000000}"
# Extra args appended to every `gpap bench` call: the uniformity arm is this same matrix with
# `--allow-soft`, and pricing a configuration change honestly means the identical code path.
# Set OUT too, or the arms overwrite each other.
EXTRA_ARGS="${EXTRA_ARGS:-}"
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
    # A few minutes touching every stage. Not a measurement: the budgets are far too small.
    N_TIME=2000; TIME_LIMIT=5; N_COV=0; CYCLES="5"; TARGETS="rare"; SEEDS=""
    GLOB_SEQ="benchmarks/circuits/bench/seq/b0[36].bench"
    GLOB_COMB="benchmarks/circuits/bench/comb/c432.bench"
    GLOB_SK="benchmarks/sampling/27.sk_3_32.cnf"
    GLOB_EPFL="benchmarks/circuits/epfl/ctrl.aig"
    GLOB_HWMCC="benchmarks/circuits/hwmcc17/texasifetch1p8.aig"
    RIVAL_SEQ="${RIVAL_SEQ:-b03}"; RIVAL_COMB="${RIVAL_COMB:-c432}"
    RIVAL_SEEDS="${RIVAL_SEEDS:-5}"; RIVAL_T="${RIVAL_T:-5}"; RIVAL_STEPS="${RIVAL_STEPS:-1}"
    # Must land elsewhere: stage files share names with the real run, and --resume would then
    # treat 5-second smoke rows as finished work.
    [ -n "$OUT_EXPLICIT" ] || OUT="$EXPERIMENTS/full_eval/${GPU_SLUG}-smoke"
    LOGS="$OUT/logs"; SCRATCH="$OUT/cnf"; mkdir -p "$OUT" "$LOGS" "$SCRATCH"
    echo "SMOKE run (tiny budgets, 1-2 instances per family) -> $OUT"
fi

CMS_ARGS=()
if [ -x "$CMSGEN" ]; then CMS_ARGS=(--cmsgen "$CMSGEN"); else
    echo "[warn] no cmsgen at $CMSGEN — running GPAP-only (scripts/build_cmsgen.sh)" >&2
fi
SEED_ARGS=(); [ -n "$SEEDS" ] && SEED_ARGS=(--seeds $SEEDS)
SOUND_ARGS=(); [ "$VERIFY_SOUND" != "0" ] && SOUND_ARGS=(--verify-sound)

banner() { echo; echo "=============== $* ==============="; date -Is; }

# Preflight: there is no Python fallback, so a missing libgpap fails every row quietly for
# hours rather than failing the sweep. `check` is exempt — reporting what is missing is its job.
if [ "$STAGE" != "check" ]; then
    if ! "$PY" -c "import sys; from gpap.engine import core_available; sys.exit(0 if core_available() else 1)" 2>/dev/null; then
        echo "[FATAL] libgpap (the engine) is not available — every GPAP row would error." >&2
        echo "        Build it:  scripts/build_core.sh" >&2
        echo "        Details:   eval/rivals/rivals_setup.sh check" >&2
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
        ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
        ${SOUND_ARGS[@]+"${SOUND_ARGS[@]}"} \
        $EXTRA_ARGS "$@" 2>&1 | tee -a "$log"
}

stage_check() { bash "$REPO_ROOT/eval/rivals/rivals_setup.sh" check; }

# Circuit stages: one bench run per (target, depth); both encoders inside each run.
stage_seq() {
    for t in $TARGETS; do for c in $CYCLES; do
        bench "seq-c${c}-${t}" "$GLOB_SEQ" \
            --encoders $ENCODERS --cycles "$c" --target "$t" --sim-baseline --portfolio circuit
    done; done
}

stage_comb() {
    # ISCAS-85: T=1 and one pinned output. Pinning all outputs over-constrains a combinational
    # block to a near-point solution set, and the single pin is DEMOTIC's own setup.
    for t in $TARGETS; do
        bench "comb-${t}" "$GLOB_COMB" \
            --encoders $ENCODERS --cycles 1 --pin-outputs 1 --target "$t" \
            --sim-baseline --portfolio circuit
    done
}

stage_sk() {
    # Projected CNF sampling (`c ind`): already CNF, so no encoder axis. The reference point
    # vs CMSGen, on the sk_ portfolio.
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
    # Recurrent's clock must not carry its host copy: unpatched, rivals_recurrent.py refuses, and
    # behind `| tee` that refusal would read as a stage that simply produced nothing. Checked
    # before DEMOTIC runs, not after it.
    local rcs_run="$GPAP_HOME/src/Recurrent-CircuitSAT-Sampler/src/pytorch/run.py"
    if [ -f "$rcs_run" ] && ! grep -q egress_runtime "$rcs_run"; then
        echo "[FATAL] the Recurrent checkout lacks eval/rivals/recurrent_gpap_eval.patch" >&2
        echo "        Apply it:  eval/rivals/rivals_setup.sh" >&2
        exit 1
    fi
    # Instance sets: Recurrent-CircuitSAT runs every b*/s* in benchmarks/circuits/bench/seq, on
    # its own table's pins where it names the instance and on our rare pin everywhere; DEMOTIC
    # runs its 11 ISCAS-85. What a tool cannot process becomes an error row, omitted from the
    # table with the count stated. b17/b19/b22 are fetched here, not committed.
    bash scripts/fetch_benchmarks.sh itc99 2>&1 | sed 's/^/[bench] /' || \
        echo "[warn] could not fetch ITC'99 b17-b22 — the Recurrent stage will report them as skipped"
    local rec_seq
    rec_seq="$(cd benchmarks/circuits/bench/seq && ls *.bench | sed 's/\.bench$//' | tr '\n' ' ')"
    local seq_insts="${RIVAL_SEQ:-$rec_seq}"
    local comb_insts="${RIVAL_COMB:-paper}"
    # GPAP at median-of-3 seeds (docs/BENCHMARKING.md §3): the CMSGen ratios on the rivals' own
    # rejection-easy pins sit in a band where seed noise could move a cell. The rival runs once.
    local rseeds="${RIVAL_SEEDS:-5 6 7}"
    local T="${RIVAL_T:-25}"
    # T=25 is the paper's depth. The T=50 stages are an additive appendix at the rival's own
    # script depth, reported beside the T=25 rows and never in their place; RIVAL_T2=50 opts in.
    local T2="${RIVAL_T2:-none}"
    # One circuit variant, `abc`, for every tool and cell: each rival gets the same
    # sequential-ABC-simplified netlist GPAP encodes from, equivalence-checked
    # (RIVAL_FAIRNESS_FINDINGS.md §16d). RIVAL_VARIANTS="raw abc" still works; it is not the
    # protocol.
    local cvars="${RIVAL_VARIANTS:-abc}"
    # Grants pin each recorded cell's batch rung and Recurrent's start_point to a previous
    # campaign's measurement: no ladder walk, no probe. RIVAL_GRANTS=none re-derives them.
    local grants="${RIVAL_GRANTS:-eval/rivals/grants/l4_protocol_20260904.json}"
    GRANT_ARGS=()
    [ "$grants" != "none" ] && [ -f "$grants" ] && GRANT_ARGS=(--grants "$grants")
    # RIVAL_HONOUR_FAILURES=1 accepts a cell the grants file records as a rival FAILURE instead
    # of re-deriving it, which costs the full init budget each (9.3 h of a 19 h campaign). OFF
    # by default: the frozen protocol measures a failure on the card in front of it. Turning it
    # on is a stated choice, never a silent one: every row that took it says `not_remeasured`
    # and verify_artifact names them. The artifact's reproduce.sh turns it on
    # (docs/papers/tacas-27/ARTIFACT.md), and
    # so did the paper's own re-run (RIVAL_FAIRNESS_FINDINGS.md §17e).
    [ "${RIVAL_HONOUR_FAILURES:-0}" = "1" ] && [ ${#GRANT_ARGS[@]} -gt 0 ] \
        && GRANT_ARGS+=(--honour-recorded-failures)
    # The CMSGen arm rides on every rival row: the patched binary on the very CNF the GPAP arm
    # sampled, same n and budget. Without it the rows carry no CMSGen columns and there is no
    # Table 1 — a warning was printed above; that is not a campaign.
    [ -x "$CMSGEN" ] || echo "[warn] rivals stage without CMSGen: no Table 1 from these rows" >&2
    # 'auto' = each rival at the num_steps its own artifacts specify; see
    # eval/rivals/rival_common.py::RIVAL_STEPS for the values and the measurement behind them.
    local rsteps="${RIVAL_STEPS:-auto}"
    # Rival subprocess cap 1,200 s per run, and a separate 1,200 s initialization budget for
    # Recurrent's start_point search (RIVAL_FAIRNESS_FINDINGS.md §11/§16d). The
    # rival's wall includes the model build its own metric excludes, which is why it is not
    # GPAP's 60 s. A timeout or failed init is an error row naming the budget, never a zero.
    local rto="${RIVAL_TIMEOUT:-1200}"
    local rib="${RIVAL_INIT_BUDGET:-1200}"
    if [ -d "$GPAP_HOME/src/Demotic/src" ]; then
        banner "rivals: DEMOTIC (their ISCAS-85 pins)"
        "$PY" eval/rivals/rivals_demotic.py --mode native $comb_insts --seeds $rseeds --steps "$rsteps" --timeout "$rto" \
            --circuit-variants $cvars ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
            --resume --output "$OUT/rivals_demotic_native.json" 2>&1 | tee -a "$LOGS/rivals.log"
        # The controlled contrast: the same ISCAS-85 circuits, our rare pin. The native arm
        # above pins the few bits their table names; this one pins all outputs to a rare
        # reachable pattern, so the only thing changing between the two rows is the task.
        banner "rivals: DEMOTIC (their ISCAS-85 circuits, OUR rare pin)"
        "$PY" eval/rivals/rivals_demotic.py --mode miter $comb_insts -T 1 --target rare \
            --bench-dir benchmarks/circuits/bench/comb \
            --seeds $rseeds --steps "$rsteps" --timeout "$rto" --circuit-variants $cvars ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} \
            ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
            --resume --output "$OUT/rivals_demotic_comb_rare.json" 2>&1 | tee -a "$LOGS/rivals.log"
        # `demotic-miter` (DEMOTIC on our unrolled b*/s* instances as single-output miters) is
        # measured but unpublished — its instances are in neither paper. RIVAL_DEMOTIC_MITER=1.
        if [ "${RIVAL_DEMOTIC_MITER:-0}" = "1" ]; then
            banner "rivals: DEMOTIC (our rare fixed-T instances, as miters)"
            "$PY" eval/rivals/rivals_demotic.py --mode miter $seq_insts -T "$T" --target rare \
                --seeds $rseeds --steps "$rsteps" --timeout "$rto" --circuit-variants $cvars ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} \
                ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
                --resume --output "$OUT/rivals_demotic_miter.json" 2>&1 | tee -a "$LOGS/rivals.log"
        fi
    else
        echo "[skip] DEMOTIC not installed (eval/rivals/rivals_setup.sh)"
    fi
    if [ -d "$GPAP_HOME/src/Recurrent-CircuitSAT-Sampler/src" ]; then
        banner "rivals: Recurrent-CircuitSAT (their target table, then our rare pins)"
        "$PY" eval/rivals/rivals_recurrent.py $seq_insts -T "$T" --seeds $rseeds --steps "$rsteps" --init-budget "$rib" \
            --timeout "$rto" \
            --circuit-variants $cvars ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
            --resume --output "$OUT/rivals_recurrent_table.json" 2>&1 | tee -a "$LOGS/rivals.log"
        "$PY" eval/rivals/rivals_recurrent.py $seq_insts -T "$T" --rare --seeds $rseeds --steps "$rsteps" --init-budget "$rib" \
            --timeout "$rto" \
            --circuit-variants $cvars ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
            --resume --output "$OUT/rivals_recurrent_rare.json" 2>&1 | tee -a "$LOGS/rivals.log"
        if [ "$T2" != "none" ] && [ "$T2" != "$T" ]; then
            banner "rivals: Recurrent-CircuitSAT at T=$T2 (their own depth limit)"
            "$PY" eval/rivals/rivals_recurrent.py $seq_insts -T "$T2" --seeds $rseeds --steps "$rsteps" --init-budget "$rib" \
                --timeout "$rto" \
                --circuit-variants $cvars ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
                --resume --output "$OUT/rivals_recurrent_table_t${T2}.json" 2>&1 | tee -a "$LOGS/rivals.log"
            "$PY" eval/rivals/rivals_recurrent.py $seq_insts -T "$T2" --rare --seeds $rseeds --steps "$rsteps" --init-budget "$rib" \
                --timeout "$rto" \
                --circuit-variants $cvars ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
                --resume --output "$OUT/rivals_recurrent_rare_t${T2}.json" 2>&1 | tee -a "$LOGS/rivals.log"
        fi
    else
        echo "[skip] Recurrent-CircuitSAT not installed (eval/rivals/rivals_setup.sh)"
    fi
    # HTS is off by default (RIVAL_HTS=1 re-enables it): its paper evaluates 60 instances and
    # none is a circuit, so in a circuits-only evaluation it can never be measured on its own
    # published ground and every cell would be our task on instances we chose.
    if [ "${RIVAL_HTS:-0}" = "1" ] && [ -d "$GPAP_HOME/src/High-Throughput-SAT-Sampler/src" ]; then
        # HTS eats CNF, so build exactly the instances it should see: the bench stages delete
        # their intermediates.
        local hts_dir="$OUT/hts_cnf"
        mkdir -p "$hts_dir"
        banner "rivals: HTS (same CNFs, same GPU)"
        # HTS is a CNF consumer, so its circuit variant is the encoder: both our Tseitin CNF
        # and the ABC-encoded one, the two GPAP is measured on.
        for inst in $seq_insts; do
            local src="benchmarks/circuits/bench/seq/$inst.bench"
            [ -f "$src" ] || continue
            for enc in tseitin abc; do
                case " $cvars " in *" ${enc/tseitin/raw} "*|*" $enc "*) ;; *) continue ;; esac
                local cnf="$hts_dir/$inst-c$T-$enc.cnf"
                [ -f "$cnf" ] || "$PY" -m gpap.cli circuit2cnf "$src" "$cnf" \
                    --cycles "$T" --target rare --encoder "$enc" --abc-opt fast --seed 5 \
                    >/dev/null 2>&1 || echo "[warn] could not build $enc CNF for $inst"
            done
        done
        "$PY" eval/rivals/rivals_hts.py "$hts_dir/*.cnf" --timeout "$rto" --steps "$rsteps" \
            --output "$OUT/rivals_hts.json" 2>&1 | tee -a "$LOGS/rivals.log"
    elif [ "${RIVAL_HTS:-0}" != "1" ]; then
        # Said for what it is: after `setup` had listed HTS as present, "not installed" read
        # as a broken install to a reviewer, when it was the default above.
        echo "[skip] HTS is off by default — its published set holds no circuits (RIVAL_HTS=1 runs it)"
    else
        echo "[skip] HTS not installed (eval/rivals/rivals_setup.sh)"
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
    campaign) stage_rivals; stage_plots ;;
    rivals) stage_rivals ;;
    plots)  stage_plots ;;
    all)    stage_check; stage_seq; stage_comb; stage_sk; stage_epfl; stage_hwmcc
            stage_rivals; stage_plots ;;
    *)      sed -n '2,40p' "$0"; exit 1 ;;
esac

banner "stage '$STAGE' done — results in $OUT"
ls -la "$OUT"/*.json 2>/dev/null | tail -20

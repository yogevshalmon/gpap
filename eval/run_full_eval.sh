#!/usr/bin/env bash
# GPAP full experimental sweep — built to run in a TERMINAL (Colab's notebook UI disconnects long
# cells; a detached shell process survives). Every stage is independently invocable and RESUMABLE:
# results are appended to a per-stage JSON after every row, and re-running the stage skips what is
# already in it. One instance blowing up never loses the rest.
#
#   # once per machine
#   eval/rivals/rivals_setup.sh
#
#   # then, one family at a time (each prints a live table; add & + nohup to detach)
#   eval/run_full_eval.sh seq
#   nohup eval/run_full_eval.sh all > full_eval.log 2>&1 &   ;  tail -f full_eval.log
#
# THE PAPER'S EXPERIMENT (converged 2026-09-03, docs/EVAL_RUNBOOK.md) is ONE stage, `campaign`:
#   1. the benchmarks are the GPU rivals' own published sets — DEMOTIC's 11 ISCAS-85, and every
#      ITC'99 b* / ISCAS-89 s* sequential circuit Recurrent-CircuitSAT evaluates (all we can
#      process; its paper's 16 are scored beside the whole set);
#   2. each rival runs on its own set, under the pins its table names AND under our rare pin,
#      one configuration each (its own script's num_steps, a granted batch, a 300 s budget);
#   3. every one of those cells is then run by GPAP (abc:fast CNF of the same circuit + pins)
#      and by CMSGen on that very CNF file — three tools, one constraint, one row.
#   What a tool cannot process is an error row, omitted from the table with the count stated.
#
# Stages
#   check    environment/readiness table (no runs)
#   smoke    every stage, 1-2 instances each, tiny budgets (~minutes) — "does this machine run
#            the whole pipeline?" Run it FIRST on a new box; the numbers are meaningless.
#   campaign THE PAPER RUN: `rivals` then `plots`
#   rivals   DEMOTIC / Recurrent-CircuitSAT head-to-heads on the same GPU, each on its own
#            paper's instances, ONE run per instance, GPAP + CMSGen arms on the same CNF
#            (HTS is off: no circuit in its paper)
#   plots    figures + summary tables from whatever JSONs exist
#   --- supplementary family sweeps (`gpap bench` vs CMSGen on instances pinned OUR way; not the
#       paper's instance set, printed by paper_tables only when present) ---
#   seq      ITC'99 b* + ISCAS-89 s* sequential unrolls
#   comb     ISCAS-85 c* combinational (T=1, single pinned output)
#   sk       benchmarks/sampling/*.cnf — projected CNF sampling vs CMSGen (no encoder axis)
#   epfl     EPFL combinational AIGs (T=1, all outputs pinned)
#   hwmcc    HWMCC'17 single-property AIGs (T=CYCLES, the witness-sampling instances)
#   all      every stage above, in order
#
# The supplementary circuit stages run the matrix: encoders x targets x unroll depths, i.e.
#   ENCODERS="abc:fast"   TARGETS="rare random"   CYCLES="25 50"
# (one `gpap bench` invocation per target x depth; every listed encoder runs inside it, back to
# back, so ENCODERS="tseitin abc:fast" gives an in-process A/B pair.) Override any of them:
#   CYCLES=25 TARGETS=rare eval/run_full_eval.sh seq
#   N_TIME=100000 TIME_LIMIT=120 eval/run_full_eval.sh sk
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export PYTHONPATH="${PYTHONPATH:-.}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPAP_HOME="${GPAP_HOME:-$REPO/.gpap}"
CMSGEN="${CMSGEN:-$GPAP_HOME/bin/cmsgen}"
GPU_SLUG="$($PY -c "
import re, torch
n = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'cpu'
print(re.sub(r'[^a-z0-9]+', '-', n.lower()).strip('-'))" 2>/dev/null || echo unknown-gpu)"
OUT_EXPLICIT="${OUT:-}"          # remember whether the caller chose the directory (smoke needs it)
OUT="${OUT:-$REPO_ROOT/.experiments/full_eval/$GPU_SLUG}"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

# abc:fast ONLY (2026-09-03, the campaign decision): the published tables are one encoder, and a
# Tseitin arm doubles every circuit stage's wall for rows nothing publishes. ENCODERS="tseitin
# abc:fast" restores the in-process A/B (docs/findings/KERNEL_AB_FINDINGS.md is where that lives).
ENCODERS="${ENCODERS:-abc:fast}"
TARGETS="${TARGETS:-rare random}"
CYCLES="${CYCLES:-25 50}"
N_TIME="${N_TIME:-50000}"
TIME_LIMIT="${TIME_LIMIT:-60}"
N_COV="${N_COV:-0}"                 # >0 adds the quality pass (2-wise + distinct); costs a 2nd pass
# The on-GPU re-check that every emitted model satisfies every clause. OFF everywhere else (it
# is a self-test, not part of a throughput number), ON here: a campaign's rows get published, so
# the extra coverage is worth its wall — which is billed to `verify_sound_s` and excluded from
# every rate, so it cannot move a measurement. Set VERIFY_SOUND=0 to skip it.
VERIFY_SOUND="${VERIFY_SOUND:-1}"
SEEDS="${SEEDS:-}"                  # e.g. "5 6 7" -> median-of-3 (docs/BENCHMARKING.md §3)
MAX_CLAUSES="${MAX_CLAUSES:-4000000}"
# Extra args appended to EVERY `gpap bench` call. The reason this exists rather than a flag: the
# uniformity arm of the sweep is the SAME matrix with `--allow-soft`, and the only honest way to
# price a configuration change is to run the identical protocol through the identical code path.
# A second copy of this script would drift. Set OUT too, or the arms overwrite each other.
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
SOUND_ARGS=(); [ "$VERIFY_SOUND" != "0" ] && SOUND_ARGS=(--verify-sound)

banner() { echo; echo "=============== $* ==============="; date -Is; }

# PREFLIGHT. The engine is a compiled library now and has no Python fallback, so a missing
# libgpap does not fail the sweep — it fails every ROW of it, quietly, for hours. Check once.
# (`check` is exempt: reporting what is missing is exactly its job.)
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
    # INSTANCE SETS (user's rule, 2026-09-03): Recurrent-CircuitSAT runs EVERY b*/s* sequential
    # circuit in benchmarks/circuits/bench/seq — on its own target-table pins where its table
    # names the instance, and on our rare pin everywhere. What a tool cannot process becomes an
    # error row and is OMITTED from the table with the count stated. DEMOTIC runs its paper's
    # 11 ISCAS-85 (`paper` -> eval/rivals/rival_common.py::PAPER_INSTANCES; paper_tables scores
    # from the same dict). b17/b19/b22 are fetched, not committed: pulled here so a fresh box
    # sweeps the full set.
    bash scripts/fetch_benchmarks.sh itc99 2>&1 | sed 's/^/[bench] /' || \
        echo "[warn] could not fetch ITC'99 b17-b22 — the Recurrent stage will report them as skipped"
    local rec_seq
    rec_seq="$(cd benchmarks/circuits/bench/seq && ls *.bench | sed 's/\.bench$//' | tr '\n' ' ')"
    local seq_insts="${RIVAL_SEQ:-$rec_seq}"
    local comb_insts="${RIVAL_COMB:-paper}"
    # GPAP at median-of-3 seeds (docs/BENCHMARKING.md §3), the same protocol as
    # eval/colab/colab_rivals_l4.sh: this stage IS the campaign now, and the CMSGen ratios on the
    # rivals' own pins (rejection-easy, p_acc 0.03-0.5) sit in the 1-10x band where +-40% seed
    # noise on a low-success instance could move a cell. The rival runs once (its own rule).
    local rseeds="${RIVAL_SEEDS:-5 6 7}"
    local T="${RIVAL_T:-25}"
    # PROTOCOL v1.0 (frozen 2026-09-06, docs/EVAL_RUNBOOK.md §0): T=25 is the paper's depth.
    # The T=50 stages are an ADDITIVE appendix run — the same protocol at the rival's own
    # script depth, reported beside the T=25 rows and never in place of them — opted into with
    # RIVAL_T2=50.
    local T2="${RIVAL_T2:-none}"
    # ONE circuit variant, `abc`, for every tool and every cell (protocol v1.0): every rival is
    # handed the same sequential-ABC-simplified netlist GPAP encodes from, equivalence-checked
    # (RIVAL_FAIRNESS_FINDINGS.md §16d). The raw axis stays implemented (RIVAL_VARIANTS="raw abc",
    # scored per instance for the rival by paper_tables.rival_cells) and is not the protocol.
    local cvars="${RIVAL_VARIANTS:-abc}"
    # GRANTS: pin every recorded cell's batch rung (and Recurrent's start_point) to a previous
    # campaign's measurement — no ladder walk, no probe (eval/rivals/extract_grants.py;
    # eval/rivals/grants/). RIVAL_GRANTS=none walks and probes from scratch.
    local grants="${RIVAL_GRANTS:-eval/rivals/grants/l4_protocol_20260904.json}"
    GRANT_ARGS=()
    [ "$grants" != "none" ] && [ -f "$grants" ] && GRANT_ARGS=(--grants "$grants")
    # THE CMSGEN ARM rides on every rival row (`--cmsgen`: the patched binary on the very CNF
    # the GPAP arm sampled, same n / budget). Without a binary the rows carry no CMSGen columns
    # and paper_tables prints no Table 1 — a warning was printed above; it is not a campaign.
    [ -x "$CMSGEN" ] || echo "[warn] rivals stage without CMSGen: no Table 1 from these rows" >&2
    # 'auto' = each rival at the num_steps its own artifacts specify; see
    # eval/rivals/rival_common.py::RIVAL_STEPS for the values and the measurement behind them.
    local rsteps="${RIVAL_STEPS:-auto}"
    # Rival subprocess cap 1,200 s per run, and a SEPARATE 1,200 s initialization budget for
    # Recurrent's start_point search (protocol v1.0; §11/§16d of RIVAL_FAIRNESS_FINDINGS.md:
    # every one of the 300 s campaign's 20 Recurrent failures was that budget expiring). The
    # rival's wall includes the PyTorch model build its OWN metric excludes, which is why it is
    # not GPAP's 60 s. Timed out / did not initialize -> error row naming the budget, never a zero.
    local rto="${RIVAL_TIMEOUT:-1200}"
    local rib="${RIVAL_INIT_BUDGET:-1200}"
    if [ -d "$GPAP_HOME/src/Demotic/src" ]; then
        banner "rivals: DEMOTIC (their ISCAS-85 pins)"
        "$PY" eval/rivals/rivals_demotic.py --mode native $comb_insts --seeds $rseeds --steps "$rsteps" --timeout "$rto" \
            --circuit-variants $cvars ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
            --resume --output "$OUT/rivals_demotic_native.json" 2>&1 | tee -a "$LOGS/rivals.log"
        # THE CONTROLLED CONTRAST: their own ISCAS-85 circuits, our rare pin. The native arm
        # above pins the 1-5 bits their table names (rejection-easy, p_acc 0.03-0.5); this one
        # pins ALL outputs to a rare reachable pattern on the SAME circuits, so the only thing
        # that changes between the two rows is the TASK. Without it the easy-pin and rare-pin
        # numbers came from different circuits (ISCAS-85 vs unrolled ITC'99) and the comparison
        # confounded task with instance.
        banner "rivals: DEMOTIC (their ISCAS-85 circuits, OUR rare pin)"
        "$PY" eval/rivals/rivals_demotic.py --mode miter $comb_insts -T 1 --target rare \
            --bench-dir benchmarks/circuits/bench/comb \
            --seeds $rseeds --steps "$rsteps" --timeout "$rto" --circuit-variants $cvars ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} \
            ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
            --resume --output "$OUT/rivals_demotic_comb_rare.json" 2>&1 | tee -a "$LOGS/rivals.log"
        # `demotic-miter` (DEMOTIC on OUR unrolled b*/s* instances as single-output miters) is
        # measured-but-unpublished (paper_tables.RIVALS: its instances are in neither paper) and
        # the most expensive DEMOTIC arm; RIVAL_DEMOTIC_MITER=1 runs it.
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
    # HTS is OFF by default since 2026-09-02 (RIVAL_HTS=1 re-enables it; nothing else changed).
    # Its paper (arXiv 2502.08673) evaluates 60 instances and NONE of them is a circuit — random
    # q-family, OR-configuration, prod-* and s15850a_* CNFs — so in a circuits-only evaluation it
    # is the one rival that can never be measured on its own published ground, and every HTS cell
    # would be our task on instances we chose. Its code is untouched and its stage still runs on
    # request; see docs/EVAL_RUNBOOK.md §6.
    if [ "${RIVAL_HTS:-0}" = "1" ] && [ -d "$GPAP_HOME/src/High-Throughput-SAT-Sampler/src" ]; then
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
                    --cycles "$T" --target rare --encoder "$enc" --abc-opt fast --seed 5 \
                    >/dev/null 2>&1 || echo "[warn] could not build $enc CNF for $inst"
            done
        done
        "$PY" eval/rivals/rivals_hts.py "$hts_dir/*.cnf" --timeout "$rto" --steps "$rsteps" \
            --output "$OUT/rivals_hts.json" 2>&1 | tee -a "$LOGS/rivals.log"
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

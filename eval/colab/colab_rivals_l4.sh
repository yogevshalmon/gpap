#!/usr/bin/env bash
# The paper campaign on a rented Colab L4, run under eval/colab/colab_supervisor.py: GPAP vs the
# GPU gradient samplers, with the CMSGen arm on every row (Table 1 is read off these files;
# CMSGEN=none runs the rivals alone). Instances, seeds, steps, timeouts and T are those of
# run_full_eval.sh's rivals stage, so the rows compare directly.
#
#   bash eval/colab/colab_rivals_l4.sh                  # under the supervisor
#   DEADLINE_H=4.3 bash eval/colab/colab_rivals_l4.sh   # one shot, on a machine you own
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # this file lives in eval/colab/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
PY="${PYTHON:-python3}"

# A fixed OUT, not the GPU slug: the supervisor mirrors this path literally (--remote-dir), so a
# slug that changes on a new VM image would write where nothing collects it.
EXPERIMENTS="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m eval.paths experiments)"
: "${EXPERIMENTS:?cannot resolve — python3 must be able to import the repo (eval/paths.py)}"
OUT="${OUT:-$EXPERIMENTS/full_eval/nvidia-l4}"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

# --- identical to run_full_eval.sh's rivals stage --------------------------------------------
# Recurrent on every b*/s* circuit, DEMOTIC on its 11 ISCAS-85; a cell a tool cannot process is
# an error row. b17-b22 are fetched, not committed.
bash "$REPO_ROOT/scripts/fetch_benchmarks.sh" itc99 2>&1 | sed 's/^/[bench] /' || \
    echo "[warn] could not fetch ITC'99 b17-b22 — the Recurrent stage will report them as skipped"
SEQ_INSTS="${SEQ_INSTS:-$(cd "$REPO_ROOT/benchmarks/circuits/bench/seq" 2>/dev/null \
    && ls *.bench 2>/dev/null | sed 's/\.bench$//' | tr '\n' ' ')}"
COMB_INSTS="${COMB_INSTS:-paper}"
# The median of 3 seeds, as the protocol says.
RSEEDS="${RSEEDS:-5 6 7}"
# auto: each rival at the num_steps it ships (rival_common.RIVAL_STEPS).
STEPS="${STEPS:-auto}"
# The rival's per-run wall, and Recurrent's initialization budget, which is in no published rate
# (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §11, §17).
TIMEOUT="${TIMEOUT:-1200}"
INIT_BUDGET="${INIT_BUDGET:-1200}"
# GRANTS=<file> pins each recorded cell's batch and Recurrent's start_point: the rule is
# unchanged, the search is skipped.
GRANT_ARGS=()
[ -n "${GRANTS:-}" ] && GRANT_ARGS=(--grants "$GRANTS")
# HONOUR_FAILURES=1 (needs GRANTS) also pins the recorded rival failures instead of re-measuring
# them; those rows say `not_remeasured`. The paper's re-run set it (RIVAL_FAIRNESS_FINDINGS.md
# §17e).
if [ "${HONOUR_FAILURES:-0}" = "1" ]; then
    if [ ${#GRANT_ARGS[@]} -gt 0 ]; then
        GRANT_ARGS+=(--honour-recorded-failures)
    else
        echo "[warn] HONOUR_FAILURES=1 without GRANTS does nothing: every failure is re-measured" >&2
    fi
fi
# RETRY_ERRORS=1 re-runs error rows; off, since a rival timeout costs its full budget again.
RETRY_ARGS=()
[ "${RETRY_ERRORS:-0}" = "1" ] && RETRY_ARGS=(--retry-errors)
T="${T:-25}"

# The circuit each side is handed (rival_common.py): `abc` for every tool and stage is the
# protocol (RIVAL_FAIRNESS_FINDINGS.md §16d); "raw abc" still runs but is not the protocol.
VARIANTS_DEM="${VARIANTS_DEM:-abc}"
VARIANTS_RCS="${VARIANTS_RCS:-abc}"
HTS_ENCS="${HTS_ENCS:-tseitin abc}"
# The default STAGES is the campaign: the four T=25 stages and the calibrations (protocol v1.0,
# docs/EVAL_RUNBOOK.md §0). The T=50 appendix, reported beside it, is STAGES="recurrent-rare-t50
# recurrent-table-t50 steps-calib-t50". Not in the paper: hts (its paper has no circuit) and
# demotic-miter. A finished stage still costs time on a restart (--resume retries error rows),
# so leave out what is done.
STAGES="${STAGES:-lr-calib demotic-native demotic-comb-rare recurrent-rare recurrent-table steps-calib}"
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) echo "[skip] $1 (not in STAGES)"; return 1 ;; esac; }

# A hard preflight for the CMSGen arm: resumed without it, a stage would append rows without the
# arm to a file that has it. CMSGEN=none opts out.
# Ask, never recompute: gpap/paths.py is the one definition (see its docstring).
GPAP_HOME="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m gpap.paths home)"
: "${GPAP_HOME:?cannot resolve — python3 must be able to import the repo (gpap/paths.py, eval/paths.py)}"
CMSGEN="${CMSGEN:-$GPAP_HOME/bin/cmsgen}"
CMS_ARGS=()
if [ "$CMSGEN" != "none" ]; then
    if [ ! -x "$CMSGEN" ]; then
        echo "[FATAL] no cmsgen at $CMSGEN — every row of this campaign carries the CMSGen arm." >&2
        echo "        Build it (scripts/build_cmsgen.sh) or set CMSGEN=none for a rivals-only run." >&2
        exit 1
    fi
    if ! "$CMSGEN" --help 2>&1 | grep -q -- --nowrite || ! "$CMSGEN" --help 2>&1 | grep -q -- --maxtime; then
        echo "[FATAL] cmsgen lacks scripts/cmsgen_gpap_eval.patch (--nowrite/--maxtime): it would be" >&2
        echo "        timed WITH sample emission and a killed run would report a FLOORED count." >&2
        exit 1
    fi
    CMS_ARGS=(--cmsgen "$CMSGEN")
fi
# ...and for Recurrent's patch: unpatched, its clock carries the host copy and rivals_recurrent.py
# refuses to run, which behind `| tee` ends every Recurrent stage with no rows and still
# RIVALS_STAGE_COMPLETE.
RCS_RUN="$GPAP_HOME/src/Recurrent-CircuitSAT-Sampler/src/pytorch/run.py"
case " $STAGES " in *" recurrent-"*|*" steps-calib"*|*" lr-calib "*)
    if ! grep -q egress_runtime "$RCS_RUN" 2>/dev/null; then
        echo "[FATAL] $RCS_RUN lacks eval/rivals/recurrent_gpap_eval.patch: its clock would carry" >&2
        echo "        the host copy. Run eval/rivals/rivals_setup.sh." >&2
        exit 1
    fi ;;
esac

# Off by default (24 h): the VM dies within the hour, which --resume and the supervisor handle,
# and an in-VM guard would skip every stage longer than one VM. For a one-shot run only.
DEADLINE_H="${DEADLINE_H:-24}"
START=$(date +%s)
DEADLINE=$($PY -c "print(int($START + $DEADLINE_H * 3600))")

left()  { echo $(( DEADLINE - $(date +%s) )); }
banner() { echo; echo "=============== $* ==============="; date -Is; echo "budget left: $(( $(left) / 60 )) min"; }

# have <minutes> <name> — refuse to start a sub-stage that cannot finish inside the budget.
have() {
    if [ "$(left)" -lt $(( $1 * 60 )) ]; then
        echo "[DEADLINE] skipping '$2' — needs ~${1}m, only $(( $(left) / 60 ))m left" | tee -a "$LOGS/rivals.log"
        return 1
    fi
    return 0
}

# The GPAP-arm cache (rival_common.cached_gpap), which makes a rival-side re-run cheap, survives
# a VM only as one file the supervisor mirrors.
CACHE_TGZ="$OUT/gpap_cache.tgz"
[ -f "$CACHE_TGZ" ] && tar xzf "$CACHE_TGZ" -C "$OUT" 2>/dev/null && \
    echo "[cache] restored $(ls "$OUT/.gpap_cache" 2>/dev/null | wc -l) GPAP arm(s)"
save_cache() { tar czf "$CACHE_TGZ.tmp" -C "$OUT" .gpap_cache 2>/dev/null \
                   && mv "$CACHE_TGZ.tmp" "$CACHE_TGZ"; }
# Packed every 4 min too, so a probe settled mid-stage survives a reclaim (the supervisor
# mirrors every 3 min).
( while true; do sleep 240; save_cache; done ) &
CACHE_PACKER=$!
trap 'kill $CACHE_PACKER 2>/dev/null' EXIT

# ---------------------------------------------------------------------------- 1. DEMOTIC
if want demotic-native && have 25 "demotic-native"; then
    banner "DEMOTIC native (their ISCAS-85 pins, raw + abc)"
    $PY eval/rivals/rivals_demotic.py --mode native $COMB_INSTS --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --circuit-variants $VARIANTS_DEM ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
        --resume --output "$OUT/rivals_demotic_native.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

# The same circuits under our rare pin: against the native arm, only the task changes.
if want demotic-comb-rare && have 25 "demotic-comb-rare"; then
    banner "DEMOTIC (their ISCAS-85 circuits, OUR rare pin)"
    $PY eval/rivals/rivals_demotic.py --mode miter $COMB_INSTS -T 1 --target rare \
        --bench-dir benchmarks/circuits/bench/comb \
        --seeds $RSEEDS --steps "$STEPS" --timeout "$TIMEOUT" --circuit-variants $VARIANTS_DEM ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} \
        ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
        --resume --output "$OUT/rivals_demotic_comb_rare.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

if want demotic-miter && have 50 "demotic-miter"; then
    banner "DEMOTIC miter (our rare fixed-T instances, raw + abc)"
    $PY eval/rivals/rivals_demotic.py --mode miter $SEQ_INSTS -T "$T" --target rare \
        --seeds $RSEEDS --steps "$STEPS" --timeout "$TIMEOUT" --circuit-variants $VARIANTS_DEM ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} \
        ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
        --resume --output "$OUT/rivals_demotic_miter.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

# ---------------------------------------------------------------------------- 2. HTS
# A CNF consumer, so its circuit variant is the encoder: build both CNFs it should see.
if want hts && have 40 "hts"; then
    banner "HTS (same CNFs, same GPU)"
    HTS_DIR="$OUT/hts_cnf"; mkdir -p "$HTS_DIR"
    for inst in $SEQ_INSTS; do
        src="benchmarks/circuits/bench/seq/$inst.bench"
        [ -f "$src" ] || continue
        for enc in $HTS_ENCS; do
            cnf="$HTS_DIR/$inst-c$T-$enc.cnf"
            [ -f "$cnf" ] || $PY -m gpap.cli circuit2cnf "$src" "$cnf" \
                --cycles "$T" --target rare --encoder "$enc" --abc-opt fast --seed 5 \
                >/dev/null 2>&1 || echo "[warn] could not build $enc CNF for $inst"
        done
    done
    $PY eval/rivals/rivals_hts.py "$HTS_DIR/*.cnf" --timeout "$TIMEOUT" --steps "$STEPS" \
        --resume --output "$OUT/rivals_hts.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

# ---------------------------------------------------------------------------- 3. Recurrent
# Rare first, the headline; every row is checkpointed, so a deadline keeps what finished.
if want recurrent-rare && have 60 "recurrent-rare"; then
    banner "Recurrent-CircuitSAT — OUR rare pins (raw)"
    $PY eval/rivals/rivals_recurrent.py $SEQ_INSTS -T "$T" --rare --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --init-budget "$INIT_BUDGET" --circuit-variants $VARIANTS_RCS ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} \
        ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} ${RETRY_ARGS[@]+"${RETRY_ARGS[@]}"} \
        --resume --output "$OUT/rivals_recurrent_rare.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

if want recurrent-table && have 30 "recurrent-table"; then
    banner "Recurrent-CircuitSAT — THEIR target table (raw)"
    $PY eval/rivals/rivals_recurrent.py $SEQ_INSTS -T "$T" --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --init-budget "$INIT_BUDGET" --circuit-variants $VARIANTS_RCS ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} \
        ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} ${RETRY_ARGS[@]+"${RETRY_ARGS[@]}"} \
        --resume --output "$OUT/rivals_recurrent_table.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

if want recurrent-rare-t50 && have 90 "recurrent-rare-t50"; then
    banner "Recurrent-CircuitSAT — OUR rare pins, T=50 (their own depth limit)"
    $PY eval/rivals/rivals_recurrent.py $SEQ_INSTS -T 50 --rare --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --init-budget "$INIT_BUDGET" --circuit-variants $VARIANTS_RCS ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} \
        ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} ${RETRY_ARGS[@]+"${RETRY_ARGS[@]}"} \
        --resume --output "$OUT/rivals_recurrent_rare_t50.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

if want recurrent-table-t50 && have 45 "recurrent-table-t50"; then
    banner "Recurrent-CircuitSAT — THEIR target table, T=50 (their own depth limit)"
    $PY eval/rivals/rivals_recurrent.py $SEQ_INSTS -T 50 --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --init-budget "$INIT_BUDGET" --circuit-variants $VARIANTS_RCS ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} \
        ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} ${RETRY_ARGS[@]+"${RETRY_ARGS[@]}"} \
        --resume --output "$OUT/rivals_recurrent_table_t50.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

# ------------------------------------------------------------ 4. the tie-breaks, on THIS card
# Evidence, not configuration: each re-derives which of two shipped values (RIVAL_STEPS, RIVAL_LR)
# is better for the rival, so the campaign carries its own proof. A flip is recorded, not acted on.
if want steps-calib && have 30 "steps-calib"; then
    banner "num_steps: script vs argparse vs PAPER (and the paper's paired config), T=$T"
    $PY eval/rivals/steps_calibration.py --timeout "$TIMEOUT" --tools demotic recurrent \
        --cycles "$T" --out "$OUT/rivals_steps_calibration.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi
# At the rival's own depth too: their Figure 5 shows the count rising over 5 iterations.
if want steps-calib-t50 && have 45 "steps-calib-t50"; then
    banner "num_steps: script vs argparse vs PAPER (and the paired config), T=50"
    $PY eval/rivals/steps_calibration.py --timeout "$TIMEOUT" --tools recurrent \
        --cycles 50 --out "$OUT/rivals_steps_calibration_t50.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

# Recurrent's learning rate: 15 in its run script, 50 in its paper.
if want lr-calib && have 15 "lr-calib"; then
    banner "learning-rate tie-break (Recurrent: their script's 15 vs their paper's 50)"
    $PY eval/rivals/steps_calibration.py --axis lr --tools recurrent --timeout "$TIMEOUT" \
        --out "$OUT/rivals_lr_calibration.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

banner "ALL DONE"
ls -la "$OUT"/*.json 2>/dev/null
echo "RIVALS_STAGE_COMPLETE"

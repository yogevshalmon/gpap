#!/usr/bin/env bash
# GPAP vs the GPU gradient samplers on a rented Colab L4, driven from a terminal.
#
# Why not `run_full_eval.sh rivals`: a rented GPU has a deadline, so this driver orders the
# sub-stages by how much each carries the headline, spends the circuit-variant axis only where it
# is cheap, and refuses to start a sub-stage it cannot finish. Everything else — instances, seeds,
# steps, timeouts, T — is identical, so rows are directly comparable to the 4080 run.
#
# The CMSGen arm rides on every row: the patched binary on the very CNF the GPAP arm sampled,
# same n and budget, so this driver IS the campaign and Table 1 is read off these files. The host
# CPU is recorded per row. CMSGEN=none runs the rival comparison alone.
#
#   DEADLINE_H=4.3 bash eval/colab/colab_rivals_l4.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # this file lives in eval/colab/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
PY="${PYTHON:-python3}"

# A fixed OUT, not the GPU slug: the supervisor mirrors this directory by LITERAL path
# (--remote-dir, see colab_supervisor.py) off a VM that dies within the hour, so a slug
# surprise on a new image — no torch, a renamed card, `unknown-gpu` — would write the sweep
# where nothing collects it and lose it with the machine. Same reasoning as
# colab_gpap_circuits_l4.sh. The name is what an L4 slug produced, so nothing moves.
EXPERIMENTS="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m eval.paths experiments)"
: "${EXPERIMENTS:?cannot resolve — python3 must be able to import the repo (eval/paths.py)}"
OUT="${OUT:-$EXPERIMENTS/full_eval/nvidia-l4}"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

# --- identical to run_full_eval.sh's rivals stage --------------------------------------------
# Instance sets: Recurrent-CircuitSAT on every b*/s* sequential circuit (their pins where their
# table names it, our rare pin everywhere); DEMOTIC on its 11 ISCAS-85. What a tool cannot
# process is an error row, omitted with the count stated. b17/b19/b22 are fetched, not
# committed, or a fresh VM reports them as skipped.
bash "$REPO_ROOT/scripts/fetch_benchmarks.sh" itc99 2>&1 | sed 's/^/[bench] /' || \
    echo "[warn] could not fetch ITC'99 b17-b22 — the Recurrent stage will report them as skipped"
SEQ_INSTS="${SEQ_INSTS:-$(cd "$REPO_ROOT/benchmarks/circuits/bench/seq" 2>/dev/null \
    && ls *.bench 2>/dev/null | sed 's/\.bench$//' | tr '\n' ' ')}"
COMB_INSTS="${COMB_INSTS:-paper}"
# 3 seeds: the 4080 rows this run is compared against were produced at median-of-3, and
# matching the protocol exactly is worth more than what it costs.
RSEEDS="${RSEEDS:-5 6 7}"
# 'auto' = each rival at the num_steps its own artifacts specify (rival_common.RIVAL_STEPS).
# Pinning one value for all of them taxes two: both saturate their batch on the first step.
STEPS="${STEPS:-auto}"
# Per-run wall for the rival subprocess, and Recurrent's separate initialization budget. The
# initialization wall bounds the search and is in no published rate (protocol v1.0,
# docs/findings/RIVAL_FAIRNESS_FINDINGS.md §11/§17).
TIMEOUT="${TIMEOUT:-1200}"
INIT_BUDGET="${INIT_BUDGET:-1200}"
# GRANTS=<file> pins each recorded cell's batch rung and Recurrent's start_point to a previous
# campaign's measurements: the rule is unchanged, the search is skipped.
GRANT_ARGS=()
[ -n "${GRANTS:-}" ] && GRANT_ARGS=(--grants "$GRANTS")
# RETRY_ERRORS=1 re-runs errored rows; off by default, because a rival timeout costs the full
# budget to rediscover on every restart.
RETRY_ARGS=()
[ "${RETRY_ERRORS:-0}" = "1" ] && RETRY_ARGS=(--retry-errors)
T="${T:-25}"

# Default off (24 h): a Colab VM is reclaimed after about an hour, so an in-VM deadline guard
# would permanently skip any stage costing more than one VM lifetime. Interruption is handled by
# --resume plus eval/colab/colab_supervisor.py, which mirrors results out each cycle and rebuilds
# the VM. Set DEADLINE_H only for a single-shot run on a machine you own.
# Which sub-stages to run. A finished stage still costs time on a restart, because --resume
# retries error rows and a rival timeout fails again identically. Set STAGES to skip what is
# done, e.g. STAGES="recurrent-rare recurrent-table".
# Circuit-variant axis (rival_common.py): which circuit each side is handed. The protocol is
# ONE variant, `abc`, for every tool and stage — the same sequential-ABC-simplified netlist GPAP
# encodes from. Measured, that choice is nobody's friend, hurting GPAP on one instance and the
# rival on another (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §16c). `VARIANTS_*="raw abc"` is
# still implemented and is not part of the protocol.
VARIANTS_DEM="${VARIANTS_DEM:-abc}"
VARIANTS_RCS="${VARIANTS_RCS:-abc}"
HTS_ENCS="${HTS_ENCS:-tseitin abc}"
# HTS is not in the default set: its paper evaluates no circuit. Nor is `demotic-miter`
# (measured but unpublished, and the most expensive DEMOTIC arm) or `steps-calib` (a calibration,
# not a measurement — add it to re-derive the tie-break on this card).
# The campaign is the four T=25 stages plus the calibrations (protocol v1.0,
# docs/EVAL_RUNBOOK.md §0). T=50 is the rival's own script depth and an additive appendix:
# STAGES="recurrent-rare-t50 recurrent-table-t50 steps-calib-t50", reported beside T=25.
STAGES="${STAGES:-lr-calib demotic-native demotic-comb-rare recurrent-rare recurrent-table steps-calib}"
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) echo "[skip] $1 (not in STAGES)"; return 1 ;; esac; }

# The CMSGen arm gets a hard preflight: a stage resumed without the binary would append armless
# rows to an armed file, and absent is not zero. CMSGEN=none opts out explicitly.
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

# The GPAP-arm cache (rival_common.cached_gpap) survives a VM only if it is mirrored like the
# JSONs, so ship it as one file the supervisor can carry. This is what makes a rival-side re-run
# cheap: most of a campaign's wall is our arm, the start_point probes and sim_baseline, all
# identical across re-runs.
CACHE_TGZ="$OUT/gpap_cache.tgz"
[ -f "$CACHE_TGZ" ] && tar xzf "$CACHE_TGZ" -C "$OUT" 2>/dev/null && \
    echo "[cache] restored $(ls "$OUT/.gpap_cache" 2>/dev/null | wc -l) GPAP arm(s)"
save_cache() { tar czf "$CACHE_TGZ.tmp" -C "$OUT" .gpap_cache 2>/dev/null \
                   && mv "$CACHE_TGZ.tmp" "$CACHE_TGZ"; }
# ...and keep packing it in the background: on the big-circuit tail the VM is reclaimed before
# the stage ends, so a probe settled mid-stage would otherwise be re-paid by every fresh VM. The
# supervisor mirrors the tarball every 3 min; 4 min here keeps the two from thrashing.
( while true; do sleep 240; save_cache; done ) &
CACHE_PACKER=$!
trap 'kill $CACHE_PACKER 2>/dev/null' EXIT

# ---------------------------------------------------------------------------- 1. DEMOTIC
# Cheap per run, so it can afford both circuit variants.
if want demotic-native && have 25 "demotic-native"; then
    banner "DEMOTIC native (their ISCAS-85 pins, raw + abc)"
    $PY eval/rivals/rivals_demotic.py --mode native $COMB_INSTS --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --circuit-variants $VARIANTS_DEM ${GRANT_ARGS[@]+"${GRANT_ARGS[@]}"} ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
        --resume --output "$OUT/rivals_demotic_native.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

# The controlled contrast: the same ISCAS-85 circuits under our rare pin, so against the native
# arm above only the task changes.
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
# The strongest rival and by far the most expensive per row, so it gets one variant only; the
# raw-vs-abc axis measures nearly flat for the rivals (eval/rivals/rival_common.py). Rare first,
# the headline regime; both harnesses checkpoint per row, so a deadline still leaves what
# finished.
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

# ------------------------------------------------------- 4. the num_steps tie-break, on THIS card
# Evidence, not configuration. RIVAL_STEPS pins one of the two values each tool ships, the one
# better for the rival; this re-derives that choice on this card so the campaign carries its own
# proof. A flip is printed and recorded, never acted on automatically.
if want steps-calib && have 30 "steps-calib"; then
    banner "num_steps: script vs argparse vs PAPER (and the paper's paired config), T=$T"
    $PY eval/rivals/steps_calibration.py --timeout "$TIMEOUT" --tools demotic recurrent \
        --cycles "$T" --out "$OUT/rivals_steps_calibration.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi
# The same evidence at the rival's own depth: their Figure 5 shows the unique count rising over
# 5 iterations, so "saturates on the first step" is re-derived at T=50.
if want steps-calib-t50 && have 45 "steps-calib-t50"; then
    banner "num_steps: script vs argparse vs PAPER (and the paired config), T=50"
    $PY eval/rivals/steps_calibration.py --timeout "$TIMEOUT" --tools recurrent \
        --cycles 50 --out "$OUT/rivals_steps_calibration_t50.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

# The learning-rate tie-break on this card. Recurrent ships two values (15 in its run script,
# 50 in its paper), so the num_steps rule applies: run it at whichever is better for the rival.
# Evidence, not configuration — a flip is reported and RIVAL_LR changed deliberately.
if want lr-calib && have 15 "lr-calib"; then
    banner "learning-rate tie-break (Recurrent: their script's 15 vs their paper's 50)"
    $PY eval/rivals/steps_calibration.py --axis lr --tools recurrent --timeout "$TIMEOUT" \
        --out "$OUT/rivals_lr_calibration.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

banner "ALL DONE"
ls -la "$OUT"/*.json 2>/dev/null
echo "RIVALS_STAGE_COMPLETE"

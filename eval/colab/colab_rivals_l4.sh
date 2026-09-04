#!/usr/bin/env bash
# GPAP vs the three GPU gradient samplers, on a Colab L4, driven from a terminal.
#
# Why this exists instead of `run_full_eval.sh rivals`: that stage is one fixed sequence with one
# set of circuit variants, and it took ~7 h on the 4080. A rented GPU has a deadline, so this
# driver (a) orders the sub-stages by how much each one carries the headline, (b) spends the
# expensive `raw abc` variant axis only where it is cheap, and (c) refuses to START a sub-stage
# it cannot finish. Everything else — instances, seeds, steps, timeouts, T — is IDENTICAL to
# run_full_eval.sh's rivals stage, so rows are directly comparable to the 4080 run.
#
# THE CMSGEN ARM RIDES ON EVERY ROW (converged protocol, 2026-09-03): the patched binary on the
# very CNF the GPAP arm sampled, same n and budget — three tools on one constraint per row, and
# Table 1 is read off these files. So this driver IS the campaign; there is no separate CMSGen
# sweep to run. The host CPU is recorded per row (a CMSGen ratio is meaningless without it).
# CMSGEN=none runs the rival comparison alone (no Table 1 from those rows).
#
#   DEADLINE_H=4.3 bash eval/colab/colab_rivals_l4.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # this file lives in eval/colab/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
PY="${PYTHON:-python3}"

GPU_SLUG="$($PY -c "
import re, torch
n = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'cpu'
print(re.sub(r'[^a-z0-9]+', '-', n.lower()).strip('-'))" 2>/dev/null || echo unknown-gpu)"
OUT="${OUT:-$REPO_ROOT/.experiments/full_eval/$GPU_SLUG}"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

# --- identical to run_full_eval.sh's rivals stage --------------------------------------------
# INSTANCE SETS: Recurrent-CircuitSAT on EVERY b*/s* sequential circuit (their pins where their
# table names the instance, our rare pin everywhere; what it cannot process is an error row,
# omitted from the table with the count stated); DEMOTIC on its paper's 11 ISCAS-85 (`paper` ->
# eval/rivals/rival_common.py::PAPER_INSTANCES, the dict paper_tables.py scores from). b17/b19/b22
# are fetched, not committed: pull them first, or a fresh VM reports them as skipped. `itc99`
# pulls only those ~25 MB and is a no-op when they are already installed.
bash "$REPO_ROOT/scripts/fetch_benchmarks.sh" itc99 2>&1 | sed 's/^/[bench] /' || \
    echo "[warn] could not fetch ITC'99 b17-b22 — the Recurrent stage will report them as skipped"
SEQ_INSTS="${SEQ_INSTS:-$(cd "$REPO_ROOT/benchmarks/circuits/bench/seq" 2>/dev/null \
    && ls *.bench 2>/dev/null | sed 's/\.bench$//' | tr '\n' ' ')}"
COMB_INSTS="${COMB_INSTS:-paper}"
# 3 seeds, NOT run_full_eval.sh's current RIVAL_SEEDS=5: the 4080 rows this run is compared
# against were produced at median-of-3, and matching the protocol exactly is worth more than the
# extra circuit-variant axis it costs (which is why Recurrent gets `raw` only below).
RSEEDS="${RSEEDS:-5 6 7}"
# 'auto' = each rival at the num_steps ITS OWN artifacts specify (rival_common.RIVAL_STEPS).
# Pinning one value for all three taxed two of them: the 2026-08-23 campaign ran 10 everywhere,
# which costs DEMOTIC 2.6x and Recurrent 8.5x (both saturate their batch on the first step).
STEPS="${STEPS:-auto}"
TIMEOUT="${TIMEOUT:-300}"
T="${T:-25}"

# Default OFF (24 h). MEASURED: a Colab L4 assignment is reclaimed after ~1 h, so an in-VM
# deadline guard would permanently skip any stage costing more than one VM lifetime. Interruption
# is handled the other way round now — every harness runs with --resume and eval/colab/colab_supervisor.py
# mirrors results to the local disk each cycle, then rebuilds the VM and continues. Set
# DEADLINE_H=<hours> only for a single-shot run on a machine you actually own.
# Which sub-stages to run. A finished stage still costs real time on a restart, because
# --resume retries ERROR rows and a rival timeout fails again identically (~300 s each) --
# measured: 3 HTS timeouts ate ~15 min of every rebuilt VM before the later stages could start.
# Set STAGES to skip what is already done, e.g. STAGES="recurrent-rare recurrent-table".
# Circuit-variant axis (see rival_common.py): which circuit each side is handed. Default keeps
# the recorded protocol (DEMOTIC raw+abc, Recurrent raw-only for cost). Override to converge a
# campaign on one encoding — e.g. VARIANTS_DEM=abc VARIANTS_RCS=abc HTS_ENCS=abc for an
# abc:fast-only sweep (2026-09-02: no tseitin arm anywhere in the tables).
# BOTH circuit variants (2026-09-04). GPAP samples an ABC-optimized CNF, so the rival must be
# able to have the ABC-optimized CIRCUIT; `paper_tables.rival_cells` now scores, per instance,
# whichever variant is BETTER FOR THE RIVAL — the same tie-break `num_steps` uses — instead of
# pooling both (which counted every instance twice, the reason this used to be raw-only).
# Measured: the ABC-simplified sequential netlist is worth geomean 1.34x to DEMOTIC.
VARIANTS_DEM="${VARIANTS_DEM:-raw abc}"
VARIANTS_RCS="${VARIANTS_RCS:-raw abc}"
HTS_ENCS="${HTS_ENCS:-tseitin abc}"
# HTS dropped 2026-09-02: its paper evaluates no circuit at all (see run_full_eval.sh).
# `demotic-miter` is NOT in the default set: it is measured-but-unpublished (see
# paper_tables.RIVALS — its instances are in neither paper), and it is the most expensive
# DEMOTIC arm at ~50 min. Add it back explicitly to refresh the archived rows.
# `steps-calib` (the rival at num_steps 1 vs 10) is a calibration, not a measurement — the
# campaign runs each rival at the ONE setting its own script names (rival_common.RIVAL_STEPS);
# add it to STAGES to re-derive the tie-break on this card.
# BOTH UNROLL DEPTHS on the sequential set (2026-09-04): T=25 was described as the rival's own
# default and it is not — their run script says `--num_clock_cycles 50` and their paper sets
# eta=50, over which their sampler pools every depth, so a shallower unroll hands them a smaller
# pool. Both are run; `paper_tables` reports them as separate rows.
STAGES="${STAGES:-lr-calib demotic-native demotic-comb-rare recurrent-rare recurrent-table \
recurrent-rare-t50 recurrent-table-t50}"
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) echo "[skip] $1 (not in STAGES)"; return 1 ;; esac; }

# The CMSGen arm: a HARD preflight, like colab_cmsgen_circuits_abc_l4.sh — a stage resumed
# without the binary would append armless rows to an armed file (absent is not zero, and
# verify_artifact fails the mix). CMSGEN=none opts out explicitly.
CMSGEN="${CMSGEN:-$REPO_ROOT/.gpap/bin/cmsgen}"
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

# The GPAP-arm cache (rival_common.cached_gpap) lives beside the JSONs, so it survives a VM
# only if it is mirrored like them. Ship it as ONE file the supervisor can carry: unpack whatever
# a previous VM built, and re-pack after every stage. This is what makes a RIVAL-side re-run
# cheap — of the 2026-08-23 campaign's ~3.3 h, only 64 min was the rivals' own subprocesses; the
# rest was our arm, the start_point probes and sim_baseline, all of it identical across re-runs.
CACHE_TGZ="$OUT/gpap_cache.tgz"
[ -f "$CACHE_TGZ" ] && tar xzf "$CACHE_TGZ" -C "$OUT" 2>/dev/null && \
    echo "[cache] restored $(ls "$OUT/.gpap_cache" 2>/dev/null | wc -l) GPAP arm(s)"
save_cache() { tar czf "$CACHE_TGZ.tmp" -C "$OUT" .gpap_cache 2>/dev/null \
                   && mv "$CACHE_TGZ.tmp" "$CACHE_TGZ"; }

# ---------------------------------------------------------------------------- 1. DEMOTIC
# Cheap (median 3.7-15.3 s per run), so it gets BOTH circuit variants.
if want demotic-native && have 25 "demotic-native"; then
    banner "DEMOTIC native (their ISCAS-85 pins, raw + abc)"
    $PY eval/rivals/rivals_demotic.py --mode native $COMB_INSTS --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --circuit-variants $VARIANTS_DEM ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
        --resume --output "$OUT/rivals_demotic_native.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

# THE CONTROLLED CONTRAST: their own ISCAS-85 circuits under OUR rare pin. Paired with the
# native arm above, the circuit is held fixed and only the TASK changes.
if want demotic-comb-rare && have 25 "demotic-comb-rare"; then
    banner "DEMOTIC (their ISCAS-85 circuits, OUR rare pin)"
    $PY eval/rivals/rivals_demotic.py --mode miter $COMB_INSTS -T 1 --target rare \
        --bench-dir benchmarks/circuits/bench/comb \
        --seeds $RSEEDS --steps "$STEPS" --timeout "$TIMEOUT" --circuit-variants $VARIANTS_DEM \
        ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
        --resume --output "$OUT/rivals_demotic_comb_rare.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

if want demotic-miter && have 50 "demotic-miter"; then
    banner "DEMOTIC miter (our rare fixed-T instances, raw + abc)"
    $PY eval/rivals/rivals_demotic.py --mode miter $SEQ_INSTS -T "$T" --target rare \
        --seeds $RSEEDS --steps "$STEPS" --timeout "$TIMEOUT" --circuit-variants $VARIANTS_DEM \
        ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
        --resume --output "$OUT/rivals_demotic_miter.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

# ---------------------------------------------------------------------------- 2. HTS
# A CNF consumer, so its "circuit variant" is the ENCODER: build both CNFs it should see.
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
# The strongest rival and by far the most expensive (~15 min/row on the 4080), so it gets the
# `raw` variant only. The 4080 run has raw rows too, so the cross-GPU comparison is unaffected;
# what is lost is the raw-vs-abc axis, which measured nearly flat for the rivals anyway
# (b03 129->127 ANDs) — the note lives in eval/rivals/rival_common.py.
# RARE FIRST: it is the headline regime. Both harnesses now checkpoint per row, so a sub-stage
# cut off by the deadline still leaves every row it finished.
if want recurrent-rare && have 60 "recurrent-rare"; then
    banner "Recurrent-CircuitSAT — OUR rare pins (raw)"
    $PY eval/rivals/rivals_recurrent.py $SEQ_INSTS -T "$T" --rare --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --circuit-variants $VARIANTS_RCS ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
        --resume --output "$OUT/rivals_recurrent_rare.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

if want recurrent-table && have 30 "recurrent-table"; then
    banner "Recurrent-CircuitSAT — THEIR target table (raw)"
    $PY eval/rivals/rivals_recurrent.py $SEQ_INSTS -T "$T" --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --circuit-variants $VARIANTS_RCS ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
        --resume --output "$OUT/rivals_recurrent_table.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

if want recurrent-rare-t50 && have 90 "recurrent-rare-t50"; then
    banner "Recurrent-CircuitSAT — OUR rare pins, T=50 (their own depth limit)"
    $PY eval/rivals/rivals_recurrent.py $SEQ_INSTS -T 50 --rare --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --circuit-variants $VARIANTS_RCS ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
        --resume --output "$OUT/rivals_recurrent_rare_t50.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

if want recurrent-table-t50 && have 45 "recurrent-table-t50"; then
    banner "Recurrent-CircuitSAT — THEIR target table, T=50 (their own depth limit)"
    $PY eval/rivals/rivals_recurrent.py $SEQ_INSTS -T 50 --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --circuit-variants $VARIANTS_RCS ${CMS_ARGS[@]+"${CMS_ARGS[@]}"} \
        --resume --output "$OUT/rivals_recurrent_table_t50.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

# ------------------------------------------------------- 4. the num_steps tie-break, on THIS card
# EVIDENCE, not configuration. RIVAL_STEPS pins one of the two values each tool itself ships,
# choosing the one better for the rival; that choice was first measured on an RTX 4080. This
# re-derives it here so the campaign carries its own proof instead of importing one. A FLIP is
# printed and recorded, never acted on automatically. Last, because it needs the HTS CNFs and
# because nothing else depends on it.
if want steps-calib && have 15 "steps-calib"; then
    banner "num_steps tie-break (each tool's own two values)"
    $PY eval/rivals/steps_calibration.py --timeout "$TIMEOUT" \
        --hts-cnfs "$OUT/hts_cnf/*.cnf" \
        --out "$OUT/rivals_steps_calibration.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

# THE LEARNING-RATE TIE-BREAK, on this card. Recurrent ships two values for the knob a gradient
# method depends on most — 15 in its run script, 50 in its paper text — so the rule that applies
# to num_steps applies here: run it at whichever is better FOR THE RIVAL. Cheap (one instance,
# two arms, our arm pinned to a token budget) and it is EVIDENCE, not configuration: a flip is
# reported loudly and RIVAL_LR is changed deliberately, never mid-sweep.
if want lr-calib && have 15 "lr-calib"; then
    banner "learning-rate tie-break (Recurrent: their script's 15 vs their paper's 50)"
    $PY eval/rivals/steps_calibration.py --axis lr --tools recurrent --timeout "$TIMEOUT" \
        --out "$OUT/rivals_lr_calibration.json" 2>&1 | tee -a "$LOGS/rivals.log"
    save_cache
fi

banner "ALL DONE"
ls -la "$OUT"/*.json 2>/dev/null
echo "RIVALS_STAGE_COMPLETE"

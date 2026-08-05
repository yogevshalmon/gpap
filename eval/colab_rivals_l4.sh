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
# CMSGen is deliberately absent: the CPU baseline belongs on the HPC cluster, not on a Colab vCPU.
#
#   DEADLINE_H=4.3 bash eval/colab_rivals_l4.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
SEQ_INSTS="${SEQ_INSTS:-b03 b04 b10 b11 b12 b13 s386 s832 s1196}"
COMB_INSTS="${COMB_INSTS:-c432 c499 c880 c1355 c1908 c2670 c3540 c5315 c6288 c7552}"
# 3 seeds, NOT run_full_eval.sh's current RIVAL_SEEDS=5: the 4080 rows this run is compared
# against were produced at median-of-3, and matching the protocol exactly is worth more than the
# extra circuit-variant axis it costs (which is why Recurrent gets `raw` only below).
RSEEDS="${RSEEDS:-5 6 7}"
STEPS="${STEPS:-10}"
TIMEOUT="${TIMEOUT:-300}"
T="${T:-25}"

# Default OFF (24 h). MEASURED: a Colab L4 assignment is reclaimed after ~1 h, so an in-VM
# deadline guard would permanently skip any stage costing more than one VM lifetime. Interruption
# is handled the other way round now — every harness runs with --resume and eval/colab_supervisor.py
# mirrors results to the local disk each cycle, then rebuilds the VM and continues. Set
# DEADLINE_H=<hours> only for a single-shot run on a machine you actually own.
# Which sub-stages to run. A finished stage still costs real time on a restart, because
# --resume retries ERROR rows and a rival timeout fails again identically (~300 s each) --
# measured: 3 HTS timeouts ate ~15 min of every rebuilt VM before the later stages could start.
# Set STAGES to skip what is already done, e.g. STAGES="recurrent-rare recurrent-table".
STAGES="${STAGES:-demotic-native demotic-miter hts recurrent-rare recurrent-table}"
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) echo "[skip] $1 (not in STAGES)"; return 1 ;; esac; }

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

# ---------------------------------------------------------------------------- 1. DEMOTIC
# Cheap (median 3.7-15.3 s per run), so it gets BOTH circuit variants.
if want demotic-native && have 25 "demotic-native"; then
    banner "DEMOTIC native (their ISCAS-85 pins, raw + abc)"
    $PY eval/rivals_demotic.py --mode native $COMB_INSTS --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --circuit-variants raw abc \
        --resume --output "$OUT/rivals_demotic_native.json" 2>&1 | tee -a "$LOGS/rivals.log"
fi

if want demotic-miter && have 50 "demotic-miter"; then
    banner "DEMOTIC miter (our rare fixed-T instances, raw + abc)"
    $PY eval/rivals_demotic.py --mode miter $SEQ_INSTS -T "$T" --target rare \
        --seeds $RSEEDS --steps "$STEPS" --timeout "$TIMEOUT" --circuit-variants raw abc \
        --resume --output "$OUT/rivals_demotic_miter.json" 2>&1 | tee -a "$LOGS/rivals.log"
fi

# ---------------------------------------------------------------------------- 2. HTS
# A CNF consumer, so its "circuit variant" is the ENCODER: build both CNFs it should see.
if want hts && have 40 "hts"; then
    banner "HTS (same CNFs, same GPU)"
    HTS_DIR="$OUT/hts_cnf"; mkdir -p "$HTS_DIR"
    for inst in $SEQ_INSTS; do
        src="benchmarks/circuits/bench/seq/$inst.bench"
        [ -f "$src" ] || continue
        for enc in tseitin abc; do
            cnf="$HTS_DIR/$inst-c$T-$enc.cnf"
            [ -f "$cnf" ] || $PY -m gpap.cli circuit2cnf "$src" "$cnf" \
                --cycles "$T" --target rare --encoder "$enc" --abc-opt fast \
                >/dev/null 2>&1 || echo "[warn] could not build $enc CNF for $inst"
        done
    done
    $PY eval/rivals_hts.py "$HTS_DIR/*.cnf" --timeout "$TIMEOUT" \
        --resume --output "$OUT/rivals_hts.json" 2>&1 | tee -a "$LOGS/rivals.log"
fi

# ---------------------------------------------------------------------------- 3. Recurrent
# The strongest rival and by far the most expensive (~15 min/row on the 4080), so it gets the
# `raw` variant only. The 4080 run has raw rows too, so the cross-GPU comparison is unaffected;
# what is lost is the raw-vs-abc axis, which measured nearly flat for the rivals anyway
# (b03 129->127 ANDs) — the note lives in eval/rival_common.py.
# RARE FIRST: it is the headline regime. Both harnesses now checkpoint per row, so a sub-stage
# cut off by the deadline still leaves every row it finished.
if want recurrent-rare && have 60 "recurrent-rare"; then
    banner "Recurrent-CircuitSAT — OUR rare pins (raw)"
    $PY eval/rivals_recurrent.py $SEQ_INSTS -T "$T" --rare --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --circuit-variants raw \
        --resume --output "$OUT/rivals_recurrent_rare.json" 2>&1 | tee -a "$LOGS/rivals.log"
fi

if want recurrent-table && have 30 "recurrent-table"; then
    banner "Recurrent-CircuitSAT — THEIR target table (raw)"
    $PY eval/rivals_recurrent.py $SEQ_INSTS -T "$T" --seeds $RSEEDS --steps "$STEPS" \
        --timeout "$TIMEOUT" --circuit-variants raw \
        --resume --output "$OUT/rivals_recurrent_table.json" 2>&1 | tee -a "$LOGS/rivals.log"
fi

banner "ALL DONE"
ls -la "$OUT"/*.json 2>/dev/null
echo "RIVALS_STAGE_COMPLETE"

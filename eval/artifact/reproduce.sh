#!/usr/bin/env bash
# THE ARTIFACT'S SINGLE ENTRY POINT — one script, one step per invocation.
#
#   eval/artifact/reproduce.sh check      # what this machine can do, and what each tier costs
#   eval/artifact/reproduce.sh setup      # build/fetch everything (network; ~20-30 min)
#   eval/artifact/reproduce.sh tables     # every table and figure, from the SHIPPED rows (CPU)
#   eval/artifact/reproduce.sh smoke      # 4 cells end to end on the GPU (~3 min)
#   eval/artifact/reproduce.sh subset     # the reviewer tier: 66 of 142 cells, T=25 (~2.5 h)
#   eval/artifact/reproduce.sh full       # the whole campaign, T=25 and T=50 (~12 h)
#   eval/artifact/reproduce.sh compare    # re-run the comparison over an existing --out
#
# Options, on either side of the step: --vram-cap l4|none|MiB   --out DIR   --reference DIR
#                                      --grants FILE|none    --remeasure-failures  --seeds "5 6 7"
#
# WHAT REPRODUCES AND WHAT DOES NOT. Rates are a property of the card: the same engine measures
# 1.82x on an A100 and 0.84x on an L4 against one 4080. So this does not ask a reviewer to
# reproduce a geomean. It asks for the three things that must hold on any machine, and
# `compare_campaign.py` is the gate: the same formula per cell, every emitted sample sound and
# replayable through the original circuit, and the same per-cell verdicts. The measured rates
# are printed as this card's speed factor against ours — evidence about the hardware, not a
# claim about the tool.
#
# THE BATCH IS NOT SEARCHED AGAIN. Each rival cell's batch rung, and Recurrent's start_point,
# are read from eval/rivals/grants/l4_protocol_20260904.json — the values the published campaign
# measured. A pinned rung this card cannot finish on is walked down from, and the row says so
# (`grant_source`); the comparator counts those cells and warns, because a rival handed a
# smaller batch than the paper gave it makes OUR ratio look better.
#
# HARDWARE. A 24 GB card reproduces the grants as recorded. Every one of the 92 pinned Recurrent
# rungs sits at the reference L4's memory ceiling (the next rung up OOM'd there), so a smaller
# card walks down on most of them. --vram-cap works the other way: on a LARGER card it holds
# our own batch to what the L4 allowed, since K is memory-bound on 63 of the campaign's 141 rows
# and an uncapped 80 GB card would hand us more parallelism while the rival stays pinned.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export PYTHONPATH="${PYTHONPATH:-.}"

# The GPAP-side VRAM budget the published campaign actually ran under: its rows record 23.45 GB
# free on every one of the 142, and the engine's model is (free - 800 MB reserve) x 0.80.
# Re-derive with:  eval/artifact/estimate_wall.py  (it reads the same rows).
L4_BUDGET_MIB=17280

VRAM_CAP="l4"
OUT=""
# The published rows, looked up in the order that makes the packaged artifact work with nothing
# set: an explicit override, then the development layout (results beside the checkout), then the
# copy `make_artifact.sh` bundles INSIDE the artifact.
REFERENCE="${GPAP_REFERENCE:-}"
if [ -z "$REFERENCE" ]; then
    for cand in "${GPAP_RESULTS:-$REPO_ROOT/../gpap-results}/paper/data" \
                "$REPO_ROOT/artifact-data/paper/data"; do
        [ -d "$cand" ] && { REFERENCE="$cand"; break; }
    done
    REFERENCE="${REFERENCE:-$REPO_ROOT/artifact-data/paper/data}"
fi
GRANTS="eval/rivals/grants/l4_protocol_20260904.json"
HONOUR_FAILURES=1                 # the artifact's default; --remeasure-failures turns it off
SEEDS="5 6 7"

# Options are accepted on EITHER side of the step name: `reproduce.sh --out D tables` and
# `reproduce.sh tables --out D` both work. A parser that only read the left-hand side dropped a
# trailing `--out` silently and wrote to the default directory, which is a wrong answer with no
# error message.
STEP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --vram-cap)  VRAM_CAP="$2"; shift 2 ;;
        --out)       OUT="$2"; shift 2 ;;
        --reference) REFERENCE="$2"; shift 2 ;;
        --grants)    GRANTS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --remeasure-failures) HONOUR_FAILURES=0; shift ;;
        -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
        --*)         echo "unknown option $1" >&2; sed -n '2,40p' "$0"; exit 2 ;;
        *)           [ -z "$STEP" ] || { echo "two steps given: $STEP and $1" >&2; exit 2; }
                     STEP="$1"; shift ;;
    esac
done
STEP="${STEP:-check}"

case "$VRAM_CAP" in
    l4)   export GPAP_VRAM_BUDGET_MIB="$L4_BUDGET_MIB" ;;
    none) unset GPAP_VRAM_BUDGET_MIB ;;
    ''|*[!0-9]*) echo "--vram-cap takes l4, none, or a number of MiB (got '$VRAM_CAP')" >&2; exit 2 ;;
    *)    export GPAP_VRAM_BUDGET_MIB="$VRAM_CAP" ;;
esac

GPU_SLUG="$($PY -c "
import re, torch
n = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'cpu'
print(re.sub(r'[^a-z0-9]+', '-', n.lower()).strip('-'))" 2>/dev/null || echo unknown-gpu)"
OUT="${OUT:-$REPO_ROOT/.experiments/artifact/$GPU_SLUG-$STEP}"
SUBSET_JSON="$REPO_ROOT/eval/artifact/subset.json"

say() { echo; echo "=== $* ==="; }

# ---------------------------------------------------------------------------- readiness
step_check() {
    say "environment"
    $PY -m gpap.cli info || echo "[FAIL] gpap info did not run — is the package installed?"
    say "components"
    bash eval/rivals/rivals_setup.sh check
    say "benchmarks"
    local nseq ncomb
    nseq="$(ls benchmarks/circuits/bench/seq/*.bench 2>/dev/null | wc -l)"
    ncomb="$(ls benchmarks/circuits/bench/comb/*.bench 2>/dev/null | wc -l)"
    echo "sequential (ITC'99 b* + ISCAS-89 s*): $nseq/30   combinational (ISCAS-85): $ncomb/11"
    [ "$nseq" -lt 30 ] && echo "  b17-b22 are fetched, not committed: scripts/fetch_benchmarks.sh itc99"
    say "GPU memory vs the reference card"
    $PY - <<'EOF'
import os
try:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device — only the `tables` step can run here")
    free, total = torch.cuda.mem_get_info()
    cap = int(os.environ.get("GPAP_VRAM_BUDGET_MIB") or 0)
    print(f"this card: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
    print(f"reference: NVIDIA L4, 23.7 GB total, 23.45 GB free on every recorded row")
    if cap:
        print(f"VRAM cap : {cap} MiB "
              f"({'binds — our batch is held to the L4 allowance' if (free-8e8)*0.8 >= cap*(1<<20) else 'inert — this card has less than the cap anyway'})")
    if total < 23e9:
        print("WARNING: under 24 GB. Every one of the 92 pinned Recurrent rungs sits at the\n"
              "         reference card's memory ceiling, so most cells will walk DOWN a rung.\n"
              "         The run still completes and every walked-down cell is reported, but the\n"
              "         rival gets less parallelism there than the paper gave it.")
except ImportError:
    raise SystemExit("torch is not installed — only the `tables` step can run here")
EOF
    say "reference rows"
    if [ -d "$REFERENCE" ]; then
        echo "$REFERENCE  ($(ls "$REFERENCE"/rivals_*.json 2>/dev/null | wc -l) stage files)"
    else
        echo "MISSING: $REFERENCE — the tables and compare steps need the published rows"
    fi
    say "what each tier costs (estimated from the reference rows, on the reference card)"
    $PY eval/artifact/estimate_wall.py "$REFERENCE" \
        --mode "$([ "$HONOUR_FAILURES" = 1 ] && echo honour || echo remeasure)" 2>/dev/null \
        || echo "(no reference rows — estimate unavailable)"
    # The measured parts come from the reference card; the overhead terms are CPU-bound and were
    # measured on one host, so a machine with a different CPU pays a different constant.
    echo
    echo "(the overhead terms are CPU-bound and were measured on one host: pass --cpu-scale N to"
    echo " estimate_wall.py to carry them to a slower (N>1) or faster (N<1) CPU)"
}

# ---------------------------------------------------------------------------- build/fetch
step_setup() {
    say "the engine (libgpap: cmake + a C++17 compiler + the NVIDIA driver; no CUDA toolkit)"
    scripts/build_core.sh || { echo "[FAIL] libgpap did not build — nothing else can run"; exit 1; }
    say "abc, patched CMSGen, and the pinned rival checkouts"
    bash eval/rivals/rivals_setup.sh
    say "benchmarks not committed to the repository (ITC'99 b17-b22)"
    bash scripts/fetch_benchmarks.sh itc99
    step_check
}

# ---------------------------------------------------------------------------- tier 1: tables
# Regenerates every published table and figure from rows, on any machine: no GPU, no CUDA
# wheels. `verify_artifact.py` is the publishability gate and prints the caveats the paper's
# text is required to carry.
step_tables() {
    local data="${1:-$REFERENCE}" out="${2:-$OUT}"
    [ -d "$data" ] || { echo "no rows at $data"; exit 1; }
    mkdir -p "$out"
    say "tables 1/2/3 (+4b where the Barbarik rows are present) -> $out"
    local bar=("$data"/barbarik/circ_abc_*.jsonl)
    local barargs=(); [ -e "${bar[0]}" ] && barargs=(--barbarik "${bar[@]}")
    $PY eval/artifact/paper_tables.py --rivals "$data" ${barargs[@]+"${barargs[@]}"} \
        --out-md "$out/RESULTS.md" --out-tex "$out/tables.tex" | tail -3
    say "the publishability gate"
    $PY eval/artifact/verify_artifact.py --rivals "$data" ${barargs[@]+"${barargs[@]}"} \
        2>&1 | tee "$out/VERIFICATION.txt" | tail -4
    say "emission-symmetry audit (no tool timed while delivering samples)"
    $PY eval/rivals/check_emission_symmetry.py --results "$data" --cycles 25 \
        2>&1 | tee "$out/emission_audit.txt" | tail -2
    say "figures"
    $PY eval/eval_plots.py "$data" --out-dir "$out/plots" 2>&1 | tail -4
    echo
    echo "wrote $out/{RESULTS.md,tables.tex,VERIFICATION.txt,emission_audit.txt,plots/}"
}

# ---------------------------------------------------------------------------- GPU tiers
# One place invokes the harnesses: eval/run_full_eval.sh, the driver the published campaign ran.
# This only chooses the cell set and the two artifact defaults (the VRAM cap and the pinned
# failures), so a tier here cannot drift from the protocol.
run_rivals() {
    local seq_insts="$1" depths="$2" comb="${3:-paper}"
    mkdir -p "$OUT"
    echo "output        : $OUT"
    echo "grants        : $GRANTS"
    echo "recorded fails: $([ "$HONOUR_FAILURES" = 1 ] && echo 'PINNED (--remeasure-failures re-derives them)' || echo 're-measured')"
    echo "VRAM cap      : ${GPAP_VRAM_BUDGET_MIB:-none} MiB"
    echo "seeds         : $SEEDS"
    echo "instances     : $(echo "$seq_insts" | wc -w) sequential, $(echo "$comb" | wc -w) combinational entry, depths $depths"
    OUT="$OUT" RIVAL_SEQ="$seq_insts" RIVAL_COMB="$comb" RIVAL_SEEDS="$SEEDS" \
        RIVAL_T=25 RIVAL_T2="$depths" RIVAL_GRANTS="$GRANTS" \
        RIVAL_HONOUR_FAILURES="$HONOUR_FAILURES" VERIFY_SOUND=1 \
        bash eval/run_full_eval.sh rivals
}

subset_instances() {
    [ -f "$SUBSET_JSON" ] || { echo "missing $SUBSET_JSON" >&2; exit 1; }
    $PY -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['instances']))" "$SUBSET_JSON"
}

step_smoke() {
    say "smoke: one instance per stage, everything else exactly as the real tiers run it"
    run_rivals "b03" none "c432"
    step_after_run
}

step_subset() {
    local insts; insts="$(subset_instances)"
    say "subset: the instances whose reference wall is under 10 minutes for both T=25 tasks"
    $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['rule'])" "$SUBSET_JSON"
    run_rivals "$insts" none
    step_after_run
}

step_full() {
    local insts
    insts="$(cd benchmarks/circuits/bench/seq && ls *.bench | sed 's/\.bench$//' | tr '\n' ' ')"
    say "full: every cell of the published campaign, T=25 and T=50"
    run_rivals "$insts" 50
    step_after_run
}

# Tables over the FRESH rows, then the cell-by-cell comparison against the published ones.
step_after_run() {
    step_tables "$OUT" "$OUT"
    step_compare
    # The last line a supervisor greps for. A rented GPU is reclaimed mid-run, so the driver has
    # to say "finished" rather than "stopped": every step above is resumable, and only this line
    # distinguishes a completed tier from a VM that died between two cells.
    echo
    echo "ARTIFACT_TIER_COMPLETE"
}

step_compare() {
    [ -d "$REFERENCE" ] || { echo "no reference rows at $REFERENCE — cannot compare"; return 1; }
    say "comparison against the published campaign"
    $PY eval/artifact/compare_campaign.py "$REFERENCE" "$OUT" --json "$OUT/comparison.json" \
        | tee "$OUT/COMPARISON.txt"
}

case "$STEP" in
    check)   step_check ;;
    setup)   step_setup ;;
    tables)  step_tables ;;
    smoke)   step_smoke ;;
    subset)  step_subset ;;
    full)    step_full ;;
    compare) step_compare ;;
    *)       echo "unknown step '$STEP'" >&2; sed -n '2,35p' "$0"; exit 2 ;;
esac

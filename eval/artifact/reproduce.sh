#!/usr/bin/env bash
# The artifact's entry point: one step per invocation.
#
#   eval/artifact/reproduce.sh check      # what this machine can do, and what each tier costs
#   eval/artifact/reproduce.sh setup      # build/fetch everything (network; ~20-30 min)
#   eval/artifact/reproduce.sh tables     # every table and figure, from the shipped rows (CPU)
#   eval/artifact/reproduce.sh smoke      # 4 cells end to end on the GPU (~3 min)
#   eval/artifact/reproduce.sh subset     # 66 of 142 cells, T=25 (~2.5 h)
#   eval/artifact/reproduce.sh full       # the whole campaign, T=25 and T=50 (~12 h)
#   eval/artifact/reproduce.sh compare    # re-run the comparison over an existing --out
#
# `tables` and each GPU tier write SUMMARY.txt, the lines to check, at the top of --out when they
# end, and everything else under details/.
#
# Options, on either side of the step: --vram-cap l4|none|MiB   --out DIR   --reference DIR
#                                      --grants FILE|none    --remeasure-failures  --seeds "5 6 7"
#                                      --tier smoke|subset|full  (for `compare`: how many cells
#                                      the directory should hold; a tier sets this itself)
#
# Rates are a property of the card, so no tier asks for a geomean back. compare_campaign.py
# checks what must hold on any machine: the same formula per cell, every sample sound and
# replayable through the circuit, and the same per-cell verdicts. Rates print as this card's
# speed factor against ours.
#
# Each rival cell's batch and Recurrent's start_point are the published campaign's
# (eval/rivals/grants/l4_protocol_20260904.json), not searched again. On a card under the 24 GB
# reference a pinned batch that does not fit walks down a rung, and the comparison names the cell,
# since a smaller rival batch flatters our ratio. On a larger card --vram-cap holds our own batch
# to the reference's (docs/findings/ARTIFACT_FINDINGS.md §1, §2).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INVOKED_FROM="$PWD"            # every step runs from the repo root; the OPTIONS do not
cd "$REPO_ROOT"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export PYTHONPATH="${PYTHONPATH:-.}"

# The GPAP-side VRAM budget the published campaign ran under: its rows record 23.45 GB free, and
# the engine's model is (free - 800 MB reserve) x 0.80. estimate_wall.py re-derives it.
L4_BUDGET_MIB=17280

VRAM_CAP="l4"
OUT=""
TIER=""                           # set by each GPU tier; sizes the comparator's --expect-cells
EXPECT_CELLS=""
# The published rows: an override, then the results beside the checkout, then the copy
# make_artifact.sh bundles inside the artifact.
REFERENCE="${GPAP_REFERENCE:-}"
if [ -z "$REFERENCE" ]; then
    # Tested for rows, not a directory, so an empty results tree cannot shadow the bundled copy.
    # Ask, never recompute: eval/paths.py is the one definition.
    RESULTS="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m eval.paths results)"
    : "${RESULTS:?cannot resolve — python3 must be able to import the repo (gpap/paths.py, eval/paths.py)}"
    for cand in "$RESULTS/paper/data" \
                "$REPO_ROOT/artifact-data/paper/data"; do
        [ -n "$(ls "$cand"/rivals_*.json 2>/dev/null)" ] && { REFERENCE="$cand"; break; }
    done
    REFERENCE="${REFERENCE:-$REPO_ROOT/artifact-data/paper/data}"
fi
GRANTS="eval/rivals/grants/l4_protocol_20260904.json"
HONOUR_FAILURES=1                 # the artifact's default; --remeasure-failures turns it off
SEEDS="5 6 7"

# The header above is the help text, up to the first line that is not a comment.
usage() { sed -n '2,${/^[^#]/q;p;}' "$0"; }

# Options go on either side of the step name.
STEP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --vram-cap)  VRAM_CAP="$2"; shift 2 ;;
        --out)       OUT="$2"; shift 2 ;;
        --reference) REFERENCE="$2"; shift 2 ;;
        --grants)    GRANTS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --remeasure-failures) HONOUR_FAILURES=0; shift ;;
        --tier)      TIER="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        --*)         echo "unknown option $1" >&2; usage; exit 2 ;;
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
EXPERIMENTS="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m eval.paths experiments)"
: "${EXPERIMENTS:?cannot resolve — python3 must be able to import the repo (eval/paths.py)}"
OUT="${OUT:-$EXPERIMENTS/artifact/$GPU_SLUG-$STEP}"
SUBSET_JSON="$REPO_ROOT/eval/artifact/subset.json"

# Every step runs from the repo root, so a relative --out or --reference is resolved where it was
# typed.
case "$OUT" in       /*) ;; *) OUT="$INVOKED_FROM/$OUT" ;; esac
case "$REFERENCE" in /*) ;; *) REFERENCE="$INVOKED_FROM/$REFERENCE" ;; esac
DETAILS="$OUT/details"
SUMMARY="$OUT/SUMMARY.txt"

say() { echo; echo "=== $* ==="; }

# The failure ledger. No `set -e`: a tier runs every step and reports. A recorded failure still
# sets the exit code and suppresses the completion marker a supervisor waits for.
FAILED=()
fail() { FAILED+=("$1"); echo "[FAIL] $1" >&2; }

# Prints a step's lines and copies them into NOTE_TO, when set, without blank lines, rules and
# `wrote` lines. Always succeeds, so a pipeline ending in `| note` fails when its producer does.
note() {
    local text; text="$(cat)"; echo "$text"
    if [ -n "${NOTE_TO:-}" ]; then
        printf '%s\n' "$text" | grep -v -E '^[[:space:]]*$|^=+$|^wrote ' >>"$NOTE_TO" || true
    fi
}

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
    print(f"this card: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
    print(f"reference: NVIDIA L4, 23.7 GB total, 23.45 GB free on every recorded row")
    # The warning first: the cap probe below asks the engine, which `setup` may not have built.
    # The memory model is the engine's (cpp/src/calibrate.cpp), never copied here.
    if total < 23e9:
        print("WARNING: under 24 GB. Every one of the 92 pinned Recurrent rungs sits at the\n"
              "         reference card's memory ceiling, so most cells will walk DOWN a rung.\n"
              "         The run still completes and every walked-down cell is reported, but the\n"
              "         rival gets less parallelism there than the paper gave it.")
    from gpap.engine.calibrate import gpu_mem_available, vram_budget_cap_mib
    cap = vram_budget_cap_mib()
    if cap:
        try:
            got = gpu_mem_available(torch.device("cuda", 0))
        except Exception as exc:                      # the core is not built yet
            print(f"VRAM cap : {cap:,} MiB — whether it binds on this card is known once the\n"
                  f"           engine is built (`setup`); not yet: {type(exc).__name__}")
        else:
            print(f"VRAM cap : {cap:,} MiB "
                  + ("binds — our batch is held to the reference card's allowance"
                     if got >= (cap << 20) else
                     "inert — this card's own budget is already below the cap"))
    elif os.environ.get("GPAP_VRAM_BUDGET_MIB"):
        print(f"VRAM cap : IGNORED — GPAP_VRAM_BUDGET_MIB="
              f"{os.environ['GPAP_VRAM_BUDGET_MIB']!r} is not a plain number of MiB")
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
    echo "grants        : $GRANTS"
    echo "recorded fails: $([ "$HONOUR_FAILURES" = 1 ] && echo PINNED || echo re-measured)"
    if [ "$GRANTS" = none ]; then
        echo
        echo "NOTE: with --grants none the estimate below does NOT apply. Every rival cell then"
        echo "      re-walks the batch ladder and re-probes its start_point, which is the search"
        echo "      the grants file exists to remove; expect several times these numbers."
    fi
    $PY eval/artifact/estimate_wall.py "$REFERENCE" \
        --mode "$([ "$HONOUR_FAILURES" = 1 ] && echo honour || echo remeasure)" 2>/dev/null \
        || echo "(no reference rows — estimate unavailable)"
    echo
    echo "(the overhead terms are CPU-bound and were measured on one host: pass --cpu-scale N to"
    echo " estimate_wall.py to carry them to a slower (N>1) or faster (N<1) CPU)"
}

# ---------------------------------------------------------------------------- build/fetch
step_setup() {
    say "the engine (libgpap: cmake + a C++17 compiler + the NVIDIA driver; no CUDA toolkit)"
    scripts/build_core.sh || { echo "[FAIL] libgpap did not build — nothing else can run"; exit 1; }
    say "abc, patched CMSGen, and the pinned rival checkouts"
    # Its own failures are warnings except two, which exit 1: no libgpap, or an unpatched
    # Recurrent (whose clock would carry its host copy).
    bash eval/rivals/rivals_setup.sh || { echo "[FAIL] rival setup failed — see above"; exit 1; }
    say "benchmarks not committed to the repository (ITC'99 b17-b22)"
    bash scripts/fetch_benchmarks.sh itc99
    step_check
}

# ---------------------------------------------------------------------------- tier 1: tables
# The paper's tables and figures from the rows, on any machine (no GPU): RESULTS.md, the check of
# every number the PDF prints, and verify_artifact.py's checks on the measurements. The console
# and SUMMARY.txt get one line from each; the rest goes to tables.log beside the files.
step_tables() {
    # $3 = "fresh": a re-measurement's own rows, held to the published rows by step_compare, never
    # to the PDF (their rates differ, and smoke and subset have no T=50 rows).
    local data="${1:-$REFERENCE}" out="${2:-$OUT}" fresh="${3:-}"
    # Rows, not a directory: over an empty one every generator used to pass.
    [ -n "$(ls "$data"/rivals_*.json 2>/dev/null)" ] || {
        fail "no rivals_*.json in $data — the tables need the campaign's rows"; return 1; }
    mkdir -p "$out"
    local log="$out/tables.log"
    : >"$log"
    # The rows must be the frozen rows, or an edited row regenerates tables that still pass. No
    # manifest is fine: a fresh campaign's rows have none.
    if [ -f "$data/MANIFEST.sha256" ]; then
        ( cd "$data" && sha256sum --quiet -c MANIFEST.sha256 ) >>"$log" 2>&1 \
            || { fail "$data does not match its MANIFEST.sha256 — these are not the frozen rows,
       so every table would describe something else. Restore them, or regenerate the
       manifest deliberately if you meant to change a row."; return 1; }
        echo "the rows match $data/MANIFEST.sha256" >>"$log"
    else
        echo "no MANIFEST.sha256 in $data — nothing pins these rows (fine for your own campaign)" \
            >>"$log"
    fi
    local bar=("$data"/barbarik/circ_abc_*.jsonl)
    local barargs=(); [ -e "${bar[0]}" ] && barargs=(--barbarik "${bar[@]}")
    $PY eval/artifact/paper_tables.py --rivals "$data" ${barargs[@]+"${barargs[@]}"} \
        --out-md "$out/RESULTS.md" --out-json "$out/paper_numbers.json" >>"$log" 2>&1 \
        || { fail "the table generator failed over $data — see $log"; return 1; }
    tail -1 "$log" | note                     # the cell count, the generator's last line
    # Does the paper still print what these rows give? Its tables are typeset by hand, so nothing
    # else checks. Fatal.
    if [ -z "$fresh" ]; then
        $PY eval/artifact/check_paper_numbers.py --generated "$out/paper_numbers.json" \
            >"$out/PAPER_NUMBERS.txt" 2>&1 \
            || fail "the paper prints numbers these rows do not produce — see $out/PAPER_NUMBERS.txt"
        grep 'published values' "$out/PAPER_NUMBERS.txt" | tail -1 | note
    fi
    $PY eval/artifact/verify_artifact.py --rivals "$data" ${barargs[@]+"${barargs[@]}"} \
        >"$out/VERIFICATION.txt" 2>&1 \
        || fail "the checks on the measurements FAILED — see $out/VERIFICATION.txt"
    grep -E '^[0-9]+ ok \|' "$out/VERIFICATION.txt" | sed 's/^/checks on the measurements: /' | note
    # Figures are not fatal: they need matplotlib, and no claim rests on a PNG.
    $PY eval/eval_plots.py "$data" --out-dir "$out/plots" >>"$log" 2>&1
    [ -n "$(ls "$out/plots" 2>/dev/null)" ] \
        || echo "figures: not drawn (they need matplotlib; the tables do not)" | note
}

# ---------------------------------------------------------------------------- GPU tiers
# One place runs the harnesses: eval/run_full_eval.sh, the published campaign's driver. A tier
# chooses only the cells and the artifact's two defaults (the VRAM cap and the pinned failures).
run_rivals() {
    local seq_insts="$1" depths="$2" comb="${3:-paper}"
    mkdir -p "$DETAILS"
    echo "output        : $OUT"
    echo "grants        : $GRANTS"
    echo "recorded fails: $([ "$HONOUR_FAILURES" = 1 ] && echo 'PINNED (--remeasure-failures re-derives them)' || echo 're-measured')"
    echo "VRAM cap      : ${GPAP_VRAM_BUDGET_MIB:-none} MiB"
    # Said here too, not only by `check`: a reviewer may go straight to a tier.
    $PY - <<'EOF'
import torch
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"this card     : {total/1e9:.1f} GB (reference: NVIDIA L4, 23.7 GB)"
          + ("\nWARNING       : under 24 GB — pinned rival rungs walk DOWN where they do not fit;"
             "\n                the run completes, and the comparison names every such cell"
             if total < 23e9 else ""))
EOF
    echo "seeds         : $SEEDS"
    echo "instances     : $(echo "$seq_insts" | wc -w) sequential, $(echo "$comb" | wc -w) combinational entry, depths $depths"
    OUT="$DETAILS" RIVAL_SEQ="$seq_insts" RIVAL_COMB="$comb" RIVAL_SEEDS="$SEEDS" \
        RIVAL_T=25 RIVAL_T2="$depths" RIVAL_GRANTS="$GRANTS" \
        RIVAL_HONOUR_FAILURES="$HONOUR_FAILURES" VERIFY_SOUND=1 \
        bash eval/run_full_eval.sh rivals \
        || fail "the measurement driver exited non-zero — the campaign is INCOMPLETE"
    # The cells this tier should produce, so the comparator can tell a finished run from one that
    # stopped early.
    EXPECT_CELLS="$($PY eval/artifact/estimate_wall.py "$REFERENCE" --tier "$TIER" --cells-only \
                    2>/dev/null || echo 0)"
}

subset_instances() {
    [ -f "$SUBSET_JSON" ] || { echo "missing $SUBSET_JSON" >&2; exit 1; }
    $PY -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['instances']))" "$SUBSET_JSON"
}

step_smoke() {
    TIER=smoke
    say "smoke: one instance per stage, everything else exactly as the real tiers run it"
    run_rivals "b03" none "c432"
    step_after_run
}

step_subset() {
    TIER=subset
    # `exit 1` inside $(...) exits only the subshell, and an empty RIVAL_SEQ makes the driver run
    # all 30 instances: the full tier, for a reviewer who asked for the short one.
    local insts; insts="$(subset_instances)" || exit 1
    [ -n "$insts" ] || { echo "the subset selection is empty — $SUBSET_JSON is unreadable" >&2
                         exit 1; }
    say "subset: the instances whose reference wall is under 10 minutes for both T=25 tasks"
    $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['rule'])" "$SUBSET_JSON"
    run_rivals "$insts" none
    step_after_run
}

step_full() {
    TIER=full
    local insts n
    insts="$(cd benchmarks/circuits/bench/seq && ls *.bench | sed 's/\.bench$//' | tr '\n' ' ')"
    # The campaign has 30 sequential instances; without `setup`, b17-b22 are missing and 48 of
    # 142 cells would silently not run.
    n="$(echo "$insts" | wc -w)"
    [ "$n" -eq 30 ] || { echo "only $n/30 sequential instances present: run" >&2
                         echo "  eval/artifact/reproduce.sh setup     (or scripts/fetch_benchmarks.sh itc99)" >&2
                         exit 1; }
    say "full: every cell of the published campaign, T=25 and T=50"
    run_rivals "$insts" 50
    step_after_run
}

# Tables over the FRESH rows, then the cell-by-cell comparison against the published ones.
step_after_run() {
    step_tables "$DETAILS" "$DETAILS" fresh
    step_compare "$SUMMARY"
    # The line a supervisor waits for, printed only when every step succeeded: it tells a
    # finished tier from a VM that died between two cells.
    echo
    local rc=0
    if [ ${#FAILED[@]} -eq 0 ]; then
        echo "ARTIFACT_TIER_COMPLETE" | tee -a "$SUMMARY"
    else
        { echo "ARTIFACT_TIER_FAILED: ${#FAILED[@]} step(s) did not succeed"
          printf '  - %s\n' "${FAILED[@]}"; } | tee -a "$SUMMARY"
        rc=1
    fi
    printf '\nThe comparison cell by cell is details/COMPARISON.txt; the rows, the tables of this run\nand the logs are in details/ too.\n' \
        >>"$SUMMARY"
    return "$rc"
}

# $1: the tier's SUMMARY.txt, which gets the report's verdict. Without it (a bare `compare`), the
# whole report is printed and no summary is touched: SUMMARY.txt belongs to the tier that ran.
step_compare() {
    local summary="${1:-}"
    [ -n "$(ls "$REFERENCE"/rivals_*.json 2>/dev/null)" ] || {
        fail "no reference rows at $REFERENCE — cannot compare"; return 1; }
    # A bare `compare` checks coverage only when told the tier (--tier); a GPU tier sets it.
    [ -z "$TIER" ] || EXPECT_CELLS="$($PY eval/artifact/estimate_wall.py "$REFERENCE" \
                                      --tier "$TIER" --cells-only 2>/dev/null || echo 0)"
    # A tier keeps its rows under details/; a directory written before that layout, at its top.
    local fresh="$DETAILS"
    [ -n "$(ls "$fresh"/rivals_*.json 2>/dev/null)" ] || fresh="$OUT"
    say "comparison against the published campaign"
    # --expect-cells turns "every cell agrees" into "the tier ran". The report goes to details/;
    # the console and SUMMARY.txt get its closing block.
    mkdir -p "$OUT"
    $PY eval/artifact/compare_campaign.py "$REFERENCE" "$fresh" --json "$fresh/comparison.json" \
        ${EXPECT_CELLS:+--expect-cells "$EXPECT_CELLS"} >"$fresh/COMPARISON.txt" 2>&1 \
        || fail "the comparison against the published campaign FAILED"
    [ -n "$summary" ] || { cat "$fresh/COMPARISON.txt"; return; }
    # Smoke and subset never run T=50 by design, so their summary drops the note that they did
    # not; the report keeps it.
    local skip='^$^'
    case "$TIER" in smoke|subset) skip='not in the fresh run — stage not attempted' ;; esac
    awk '/^=+$/ {block = ""; next} !/^report:/ {block = block $0 "\n"} END {printf "%s", block}' \
        "$fresh/COMPARISON.txt" | grep -v -e "$skip" | tee "$summary"
}

# `tables` over the published rows: SUMMARY.txt holds the lines the guide quotes, then what each
# file is.
step_published_tables() {
    mkdir -p "$OUT" && : >"$SUMMARY"
    echo "tables: the paper's tables and figures, regenerated from the recorded measurements"
    NOTE_TO="$SUMMARY" step_tables "$REFERENCE" "$DETAILS"
    [ ${#FAILED[@]} -eq 0 ] || printf 'FAILED: %s\n' "${FAILED[@]}" >>"$SUMMARY"
    cat >>"$SUMMARY" <<'GUIDE'

In details/:
  RESULTS.md          the paper's Tables 1-3, its uniformity counts and the numbers its text
                      quotes, regenerated from the recorded measurements: read it beside the PDF
  PAPER_NUMBERS.txt   every number the PDF prints, checked against the regenerated one
  VERIFICATION.txt    the checks on the measurements: every sample sound, every count exact,
                      every rival failure reported
  plots/              the figures; the paper's Figure 3 is sps_scatter_cmsgen.pdf
  paper_numbers.json  the regenerated numbers, machine-readable
  tables.log          everything else this step printed
GUIDE
    echo "Read $SUMMARY"
}

# A tier writes SUMMARY.txt when it ends, so one left by an earlier run into the same --out must
# not stand for this run if it stops first.
case "$STEP" in smoke|subset|full) rm -f "$SUMMARY" ;; esac

case "$STEP" in
    check)   step_check ;;
    setup)   step_setup ;;
    tables)  step_published_tables ;;
    smoke)   step_smoke ;;
    subset)  step_subset ;;
    full)    step_full ;;
    compare) step_compare ;;
    *)       echo "unknown step '$STEP'" >&2; usage; exit 2 ;;
esac
rc=$?
# Anything in the ledger fails the invocation, whatever the step returned.
[ ${#FAILED[@]} -eq 0 ] || rc=1
exit "$rc"

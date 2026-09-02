#!/usr/bin/env bash
# Package the experiment data as a self-describing artifact archive.
#
# The archive is the evidence behind the paper's numbers, so it carries three things a bare `zip`
# of a results directory does not:
#   * a MANIFEST with the commit, host, per-file sizes and SHA-256 of every JSON, so a number can
#     be traced to the exact bytes it came from;
#   * the VERIFICATION output (eval/artifact/verify_artifact.py), so the caveats travel WITH the data
#     rather than living only in someone's memory;
#   * the regeneration commands, so every table can be rebuilt from the JSONs alone.
#
#   eval/artifact/make_artifact.sh [OUT_DIR]
#
# The data lives OUTSIDE the repo (it is ~9 MB compressed and much larger raw), in
# `$GPAP_RESULTS` — a sibling directory, default `../gpap-results`. Each dataset is looked up in
# the already-organised archive layout first and in the raw `.experiments/` run directories
# second, so this works both straight after a sweep and long after the results were moved out.
set -uo pipefail

# This file lives in eval/artifact/, so the repo root is TWO levels up. It was `..` — the same
# off-by-one the eval/ reorganisation left in eleven other files, missed here because nothing
# had run the packager since. With it wrong, $RESULTS pointed inside the repo, every dataset
# came back "no source found", and `cd "$REPO_ROOT"` landed in eval/ so the table copies and
# the verifier path were wrong too.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

STAMP="$(date +%Y%m%d)"
RESULTS="${GPAP_RESULTS:-$(cd "$REPO_ROOT/.." && pwd)/gpap-results}"
OUT="${1:-$RESULTS}"
NAME="gpap_experiments_l4_${STAMP}"
FINAL="$OUT/$NAME"
# Stage into a SEPARATE directory and swap at the end. Rebuilding on the same day as a previous
# run makes $FINAL both the destination AND (via $PRIOR below) the SOURCE of the data — so a
# `rm -rf "$FINAL"` up front deletes the results it is about to copy from. Found before it fired;
# a re-run is meant to be safe, and losing the data would mean re-renting the GPUs.
STAGE="$OUT/.staging_$NAME.$$"
mkdir -p "$OUT"
rm -rf "$STAGE"; mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
echo "results root: $RESULTS"
echo "output      : $OUT"

# --- the data ---------------------------------------------------------------------------------
# A CAMPAIGN is one dated directory holding every arm measured together (the layout since
# 2026-08-23). When one is given — or found — it is the ONLY source: mixing arms across campaigns
# is how a table ends up describing two different protocols at once.
#
# ⚠️ This used to prefer `$PRIOR`, the most recent already-archived copy, ahead of any fresh run
# directory. With an August archive on disk that silently packaged 20-day-old data for EVERY arm
# — including rival rows measured under the old fixed-5-GiB batch grant — while printing `[ok]`
# for each one. A packager that ships the wrong data quietly is worse than one that fails.
CAMPAIGN="${GPAP_CAMPAIGN:-}"
if [ -z "$CAMPAIGN" ]; then
    CAMPAIGN="$(ls -d "$RESULTS"/l4_campaign_* "$REPO_ROOT"/.experiments/l4_campaign_* \
                2>/dev/null | sort | tail -1)"
fi
[ -n "$CAMPAIGN" ] && echo "campaign    : $CAMPAIGN" || echo "campaign    : none found (legacy layout)"

# How stale a source may be before the packager refuses. A sweep is re-run precisely because the
# protocol changed, so silently shipping the previous one defeats the point.
MAX_AGE_DAYS="${GPAP_ARTIFACT_MAX_AGE_DAYS:-14}"
STALE=0
# Datasets whose age is the POINT: a prior campaign kept for comparison, or a reference sweep on
# another card. Named explicitly rather than exempted by a blanket flag, so adding a stale arm
# stays a deliberate act.
ARCHIVAL=" uniformity_prior_campaign gpap_vs_cmsgen_4080_reference "

# find_src <dest-name> <candidate...> -> first candidate that exists
copy() {  # copy <dest-name> <candidate-src>...
    local dest="$1"; shift
    local src=""
    # A campaign directory always wins, whatever the caller listed.
    if [ -n "$CAMPAIGN" ] && [ -e "$CAMPAIGN/$dest" ]; then
        src="$CAMPAIGN/$dest"
    else
        for c in "$@"; do [ -e "$c" ] && { src="$c"; break; }; done
    fi
    [ -n "$src" ] || { echo "  [skip] $dest (no source found)"; return; }
    local age=$(( ( $(date +%s) - $(stat -c %Y "$src" 2>/dev/null || echo 0) ) / 86400 ))
    if [ "$age" -gt "$MAX_AGE_DAYS" ] && [[ "$ARCHIVAL" != *" $dest "* ]]; then
        echo "  [STALE] $dest  <- $src  (${age}d old, limit ${MAX_AGE_DAYS}d)"
        STALE=$((STALE + 1))
    fi
    mkdir -p "$STAGE/$dest"
    # JSON + the summary markdown/plots; never the multi-GB converted CNFs
    rsync -a --exclude 'cnf/' --exclude 'hts_cnf/' --exclude 'stage/' --exclude '*.cnf' \
          --exclude '*.bench' "$src"/ "$STAGE/$dest"/ 2>/dev/null \
      || cp -r "$src"/. "$STAGE/$dest"/
    echo "  [ok]   $dest  <- $src  (${age}d old)"
}
# Prefer the most recent already-archived copy, since raw run dirs get cleaned.
PRIOR="$(ls -d "$RESULTS"/gpap_experiments_l4_* 2>/dev/null | grep -v '\.tar\.gz$' | sort | tail -1)"
echo "prior archive: ${PRIOR:-none}"
echo "staging data ..."
copy gpap_vs_cmsgen_l4          "$PRIOR/gpap_vs_cmsgen_l4"          "${CMSGEN_RUN_DIR:-$REPO_ROOT/.experiments/cmsgen_l4}"
copy gpap_vs_gpu_rivals_l4      "$PRIOR/gpap_vs_gpu_rivals_l4"      "$REPO_ROOT/.experiments/l4_rivals"
copy reproducibility_recheck_l4 "$PRIOR/reproducibility_recheck_l4" "$REPO_ROOT/.experiments/recheck"
copy uniformity_l4              "$PRIOR/uniformity_l4"              "$REPO_ROOT/.experiments/uniformity"
copy uniformity_prior_campaign  "$PRIOR/uniformity_prior_campaign"  "$REPO_ROOT/.experiments/fuzz"
copy gpap_vs_cmsgen_4080_reference "$PRIOR/gpap_vs_cmsgen_4080_reference" \
     "$REPO_ROOT/.experiments/full_eval/nvidia-geforce-rtx-4080-laptop-gpu"
# The WAVE-vs-SCALAR kernel A/B (docs/paper/KERNEL_AB.md + kernel_ab.tex are generated from it).
# It had a paper table but was never packaged, so the archive could not regenerate that table.
copy kernel_ab_l4               "$PRIOR/kernel_ab_l4"               "$REPO_ROOT/.experiments/kernel_ab_l4"

# The uniformity-PRICE arm: the same matrix with --allow-soft (eval/colab/colab_cmsgen_soft_l4.sh).
copy gpap_vs_cmsgen_l4_softarm  "${SOFT_RUN_DIR:-$RESULTS/cmsgen_soft_l4}" \
     "$PRIOR/gpap_vs_cmsgen_l4_softarm" "$REPO_ROOT/.experiments/full_eval/nvidia-l4-soft"

if [ "$STALE" -gt 0 ]; then
    echo
    echo "[FATAL] $STALE dataset(s) older than ${MAX_AGE_DAYS} days — refusing to package." >&2
    echo "        Point GPAP_CAMPAIGN at the campaign you mean, or raise" >&2
    echo "        GPAP_ARTIFACT_MAX_AGE_DAYS if the age is genuinely intended." >&2
    exit 1
fi

# --- generated tables -------------------------------------------------------------------------
mkdir -p "$STAGE/tables"
for f in RESULTS.md tables.tex SOFT_PRICE.md soft_price.tex KERNEL_AB.md kernel_ab.tex; do
    cp -f "$REPO_ROOT/docs/paper/$f" "$STAGE/tables/" 2>/dev/null || true
done
cp -f "$REPO_ROOT/docs/findings/L4_EVAL_FINDINGS.md" "$STAGE/" 2>/dev/null || true

# --- verification, captured INTO the archive ---------------------------------------------------
# Run against the STAGED copy, so the verdict describes the bytes actually shipped rather than
# whatever happens to be in a run directory that may have moved since.
echo "verifying ..."
VERIFY_OK=1
$PY eval/artifact/verify_artifact.py "$STAGE/gpap_vs_cmsgen_l4" \
    --rivals "$STAGE/gpap_vs_gpu_rivals_l4" \
    --recheck "$STAGE/reproducibility_recheck_l4" \
    --uniformity "$STAGE/uniformity_l4"/*.json "$STAGE/uniformity_prior_campaign/round2.json" \
    > "$STAGE/VERIFICATION.txt" 2>&1 || VERIFY_OK=0
VERDICT="$(tail -1 "$STAGE/VERIFICATION.txt")"
# A FAILING verifier is a refusal, not a footnote: the exit code used to be discarded here, so
# an artifact whose own checker said "NOT publishable" was tarred and stamped anyway, with the
# verdict buried in a file nobody is forced to read. Same rule as the STALE refusal above.
if [ "$VERIFY_OK" != 1 ] && [ "${GPAP_ARTIFACT_ALLOW_FAIL:-0}" != "1" ]; then
    echo
    grep '^  FAIL' "$STAGE/VERIFICATION.txt" >&2 || tail -25 "$STAGE/VERIFICATION.txt" >&2
    echo "[FATAL] verify_artifact FAILED — refusing to package an artifact its own verifier" >&2
    echo "        calls unpublishable. Fix the failures above (full report:" >&2
    echo "        VERIFICATION.txt), or set GPAP_ARTIFACT_ALLOW_FAIL=1 to package anyway;" >&2
    echo "        the failing verdict ships inside the archive either way." >&2
    exit 1
fi

# --- manifest ----------------------------------------------------------------------------------
{
  echo "# GPAP experiment artifact — $NAME"
  echo
  echo "Generated $(date -Is) from commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "on $(uname -sr)."
  echo
  echo "## Verdict"
  echo
  echo "    $VERDICT"
  echo
  echo "See VERIFICATION.txt — every warning there must be stated in the paper text."
  echo
  echo "## Contents"
  echo
  echo '| directory | what it is |'
  echo '|---|---|'
  echo '| `gpap_vs_cmsgen_l4/` | the CMSGen head-to-head on a Colab L4 (all families) |'
  echo '| `gpap_vs_cmsgen_l4_softarm/` | the SAME matrix with `--allow-soft`: the uniformity price |'
  echo '| `gpap_vs_gpu_rivals_l4/` | DEMOTIC / Recurrent-CircuitSAT / HTS on the same L4 |'
  echo '| `reproducibility_recheck_l4/` | the same cells re-measured on a FRESH L4 |'
  echo '| `uniformity_l4/` | exact-oracle uniformity campaign + the `soft` throughput price |'
  echo '| `uniformity_prior_campaign/` | the earlier uniformity rounds (4080) |'
  echo '| `gpap_vs_cmsgen_4080_reference/` | the RTX 4080 laptop reference sweep |'
  echo '| `kernel_ab_l4/` | WAVE vs SCALAR propagate, forced-kernel A/B at pinned K |'
  echo '| `tables/` | the paper tables, generated from the JSONs above |'
  echo
  echo "## Regenerating every table from this data"
  echo
  echo '```bash'
  echo "eval/artifact/paper_tables.py --cmsgen gpap_vs_cmsgen_l4 \\"
  echo "    --rivals gpap_vs_gpu_rivals_l4 --recheck reproducibility_recheck_l4 \\"
  echo "    --uniformity uniformity_l4/uniformity_campaign.json \\"
  echo "    --soft uniformity_l4/soft_profile.json \\"
  echo "    --out-md RESULTS.md --out-tex tables.tex"
  echo "eval/artifact/soft_price_tables.py --safe gpap_vs_cmsgen_l4 --soft gpap_vs_cmsgen_l4_softarm \\"
  echo "    --out-md SOFT_PRICE.md --out-tex soft_price.tex"
  echo "eval/artifact/kernel_ab_tables.py kernel_ab_l4/kernel_ab.json \\"
  echo "    --out-md KERNEL_AB.md --out-tex kernel_ab.tex"
  echo "eval/artifact/verify_artifact.py gpap_vs_cmsgen_l4 --rivals gpap_vs_gpu_rivals_l4 \\"
  echo "    --recheck reproducibility_recheck_l4"
  echo '```'
  echo
  echo "## Files (SHA-256)"
  echo
  echo '| file | bytes | sha256 |'
  echo '|---|---:|---|'
  (cd "$STAGE" && find . -name '*.json' | sort | while read -r f; do
      printf '| `%s` | %s | `%s` |\n' "${f#./}" "$(stat -c%s "$f")" "$(sha256sum "$f" | cut -c1-16)"
   done)
} > "$STAGE/MANIFEST.md"

# --- swap into place, then archive ---------------------------------------------------------------
# Only now is the old copy removed: everything above read from it.
NJSON="$(find "$STAGE" -name '*.json' | wc -l)"
NFILE="$(find "$STAGE" -type f | wc -l)"
if [ "$NJSON" -lt 10 ]; then
    echo "[FATAL] only $NJSON JSONs staged — refusing to replace $FINAL" >&2
    exit 1
fi
rm -rf "$FINAL"
mv "$STAGE" "$FINAL"
trap - EXIT
( cd "$OUT" && tar czf "$NAME.tar.gz" "$NAME" )
SIZE="$(du -h "$OUT/$NAME.tar.gz" | cut -f1)"
echo
echo "artifact: $OUT/$NAME.tar.gz  ($SIZE)"
echo "verdict : $VERDICT"
echo "files   : $NJSON JSON, $NFILE total"

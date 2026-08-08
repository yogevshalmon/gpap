#!/usr/bin/env bash
# Package the experiment data as a self-describing artifact archive.
#
# The archive is the evidence behind the paper's numbers, so it carries three things a bare `zip`
# of a results directory does not:
#   * a MANIFEST with the commit, host, per-file sizes and SHA-256 of every JSON, so a number can
#     be traced to the exact bytes it came from;
#   * the VERIFICATION output (eval/verify_artifact.py), so the caveats travel WITH the data
#     rather than living only in someone's memory;
#   * the regeneration commands, so every table can be rebuilt from the JSONs alone.
#
#   eval/make_artifact.sh [OUT_DIR]
#
# The data lives OUTSIDE the repo (it is ~9 MB compressed and much larger raw), in
# `$GPAP_RESULTS` — a sibling directory, default `../gpap-results`. Each dataset is looked up in
# the already-organised archive layout first and in the raw `.experiments/` run directories
# second, so this works both straight after a sweep and long after the results were moved out.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
# find_src <dest-name> <candidate...> -> first candidate that exists
copy() {  # copy <dest-name> <candidate-src>...
    local dest="$1"; shift
    local src=""
    for c in "$@"; do [ -e "$c" ] && { src="$c"; break; }; done
    [ -n "$src" ] || { echo "  [skip] $dest (no source found)"; return; }
    mkdir -p "$STAGE/$dest"
    # JSON + the summary markdown/plots; never the multi-GB converted CNFs
    rsync -a --exclude 'cnf/' --exclude 'hts_cnf/' --exclude 'stage/' --exclude '*.cnf' \
          --exclude '*.bench' "$src"/ "$STAGE/$dest"/ 2>/dev/null \
      || cp -r "$src"/. "$STAGE/$dest"/
    echo "  [ok]   $dest  <- $src"
}
# Prefer the most recent already-archived copy, since raw run dirs get cleaned.
PRIOR="$(ls -d "$RESULTS"/gpap_experiments_l4_* 2>/dev/null | grep -v '\.tar\.gz$' | sort | tail -1)"
echo "prior archive: ${PRIOR:-none}"
echo "staging data ..."
copy gpap_vs_cmsgen_l4          "$PRIOR/gpap_vs_cmsgen_l4"          "$REPO_ROOT/.experiments/cmsgen_l4_2026-08-03"
copy gpap_vs_gpu_rivals_l4      "$PRIOR/gpap_vs_gpu_rivals_l4"      "$REPO_ROOT/.experiments/l4_rivals"
copy reproducibility_recheck_l4 "$PRIOR/reproducibility_recheck_l4" "$REPO_ROOT/.experiments/recheck"
copy uniformity_l4              "$PRIOR/uniformity_l4"              "$REPO_ROOT/.experiments/uniformity"
copy uniformity_prior_campaign  "$PRIOR/uniformity_prior_campaign"  "$REPO_ROOT/.experiments/fuzz"
copy gpap_vs_cmsgen_4080_reference "$PRIOR/gpap_vs_cmsgen_4080_reference" \
     "$REPO_ROOT/.experiments/full_eval/nvidia-geforce-rtx-4080-laptop-gpu"
# The uniformity-PRICE arm: the same matrix with --allow-soft (eval/colab_cmsgen_soft_l4.sh).
copy gpap_vs_cmsgen_l4_softarm  "$RESULTS/cmsgen_soft_l4_2026-08-05" \
     "$PRIOR/gpap_vs_cmsgen_l4_softarm" "$REPO_ROOT/.experiments/full_eval/nvidia-l4-soft"

# --- generated tables -------------------------------------------------------------------------
mkdir -p "$STAGE/tables"
for f in RESULTS.md tables.tex SOFT_PRICE.md soft_price.tex; do
    cp -f "$REPO_ROOT/docs/paper/$f" "$STAGE/tables/" 2>/dev/null || true
done
cp -f "$REPO_ROOT/docs/findings/L4_EVAL_FINDINGS.md" "$STAGE/" 2>/dev/null || true

# --- verification, captured INTO the archive ---------------------------------------------------
# Run against the STAGED copy, so the verdict describes the bytes actually shipped rather than
# whatever happens to be in a run directory that may have moved since.
echo "verifying ..."
$PY eval/verify_artifact.py "$STAGE/gpap_vs_cmsgen_l4" \
    --rivals "$STAGE/gpap_vs_gpu_rivals_l4" \
    --uniformity "$STAGE/uniformity_l4"/*.json "$STAGE/uniformity_prior_campaign/round2.json" \
    > "$STAGE/VERIFICATION.txt" 2>&1
VERDICT="$(tail -1 "$STAGE/VERIFICATION.txt")"

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
  echo '| `tables/` | the paper tables, generated from the JSONs above |'
  echo
  echo "## Regenerating every table from this data"
  echo
  echo '```bash'
  echo "eval/paper_tables.py --cmsgen gpap_vs_cmsgen_l4 \\"
  echo "    --cmsgen-ref gpap_vs_cmsgen_4080_reference \\"
  echo "    --rivals gpap_vs_gpu_rivals_l4 --recheck reproducibility_recheck_l4 \\"
  echo "    --uniformity uniformity_l4/uniformity_campaign.json \\"
  echo "    --soft uniformity_l4/soft_profile.json \\"
  echo "    --out-md RESULTS.md --out-tex tables.tex"
  echo "eval/soft_price_tables.py --safe gpap_vs_cmsgen_l4 --soft gpap_vs_cmsgen_l4_softarm \\"
  echo "    --out-md SOFT_PRICE.md --out-tex soft_price.tex"
  echo "eval/verify_artifact.py gpap_vs_cmsgen_l4 --rivals gpap_vs_gpu_rivals_l4"
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

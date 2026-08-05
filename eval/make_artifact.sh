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
#   eval/make_artifact.sh [OUT_DIR]      # default .experiments/artifact
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

STAMP="$(date +%Y%m%d)"
# Default to a TRACKABLE path: `.experiments/` is gitignored, and the whole point of the archive
# is that it can be committed alongside the paper if we choose to.
OUT="${1:-$REPO_ROOT/artifacts}"
NAME="gpap_experiments_l4_${STAMP}"
STAGE="$OUT/$NAME"
rm -rf "$STAGE"; mkdir -p "$STAGE"

# --- the data ---------------------------------------------------------------------------------
copy() {  # copy <src> <dest-name>
    [ -e "$1" ] || { echo "  [skip] $1 (absent)"; return; }
    mkdir -p "$STAGE/$2"
    # JSON + the summary markdown/plots; never the multi-GB converted CNFs
    rsync -a --exclude 'cnf/' --exclude 'hts_cnf/' --exclude 'stage/' --exclude '*.cnf' \
          --exclude '*.bench' "$1"/ "$STAGE/$2"/ 2>/dev/null \
      || cp -r "$1"/. "$STAGE/$2"/
    echo "  [ok]   $2"
}
echo "staging data ..."
copy "$REPO_ROOT/.experiments/cmsgen_l4_2026-08-03"                      gpap_vs_cmsgen_l4
copy "$REPO_ROOT/.experiments/l4_rivals"                                  gpap_vs_gpu_rivals_l4
copy "$REPO_ROOT/.experiments/recheck"                                    reproducibility_recheck_l4
copy "$REPO_ROOT/.experiments/uniformity"                                 uniformity_l4
copy "$REPO_ROOT/.experiments/fuzz"                                       uniformity_prior_campaign
copy "$REPO_ROOT/.experiments/full_eval/nvidia-geforce-rtx-4080-laptop-gpu" gpap_vs_cmsgen_4080_reference

# --- generated tables -------------------------------------------------------------------------
mkdir -p "$STAGE/tables"
cp -f "$REPO_ROOT/docs/paper/RESULTS.md" "$STAGE/tables/" 2>/dev/null || true
cp -f "$REPO_ROOT/docs/paper/tables.tex" "$STAGE/tables/" 2>/dev/null || true
cp -f "$REPO_ROOT/docs/findings/L4_EVAL_FINDINGS.md" "$STAGE/" 2>/dev/null || true

# --- verification, captured INTO the archive ---------------------------------------------------
echo "verifying ..."
$PY eval/verify_artifact.py "$REPO_ROOT/.experiments/cmsgen_l4_2026-08-03" \
    --rivals "$REPO_ROOT/.experiments/l4_rivals" \
    --uniformity "$REPO_ROOT/.experiments/uniformity/*.json" "$REPO_ROOT/.experiments/fuzz/round2.json" \
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
  echo "    --uniformity uniformity_l4/*.json \\"
  echo "    --out-md RESULTS.md --out-tex tables.tex"
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

# --- archive ------------------------------------------------------------------------------------
( cd "$OUT" && tar czf "$NAME.tar.gz" "$NAME" )
SIZE="$(du -h "$OUT/$NAME.tar.gz" | cut -f1)"
echo
echo "artifact: $OUT/$NAME.tar.gz  ($SIZE)"
echo "verdict : $VERDICT"
echo "files   : $(find "$STAGE" -name '*.json' | wc -l) JSON, $(find "$STAGE" -type f | wc -l) total"

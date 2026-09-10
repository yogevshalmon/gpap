#!/usr/bin/env bash
# Package the SELF-CONTAINED artifact: the code at HEAD, the published rows, and the benchmarks
# the repository does not carry — one directory a reviewer can unpack and run.
#
#   eval/artifact/make_artifact.sh [OUT_DIR]        # default: $GPAP_RESULTS
#
# What goes in, and why each piece is needed:
#   the repository at HEAD   `git archive`, so nothing untracked or stale can ride along
#   artifact-data/           the frozen campaign rows the tables are generated from, with the
#                            Barbarik arm, sha256-manifested. reproduce.sh finds them here with
#                            no environment set (its bundled-reference fallback).
#   benchmarks .../b1[789],b2[012]  the six ITC'99 circuits too large to commit, which every
#                            sequential stage needs and which the repository fetches at run time
#                            from an unpinned URL. A self-contained artifact carries them.
#   README.md                ARTIFACT.md, so the entry point is the first file in the archive.
#
# What deliberately stays out: the rival checkouts (DEMOTIC, Recurrent-CircuitSAT, HTS ship no
# licence, so they are fetched at their pinned commits by rivals_setup.sh, and every result row
# records the commit that actually ran) and the built binaries (abc, CMSGen — built from pinned
# sources by the same script, and architecture-specific anyway).
#
# The Tier-1 gate runs against the STAGED copy before anything is archived, so the verdict
# describes the bytes actually shipped, and a failing verdict refuses to package.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

RESULTS="${GPAP_RESULTS:-$(cd "$REPO_ROOT/.." && pwd)/gpap-results}"
OUT="${1:-$RESULTS}"
SRC_DATA="${GPAP_PAPER_DIR:-$RESULTS/paper}"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
NAME="gpap-artifact-$(date +%Y%m%d)-$COMMIT"
STAGE="$OUT/.staging_$NAME.$$"
FINAL="$OUT/$NAME"

echo "repository : $REPO_ROOT @ $COMMIT"
echo "rows       : $SRC_DATA/data"
echo "output     : $OUT/$NAME"

[ -d "$SRC_DATA/data" ] || { echo "[FATAL] no published rows at $SRC_DATA/data" >&2; exit 1; }
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "[warn] the working tree is dirty; git archive ships HEAD, not what is on disk" >&2
fi

rm -rf "$STAGE"; mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT

# --- the code, exactly as committed ------------------------------------------------------------
git archive HEAD | tar -x -C "$STAGE"

# --- the rows the tables are generated from -----------------------------------------------------
mkdir -p "$STAGE/artifact-data/paper"
cp -r "$SRC_DATA/data" "$STAGE/artifact-data/paper/data"
for f in README.md RESULTS.md tables.tex VERIFICATION.txt emission_audit.txt; do
    [ -f "$SRC_DATA/$f" ] && cp "$SRC_DATA/$f" "$STAGE/artifact-data/paper/$f"
done
[ -d "$SRC_DATA/plots" ] && cp -r "$SRC_DATA/plots" "$STAGE/artifact-data/paper/plots"

# --- the benchmarks the repository does not carry ------------------------------------------------
# Gitignored because of their size, fetched at run time from a raw GitHub URL that is neither
# pinned nor guaranteed to exist in three years. Six files, 25 MB, and without them a third of
# the sequential set silently does not run.
n_extra=0
for b in b17 b18 b19 b20 b21 b22; do
    src="benchmarks/circuits/bench/seq/$b.bench"
    [ -f "$src" ] || { echo "[warn] $b.bench is not installed — run scripts/fetch_benchmarks.sh itc99"; continue; }
    cp "$src" "$STAGE/benchmarks/circuits/bench/seq/$b.bench" && n_extra=$((n_extra + 1))
done
echo "benchmarks : $n_extra/6 large ITC'99 circuits bundled"

# --- the entry point is the first file a reviewer sees --------------------------------------------
cp "$STAGE/ARTIFACT.md" "$STAGE/README-ARTIFACT.md"

# --- the Tier-1 gate, over the STAGED rows --------------------------------------------------------
echo
echo "== verifying the staged rows (the publishability gate)"
VERIFY_OK=1
$PY eval/artifact/verify_artifact.py --rivals "$STAGE/artifact-data/paper/data" \
    --barbarik "$STAGE"/artifact-data/paper/data/barbarik/circ_abc_*.jsonl \
    > "$STAGE/artifact-data/VERIFICATION.txt" 2>&1 || VERIFY_OK=0
VERDICT="$(tail -1 "$STAGE/artifact-data/VERIFICATION.txt")"
echo "$VERDICT"
if [ "$VERIFY_OK" != 1 ] && [ "${GPAP_ARTIFACT_ALLOW_FAIL:-0}" != "1" ]; then
    grep '^  FAIL' "$STAGE/artifact-data/VERIFICATION.txt" >&2 \
        || tail -20 "$STAGE/artifact-data/VERIFICATION.txt" >&2
    echo "[FATAL] the gate FAILED — refusing to package an artifact its own verifier calls" >&2
    echo "        unpublishable. GPAP_ARTIFACT_ALLOW_FAIL=1 packages anyway; the verdict" >&2
    echo "        ships inside either way." >&2
    exit 1
fi

# --- manifest --------------------------------------------------------------------------------------
( cd "$STAGE" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z \
    | xargs -0 sha256sum > MANIFEST.sha256 )
NFILES="$(wc -l < "$STAGE/MANIFEST.sha256")"
{
  echo "# $NAME"
  echo
  echo "Built $(date -Is) from commit \`$COMMIT\` on $(uname -sr)."
  echo
  echo "    $VERDICT"
  echo
  echo "Start with **README-ARTIFACT.md** (the same file as \`ARTIFACT.md\`)."
  echo
  echo "| path | what it is |"
  echo "|---|---|"
  echo "| \`eval/artifact/reproduce.sh\` | the single entry point: check / setup / tables / smoke / subset / full |"
  echo "| \`artifact-data/paper/data/\` | the published campaign's rows — every table is generated from these |"
  echo "| \`artifact-data/paper/\` | the generated tables, gate output and figures as published |"
  echo "| \`benchmarks/\` | every circuit, including the six ITC'99 files the repository fetches |"
  echo "| \`MANIFEST.sha256\` | $NFILES files, hashed |"
  echo
  echo "Verify the contents with:  \`sha256sum -c MANIFEST.sha256\`"
} > "$STAGE/MANIFEST.md"

rm -rf "$FINAL"
mkdir -p "$OUT"
mv "$STAGE" "$FINAL"
trap - EXIT
( cd "$OUT" && tar czf "$NAME.tar.gz" "$NAME" )
echo
echo "artifact: $OUT/$NAME.tar.gz  ($(du -h "$OUT/$NAME.tar.gz" | cut -f1), $NFILES files)"
echo "verdict : $VERDICT"
echo
echo "Next: upload to Zenodo for the DOI (the Available badge), and put that DOI in"
echo "      CITATION.cff and the paper's Data Availability Statement."

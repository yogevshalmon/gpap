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
#   README-ARTIFACT.md       a copy of ARTIFACT.md, so the entry point is the first file a
#                            reviewer sees in the archive listing.
#
# What deliberately stays out: the rival checkouts (DEMOTIC, Recurrent-CircuitSAT, HTS ship no
# licence, so they are fetched at their pinned commits by rivals_setup.sh, and every result row
# records the commit that actually ran) and the built binaries (abc, CMSGen — built from pinned
# sources by the same script, and architecture-specific anyway).
#
# THE TIER-1 GATE RUNS INSIDE THE STAGED TREE — staged code over staged rows — before anything is
# archived, and the shipped tables are REGENERATED there rather than copied. So the verdict and
# the tables describe the bytes actually shipped, not the checkout they were built from, and a
# failing verdict refuses to package.
set -uo pipefail
# Nothing that runs inside the staged tree may leave bytecode in it: every artifact built before
# this line shipped nine `__pycache__/*.pyc` files, written by the generators and the gate as they
# ran over the stage. The scrub before the leak gate is the other half, for anything a subprocess
# writes regardless.
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

# Ask, never recompute: eval/paths.py is the one definition (physical, `$GPAP_RESULTS`-aware).
RESULTS="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m eval.paths results)"
: "${RESULTS:?cannot resolve — python3 must be able to import the repo (gpap/paths.py, eval/paths.py)}"
OUT="${1:-$RESULTS}"
SRC_DATA="${GPAP_PAPER_DIR:-$RESULTS/paper}"
# What is archived: HEAD, unless GPAP_ARTIFACT_TREE names another tree-ish — the dry-run knob for
# packaging UNCOMMITTED work (`git write-tree` on a temporary index gives the working tree as a
# tree object) so the gates below can be run against a change before it is committed. The name
# and the manifest then carry that object's id, not HEAD's, so a dry run cannot pass as a release.
TREE="${GPAP_ARTIFACT_TREE:-HEAD}"
COMMIT="$(git rev-parse --short "$TREE" 2>/dev/null || echo unknown)"
NAME="gpap-artifact-$(date +%Y%m%d)-$COMMIT"
STAGE="$OUT/.staging_$NAME.$$"
FINAL="$OUT/$NAME"

echo "repository : $REPO_ROOT @ $COMMIT"
echo "rows       : $SRC_DATA/data"
echo "output     : $OUT/$NAME"

[ -d "$SRC_DATA/data" ] || { echo "[FATAL] no published rows at $SRC_DATA/data" >&2; exit 1; }
# `git archive HEAD` ships committed bytes. Packaging from a dirty tree therefore produces an
# artifact that differs from the checkout the operator is looking at and testing — silently, and
# in whichever direction the uncommitted edits went. That is a refusal, not a warning.
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    if [ "${GPAP_ARTIFACT_ALLOW_DIRTY:-0}" != "1" ]; then
        echo "[FATAL] the working tree is dirty and git archive ships HEAD, so the package would" >&2
        echo "        not be the code you are testing. Commit, stash, or set" >&2
        echo "        GPAP_ARTIFACT_ALLOW_DIRTY=1 to package HEAD regardless." >&2
        git status --short >&2
        exit 1
    fi
    echo "[warn] the working tree is dirty; git archive ships HEAD, not what is on disk" >&2
fi

rm -rf "$STAGE"; mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT

# --- the code, exactly as committed ------------------------------------------------------------
git archive "$TREE" | tar -x -C "$STAGE"

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

# --- Tier 1, run INSIDE the staged tree ----------------------------------------------------------
# Every command below runs from $STAGE, so it is the staged copy of the generator and of the gate
# reading the staged rows. Run from the checkout instead — as this did — and a dirty working tree
# silently certifies the package with code that is not in it.
echo
echo "== regenerating the tables and running the gate, inside the staged tree"
BAR=("$STAGE"/artifact-data/paper/data/barbarik/circ_abc_*.jsonl)
BARARGS=(); [ -e "${BAR[0]}" ] && BARARGS=(--barbarik "${BAR[@]}")
TABLES_OK=1
( cd "$STAGE" && $PY eval/artifact/paper_tables.py --rivals artifact-data/paper/data \
    ${BARARGS[@]+"${BARARGS[@]}"} \
    --out-md artifact-data/paper/RESULTS.md --out-tex artifact-data/paper/tables.tex ) \
    > "$STAGE/artifact-data/TABLES.log" 2>&1 || TABLES_OK=0
if [ "$TABLES_OK" != 1 ]; then
    tail -20 "$STAGE/artifact-data/TABLES.log" >&2
    echo "[FATAL] the staged code could not regenerate the tables from the staged rows —" >&2
    echo "        the package would ship tables it cannot reproduce." >&2
    cp "$STAGE/artifact-data/TABLES.log" "$OUT/$NAME.TABLES.log" 2>/dev/null
    exit 1
fi
VERIFY_OK=1
( cd "$STAGE" && $PY eval/artifact/verify_artifact.py --rivals artifact-data/paper/data \
    ${BARARGS[@]+"${BARARGS[@]}"} ) > "$STAGE/artifact-data/VERIFICATION.txt" 2>&1 || VERIFY_OK=0
VERDICT="$(tail -1 "$STAGE/artifact-data/VERIFICATION.txt")"
echo "$VERDICT"
if [ "$VERIFY_OK" != 1 ] && [ "${GPAP_ARTIFACT_ALLOW_FAIL:-0}" != "1" ]; then
    grep '^  FAIL' "$STAGE/artifact-data/VERIFICATION.txt" >&2 \
        || tail -20 "$STAGE/artifact-data/VERIFICATION.txt" >&2
    # The EXIT trap is about to delete the staging tree, and with it the only copy of the
    # evidence for this refusal.
    cp "$STAGE/artifact-data/VERIFICATION.txt" "$OUT/$NAME.VERIFICATION.txt" 2>/dev/null \
        && echo "        the failing verdict is kept at $OUT/$NAME.VERIFICATION.txt" >&2
    echo "[FATAL] the gate FAILED — refusing to package an artifact its own verifier calls" >&2
    echo "        unpublishable. GPAP_ARTIFACT_ALLOW_FAIL=1 packages anyway; the verdict" >&2
    echo "        ships inside either way." >&2
    exit 1
fi

# --- nothing the staging steps themselves wrote may ship --------------------------------------------
# Before the gate and the manifest, so both see exactly the bytes that ship.
find "$STAGE" -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) \
    -prune -exec rm -rf {} +

# --- nothing that names one machine or one person may ship ---------------------------------------
# A path in a shipped file names the disk it was generated on, and puts a username into an upload
# that is permanent once it has a DOI. It has happened three times, each from a generator writing
# the path it was handed (a committed slide table, `emission_audit.txt`, three Barbarik error
# rows), and the third had a shape the first version of this check — one `/home/` pattern inline
# here — did not know. The shapes now live in ONE place, `check_leaks.sh` (gated by
# tests/test_artifact_gate.py); the generators are fixed at the source
# (`eval/paths.py::display_path`) and this is what says they stay fixed — over the STAGED BYTES,
# like the verdict above, because that is what a reviewer receives. `GPAP_ARTIFACT_DENYLIST=<file>`
# adds literal strings (a name, a hostname) that cannot be spelled in a script that ships.
leaks="$(bash eval/artifact/check_leaks.sh "$STAGE" \
             ${GPAP_ARTIFACT_DENYLIST:+--denylist "$GPAP_ARTIFACT_DENYLIST"})"
gate=$?
if [ "$gate" = 2 ]; then
    echo "[FATAL] the leak gate could not run — see above" >&2
    exit 1
elif [ "$gate" != 0 ]; then
    echo "[FATAL] these staged files carry a string that names one machine or one person:" >&2
    echo "$leaks" | sed 's/^/          /' >&2
    echo "        Fix the GENERATOR (eval/paths.py::display_path), regenerate the rows," >&2
    echo "        and re-package. GPAP_ARTIFACT_ALLOW_LEAK=1 overrides." >&2
    [ "${GPAP_ARTIFACT_ALLOW_LEAK:-0}" = "1" ] || exit 1
fi

# --- manifest --------------------------------------------------------------------------------------
# MANIFEST.md is written FIRST so that the hashes cover it: a manifest that advertises
# `sha256sum -c` while being the one file nobody can check is a manifest with a hole in it. The
# count comes from find, not from the hash file that does not exist yet.
NFILES="$(cd "$STAGE" && find . -type f ! -name MANIFEST.sha256 | wc -l)"
NFILES=$((NFILES + 1))                                   # MANIFEST.md itself, written next
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

( cd "$STAGE" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z \
    | xargs -0 sha256sum > MANIFEST.sha256 )

rm -rf "$FINAL"
mkdir -p "$OUT"
mv "$STAGE" "$FINAL"
trap - EXIT
( cd "$OUT" && tar czf "$NAME.tar.gz" "$NAME" )
echo
echo "artifact: $OUT/$NAME.tar.gz  ($(du -h "$OUT/$NAME.tar.gz" | cut -f1), $NFILES files)"
echo "verdict : $VERDICT"
echo
echo "Next: for a double-blind review copy, \$GPAP_RESULTS/artifact/anonymize.sh <this tarball>"
echo "      (outside the repository: it spells the author's name). Then upload to Zenodo for the"
echo "      DOI (the Available badge), and put that DOI in CITATION.cff and the paper's Data"
echo "      Availability Statement."

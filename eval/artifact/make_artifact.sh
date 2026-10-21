#!/usr/bin/env bash
# Package the self-contained artifact: the code at HEAD, the published rows and the benchmarks
# the repository does not carry, in one directory a reviewer can unpack and run.
#
#   eval/artifact/make_artifact.sh [OUT_DIR]        # default: $GPAP_RESULTS
#
#   the repository at HEAD   `git archive`, so nothing untracked rides along
#   artifact-data/           the frozen rows, with the Barbarik arm, sha256-manifested;
#                            reproduce.sh finds them there with nothing set
#   benchmarks .../b1[789],b2[012]  the six ITC'99 circuits too large to commit
#   README-ARTIFACT.md       a pointer to the guide, docs/papers/tacas-27/AE_README.md
#
# Beside the archive it writes gpap-ae-<date>-<commit>.zip, the upload the venue's artifact form
# takes: the paper folder's AE_README.md as README.md, and LICENSE.
#
# Left out: the compared samplers (no licence; rivals_setup.sh fetches them at pinned commits)
# and built binaries (abc, CMSGen, built from pinned sources).
#
# The tables are regenerated and the gates run inside the staged tree, staged code over staged
# rows, so they describe the bytes that ship; a failing gate refuses to package.
set -uo pipefail
# Nothing run inside the stage may leave bytecode there; the scrub before the leak gate takes the
# rest.
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INVOKED_FROM="$PWD"            # a relative OUT_DIR or PYTHON is the caller's, not the checkout's
cd "$REPO_ROOT"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
# Absolute, since the gates run after `cd "$STAGE"`; a bare name is looked up on PATH.
case "$PY" in /*) ;; */*) PY="$INVOKED_FROM/$PY" ;; esac
[ -x "$PY" ] || PY="$(command -v python3)"

# Ask, never recompute: eval/paths.py is the one definition (physical, `$GPAP_RESULTS`-aware).
RESULTS="$(PYTHONPATH="$REPO_ROOT" "$PY" -m eval.paths results)"
: "${RESULTS:?cannot resolve — python3 must be able to import the repo (gpap/paths.py, eval/paths.py)}"
OUT="${1:-$RESULTS}"
# Absolute, since the gates run after `cd "$STAGE"`, and resolved where it was typed.
case "$OUT" in /*) ;; *) OUT="$INVOKED_FROM/$OUT" ;; esac
mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)" || { echo "[FATAL] cannot create $OUT" >&2; exit 1; }
SRC_DATA="${GPAP_PAPER_DIR:-$RESULTS/paper}"
# GPAP_ARTIFACT_TREE archives another tree-ish than HEAD: the dry-run knob for uncommitted work
# (`git write-tree` on a temporary index). The name and the manifest then carry that object's id.
TREE="${GPAP_ARTIFACT_TREE:-HEAD}"
COMMIT="$(git rev-parse --short "$TREE" 2>/dev/null || echo unknown)"
# A dry run says so in its name: a tree id looks like a commit id, and a dry run uploaded to
# Zenodo cannot be taken back once it has a DOI.
DRYRUN=""
[ -z "${GPAP_ARTIFACT_TREE:-}" ] || DRYRUN="-dryrun"
NAME="gpap-artifact-$(date +%Y%m%d)-$COMMIT$DRYRUN"
STAGE="$OUT/.staging_$NAME.$$"
FINAL="$OUT/$NAME"

echo "repository : $REPO_ROOT @ $COMMIT"
echo "rows       : $SRC_DATA/data"
echo "output     : $OUT/$NAME"
echo "python     : $PY"

[ -d "$SRC_DATA/data" ] || { echo "[FATAL] no published rows at $SRC_DATA/data" >&2; exit 1; }
# git archive ships HEAD, so a dirty tree would package code other than the code being tested.
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
# `reproduce.sh tables` (the folder's regenerate.sh) writes these into details/; an older folder
# has them at its top.
for f in README.md RESULTS.md VERIFICATION.txt plots; do
    for src in "$SRC_DATA/details/$f" "$SRC_DATA/$f"; do
        [ -e "$src" ] && { cp -r "$src" "$STAGE/artifact-data/paper/$f"; break; }
    done
done

# The rows must be the frozen rows.
[ -f "$STAGE/artifact-data/paper/data/MANIFEST.sha256" ] || {
    echo "[FATAL] no data/MANIFEST.sha256 beside the rows in $SRC_DATA/data — without it" >&2
    echo "        nothing here can tell the frozen campaign from a working copy." >&2
    exit 1
}
( cd "$STAGE/artifact-data/paper/data" && sha256sum --quiet -c MANIFEST.sha256 ) || {
    echo "[FATAL] the rows do not match data/MANIFEST.sha256 — these are not the frozen rows." >&2
    echo "        If a row was edited deliberately, re-run the paper folder's regenerate.sh" >&2
    echo "        (tables and checks) and update the manifest, then re-package." >&2
    exit 1
}
echo "manifest   : the rows match data/MANIFEST.sha256"

# --- the benchmarks the repository does not carry ------------------------------------------------
# Gitignored for their size and fetched at run time from a pinned URL that may not last; without
# them a third of the sequential set does not run.
n_extra=0
for b in b17 b18 b19 b20 b21 b22; do
    src="benchmarks/circuits/bench/seq/$b.bench"
    [ -f "$src" ] || { echo "[warn] $b.bench is not installed — run scripts/fetch_benchmarks.sh itc99"; continue; }
    cp "$src" "$STAGE/benchmarks/circuits/bench/seq/$b.bench" && n_extra=$((n_extra + 1))
done
echo "benchmarks : $n_extra/6 large ITC'99 circuits bundled"

# --- the entry point is the first file a reviewer sees --------------------------------------------
# A pointer, not a copy: the guide's relative links resolve only where it lives.
for f in AE_README.md ARTIFACT.md; do     # the guide (the venue zip's README), its detail
    [ -f "$STAGE/docs/papers/tacas-27/$f" ] || {
        echo "[FATAL] docs/papers/tacas-27/$f is not in the staged tree" >&2; exit 1; }
done
cat > "$STAGE/README-ARTIFACT.md" <<'POINTER'
# GPAP — artifact for the TACAS 2027 tool paper

Start with [docs/papers/tacas-27/AE_README.md](docs/papers/tacas-27/AE_README.md): what to run,
what it prints, and how to read it. [ARTIFACT.md](docs/papers/tacas-27/ARTIFACT.md) beside it
holds the detail.
POINTER

# --- Tier 1, run INSIDE the staged tree ----------------------------------------------------------
# From $STAGE: the staged generator and gate over the staged rows. Run from the checkout, a dirty
# tree would certify the package with code that is not in it.
echo
echo "== regenerating the tables and running the gate, inside the staged tree"
BAR=("$STAGE"/artifact-data/paper/data/barbarik/circ_abc_*.jsonl)
BARARGS=(); [ -e "${BAR[0]}" ] && BARARGS=(--barbarik "${BAR[@]}")
TABLES_OK=1
( cd "$STAGE" && $PY eval/artifact/paper_tables.py --rivals artifact-data/paper/data \
    ${BARARGS[@]+"${BARARGS[@]}"} \
    --out-md artifact-data/paper/RESULTS.md \
    --out-json artifact-data/paper/paper_numbers.json ) \
    > "$STAGE/artifact-data/TABLES.log" 2>&1 || TABLES_OK=0
if [ "$TABLES_OK" != 1 ]; then
    tail -20 "$STAGE/artifact-data/TABLES.log" >&2
    echo "[FATAL] the staged code could not regenerate the tables from the staged rows —" >&2
    echo "        the package would ship tables it cannot reproduce." >&2
    cp "$STAGE/artifact-data/TABLES.log" "$OUT/$NAME.TABLES.log" 2>/dev/null
    exit 1
fi
VERIFY_OK=1
# Over the VERIFICATION.txt the copy loop brought, so the artifact ships one, regenerated here.
( cd "$STAGE" && $PY eval/artifact/verify_artifact.py --rivals artifact-data/paper/data \
    ${BARARGS[@]+"${BARARGS[@]}"} ) > "$STAGE/artifact-data/paper/VERIFICATION.txt" 2>&1 \
    || VERIFY_OK=0
VERDICT="$(tail -1 "$STAGE/artifact-data/paper/VERIFICATION.txt")"
echo "$VERDICT"
if [ "$VERIFY_OK" != 1 ] && [ "${GPAP_ARTIFACT_ALLOW_FAIL:-0}" != "1" ]; then
    grep '^  FAIL' "$STAGE/artifact-data/paper/VERIFICATION.txt" >&2 \
        || tail -20 "$STAGE/artifact-data/paper/VERIFICATION.txt" >&2
    # The EXIT trap deletes the stage, and with it the only copy of the evidence.
    cp "$STAGE/artifact-data/paper/VERIFICATION.txt" "$OUT/$NAME.VERIFICATION.txt" 2>/dev/null \
        && echo "        the failing verdict is kept at $OUT/$NAME.VERIFICATION.txt" >&2
    echo "[FATAL] the gate FAILED — refusing to package an artifact its own verifier calls" >&2
    echo "        unpublishable. GPAP_ARTIFACT_ALLOW_FAIL=1 packages anyway; the verdict" >&2
    echo "        ships inside either way." >&2
    exit 1
fi

# --- do the staged tables still say what the PAPER says? ------------------------------------
# Are the numbers the PDF prints the ones this tarball's code derives from its rows? The paper's
# tables are typeset by hand, so nothing else checks. Fatal, like the gate.
NUMBERS_OK=1
( cd "$STAGE" && $PY eval/artifact/check_paper_numbers.py \
    --generated artifact-data/paper/paper_numbers.json ) \
    > "$STAGE/artifact-data/paper/PAPER_NUMBERS.txt" 2>&1 || NUMBERS_OK=0
# The count line, not `tail -1`: on a failure the last line is advice, not a verdict.
echo "numbers    : $(grep 'published values' \
    "$STAGE/artifact-data/paper/PAPER_NUMBERS.txt" | tail -1)"
if [ "$NUMBERS_OK" != 1 ]; then
    grep 'MISMATCH' "$STAGE/artifact-data/paper/PAPER_NUMBERS.txt" >&2 \
        || tail -20 "$STAGE/artifact-data/paper/PAPER_NUMBERS.txt" >&2
    # The EXIT trap deletes the stage, and with it the only copy of the evidence.
    cp "$STAGE/artifact-data/paper/PAPER_NUMBERS.txt" "$OUT/$NAME.PAPER_NUMBERS.txt" 2>/dev/null \
        && echo "        the mismatches are kept at $OUT/$NAME.PAPER_NUMBERS.txt" >&2
    echo "[FATAL] the paper prints numbers these rows do not produce — refusing to package an" >&2
    echo "        artifact that disagrees with the paper it supports. Find which row moved;" >&2
    echo "        re-transcribing docs/papers/*/published_numbers.json to make this pass mints" >&2
    echo "        a number no campaign produced." >&2
    exit 1
fi

# --- nothing the staging steps themselves wrote may ship ------------------------------------------
# Before the gate and the manifest, so both see exactly the bytes that ship.
find "$STAGE" -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) \
    -prune -exec rm -rf {} +

# --- nothing that names one machine or one person may ship ---------------------------------------
# A path in a shipped file names the machine that built it, in an upload a DOI makes permanent.
# The shapes live in check_leaks.sh; the generators are fixed at the source
# (eval/paths.py::display_path), and this says they stay fixed. GPAP_ARTIFACT_DENYLIST=<file> adds
# strings a shipped script cannot spell, such as a username or a hostname.
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

# --- manifest -------------------------------------------------------------------------------------
# MANIFEST.md first, so the hashes cover it; the count comes from find, since the hash file does
# not exist yet.
NFILES="$(cd "$STAGE" && find . -type f ! -name MANIFEST.sha256 | wc -l)"
NFILES=$((NFILES + 1))                                   # MANIFEST.md itself, written next
{
  echo "# $NAME"
  echo
  if [ -n "$DRYRUN" ]; then
    echo "**DRY RUN** — built $(date -Is) from the TREE OBJECT \`$COMMIT\` on $(uname -sr):"
    echo "uncommitted work, packaged to exercise the gates. NOT a release, and not for upload."
  else
    echo "Built $(date -Is) from commit \`$COMMIT\` on $(uname -sr)."
  fi
  echo
  echo "    $VERDICT"
  echo
  echo "Start with **docs/papers/tacas-27/AE_README.md** (\`README-ARTIFACT.md\` points to it)."
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

# --- the venue's upload: one small zip that points to the DOI --------------------------------------
# From the archive's own copies, so the zip says what the archive says.
ZIP="$OUT/gpap-ae-${NAME#gpap-artifact-}.zip"
"$PY" - "$ZIP" "$FINAL/docs/papers/tacas-27/AE_README.md" "$FINAL/LICENSE" <<'ZIP' \
    || { echo "[FATAL] could not write $ZIP" >&2; exit 1; }
import sys, zipfile
out, readme, licence = sys.argv[1:]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(readme, "README.md")
    z.write(licence, "LICENSE")
ZIP
echo
echo "artifact: $OUT/$NAME.tar.gz  ($(du -h "$OUT/$NAME.tar.gz" | cut -f1), $NFILES files)"
echo "upload  : $ZIP  (README.md and LICENSE, for the venue's artifact form)"
echo "verdict : $VERDICT"
echo
echo "Next: upload to Zenodo for the DOI (the Available badge), and put the CONCEPT DOI — the"
echo "      one Zenodo labels 'all versions', never a version DOI — in CITATION.cff and the"
echo "      paper's Data Availability Statement. A version DOI goes stale the moment a corrected"
echo "      artifact is uploaded, and the paper's link with it."

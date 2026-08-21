#!/usr/bin/env bash
# Build meelgroup/cmsgen — the CPU sampler GPAP is benchmarked against — at a PINNED commit,
# with the GPAP evaluation patch applied (scripts/cmsgen_gpap_eval.patch).
#
# The patch adds two OFF-BY-DEFAULT flags; stock CMSGen behaviour is unchanged without them:
#   --nowrite       generate samples, emit none  (the fair mode vs GPAP's non-emitting timed run:
#                   stock CMSGen spends most of its wall formatting full N-var assignments)
#   --statsevery N  a flushed, parseable `c gpap-stats ... props= decisions= conflicts= ...` line
#                   every N samples — the CPU work rates our impl/s and dec/s are compared to
# `gpap bench` auto-detects both (it probes --help) and falls back to stock behaviour without them.
#
#   scripts/build_cmsgen.sh                     # -> <repo>/.gpap/bin/cmsgen
#   GPAP_CMSGEN_FORCE=1 scripts/build_cmsgen.sh # rebuild from scratch
#   GPAP_CMSGEN_STOCK=1 scripts/build_cmsgen.sh # build UNPATCHED upstream (-> bin/cmsgen-stock)
#
# CMSGen is fetched from upstream at build time — GPAP redistributes no CMSGen code or binaries;
# the patch file in this repo is a diff against the pinned upstream commit.
set -euo pipefail

CMSGEN_COMMIT="${GPAP_CMSGEN_COMMIT:-8210aa90b7535c75cbde8e6a3c5bb9763b233eda}"
CMSGEN_REPO="${GPAP_CMSGEN_REPO:-https://github.com/meelgroup/cmsgen.git}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${GPAP_HOME:-$REPO/.gpap}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="${GPAP_CMSGEN_PATCH:-$REPO_ROOT/scripts/cmsgen_gpap_eval.patch}"

STOCK="${GPAP_CMSGEN_STOCK:-0}"
if [ "$STOCK" = "1" ]; then
    SRC="$PREFIX/src/cmsgen-stock"; BIN="$PREFIX/bin/cmsgen-stock"
else
    SRC="$PREFIX/src/cmsgen"; BIN="$PREFIX/bin/cmsgen"
fi
BIN_DIR="$PREFIX/bin"

if [ -x "$BIN" ] && [ "${GPAP_CMSGEN_FORCE:-0}" != "1" ]; then
    echo "cmsgen already built: $BIN"
    "$BIN" --help 2>&1 | grep -q -- --nowrite \
        && echo "  (GPAP eval patch present: --nowrite / --statsevery available)" \
        || echo "  (stock build: no --nowrite; rerun with GPAP_CMSGEN_FORCE=1 to apply the patch)"
    exit 0
fi

for tool in git cmake make g++; do
    command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: '$tool' is required to build cmsgen" >&2; exit 1; }
done

mkdir -p "$SRC" "$BIN_DIR"
cd "$SRC"
[ -d .git ] || git init -q .
git remote add origin "$CMSGEN_REPO" 2>/dev/null || git remote set-url origin "$CMSGEN_REPO"
echo "fetching cmsgen ..."
# The pinned commit is a merge on master; a depth-1 fetch of the SHA works on GitHub, but fall
# back to the branch tip if the server refuses SHA fetches (older git / mirrors).
git fetch -q --depth 1 origin "$CMSGEN_COMMIT" 2>/dev/null || git fetch -q --depth 1 origin master
git checkout -q FETCH_HEAD 2>/dev/null || git checkout -q origin/master
git checkout -q -- .                       # drop any previously applied patch (idempotent)

if [ "$STOCK" != "1" ]; then
    [ -f "$PATCH" ] || { echo "ERROR: patch not found: $PATCH" >&2; exit 1; }
    echo "applying $(basename "$PATCH") ..."
    git apply --verbose "$PATCH" || {
        echo "ERROR: the GPAP eval patch does not apply to this cmsgen revision." >&2
        echo "       Either pin GPAP_CMSGEN_COMMIT back, or regenerate the patch." >&2
        exit 1
    }
fi

echo "building cmsgen ..."
mkdir -p build && cd build
cmake .. >"$SRC/build.log" 2>&1
make -j"$(nproc 2>/dev/null || echo 4)" >>"$SRC/build.log" 2>&1 || {
    echo "ERROR: cmsgen build failed; tail of $SRC/build.log:" >&2
    tail -25 "$SRC/build.log" >&2
    exit 1
}

install -m755 "$SRC/build/cmsgen" "$BIN"
echo "installed: $BIN"
"$BIN" --help 2>&1 | head -1
if [ "$STOCK" != "1" ]; then
    "$BIN" --help 2>&1 | grep -q -- --nowrite \
        && echo "GPAP eval patch active (--nowrite / --statsevery)" \
        || { echo "ERROR: patched build lacks --nowrite" >&2; exit 1; }
fi

#!/usr/bin/env bash
# Build berkeley-abc for GPAP's OPTIONAL circuit-preprocessing pass (gpap.circuits.abc_backend).
#
# ABC is never vendored and never a hard dependency: GPAP falls back to its native Tseitin
# encoder when no `abc` binary is found. This script fetches a PINNED commit (reproducible),
# builds it, and installs the single static binary into the GPAP cache dir.
#
#   scripts/build_abc.sh              # -> $HOME/.cache/gpap/bin/abc
#   GPAP_ABC_HOME=/opt/gpap scripts/build_abc.sh
#
# GPAP finds the binary via, in order:  $GPAP_ABC  ->  `abc` on PATH  ->  $GPAP_ABC_HOME/bin/abc
#
# ABC is distributed by UC Berkeley under a permissive BSD-style licence; it is fetched from
# upstream at build time, so GPAP redistributes no ABC code or binaries.
set -euo pipefail

# Pinned upstream commit (berkeley-abc/abc master @ 2026-07-09). Override to track a newer ABC.
ABC_COMMIT="${GPAP_ABC_COMMIT:-bcfdf592289a408cd67ec19260f8a60a37b085b6}"
ABC_REPO="${GPAP_ABC_REPO:-https://github.com/berkeley-abc/abc.git}"
PREFIX="${GPAP_ABC_HOME:-$HOME/.cache/gpap}"

SRC="$PREFIX/src/abc"
BIN_DIR="$PREFIX/bin"
BIN="$BIN_DIR/abc"

if [ -x "$BIN" ] && [ "${GPAP_ABC_FORCE:-0}" != "1" ]; then
    echo "abc already built: $BIN"
    "$BIN" -q "version" || true
    exit 0
fi

for tool in git make gcc; do
    command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: '$tool' is required to build abc" >&2; exit 1; }
done

mkdir -p "$SRC" "$BIN_DIR"
cd "$SRC"
# Shallow fetch of exactly the pinned commit (keeps the checkout small and reproducible).
[ -d .git ] || git init -q .
git remote add origin "$ABC_REPO" 2>/dev/null || git remote set-url origin "$ABC_REPO"
echo "fetching abc @ ${ABC_COMMIT:0:12} ..."
git fetch -q --depth 1 origin "$ABC_COMMIT"
git checkout -q FETCH_HEAD

# ABC links GNU readline only for its interactive shell; we drive it with `abc -q "<script>"`,
# and the readline dev headers are absent on many boxes (incl. this WSL2 one), so disable it.
echo "building abc (this takes a few minutes) ..."
make ABC_USE_NO_READLINE=1 -j"$(nproc 2>/dev/null || echo 4)" abc >"$SRC/build.log" 2>&1 || {
    echo "ERROR: abc build failed; tail of $SRC/build.log:" >&2
    tail -20 "$SRC/build.log" >&2
    exit 1
}

install -m755 "$SRC/abc" "$BIN"
echo "installed: $BIN"
"$BIN" -q "version"

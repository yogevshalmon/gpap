#!/usr/bin/env bash
# Build berkeley-abc for GPAP's OPTIONAL circuit-preprocessing pass (gpap.circuits.abc_backend).
#
# ABC is never vendored and never a hard dependency: GPAP falls back to its native Tseitin
# encoder when no `abc` binary is found. This fetches a pinned commit, builds it, and installs
# the static binary into the repo-local GPAP cache (not $HOME, which these artifacts would fill).
#
#   scripts/build_abc.sh              # -> <repo>/.gpap/bin/abc
#   GPAP_HOME=/opt/gpap scripts/build_abc.sh
#
# GPAP finds the binary via $GPAP_ABC, then `abc` on PATH, then $GPAP_HOME/bin/abc. ABC is
# fetched from upstream at build time under its BSD-style licence; nothing is redistributed.
set -euo pipefail

# Pinned upstream commit — the hash IS the pin. Override to track a newer ABC.
ABC_COMMIT="${GPAP_ABC_COMMIT:-bcfdf592289a408cd67ec19260f8a60a37b085b6}"
ABC_REPO="${GPAP_ABC_REPO:-https://github.com/berkeley-abc/abc.git}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Ask, never recompute: gpap/paths.py is the one definition (see its docstring).
PREFIX="$(PYTHONPATH="$REPO" "${PYTHON:-python3}" -m gpap.paths home)" || { echo "$0: cannot resolve the GPAP root — is python3 able to import the repo?" >&2; exit 1; }

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

# ABC links readline only for its interactive shell, which we never use, and its dev headers
# are absent on many systems.
echo "building abc (this takes a few minutes) ..."
make ABC_USE_NO_READLINE=1 -j"$(nproc 2>/dev/null || echo 4)" abc >"$SRC/build.log" 2>&1 || {
    echo "ERROR: abc build failed; tail of $SRC/build.log:" >&2
    tail -20 "$SRC/build.log" >&2
    exit 1
}

install -m755 "$SRC/abc" "$BIN"
echo "installed: $BIN"
"$BIN" -q "version"

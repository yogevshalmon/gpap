#!/usr/bin/env bash
# Refuse a directory that carries a string naming one machine or one person.
#
#   eval/artifact/check_leaks.sh DIR [--denylist FILE]
#
# Exit 0: clean. Exit 1: leaks, listed on stdout as `<shape>  <path>`. Exit 2: usage.
#
# make_artifact.sh runs it over the staged tree, so the verdict is about the bytes a reviewer
# receives. A generator prints the path it was handed, and every shape below is one that shipped
# or reached staging (docs/findings/ARTIFACT_FINDINGS.md §9).
#
# The operator's identity, not the authors' (named on purpose: this is not a double-blind
# submission). A username or a hostname cannot be spelled in a file that ships, so they come from
# a denylist kept outside the repository: one fixed string per line, case-insensitive. Blank lines,
# `#` comments and CRs are stripped, since to `grep -F -f` an empty line matches every line and a
# trailing CR is part of the pattern. Gated by tests/experiments/test_artifact_gate.py.
set -uo pipefail

usage() { echo "usage: check_leaks.sh DIR [--denylist FILE]" >&2; exit 2; }
[ $# -ge 1 ] || usage
DIR="$1"; shift
DENY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --denylist) [ $# -ge 2 ] || usage; DENY="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[ -d "$DIR" ] || { echo "not a directory: $DIR" >&2; exit 2; }
[ -z "$DENY" ] || [ -f "$DENY" ] || { echo "no such denylist: $DENY" >&2; exit 2; }

# THE SHAPES: a path form that names a machine without saying so. Text files only (`-I`): an
# image's bytes are not a path. Mirrored by tests/repo/test_docs_hygiene.py over the checkout.
SHAPES=(
    '/home/[a-z][A-Za-z0-9_.-]*/'          # a Linux home directory
    '/Users/[A-Za-z][A-Za-z0-9_.-]*/'      # a macOS home directory
    '-home-[a-z][A-Za-z0-9_.-]*-'          # a home path flattened into one directory name
    '/claude-[0-9]+/'                      # an agent's per-user scratch root
)

status=0
for shape in "${SHAPES[@]}"; do
    hits="$(cd "$DIR" && grep -rlIE -- "$shape" . 2>/dev/null || true)"
    if [ -n "$hits" ]; then
        status=1
        echo "$hits" | sed "s|^\./||; s|^|$shape  |"
    fi
done
if [ -n "$DENY" ]; then
    # Comments, blank lines and CRs are stripped before grep sees a pattern (see the header).
    pats="$(tr -d '\r' < "$DENY" | grep -vE '^[[:space:]]*(#|$)' || true)"
    if [ -n "$pats" ]; then
        hits="$(cd "$DIR" && printf '%s\n' "$pats" | grep -rlIiF -f - . 2>/dev/null || true)"
        if [ -n "$hits" ]; then
            status=1
            echo "$hits" | sed 's|^\./||; s|^|denylist  |'
        fi
    fi
fi
exit $status

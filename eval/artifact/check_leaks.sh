#!/usr/bin/env bash
# Refuse a directory that carries a string naming one machine or one person.
#
#   eval/artifact/check_leaks.sh DIR [--denylist FILE]
#
# Exit 0: clean. Exit 1: leaks, listed on stdout as `<shape>  <path>`. Exit 2: usage.
#
# `make_artifact.sh` runs this over the STAGED tree, so the verdict is about the bytes a reviewer
# receives, not about the checkout. It exists because a generator prints the path it was handed,
# and that path names the disk it ran on: a committed slide table carried a home directory for a
# month, `emission_audit.txt` reached a staging directory with one, and three rows of the
# published Barbarik campaign carried an agent's scratch directory with the operator's username
# flattened into its NAME — a shape with no `/home/` in it, which the first version of this check
# (one pattern inline in the packager) let through. Every shape below is one that shipped, or
# reached staging.
#
# WHOSE identity: the operator's, not the authors'. The authors are named on purpose — this is not
# a double-blind submission — and what must not ship is the machine that built the tarball. A
# username or a hostname cannot be spelled here anyway, since this file ships inside the tarball it
# checks, so they come from a denylist kept OUTSIDE the repository: one fixed string per line,
# matched case-insensitively; blank lines, `#` comments and carriage returns are stripped, because
# to `grep -F -f` an empty line is a pattern that matches EVERY line and a trailing CR is part of
# the pattern — the two ways this gate can fail open. Gated by tests/experiments/test_artifact_gate.py.
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
    # Comments and blank lines are stripped BEFORE grep sees a pattern, and so is a carriage
    # return: `grep -F` takes a trailing CR as part of the pattern, so a denylist saved on Windows
    # matched nothing and this returned CLEAN — see the header.
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

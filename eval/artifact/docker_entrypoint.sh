#!/usr/bin/env bash
# The artifact image's entry point (eval/artifact/Dockerfile). The first argument is a step of
# reproduce.sh and the rest are its options:
#
#   docker run --rm --gpus all IMAGE check
#   docker run --rm IMAGE tables                                  # needs no GPU
#   docker run --rm --gpus all -v "$PWD/out:/out" IMAGE smoke     # and likewise subset, full
#
# The steps that run on the GPU first ask the tool image's entry point whether there is one GPAP
# can use, so a missing --gpus is one line, not a traceback; `check` also gets its first-use
# compile, so it shows the engine builds for this GPU. A GPU tier runs `setup` first, in the same
# container, as the guide's path without Docker does: it compiles libgpap for this GPU and clones
# the compared samplers at their pinned commits (they state no licence, so no image carries
# them). Anything that is not a step (`gpap sample ...`, `bash`) goes to the tool image's entry
# point. Results go under the experiments root, which the image points at the directory to
# mount; run as root, what the steps create there is given back to that directory's owner, also
# when `docker stop` or Ctrl-C ends them.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="$(PYTHONPATH="$REPO" python3 -m eval.paths experiments)"
: "${OUT_ROOT:?cannot resolve the experiments root — python3 must import the repo (eval/paths.py)}"
REPRODUCE="$REPO/eval/artifact/reproduce.sh"

STEP="${1:-check}"
case "$STEP" in
    check|setup|tables|smoke|subset|full|compare) [ $# -gt 0 ] && shift ;;
    *) exec "$REPO/scripts/docker_entrypoint.sh" "$@" ;;
esac
. "$REPO/scripts/docker_entrypoint.sh"          # usable_cc, ensure_core

# PID 1 gets no default action for TERM or INT: without this, `docker stop` waited out its grace
# period and killed the step, and Ctrl-C did nothing. Each step runs in its own process group
# (setsid), which the trap signals as a whole; a signal that arrives between steps keeps the next
# one from starting. The results are still given back below.
child="" stopped=0
stop_child() { kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null; }
trap 'stopped=1; [ -z "$child" ] || stop_child' TERM INT
run() {
    [ "$stopped" = 0 ] || return 143
    setsid "$@" & child=$!
    [ "$stopped" = 0 ] || stop_child         # a signal that came before $child was known
    local rc=0
    wait "$child" || rc=$?
    while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done   # a trap ended the wait
    child=""
    return "$rc"
}

case "$STEP" in
    check)
        # In the foreground, so a stop waits for the compile (under a minute) to end.
        [ "${GPAP_SKIP_BUILD:-0}" = 1 ] || ensure_core || exit 1
        [ "$stopped" = 0 ] || exit 143 ;;
    setup|smoke|subset|full)
        usable_cc >/dev/null || { echo "[FAIL] $STEP runs on the GPU; see the line above." >&2; exit 1; } ;;
esac
umask 022                                       # ensure_core compiles under 000
case "$STEP" in
    tables|smoke|subset|full|compare)
        mountpoint -q "$OUT_ROOT" || echo "NOTE: nothing is mounted at $OUT_ROOT, so the results" \
            "stay inside this container; add  -v \"\$PWD/out:$OUT_ROOT\"  to keep them." >&2 ;;
esac

rc=0
case "$STEP" in
    smoke|subset|full)
        # Its output is long and only matters when it fails, so it goes to a log beside the steps'.
        echo "setup: compiling the engine for this GPU, cloning the compared samplers ($OUT_ROOT/setup.log)"
        run bash "$REPRODUCE" setup >"$OUT_ROOT/setup.log" 2>&1 || rc=$?
        if [ "$rc" = 0 ]; then
            echo "setup: done"
        elif [ "$stopped" = 0 ]; then               # failed, rather than stopped
            tail -25 "$OUT_ROOT/setup.log" >&2
        fi
        # The compared samplers come from GitHub, once per container. Said here, not left to
        # surface as error rows an hour into a tier, and only when a clone did fail.
        if grep -q '^\[warn\] clone failed:' "$OUT_ROOT/setup.log"; then
            echo "NOTE: a compared sampler could not be cloned: a GPU step needs network access to" \
                 "github.com, once per container ($OUT_ROOT/setup.log)." >&2
        fi ;;
esac
if [ "$rc" = 0 ]; then
    run bash "$REPRODUCE" "$STEP" "$@" || rc=$?
elif [ "$stopped" = 0 ]; then
    echo "[FAIL] setup failed, so $STEP did not run; see above." >&2
fi
# Only what this container created, which is root's: Docker creates a missing bind source as
# root, and then there is no owner to give it to (the guide's `mkdir -p out` comes first).
if [ "$(id -u)" = 0 ] && mountpoint -q "$OUT_ROOT" \
        && owner="$(stat -c %u:%g "$OUT_ROOT")" && [ "$owner" != 0:0 ]; then
    find "$OUT_ROOT" -uid 0 -exec chown -h "$owner" {} + 2>/dev/null || true
fi
exit "$rc"

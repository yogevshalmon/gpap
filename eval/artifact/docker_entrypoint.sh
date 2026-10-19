#!/usr/bin/env bash
# The artifact image's entry point (eval/artifact/Dockerfile). The first argument is a step of
# reproduce.sh and the rest are its options:
#
#   docker run --rm --gpus all IMAGE check
#   docker run --rm IMAGE tables                                  # needs no GPU
#   docker run --rm --gpus all -v "$PWD/out:/out" IMAGE smoke     # and likewise subset, full
#
# A GPU tier runs `setup` first, in the same container, as the guide's path without Docker does:
# it compiles libgpap for this GPU and clones the compared samplers at their pinned commits (they
# state no licence, so no image carries them). Anything that is not a step (`gpap sample ...`,
# `bash`) goes to the tool image's entry point, which compiles the engine for the GPU first.
# Results go under the experiments root, which the image points at the directory to mount; run as
# root, this gives them back to that directory's owner.
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
        if bash "$REPRODUCE" setup >"$OUT_ROOT/setup.log" 2>&1; then
            echo "setup: done"
        else
            rc=1; tail -25 "$OUT_ROOT/setup.log" >&2
        fi
        # The compared samplers come from GitHub, once per container. Said here, not left to
        # surface as error rows an hour into a tier.
        if bash "$REPO/eval/rivals/rivals_setup.sh" check 2>/dev/null | grep -q 'MISSING (https://'; then
            echo "NOTE: a compared sampler could not be cloned: a GPU step needs network access to" \
                 "github.com, once per container ($OUT_ROOT/setup.log)." >&2
        fi ;;
esac
if [ "$rc" = 0 ]; then
    bash "$REPRODUCE" "$STEP" "$@"; rc=$?
else
    echo "[FAIL] setup failed, so $STEP did not run; see above." >&2
fi
if [ "$(id -u)" = 0 ] && mountpoint -q "$OUT_ROOT"; then
    chown -R "$(stat -c %u:%g "$OUT_ROOT")" "$OUT_ROOT" 2>/dev/null || true
fi
exit "$rc"

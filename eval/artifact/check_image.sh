#!/usr/bin/env bash
# Gate the tool image and an artifact image before they are saved (make_image.sh) or submitted:
# what they must do on any machine, with no GPU and no network, and with --gpu what they must do
# on this machine's GPU, which needs the NVIDIA Container Toolkit.
#
#   eval/artifact/check_image.sh TOOL_IMAGE ARTIFACT_IMAGE [--gpu]
#
# One [ok] or [FAIL] line per gate, a log of each in a kept directory, and exit 1 if any gate
# failed. What `tables` must write into its SUMMARY.txt is read from the guide inside the image
# (docs/papers/<tag>/ARTIFACT.md, <tag> being the artifact image's: the block after the comment
# naming this script), so the gate and the guide cannot disagree. GPAP_ARTIFACT_DENYLIST, a file kept outside the repository (check_leaks.sh says
# why), adds the operator's names to the leak gate.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() { sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 2; }
[ $# -ge 2 ] || usage
TOOL="$1"; ART="$2"; GPU=0
[ "${3:-}" = "--gpu" ] && GPU=1
TAG="${ART##*:}"
GUIDE="/opt/gpap/docs/papers/$TAG/ARTIFACT.md"
DENY="${GPAP_ARTIFACT_DENYLIST:-}"
WORK="$(mktemp -d)"
LOGS="$WORK/logs"; mkdir -p "$LOGS" "$WORK/out" "$WORK/data"
ME="$(id -u):$(id -g)"
VOL="gpap-check-$$"

FAILED=()
# gate NAME CMD...: runs CMD with its output in $LOGS/NAME.log and prints one line.
gate() {
    local name="$1"; shift
    if "$@" >"$LOGS/$name.log" 2>&1; then echo "[ok]   $name"
    else echo "[FAIL] $name   (log: $LOGS/$name.log)"; FAILED+=("$name"); fi
}
has() { grep -qF -- "$2" "$LOGS/$1.log"; }     # has GATE TEXT: the gate's log contains TEXT

# --------------------------------------------------------------------- no GPU, no network
tool_starts() {
    docker run --rm --network none "$TOOL" && has tool-starts-without-gpu "no NVIDIA GPU in this container" \
        && has tool-starts-without-gpu "usage:"
}
gate tool-starts-without-gpu tool_starts
gate tool-imports-gpap docker run --rm --network none "$TOOL" python -c "import gpap"
gate archive-intact docker run --rm --network none --entrypoint bash "$ART" \
    -c 'cd /opt/gpap && sha256sum --quiet -c MANIFEST.sha256'
leaks() {
    docker run --rm --network none --entrypoint bash ${DENY:+-v "$DENY:/run/denylist:ro"} "$ART" \
        -c "eval/artifact/check_leaks.sh /opt/gpap ${DENY:+--denylist /run/denylist}" || return 1
    mkdir -p "$WORK/meta"
    docker history --no-trunc "$ART" >"$WORK/meta/history.txt"
    docker image inspect "$TOOL" "$ART" >"$WORK/meta/inspect.json"
    bash "$HERE/check_leaks.sh" "$WORK/meta" ${DENY:+--denylist "$DENY"}
}
gate no-leaks leaks
check_step() { docker run --rm --network none "$ART" check && has check-without-gpu "only the \`tables\` step can run here"; }
gate check-without-gpu check_step
tables() {
    local expected line summary="$WORK/out/tables/SUMMARY.txt"
    docker run --rm --network none -v "$WORK/out:/out" "$ART" tables --out /out/tables || return 1
    expected="$(docker run --rm --entrypoint awk "$ART" '/check_image.sh: the lines/ {f=1; next}
        f && /^[ \t]*```/ {n++; if (n == 2) exit; next}
        f && n == 1 {sub(/^[ \t]+/, ""); print}' "$GUIDE")"
    [ -n "$expected" ] || { echo "no expected lines found in $GUIDE"; return 1; }
    while IFS= read -r line; do
        grep -qxF -- "$line" "$summary" || { echo "MISSING from $summary: $line"; return 1; }
    done <<<"$expected"
}
gate tables tables
# The shipped tables were generated from the same rows when the archive was built: only a date
# may differ. Every shipped figure is drawn again, byte for byte.
as_shipped() {
    docker run --rm --network none -v "$WORK/out:/out" --entrypoint bash "$ART" -c '
        cd /opt/gpap/artifact-data/paper
        diff /out/tables/details/RESULTS.md RESULTS.md | grep "^[<>]" | grep -vE "[0-9]{4}-[0-9]{2}-[0-9]{2}" && exit 1
        for f in plots/*; do cmp "$f" "/out/tables/details/$f" || exit 1; done
        echo "RESULTS.md and $(ls plots | wc -l) figures as shipped"'
}
gate tables-as-shipped as_shipped
gate host-test-suites docker run --rm --network none --entrypoint bash "$ART" -c '
    pytest -q -p no:cacheprovider -m "not gpu and not slow" &&
    pytest -q -p no:cacheprovider tests/experiments -m "not gpu and not slow"'

# --------------------------------------------------------------------- this machine's GPU
if [ "$GPU" = 1 ]; then
    if ! docker run --rm --gpus all "$TOOL" true >"$LOGS/gpus.log" 2>&1; then
        echo "[FAIL] --gpus does not work here: install the NVIDIA Container Toolkit (log: $LOGS/gpus.log)"
        FAILED+=(gpus)
    else
        compiles() { docker run --rm --gpus all -v "$VOL:/cache" "$TOOL" info && has tool-compiles-on-first-use "compiling the engine"; }
        gate tool-compiles-on-first-use compiles
        reuses() { docker run --rm --gpus all -v "$VOL:/cache" "$TOOL" info && ! has tool-reuses-the-compile "compiling the engine"; }
        gate tool-reuses-the-compile reuses
        as_user() {
            docker run --rm --gpus all --user "$ME" -v "$VOL-user:/cache" -v "$WORK/data:/data" "$TOOL" \
                sample /opt/gpap/benchmarks/circuits/bench/seq/b10.bench --cycles 25 -n 10000 \
                --out /data/b10.txt --sim-check || return 1
            docker run --rm --gpus all --user "$ME" -v "$VOL-user:/cache" -v "$WORK/data:/data" "$TOOL" \
                sample /opt/gpap/benchmarks/sampling/27.sk_3_32.cnf -n 10000 --out /data/cnf.txt || return 1
            [ -s "$WORK/data/b10.txt" ] && [ -s "$WORK/data/cnf.txt" ] \
                && [ -z "$(find "$WORK/data" ! -user "$(id -u)")" ]
        }
        gate tool-samples-as-a-user as_user
        gate tool-gpu-test-suite docker run --rm --gpus all -v "$VOL:/cache" "$TOOL" pytest -q -p no:cacheprovider
        smoke() {
            docker run --rm --gpus all -v "$WORK/out:/out" "$ART" smoke --out /out/smoke || return 1
            grep -qF "PASS — 4 cells compared" "$WORK/out/smoke/SUMMARY.txt" \
                && grep -qxF "ARTIFACT_TIER_COMPLETE" "$WORK/out/smoke/SUMMARY.txt" \
                && [ -z "$(find "$WORK/out" ! -user "$(id -u)")" ]
        }
        gate smoke smoke
        offline() {
            local c="$VOL-offline"
            docker run -d --gpus all --name "$c" "$ART" sleep infinity || return 1
            docker exec "$c" bash eval/artifact/reproduce.sh setup \
                && docker network disconnect bridge "$c" \
                && docker exec "$c" bash eval/artifact/reproduce.sh smoke --out /out/offline \
                && has smoke-offline-after-setup "PASS — 4 cells compared"
            local rc=$?
            docker rm -f "$c" >/dev/null
            return "$rc"
        }
        gate smoke-offline-after-setup offline
        says_why() {
            ! docker run --rm --gpus all --network none "$ART" smoke \
                && has no-network-says-why "needs network access to github.com"
        }
        gate no-network-says-why says_why
        docker volume rm -f "$VOL" "$VOL-user" >/dev/null 2>&1
    fi
fi

echo
docker image inspect --format '{{index .RepoTags 0}}  {{.Size}} bytes' "$TOOL" "$ART"
echo "logs: $LOGS"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi
echo "every gate passed"

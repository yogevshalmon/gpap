#!/usr/bin/env bash
# Build the tool image and one paper's artifact image from an unpacked artifact archive, gate
# them (check_image.sh), and save both in one file for the archive's record (Zenodo).
#
#   eval/artifact/make_image.sh <unpacked archive> [OUT_DIR]      # OUT_DIR: beside the archive
#
# Both come from the archive, so every file in them passed make_artifact.sh's gates: the tool
# image from the whole archive (the root .dockerignore leaves the paper's files out), and the
# artifact layer from a context holding only the files its Dockerfile copies. GPAP_IMAGE_TAG
# names the artifact image, by default tacas-27, the paper's folder under docs/papers/.
# GPAP_IMAGE_GPU=1 adds the gates that need this machine's GPU. Nothing is saved unless every
# gate passes.
set -euo pipefail

ARCHIVE="$(cd "${1:?usage: make_image.sh <unpacked archive> [OUT_DIR]}" && pwd)"
OUT="$(cd "${2:-$ARCHIVE/..}" && pwd)"
NAME="$(basename "$ARCHIVE")"                         # gpap-artifact-<date>-<commit>[-dryrun]
TAG="${GPAP_IMAGE_TAG:-tacas-27}"
[ -f "$ARCHIVE/MANIFEST.sha256" ] || { echo "$ARCHIVE is not an unpacked artifact archive" >&2; exit 1; }
[ -d "$ARCHIVE/docs/papers/$TAG" ] || { echo "no docs/papers/$TAG in the archive: set GPAP_IMAGE_TAG" >&2; exit 1; }
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$ARCHIVE/pyproject.toml")"
REVISION="$(echo "$NAME" | cut -d- -f4)"
TOOL="gpap:$VERSION"
ART="gpap-artifact:$TAG"
SAVED="$OUT/gpap-image-${NAME#gpap-artifact-}.tar.gz"   # never matched by gpap-artifact-*

echo "archive : $ARCHIVE"
echo "images  : $TOOL  $ART"
echo "output  : $SAVED"

docker build -t "$TOOL" --build-arg GPAP_VERSION="$VERSION" --build-arg GPAP_REVISION="$REVISION" \
    "$ARCHIVE"
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
cp "$ARCHIVE/eval/artifact/Dockerfile" "$CTX/Dockerfile"
for src in $(sed -n 's|^COPY \(.*\) /opt/gpap/\{0,1\}[^ ]*$|\1|p' "$CTX/Dockerfile"); do
    cp -r "$ARCHIVE/$src" "$CTX/"
done
docker build -t "$ART" --build-arg BASE="$TOOL" "$CTX"

bash "$ARCHIVE/eval/artifact/check_image.sh" "$TOOL" "$ART" ${GPAP_IMAGE_GPU:+--gpu}

docker save "$TOOL" "$ART" | gzip >"$SAVED"
( cd "$OUT" && sha256sum "$(basename "$SAVED")" >"$(basename "$SAVED").sha256" )
echo "saved   : $SAVED ($(du -h "$SAVED" | cut -f1))"
echo "load it : docker load -i $(basename "$SAVED")"

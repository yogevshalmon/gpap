#!/usr/bin/env bash
# Fetch the FULL upstream circuit-benchmark sets that back the curated subset committed under
# benchmarks/circuits/ (see benchmarks/circuits/README.md for provenance/licenses).
# Everything lands in <repo>/.gpap/benchmarks: durable across reboots, and repo-local so a
# quota'd home directory does not silently fill up. Override with $GPAP_HOME.
#
#   scripts/fetch_benchmarks.sh              # every set
#   scripts/fetch_benchmarks.sh itc99        # just the big ITC'99 circuits (~25 MB)
#
# The set filter exists so a driver that needs only b17-b22 does not also pull the AIGER sets:
# eval/colab/colab_rivals_l4.sh calls it on a fresh VM, and a rival sweep touches none of them.
#
# Every source is pinned, so a fetch on any machine gets the files the recorded campaigns ran
# on: the GitHub sets by commit (the raw URL names it), the HWMCC'17 tarball, which has no
# commit, by its sha256. Moving a pin changes the instances.
set -euo pipefail

EPFL_COMMIT="82d8cc6910419298e713a46644ed59fd3df53038"     # lsils/benchmarks
I99T_COMMIT="8a2c3b500ee7ff20e7031de92592b737bedc6d8c"     # cad-polito-it/I99T
HWMCC17_SHA256="8bcf9fd4413965661a9c23b0d086a97fa13145b8d9f75fab23020336f4b5fcfe"

WANT="${1:-all}"
want_set() { [ "$WANT" = "all" ] || [ "$WANT" = "$1" ]; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Ask, never recompute: gpap/paths.py is the one definition (see its docstring).
CACHE="$(PYTHONPATH="$REPO" "${PYTHON:-python3}" -m gpap.paths benchmarks)" || { echo "$0: cannot resolve the GPAP root — is python3 able to import the repo?" >&2; exit 1; }
mkdir -p "$CACHE"
cd "$CACHE"

if want_set epfl; then
echo "== EPFL combinational suite (lsils/benchmarks, AIG only) -> $CACHE/epfl"
mkdir -p epfl
for f in adder bar div hyp log2 max multiplier sin sqrt square; do
    [ -f "epfl/$f.aig" ] || curl -sf -o "epfl/$f.aig" \
        "https://raw.githubusercontent.com/lsils/benchmarks/$EPFL_COMMIT/arithmetic/$f.aig"
done
for f in arbiter cavlc ctrl dec i2c int2float mem_ctrl priority router voter; do
    [ -f "epfl/$f.aig" ] || curl -sf -o "epfl/$f.aig" \
        "https://raw.githubusercontent.com/lsils/benchmarks/$EPFL_COMMIT/random_control/$f.aig"
done

fi

if want_set hwmcc; then
echo "== HWMCC'17 single-property AIGs (300 instances, ~27 MB) -> $CACHE/hwmcc17"
if [ ! -d hwmcc17 ] || [ "$(find hwmcc17 -maxdepth 1 -type f -name '*.aig' | wc -l)" -lt 300 ]; then
    curl -sf -o hwmcc17-single.tar.xz https://fmv.jku.at/hwmcc17/hwmcc17-single-benchmarks.tar.xz
    echo "$HWMCC17_SHA256  hwmcc17-single.tar.xz" | sha256sum --quiet -c - || {
        echo "ERROR: hwmcc17-single.tar.xz does not match its pinned sha256" >&2
        rm -f hwmcc17-single.tar.xz; exit 1; }
    mkdir -p hwmcc17 && tar -xf hwmcc17-single.tar.xz -C hwmcc17
    # the tarball extracts flat *.aig; normalize if it created a subdir
    find hwmcc17 -mindepth 2 -name '*.aig' -exec mv {} hwmcc17/ \; 2>/dev/null || true
fi

fi

if want_set itc99; then
echo "== ITC'99 big instances (I99T, EUPL-1.2) -> $CACHE/"
for b in b17 b18 b19 b20 b21 b22; do
    [ -f "$b.bench" ] || curl -sf -o "$b.bench" \
        "https://raw.githubusercontent.com/cad-polito-it/I99T/$I99T_COMMIT/i99t/$b/$b.bench"
done

# ...and install them beside the committed circuits, the one directory every driver reads: both
# campaign drivers build their instance list from `benchmarks/circuits/bench/seq/*.bench`, so a
# big instance sitting only in $CACHE is silently absent — no error, no skip line, just a shorter
# sweep. They stay out of git, so this is an install step and `git status` is unchanged.
SEQ_DIR="$REPO/benchmarks/circuits/bench/seq"
if [ -d "$SEQ_DIR" ]; then
    n=0
    for b in b17 b18 b19 b20 b21 b22; do
        if [ -f "$b.bench" ] && [ ! -f "$SEQ_DIR/$b.bench" ]; then
            cp "$b.bench" "$SEQ_DIR/$b.bench" && n=$((n + 1))
        fi
    done
    echo "   installed $n big ITC'99 bench(es) into benchmarks/circuits/bench/seq/ "\
         "(gitignored; $(ls "$SEQ_DIR"/*.bench 2>/dev/null | wc -l) sequential circuits now visible)"
fi

fi

echo "done. Sets under $CACHE:"
du -sh "$CACHE"/* 2>/dev/null

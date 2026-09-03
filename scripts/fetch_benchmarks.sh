#!/usr/bin/env bash
# Fetch the FULL upstream circuit-benchmark sets that back the curated subset committed under
# benchmarks/circuits/ (see benchmarks/circuits/README.md for provenance/licenses).
# Everything lands in <repo>/.gpap/benchmarks: durable across reboots, and repo-local so a
# quota'd home directory does not silently fill up. Override with $GPAP_HOME.
#
#   scripts/fetch_benchmarks.sh              # every set
#   scripts/fetch_benchmarks.sh itc99        # just the big ITC'99 circuits (~25 MB)
#
# The set filter exists so a driver that needs only b17-b22 does not also pull 93 MB of HWMCC:
# eval/colab/colab_rivals_l4.sh calls it on a fresh VM, and a rival sweep touches no AIGER set.
set -euo pipefail

WANT="${1:-all}"
want_set() { [ "$WANT" = "all" ] || [ "$WANT" = "$1" ]; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${GPAP_HOME:-$REPO/.gpap}/benchmarks"
mkdir -p "$CACHE"
cd "$CACHE"

if want_set epfl; then
echo "== EPFL combinational suite (lsils/benchmarks, AIG only) -> $CACHE/epfl"
mkdir -p epfl
for f in adder bar div hyp log2 max multiplier sin sqrt square; do
    [ -f "epfl/$f.aig" ] || curl -sf -o "epfl/$f.aig" \
        "https://raw.githubusercontent.com/lsils/benchmarks/master/arithmetic/$f.aig"
done
for f in arbiter cavlc ctrl dec i2c int2float mem_ctrl priority router voter; do
    [ -f "epfl/$f.aig" ] || curl -sf -o "epfl/$f.aig" \
        "https://raw.githubusercontent.com/lsils/benchmarks/master/random_control/$f.aig"
done

fi

if want_set hwmcc; then
echo "== HWMCC'17 single-property AIGs (300 instances, ~27 MB) -> $CACHE/hwmcc17"
if [ ! -d hwmcc17 ] || [ "$(find hwmcc17 -maxdepth 1 -type f -name '*.aig' | wc -l)" -lt 300 ]; then
    curl -sf -o hwmcc17-single.tar.xz https://fmv.jku.at/hwmcc17/hwmcc17-single-benchmarks.tar.xz
    mkdir -p hwmcc17 && tar -xf hwmcc17-single.tar.xz -C hwmcc17
    # the tarball extracts flat *.aig; normalize if it created a subdir
    find hwmcc17 -mindepth 2 -name '*.aig' -exec mv {} hwmcc17/ \; 2>/dev/null || true
fi

fi

if want_set itc99; then
echo "== ITC'99 big instances (I99T, EUPL-1.2) -> $CACHE/"
for b in b17 b18 b19 b20 b21 b22; do
    [ -f "$b.bench" ] || curl -sf -o "$b.bench" \
        "https://raw.githubusercontent.com/cad-polito-it/I99T/master/i99t/$b/$b.bench"
done

# ...and INSTALL them beside the committed circuits, because that is the ONE directory every
# driver reads. `eval/run_full_eval.sh` and `eval/colab/colab_rivals_l4.sh` both build their
# instance list from `benchmarks/circuits/bench/seq/*.bench` — the colab driver derives the whole
# sequential family from it — so a big instance sitting only in $CACHE is SILENTLY ABSENT: no
# error, no skip line, just a sweep six instances shorter than the one the comments describe.
# That is what happened to b17-b22 in the 2026-09-02 campaign; they ran only because they had
# been copied over by hand. They stay out of git (.gitignore), so this is an install step, not a
# commit: `git status` is unchanged after running it.
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

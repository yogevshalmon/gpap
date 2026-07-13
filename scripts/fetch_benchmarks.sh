#!/usr/bin/env bash
# Fetch the FULL upstream circuit-benchmark sets that back the curated subset committed under
# benchmarks/circuits/ (see benchmarks/circuits/README.md for provenance/licenses).
# Everything lands in ~/.cache/gpap/benchmarks (durable — /tmp is reaped on WSL2 boxes).
set -euo pipefail

CACHE="${GPAP_BENCH_HOME:-$HOME/.cache/gpap/benchmarks}"
mkdir -p "$CACHE"
cd "$CACHE"

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

echo "== HWMCC'17 single-property AIGs (300 instances, ~27 MB) -> $CACHE/hwmcc17"
if [ ! -d hwmcc17 ] || [ "$(find hwmcc17 -maxdepth 1 -type f -name '*.aig' | wc -l)" -lt 300 ]; then
    curl -sf -o hwmcc17-single.tar.xz https://fmv.jku.at/hwmcc17/hwmcc17-single-benchmarks.tar.xz
    mkdir -p hwmcc17 && tar -xf hwmcc17-single.tar.xz -C hwmcc17
    # the tarball extracts flat *.aig; normalize if it created a subdir
    find hwmcc17 -mindepth 2 -name '*.aig' -exec mv {} hwmcc17/ \; 2>/dev/null || true
fi

echo "== ITC'99 big instances (I99T, EUPL-1.2) -> $CACHE/"
for b in b17 b18 b19 b20 b21 b22; do
    [ -f "$b.bench" ] || curl -sf -o "$b.bench" \
        "https://raw.githubusercontent.com/cad-polito-it/I99T/master/i99t/$b/$b.bench"
done

echo "done. Sets under $CACHE:"
du -sh "$CACHE"/* 2>/dev/null

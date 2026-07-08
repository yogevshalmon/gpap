#!/usr/bin/env python3
"""Thin CLI over gpap.circuits.bench: unroll .bench circuits into projected-sampling DIMACS CNFs.

Usage:
  # single instance
  python scripts/bench_to_cnf.py --bench s386.bench --cycles 25 --out s386-c25.cnf --seed 1
  # whole folder of .bench files -> folder of .cnf files (named <stem>-c<cycles>.cnf)
  python scripts/bench_to_cnf.py --bench benches/ --cycles 25 --out cnf_out/ --seed 1
"""
import sys

sys.path.insert(0, ".")
from gpap.circuits.bench import main

if __name__ == "__main__":
    main()

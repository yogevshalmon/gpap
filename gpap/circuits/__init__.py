"""Circuit-to-CNF tooling: parse ISCAS-85/89 + ITC'99 `.bench`, unroll T cycles, Tseitin-encode,
pin outputs to a simulated reachable pattern, emit projected-sampling DIMACS (`c ind` support).
"""
from gpap.circuits.bench import build_cnf, parse_bench, simulate, write_dimacs, convert_one

__all__ = ["parse_bench", "build_cnf", "simulate", "write_dimacs", "convert_one"]

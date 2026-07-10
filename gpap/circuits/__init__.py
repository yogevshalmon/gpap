"""Circuit-to-CNF tooling: parse `.bench` (ISCAS-85/89, ITC'99) and AIGER `.aig`/`.aag` circuits,
unroll T cycles, Tseitin-encode, pin outputs to a simulated reachable pattern, emit
projected-sampling DIMACS (`c ind` support + `c wire` labels + `c encoder` provenance).

Two encoders, same projected solution space (`convert_one` / `convert_aig_one`, `encoder=...`):
the native gate-by-gate Tseitin, and an optional berkeley-abc pass (unroll + `dc2`/`fraig`, then
Tseitin its AIG) that shrinks the CNF 2-8x. See gpap/circuits/abc_backend.py + scripts/build_abc.sh.
"""
from gpap.circuits.abc_backend import abc_available, build_cnf_abc, build_cnf_abc_aig, find_abc
from gpap.circuits.aig2cnf import build_cnf_aig, convert_aig_one, simulate_aig
from gpap.circuits.aiger import Aig, read_aiger
from gpap.circuits.bench import (
    build_cnf, convert_one, parse_bench, parse_encoder_comment, parse_support_labels, simulate,
    write_dimacs,
)

__all__ = ["parse_bench", "build_cnf", "simulate", "write_dimacs", "convert_one",
           "parse_support_labels", "parse_encoder_comment",
           "Aig", "read_aiger", "build_cnf_aig", "convert_aig_one", "simulate_aig",
           "abc_available", "build_cnf_abc", "build_cnf_abc_aig", "find_abc"]

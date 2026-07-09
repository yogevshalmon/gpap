"""GPAP's flagship application: GPU-parallel SAT solution sampling (CircuitSAT + projected).

`GPUSampler` runs K randomized BCP-completions per chunk (CMSGen's algorithm, massively
parallel); every emitted model is guaranteed valid. `pipeline` is the full end-to-end flow
(parse → simplify → warmup → entailed-clause mining → contract-timed run) that the CLI,
the benchmark script and the Colab driver all share.
"""
from gpap.sampling.sampler import GPUSampler, parse_independent_support
from gpap.sampling.output import write_samples

__all__ = ["GPUSampler", "parse_independent_support", "write_samples"]

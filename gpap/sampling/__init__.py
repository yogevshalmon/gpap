"""GPAP's flagship application: GPU-parallel SAT solution sampling (CircuitSAT + projected).

`GPUSampler` runs K randomized BCP-completions per chunk; every emitted model is guaranteed
valid, because it is read off a completed propagation rather than repaired into one. `pipeline`
is the end-to-end flow the CLI, the benchmark driver and the Colab driver all share:

    parse -> simplify -> warmup -> entailed-clause mining (gpap.mining) -> contract-timed run

Mining is a separate application, not a sampling detail — the pipeline just happens to be a
consumer of it.
"""
from gpap.sampling.sampler import GPUSampler, parse_independent_support
from gpap.sampling.output import write_samples

__all__ = ["GPUSampler", "parse_independent_support", "write_samples"]

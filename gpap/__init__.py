"""GPAP — GPU-parallel BCP (Boolean Constraint Propagation) and CircuitSAT sampling.

The reusable primitive is `gpap.engine` (K-parallel sum-trick BCP over independent assumption
sets, with fused Triton trajectory kernels); the flagship application is `gpap.sampling`
(guaranteed-valid SAT solution sampling, GPU-parallel CMSGen). `gpap.circuits` converts
sequential/combinational `.bench` circuits into unrolled projected-sampling CNFs.
"""
import os

# CUDA expandable-segments allocation must be configured BEFORE torch initializes its CUDA
# allocator. The sampler's hot loop churns variable-sized temporaries every round, which
# fragments the default caching allocator badly (measured: peak reserved 6.5 GB vs 2.5 GB live,
# growing ~K^2.3); one growable virtual segment makes reserved track allocated and linear in K
# (~2.3x more usable K for the same VRAM). setdefault so an explicit user setting wins.
# Importing gpap before any CUDA use guarantees the ordering.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"


def sample(path, n=50_000, *, device=None, seed=5, **kw):
    """Sample `n` guaranteed-valid models from a DIMACS CNF file. Returns (models, info).

    Thin wrapper over gpap.sampling.pipeline.sample_file (the full e2e flow: parse, simplify,
    warmup, entailed-clause mining, contract-timed run). `info` is the pipeline's rich dict
    (rates, phases, mining stats, soundness); models come from the quality/coverage pass, so
    pass n_cov=<count> to collect them (default collects none — throughput only)."""
    from gpap.sampling.pipeline import sample_file
    info = sample_file(path, _resolve_device(device), n_time=n, seed=seed, **kw)
    return info.pop("models"), info


def sample_clauses(num_vars, clauses, n=50_000, *, support=None, device=None, seed=5, **kw):
    """Like sample(), for an in-memory clause list. `support` = 0-based projection vars."""
    from gpap.sampling.pipeline import sample_clauses as _sc
    info = _sc(num_vars, clauses, support, _resolve_device(device), n_time=n, seed=seed, **kw)
    return info.pop("models"), info


def sample_circuit(bench_path, cycles=25, n=50_000, *, pin_outputs=None, seed=5,
                   device=None, **kw):
    """Unroll a .bench circuit `cycles` times (Tseitin, outputs pinned to a simulated reachable
    pattern) and sample the primary-input support. Returns (models, info)."""
    from gpap.circuits.bench import parse_bench, build_cnf
    circ = parse_bench(bench_path)
    b, support, _target, _labels = build_cnf(circ, cycles, seed, pin_outputs=pin_outputs)
    return sample_clauses(b.nv, b.clauses, n, support=[v - 1 for v in support],
                          device=device, seed=seed, **kw)


def _resolve_device(device):
    import torch
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

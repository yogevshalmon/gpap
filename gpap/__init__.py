"""GPAP — GPU-parallel (partial) assignment propagation and CircuitSAT sampling.

The reusable primitive is `gpap.engine`: K-parallel sum-trick BCP over independent partial
assignments, with fused trajectory kernels. The flagship application is `gpap.sampling`,
guaranteed-valid SAT solution sampling. `gpap.circuits` converts sequential and combinational
circuits into unrolled projected-sampling CNFs.
"""
import os

# Expandable-segments allocation must be configured before torch initializes its CUDA
# allocator: the sampler's hot loop churns variable-sized temporaries, which fragments the
# default caching allocator badly and costs usable K. setdefault so an explicit user setting
# wins; importing gpap before any CUDA use guarantees the ordering.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
del os                     # not part of the package namespace (`__all__` is the contract)

__version__ = "0.3.1"

# THE tier-0 contract (docs/PYTHON_API.md: `__all__` is the supported API).
__all__ = ["sample", "sample_clauses", "sample_circuit", "MAX_SAMPLES", "__version__"]


# ---------------------------------------------------------------------------- the top-level API
# The contract of these three: ask for n samples, get n samples. Nothing here names a batch size,
# a portfolio or a memory model — the pipeline sizes those to the GPU and the formula. `samples`
# is a numpy (rows, S) uint8 matrix, one column per support variable (`info["cols"]` names them),
# projected on the GPU so only those bits cross the bus. full_models=True returns full DIMACS
# models instead.
#: Default ceiling on rows returned, on both return shapes. Peak host RAM of the
#: `full_models=True` path is bounded separately by the coverage run's own batch and chunk caps,
#: since a chunk is never abandoned half-written.
MAX_SAMPLES = 100_000


def sample(path, n=50_000, *, device=None, seed=5, max_samples=MAX_SAMPLES,
           full_models=False, **kw):
    """Sample `n` guaranteed-valid models from a DIMACS CNF file. Returns (samples, info).

    Thin wrapper over gpap.sampling.pipeline.sample_file (parse, simplify, warmup, mining, timed
    run); `info` is that pipeline's dict of rates, phases, mining stats and soundness. `samples`
    is a numpy (rows, S) uint8 matrix of support projections, at most min(n, `max_samples`) rows,
    with `info["cols"]` naming each column's DIMACS variable. Collection is off the timing path,
    so `info["samples_per_s"]` is unaffected. `full_models=True` returns the coverage pass's full
    DIMACS models instead: more informative, considerably more expensive, same row bound."""
    from gpap.sampling.pipeline import sample_file
    return _samples(sample_file, (path, _resolve_device(device)), n, seed, max_samples,
                    full_models, kw)


def sample_clauses(num_vars, clauses, n=50_000, *, support=None, device=None, seed=5,
                   max_samples=MAX_SAMPLES, full_models=False, **kw):
    """Like sample(), for an in-memory clause list. `support` = 0-based projection vars."""
    if support is not None:
        bad = [v for v in support if not 0 <= int(v) < num_vars]
        if bad:
            raise ValueError(
                f"support variables are 0-BASED indices in [0, {num_vars - 1}] "
                f"(DIMACS var v is index v-1); out of range: {bad[:5]}")
    from gpap.sampling.pipeline import sample_clauses as _sc
    return _samples(_sc, (num_vars, clauses, support, _resolve_device(device)), n, seed,
                    max_samples, full_models, kw)


def sample_circuit(circuit_path, cycles=25, n=50_000, *, pin_outputs=None, seed=5,
                   device=None, **kw):
    """Unroll a circuit (`.bench`, or AIGER `.aig`/`.aag`) `cycles` times (Tseitin, outputs
    pinned to a simulated reachable pattern) and sample the primary-input support. Returns
    (samples, info); the columns are the circuit's primary inputs per cycle, named by
    `info["wire_map"]`."""
    if str(circuit_path).endswith((".aig", ".aag")):
        from gpap.circuits.aig2cnf import build_cnf_aig
        from gpap.circuits.aiger import read_aiger
        b, support, _target, labels = build_cnf_aig(read_aiger(circuit_path), cycles, seed,
                                                    pin_outputs=pin_outputs)
    else:
        from gpap.circuits.bench import parse_bench, build_cnf
        circ = parse_bench(circuit_path)
        b, support, _target, labels = build_cnf(circ, cycles, seed, pin_outputs=pin_outputs)
    # Name the columns here rather than making the caller re-derive which CNF variable is which
    # input at which cycle: `sample_file` recovers the same map from the CNF's `c wire`
    # comments, and this path never writes a file.
    kw.setdefault("_wire_map", dict(zip(support, labels)))
    return sample_clauses(b.nv, b.clauses, n, support=[v - 1 for v in support],
                          device=device, seed=seed, **kw)


def _samples(fn, args, n, seed, max_samples, full_models, kw):
    """Run a pipeline entry point and hand back (samples, info).

    `full_models` picks which pass collects: `n_cov` reconstructs full DIMACS models on the host,
    `out_cap` projects support bits on the GPU. An explicitly passed value always wins, which is
    how the benchmark driver asks for a small coverage sample beside a large timed run."""
    key = "n_cov" if full_models else "out_cap"
    kw.setdefault(key, max(0, min(int(n), int(max_samples))))
    info = fn(*args, n_time=n, seed=seed, **kw)
    if full_models:
        # Truncate to what was asked for: a chunk is the smallest unit of work, so the coverage
        # run overshoots its target by up to one batch. One ceiling for both shapes, and an
        # explicitly passed `n_cov` still wins.
        return info.pop("models")[:int(kw[key])], info
    info.pop("models", None)
    s, cols = info.get("samples"), info.get("cols")
    if s is None:      # out_cap=0 explicitly, or a path that collected nothing
        import numpy as np
        info["samples"] = s = np.zeros((0, len(cols) if cols else 0), dtype=np.uint8)
    return s, info


def _resolve_device(device):
    import torch
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

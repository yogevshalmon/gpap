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
del os                     # not part of the package namespace (`__all__` is the contract)

__version__ = "0.1.0"

# THE tier-0 contract (docs/PYTHON_API.md: `__all__` is the supported API).
__all__ = ["sample", "sample_clauses", "sample_circuit", "MAX_SAMPLES", "__version__"]


# ---------------------------------------------------------------------------- the top-level API
# THE contract of these three: you ask for n samples, you get n samples. Nothing here mentions a
# batch size, a portfolio or a memory model — the pipeline sizes all of it to the GPU and the
# formula. `samples` is a numpy (rows, S) uint8 bit matrix, one column per SUPPORT variable
# (`info["cols"]` names them), which is both the cheap artifact and the one people want: it is
# GPU-projected, so only those bits cross the bus, ~10x cheaper than rebuilding full N-variable
# assignments on the host. Pass full_models=True for the old list[list[DIMACS literal]] form.
MAX_SAMPLES = 100_000    # default ceiling on rows materialized to host RAM


def sample(path, n=50_000, *, device=None, seed=5, max_samples=MAX_SAMPLES,
           full_models=False, **kw):
    """Sample `n` guaranteed-valid models from a DIMACS CNF file. Returns (samples, info).

    Thin wrapper over gpap.sampling.pipeline.sample_file (the full e2e flow: parse, simplify,
    warmup, entailed-clause mining, contract-timed run). `info` is the pipeline's rich dict
    (rates, phases, mining stats, soundness).

    `samples` is a numpy (rows, S) uint8 matrix of support projections, rows <= min(n,
    `max_samples`); `info["cols"]` gives each column's original DIMACS variable. Collecting them
    happens OFF the timing path, so `info["samples_per_s"]` is unaffected. `full_models=True`
    switches to the coverage pass's full DIMACS models instead (list[list[int]], the historical
    return) — more informative, considerably more expensive."""
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
    # The labels exist right here, so name the columns rather than making the caller re-derive
    # which CNF variable is which input at which cycle. `sample_file` recovers the same map by
    # re-parsing the CNF's `c wire` comments; this path never writes a file.
    kw.setdefault("_wire_map", dict(zip(support, labels)))
    return sample_clauses(b.nv, b.clauses, n, support=[v - 1 for v in support],
                          device=device, seed=seed, **kw)


def _samples(fn, args, n, seed, max_samples, full_models, kw):
    """Run a pipeline entry point and hand back (samples, info).

    `full_models` picks WHICH pass collects: `n_cov` reconstructs full DIMACS models on the host,
    `out_cap` projects support bits on the GPU. A caller may still set either explicitly — an
    explicit value always wins, since that is how the benchmark driver asks for a small coverage
    sample alongside a large timed run."""
    key = "n_cov" if full_models else "out_cap"
    kw.setdefault(key, max(0, min(int(n), int(max_samples))))
    info = fn(*args, n_time=n, seed=seed, **kw)
    if full_models:
        return info.pop("models"), info
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

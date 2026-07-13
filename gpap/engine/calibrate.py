"""GPU memory budget utilities: free-memory query and K (batch size) computation."""
import torch

# Estimate of average literals per clause, used for formula_bytes sizing.
AVG_LITS_PER_CLAUSE = 4.0


def formula_mem_bytes(M: int, N: int) -> int:
    """One-time GPU bytes for CnfFormula structures (occ_list, init_lit_sum, etc.)."""
    total_lits = int(AVG_LITS_PER_CLAUSE * M)
    return 8 * M + 16 * N + 8 * total_lits


def per_batch_mem_bytes(M: int, N: int, max_clause_size: int = 127,
                        total_lits: int | None = None) -> int:
    """GPU bytes per batch slot — the CONSERVATIVE analytic K bound.

    HISTORY: this model was derived and empirically calibrated (goldb/sat24) on the retired
    PyTorch propagate path, whose scatter temporaries dominated the peak. The packed-kernel
    engine's true footprint is far smaller (~4*M packed word + 2*N assignment + ~4*N trail,
    no per-round temporaries), so this bound now over-reserves — DELIBERATELY kept unchanged:
    every produced K was validated spill-safe under the nvidia-smi kill-switch with THESE
    numbers, and any tightening must be re-validated the same way (this box spills to host
    RAM instead of OOMing, so an optimistic model silently halves throughput). The fused-K
    boost that recovers the headroom lives in GPUSampler.__init__ (K_fused), already
    kill-switch-validated.

    Model (historical derivation):
      static           (sat_b + 4)*M + 2*N
      apply peak       8*N   (four K×2N bool temporaries)
      propagate peak   1000*total_lits//N   (scatter temporaries, empirical 570 B/unit + margin)
      sat_b: 1 if max_clause_size ≤ 127, 2 if ≤ 32767, else 4.

    Formula: (sat_b + 4)*M + 2*N + max(8*N, 1000*total_lits//N) + 3
    """
    if total_lits is None:
        total_lits = int(AVG_LITS_PER_CLAUSE * M)
    if max_clause_size <= 127:
        sat_b = 1
    elif max_clause_size <= 32767:
        sat_b = 2
    else:
        sat_b = 4
    static     = (sat_b + 4) * M + 2 * N
    apply_peak = 8 * N
    prop_peak  = 1000 * total_lits // max(N, 1)
    return max(1, static + max(apply_peak, prop_peak) + 3)


_WDDM_RESERVE_BYTES = int(0.8e9)  # 0.8 GB reserved for WDDM/WSL2 driver overhead


def gpu_mem_available(device: torch.device, fraction: float = 0.75) -> int:
    """Return bytes available for new tensor allocations on `device`.

    Uses torch.cuda.mem_get_info() which reports actual free CUDA memory from
    the driver — already accounting for all previously allocated tensors
    (formula data, PyTorch caching allocator, etc.).

    fraction=0.75: leaves 25% of free VRAM as headroom.  _WDDM_RESERVE_BYTES (0.8 GB)
    is subtracted first to account for WDDM/WSL2 display driver overhead that isn't
    visible to CUDA's mem_get_info but still reduces usable VRAM.
    """
    if device.type != "cuda":
        return int(8e9)  # conservative fallback for CPU / MPS
    free, _ = torch.cuda.mem_get_info(device)
    return int(max(0, free - _WDDM_RESERVE_BYTES) * fraction)


def compute_k_safe(
    M: int,
    N: int,
    gpu_mem_bytes: float,
    cap: int | None = 65536,
    max_clause_size: int = 127,
    total_lits: int | None = None,
    deduct_formula: bool = True,
) -> int:
    """Largest K (batch size) fitting in `gpu_mem_bytes` for a formula with M clauses, N vars.

    Memory model (see per_batch_mem_bytes for full derivation):
      formula_bytes   = 8*M + 16*N + 8*total_lits_est       (one-time, estimated)
      per_batch_bytes = (sat_b+4)*M + 2*N                   (static)
                      + max(8*N, 1000*total_lits//N) + 3    (peak: apply or propagate)

    max_clause_size: pass formula.clause_sizes.max().item() for accurate sat_b selection.
    total_lits: pass formula.clause_sizes.sum().item() for accurate prop-peak estimation.
                Default: AVG_LITS_PER_CLAUSE * M (may underestimate for dense formulas).
    cap: upper bound on K (default 65536 for benchmarks; pass None for clause
         learning where larger K = fewer chunks = faster).
    deduct_formula: if True (default), subtracts estimated formula bytes from gpu_mem_bytes.
                    Set False when gpu_mem_bytes comes from gpu_mem_available() called AFTER
                    the formula is already loaded — mem_get_info().free already excludes it.

    Returns max(1, min(K_safe, cap)).
    """
    f_bytes = formula_mem_bytes(M, N)
    p_bytes = per_batch_mem_bytes(M, N, max_clause_size, total_lits)
    available = gpu_mem_bytes - f_bytes if deduct_formula else gpu_mem_bytes
    if available <= 0:
        return 1
    k_safe = int(available / p_bytes)
    return max(1, k_safe if cap is None else min(k_safe, cap))

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
    """GPU bytes per batch slot in BCPEngine, including peak intermediate allocations.

    Three phases contribute to peak memory — we take the max across sequential phases:

    1. Static tensors (always alive):
         sat_count (K,M): sat_b B/element  (auto-selected from max_clause_size)
         lit_sum   (K,M): int32 → 4 B/element
         assignment(K,2N): int8 → 1 B/element
         Subtotal: (sat_b + 4)*M + 2*N

    2. apply_assumptions_bulk peak: +8N per batch
         4 simultaneous K×2N bool tensors (assum_mat, assum_neg, bool view, &-result).

    3. _bulk_propagate_lits peak (DOMINANT for dense/small-N formulas): +1000*total_lits//N
         7 int64 + 2 int32 temporary tensors, all of size `total` per propagation round.
         total ≈ K × P × avg_occ_length, P ≈ 10 propagations/batch/round,
         avg_occ_length = total_lits / N.
         Empirical calibration (goldb, peak_prop measured): 570 bytes per avg_occ_length
         unit; coefficient 1000 adds ~75% safety margin for dense/high-M/N formulas.

    BCPEngine auto-selects sat_count dtype:
      max_clause_size ≤ 127  → int8  (sat_b = 1)
      max_clause_size ≤ 32767 → int16 (sat_b = 2)
      else                   → int32 (sat_b = 4)

    Dominant phase:
      apply-peak (8N) dominates when N² > 90*total_lits (sparse/large-N formulas, e.g. goldb)
      prop-peak  dominates when N² < 90*total_lits (dense/small-N formulas, e.g. sat24)

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

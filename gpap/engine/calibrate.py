"""How big a batch, and which propagate kernel — the engine's sizing policy, in ONE place.

Nothing above this module should re-derive "how many rows fit on this GPU". Every consumer
(`BCPEngine`, `AssumptionChecker`, `GPUSampler`, and the C++ `Sampler` through the mirrored
`cpp/src/calibrate.cpp`) asks here, so a new application gets the same answer without copying
the reasoning — and so a correction lands once.

Three questions, three answers:

  * `compute_k_safe` — how many rows of engine state fit in a memory budget (the conservative,
    kill-switch-validated analytic model; see per_batch_mem_bytes for why it stays conservative).
  * `default_k_cap` / `resolve_k_cap` — the ceiling, sized to the card so a bigger GPU is
    allowed to use it. `"auto"` is what every entry point defaults to.
  * `dispatch_mode` — scalar or wave propagate for a given batch, from the device's occupancy.

A caller who does not care about any of it should never have to name a number: that is the
contract these three exist to keep.
"""
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


# ------------------------------------------------------------------ K cap (VRAM-proportional)
# The benchmark drivers used to hardcode a 49,152 cap, which silently pinned a 48 GB L40S or an
# 80 GB A100 to the 12 GB laptop's batch size. Throughput is ~linear in K, so a bigger GPU
# reproduced the laptop's numbers and the "does this scale with the GPU?" question could not be
# answered at all. K_CAP_REF is that historical constant and K_CAP_VRAM_REF the TOTAL VRAM of the
# box it was validated on (RTX 4080 Laptop, 11.994 GiB). Scaling one by the other leaves the
# reference GPU at EXACTLY 49,152 — and max() floors every smaller card there too, so no existing
# result moves — while a larger card may use the VRAM it has (L40S 48 GB ~4x, A100 80 GB ~6.7x).
#
# TOTAL memory, deliberately, NOT gpu_mem_available()'s live free budget: free memory moves with
# whatever else touches the GPU, so a free-based cap made K jitter run-to-run on an idle box
# (measured 49,526 vs 49,671 across two runs in the same session) and the sweep stopped being
# reproducible. Total VRAM is a fixed property of the card.
#
# This is a CAP, not a K. The actual K is min(compute_k_safe(...), cap) against the LIVE free
# budget, so raising the cap can only ever expose the analytic memory model's own bound; it never
# authorises an allocation that model says does not fit. Per CLAUDE.md's ground rules a K change
# still wants nvidia-smi kill-switch validation on any box where this cap stops binding.
K_CAP_REF = 49_152
K_CAP_VRAM_REF = 12_878_086_144   # RTX 4080 Laptop, torch.cuda.get_device_properties().total_memory


def default_k_cap(device: torch.device) -> int:
    """K cap sized to this GPU: K_CAP_REF scaled by its total VRAM vs the reference box."""
    if device.type != "cuda":
        return K_CAP_REF
    total = torch.cuda.get_device_properties(device).total_memory
    return max(K_CAP_REF, int(K_CAP_REF * total / K_CAP_VRAM_REF))


def resolve_k_cap(k_cap: int | str | None, device: torch.device) -> int | None:
    """Resolve the "auto" sentinel to `default_k_cap(device)`; pass ints and None through.

    Deliberately separate from default_k_cap so GPUSampler's own `k_cap` contract is untouched:
    there, None still means "no cap" (the analytic bound alone). "auto" is an app-level default,
    resolved once at the pipeline/CLI boundary where the device is known.
    """
    return default_k_cap(device) if k_cap == "auto" else k_cap


# ------------------------------------------------------------------ propagate-kernel dispatch
# Below memory saturation each row's serial trail leaves requests-in-flight on the table, and the
# warp-wide WAVE propagate multiplies them. The crossover was RETUNED 2026-07-14 after the
# assignment-BITMAP change removed wave's WxW dedup tax: wave now wins well past saturation
# (b03 at K=49k = 17.7x resident warps: wave 1.41x, was 0.9x pre-bitmap) and scalar only holds the
# huge-K/tiny-M extreme (b06 at K=129k = 46x resident: scalar 1.13x, interleaved A/Bs) — so the
# boundary sits mid-gap at ~30x resident warps. Derived from the DEVICE, so it transfers to bigger
# GPUs rather than encoding this laptop.
#
# Both variants share the same state and reach the same fixpoints (BCP is confluent), so this is a
# pure launch-flag choice that can be made per chunk and never changes an answer — it is
# gate-validated bit-identical. `GPAP_KERNEL=scalar|wave` forces one.
WAVE_RESIDENT_MULTIPLE = 30


def resident_warps(device: torch.device) -> int:
    """Warps this GPU can keep resident at once — SMs x (max threads per SM / 32)."""
    if device.type != "cuda":
        return 0
    p = torch.cuda.get_device_properties(device)
    return p.multi_processor_count * (p.max_threads_per_multi_processor // 32)


def dispatch_mode(batch: int, device: torch.device) -> str:
    """"wave" or "scalar" for a chunk of `batch` rows — the measured occupancy rule above."""
    resident = resident_warps(device)
    return "wave" if resident and batch < WAVE_RESIDENT_MULTIPLE * resident else "scalar"


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

/* Batch sizing: how many rows of engine state fit in a memory budget, and how wide a batch the
 * card should run. THE one implementation — Python calls these through the ABI rather than
 * mirroring the arithmetic (gpap/engine/calibrate.py), because two copies of a memory model
 * drift silently and the symptom is a spill, not an error.
 *
 * The model is the ALLOCATION LIST of gpap_engine's per-chunk buffers, not an estimate. Verified
 * against measured device memory: a least-squares fit over synthetic formulas returns
 * 4.00*M + 4.25*N + 285 bytes/row, which is exactly what ensure_state() reserves, and the
 * prediction matches cuMemGetInfo deltas to 1.00x on real instances.
 *
 * HISTORY, because the previous model was wrong in both directions at once and nobody could see
 * it: it charged (sat_b+4)*M for a separate sat_count array that the packed word replaced, 2*N
 * for an assignment that became a bitmap (8x smaller), and max(8*N, 1000*total_lits/N) for
 * scatter temporaries belonging to a PyTorch propagate path deleted in 2026-07 — while omitting
 * the trail, which is the second-largest real term. Net 1.57x over on real instances and 17x on
 * small ones, so K was ~a third of what fits. Do not reintroduce an "empirical margin" here: if
 * the engine allocates something, add it to the list; if it does not, do not charge for it.
 */
#include <cuda.h>

#include <algorithm>
#include <cstdint>

#include "gpap/gpap.h"

namespace {
constexpr int64_t kWddmReserveBytes = 800000000ll;   /* WDDM/WSL2 driver overhead */

/* The VRAM-proportional batch cap (mirrored by calibrate.py's K_CAP_REF / K_CAP_VRAM_REF).
 * kKCapRef is the historical hardcoded cap and kKCapVramRef the TOTAL VRAM of the box it was
 * validated on (RTX 4080 Laptop), so the reference card gets back exactly 49,152 and every
 * smaller card is floored there — no existing measurement moves — while a larger card may use
 * the memory it has.
 *
 * TOTAL VRAM, never free: a free-memory cap made K jitter run-to-run on an idle box (49,526 vs
 * 49,671 in one session) and a sweep whose batch depends on what else is resident is not a
 * measurement. And it is a CAP, not a K — the analytic bound below still decides what fits. */
constexpr int32_t kKCapRef = 49152;
constexpr int64_t kKCapVramRef = 12878086144ll;

/* Floor on the batch cap. This is the value that ACTUALLY bounded chunks before the sizing
 * collapse: the sampler capped its base K at kKCapRef but then ran a "fused" batch capped at
 * max(262144, k_cap), and the fused batch is what every chunk used. Both the 4080 (kKCapRef
 * scaled = 49,152) and the L4 (90,300) therefore ran against an effective ceiling of 262,144 —
 * which is why the recorded L4 sweeps show batch 258,316, memory-bound just under it. Keeping
 * the floor here preserves that behaviour with one cap instead of two.
 *
 * MEASURED 2026-08-18 on an L4 (23,034 MiB), because "kept for reproducibility" is not a reason to
 * keep a number: the cap BINDS on 7 of 16 ABC circuit instances there, and where it binds hardest
 * (b03-c25, M=2,087) the memory model would allow 1,505,739 rows while the run peaks at 13.6% of
 * VRAM. So it is not a safety limit — it is a saturation limit, and the saturation is real: at
 * 262,144 rows the card is ~94x oversubscribed against its 2,784 resident warps, and a 4x batch
 * was separately measured at +2.8% (see CLAUDE.md, Colab A100+L4). Raising it would buy ~nothing
 * and cost the comparability of every recorded sweep. It also bounds host-side readout, which is
 * NOT VRAM-proportional. Revisit only with a throughput measurement that shows the cap binding on
 * something that is not already occupancy-bound. */
constexpr int32_t kBatchCapFloor = 262144;

/* Clause count at or above which the warp-wide WAVE propagate beats the scalar one. Measured, see
 * gpap_dispatch_wave below for the whole calibration table and why the variable is M, not batch. */
constexpr int32_t kWaveMinClauses = 2048;
}  // namespace

/* NOTE — there is deliberately NO formula-bytes deduction here, and no gpap_formula_mem_bytes().
 * There used to be one, and it was a trap: gpap_mem_available() reports memory that is free
 * RIGHT NOW, which is already after the formula's occurrence lists and packed init have been
 * uploaded. Subtracting an estimate of them again double-counts and shrinks K for nothing. Every
 * in-tree caller had to remember to pass deduct_formula=False to avoid it, which is not a
 * default anyone should have to defend against. Size the batch against live free memory; the
 * formula is already paid for. */

extern "C" int32_t gpap_trail_cap(int32_t num_vars, int32_t decisions_per_round) {
    /* Mirrors engine.cpp::trail_cap_for — see the proof there for why the assumption width
     * does not enter: every trail push is deduplicated against the assignment bitmap, so a
     * variable can contribute at most one entry. */
    const int32_t floor_b = 64;
    return num_vars + std::max(decisions_per_round, floor_b) + 8;
}

extern "C" int64_t gpap_row_bytes(int32_t M, int32_t N, int32_t trail_cap) {
    if (trail_cap <= 0) trail_cap = gpap_trail_cap(N, 0);
    const int64_t words = (2ll * N + 31) / 32;          /* assignment bitmap, int32 words */
    return 4ll * M                                      /* packed clause word   (K, M)      */
           + 4ll * words                                /* assignment bitmap    (K, ceil2N/32)*/
           + 4ll * trail_cap                            /* trail                (K, cap)    */
           + 2                                          /* active + conflict                */
           + 16                                         /* nprops/nops/ndecs/nrounds        */
           + GPAP_SLOTS;                                /* one selection mask per slot      */
}

extern "C" int64_t gpap_mem_available(int device, double fraction) {
    CUdevice dev;
    CUcontext ctx;
    if (cuInit(0) != CUDA_SUCCESS) return 0;
    if (cuDeviceGet(&dev, device) != CUDA_SUCCESS) return 0;
    if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) return 0;
    cuCtxSetCurrent(ctx);
    size_t free_b = 0, total_b = 0;
    CUresult rc = cuMemGetInfo(&free_b, &total_b);
    cuDevicePrimaryCtxRelease(dev);
    if (rc != CUDA_SUCCESS) return 0;
    const int64_t usable = std::max<int64_t>(0, (int64_t)free_b - kWddmReserveBytes);
    return (int64_t)(usable * fraction);
}

extern "C" int32_t gpap_default_k_cap(int device) {
    CUdevice dev;
    size_t total = 0;
    if (cuInit(0) != CUDA_SUCCESS) return kBatchCapFloor;
    if (cuDeviceGet(&dev, device) != CUDA_SUCCESS) return kBatchCapFloor;
    if (cuDeviceTotalMem(&total, dev) != CUDA_SUCCESS || total == 0) return kBatchCapFloor;
    const int64_t k = (int64_t)((double)kKCapRef * (double)total / (double)kKCapVramRef);
    return (int32_t)std::max<int64_t>(kBatchCapFloor, std::max<int64_t>(kKCapRef, k));
}

extern "C" int32_t gpap_compute_k_safe(int32_t M, int32_t N, int64_t bytes, int32_t cap,
                                       int32_t decisions_per_round) {
    if (bytes <= 0) return 1;
    const int64_t per_row = gpap_row_bytes(M, N, gpap_trail_cap(N, decisions_per_round));
    int64_t k = bytes / std::max<int64_t>(per_row, 1);
    if (cap > 0) k = std::min<int64_t>(k, cap);
    return (int32_t)std::max<int64_t>(1, k);
}

extern "C" int32_t gpap_resident_warps(int device) {
    CUdevice dev;
    if (cuInit(0) != CUDA_SUCCESS) return 0;
    if (cuDeviceGet(&dev, device) != CUDA_SUCCESS) return 0;
    int sms = 0, threads_per_sm = 0;
    cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&threads_per_sm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
    return sms * (threads_per_sm / 32);
}

/* Which propagate kernel should a chunk of `batch` rows of THIS formula run? 1 = WAVE, 0 = SCALAR.
 * ONE definition of the rule: gpap::Engine::plan_for and Python's calibrate.dispatch_mode both
 * call this instead of repeating the arithmetic, for the same reason the memory model lives here.
 * Both variants share the same state and reach the same fixpoints (BCP is confluent), so this is
 * a pure launch-flag choice that can never change an answer -- only how fast it arrives.
 *
 * The rule keys off M, the CLAUSE COUNT, because M is what decides whether a warp-wide propagate
 * has work to spread: WAVE hands ONE row 32 lanes, and a row with few clauses cannot fill them,
 * so SCALAR's one-thread-per-row (far more rows in flight) wins there.
 *
 * HISTORY -- this rule used to read `batch < 30 * resident_warps`, and that was a PROXY for M, not
 * a statement about occupancy. Because K ~ budget/(4*M), a small formula gets a huge batch, so
 * "huge batch" and "small M" were THE SAME ROWS and the proxy held. The 2026-08-18 sizing collapse
 * broke it: with K now 1.4-2.5x larger, mid-size formulas sit AT the batch cap, where `batch` is
 * pinned to a constant and says nothing about M -- so every one of them was classified "scalar".
 * Measured cost of exactly that misclassification (ABC-encoded circuits, fixed wall, interleaved
 * best-of-2, base vs the collapse): 0.727x geomean on the 4 rows whose kernel flipped on the 4080
 * and 0.615x/0.698x on the 2 that flipped on an L4, while the 11 rows that KEPT their kernel came
 * in at 0.993x. Do not reintroduce a batch-only rule.
 *
 * CALIBRATION -- forced-kernel A/B at FIXED K (so only the kernel differs), 25 instances, RTX 4080
 * Laptop + L4, three families (ABC circuit unrolls, sk_ CNFs, tiny circuits). wave/scalar
 * samples-per-second AT THE BATCH CAP, where batch is constant so M is the only variable:
 *      M=32    0.772 | M=483    0.918 | M=1,982  0.960 | M=2,098  1.041 | M=2,215  1.092
 *      M=2,882 1.407 | M=6,260  1.270 | M=14,012 1.572 | ... 1.27-1.66 for every larger M
 * WAVE wins from M ~ 2,000 upward and keeps winning out to M = 107,065; below it SCALAR wins by up
 * to 1.30x. Whole-set geomean wave/scalar was 1.39x (4080) / 1.46x (L4) -- i.e. wherever the old
 * batch-only rule fired on a mid-size formula it was leaving ~1.4x on the table.
 *
 * The batch term SURVIVES only for the regime M cannot speak for: a batch too small to fill even
 * one warp per SM, where SCALAR's extra rows do not exist to be had. Measured on M=483: WAVE is
 * 1.037x at batch 1,024 but 0.918x at the cap. */
extern "C" int32_t gpap_dispatch_wave(int32_t num_clauses, int32_t batch, int device) {
    if (num_clauses >= kWaveMinClauses) return 1;
    /* Only the small-M path needs the device query, so the common case costs nothing. */
    const int32_t resident = gpap_resident_warps(device);
    return (resident > 0 && batch <= resident) ? 1 : 0;
}

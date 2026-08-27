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
 * kKCapRef is the historical hardcoded cap and kKCapVramRef the TOTAL VRAM of the reference
 * card it was validated on, so that card gets back exactly kKCapRef and every
 * smaller card is floored there — no existing measurement moves — while a larger card may use
 * the memory it has.
 *
 * TOTAL VRAM, never free: a free-memory cap made K jitter run-to-run on an idle box (49,526 vs
 * 49,671 in one session) and a sweep whose batch depends on what else is resident is not a
 * measurement. And it is a CAP, not a K — the analytic bound below still decides what fits. */
constexpr int32_t kKCapRef = 49152;
constexpr int64_t kKCapVramRef = 12878086144ll;

/* Floor on the batch cap.
 *
 * This is a SATURATION limit, not a safety limit. Where it binds hardest the memory model would
 * allow orders of magnitude more rows than the run actually uses, so it is not protecting VRAM:
 * at this batch the card is already ~2 orders of magnitude oversubscribed against its resident
 * warps, and a 4x larger batch measured at +2.8%. Raising it would buy ~nothing and cost the
 * comparability of every recorded sweep. It also bounds host-side readout, which is NOT
 * VRAM-proportional.
 *
 * Revisit only with a throughput measurement showing the cap binding on something that is not
 * already occupancy-bound. Numbers: docs/findings/CROSS_GPU_FINDINGS.md. */
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

/* `M` here is the STATE clause count — clauses of length >= 3, `gpap_formula_info.num_long` —
 * because only those have a packed word per row. Binary clauses are implication edges in the
 * shared occ CSR and unit clauses live in the flush cache's bitmap: zero per-row bytes. Feeding
 * total num_clauses over-charges ~4-10x of the first term on circuit CNFs (65-82% of their
 * clauses are binary) and was exactly the kind of phantom charge the HISTORY note above bans. */
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
 * ⚠️ DO NOT reintroduce a batch-only rule. One existed (`batch < 30 * resident_warps`) and was a
 * PROXY for M rather than a statement about occupancy: because K ~ budget/(4*M), a small formula
 * gets a huge batch, so "huge batch" and "small M" were THE SAME ROWS. A later batch-sizing change
 * broke the proxy -- mid-size formulas came to sit AT the batch cap, where `batch` is pinned to a
 * constant and says nothing about M, so every one of them was misclassified "scalar" and lost
 * ~1.3x. The proxy held only by coincidence of the sizing at the time.
 *
 * Calibrated by forced-kernel A/B at FIXED K, so only the kernel differs, across three families
 * and two GPUs: WAVE wins from M ~ 2,000 upward and keeps winning for every larger M; below that
 * SCALAR wins by up to 1.30x. Full per-M table: docs/findings/INSIGHTS.md.
 *
 * The batch term SURVIVES only for the regime M cannot speak for: a batch too small to fill even
 * one warp per SM, where SCALAR's extra rows do not exist to be had. */
extern "C" int32_t gpap_dispatch_wave(int32_t num_clauses, int32_t batch, int device) {
    if (num_clauses >= kWaveMinClauses) return 1;
    /* Only the small-M path needs the device query, so the common case costs nothing. */
    const int32_t resident = gpap_resident_warps(device);
    return (resident > 0 && batch <= resident) ? 1 : 0;
}

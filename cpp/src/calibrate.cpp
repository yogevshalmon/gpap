/* Batch sizing: how many rows of engine state fit a memory budget, and how wide a batch the card
 * should run. The one implementation — Python calls these through the ABI instead of mirroring
 * the arithmetic, because two copies of a memory model drift silently and the symptom is a spill,
 * not an error. The model is the ALLOCATION LIST of gpap_engine's per-chunk buffers, verified
 * against cuMemGetInfo deltas: if the engine allocates something, add it; if not, do not. */
#include <cuda.h>

#include <algorithm>
#include <cstdint>

#include "gpap/gpap.h"

namespace {
/* Absolute slack between what cuMemGetInfo reports free and what we then allocate: `free` is a
 * snapshot of somebody else's bookkeeping, and a paging driver or another process can take memory
 * back underneath us (`fraction` is the proportional margin). Flat on every platform on purpose:
 * making it conditional changes gpap_compute_k_safe's input, i.e. K, unvalidated. */
constexpr int64_t kFreeVramReserveBytes = 800000000ll;

/* The batch cap: one number, the same on every GPU. A SATURATION limit, not a safety one — the
 * analytic bound in gpap_compute_k_safe protects memory, and at this batch a card is already
 * heavily oversubscribed against its resident warps. Device-independent on purpose, or `--k-cap
 * auto` resolves differently per machine. Revisit only with a measurement showing it bind off
 * the occupancy plateau: docs/findings/CROSS_GPU_FINDINGS.md. */
constexpr int32_t kBatchCap = 262144;

/* Clause count at or above which the warp-wide WAVE propagate beats the scalar one; measured,
 * docs/findings/KERNEL_AB_FINDINGS.md. See gpap_dispatch_wave for why the variable is M. */
constexpr int32_t kWaveMinClauses = 2048;

/* Trail headroom past the exact bound — one wavefront; see gpap_trail_cap for why that is the
 * unit. Keep in step with WAVE_W (cpp/tools/build_kernels.py). */
constexpr int32_t kTrailSlack = 32;
}  // namespace

/* There is deliberately no formula-bytes deduction here: gpap_mem_available() reports memory
 * free right now, i.e. after the occ lists and packed init were uploaded — already paid for. */

/* The trail sizing rule — one definition, called by the engine's own ensure_state() as well as
 * by every sizing consumer, so the allocation and the model cannot disagree.
 *
 * BOUND = num_vars, and it is tight. Every push is gated on the assignment bitmap flipping that
 * literal's bit from 0 to 1 (the scalar paths test-then-set within one program, the wave paths
 * take the `atomic_or` return), so exactly one lane can observe the transition for a literal;
 * pushing a literal whose negation is already set is a contradiction and conflicts the row
 * instead; and l and ~l share a bitmap word. So a VARIABLE contributes at most one trail entry
 * per launch, whatever the decision batch B or the assumption width L — which is why neither is
 * a parameter here.
 *
 * SLACK = one WAVEFRONT. The wave propagate and the warp-wide decision scan append a whole
 * wavefront of pushes in one masked store at `tail + prefix_sum`, and unlike the scalar path
 * (`elif tail < TRAIL_CAP`) a vectorised append has no bound test — so a wavefront is the
 * granularity by which a broken dedup invariant would overshoot, and one of headroom keeps that
 * step inside this row. Must not be smaller than WAVE_W in cpp/tools/build_kernels.py. */
extern "C" int32_t gpap_trail_cap(int32_t num_vars) {
    return num_vars + kTrailSlack;
}

/* `num_long` is the STATE clause count — length >= 3 — because only those carry a packed word
 * per row; binary clauses are implication edges in the occ CSR and units live in the flush
 * cache's bitmap, at zero per-row bytes. It is deliberately not called `M`, which means the
 * total clause count everywhere else here. */
extern "C" int64_t gpap_row_bytes(int32_t num_long, int32_t N, int32_t trail_cap) {
    if (trail_cap <= 0) trail_cap = gpap_trail_cap(N);
    const int64_t words = (2ll * N + 31) / 32;          /* assignment bitmap, int32 words */
    return 4ll * num_long                               /* packed clause word (K, num_long) */
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
    const int64_t usable = std::max<int64_t>(0, (int64_t)free_b - kFreeVramReserveBytes);
    return (int64_t)(usable * fraction);
}

extern "C" int32_t gpap_default_k_cap(int device) {
    (void)device;                 /* kept in the signature: the cap is device-independent, and a
                                     caller should not have to care that it became so */
    return kBatchCap;
}

extern "C" int32_t gpap_compute_k_safe(int32_t num_long, int32_t N, int64_t bytes, int32_t cap) {
    if (bytes <= 0) return 1;
    const int64_t per_row = gpap_row_bytes(num_long, N, gpap_trail_cap(N));
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
 * One definition of the rule: gpap::Engine::plan_for and Python's calibrate.dispatch_mode call
 * this instead of repeating the arithmetic. Both variants share the same state and reach the same
 * fixpoints (BCP is confluent), so this is a pure launch-flag choice — only speed changes.
 *
 * The rule keys off M, the CLAUSE COUNT, never the batch alone: M decides whether a warp-wide
 * propagate has work to spread, since WAVE hands one row 32 lanes and a row with few clauses
 * cannot fill them, while SCALAR keeps far more rows in flight. Do not reintroduce a batch-only
 * rule: the old `batch < 30 * resident_warps` was a proxy for M (K ~ budget/4M, so "huge batch"
 * and "small M" were the same rows) that a later sizing change broke, misclassifying every
 * mid-size formula sitting at the batch cap. A/B at fixed K: docs/findings/KERNEL_AB_FINDINGS.md.
 * The batch term survives only for the regime M cannot speak for: a batch too small to fill even
 * one warp per SM, where SCALAR's extra rows do not exist to be had. */
extern "C" int32_t gpap_dispatch_wave(int32_t num_clauses, int32_t batch, int device) {
    if (num_clauses >= kWaveMinClauses) return 1;
    /* Only the small-M path needs the device query, so the common case costs nothing. */
    const int32_t resident = gpap_resident_warps(device);
    return (resident > 0 && batch <= resident) ? 1 : 0;
}

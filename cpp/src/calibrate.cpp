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
/* Absolute slack between what cuMemGetInfo reports free and what we then allocate. `free` is a
 * SNAPSHOT of somebody else's bookkeeping: on a WDDM-paged GPU (Windows, WSL2) the display driver
 * can take memory back underneath us, and on any shared or preemptible host another process can.
 * The `fraction` argument is the proportional margin; this is the absolute one.
 *
 * FLAT ON EVERY PLATFORM, on purpose. It was briefly made conditional on WDDM — the driver it is
 * named for — which is defensible on the naming but is a change to gpap_compute_k_safe's INPUT,
 * i.e. a K change on every native-Linux card and only on the cards this project's development box
 * cannot test. K changes need kill-switch re-validation, and they make recorded sweeps
 * incomparable at the same --k-cap. The saving would have been ~4% of K on a 23 GB card, worth
 * ~1% of throughput (measured: a 4x batch buys 2.8%). Not a trade worth an unvalidated default.
 * If it is ever revisited, make it an explicit sizing PARAMETER that a row can record, not an
 * invisible platform sniff inside the allocator. */
constexpr int64_t kFreeVramReserveBytes = 800000000ll;

/* THE batch cap: one number, the same on every GPU.
 *
 * It is a SATURATION limit, not a safety limit — the analytic bound in gpap_compute_k_safe is
 * what protects memory, and where this cap binds the memory model would allow orders of
 * magnitude more rows than the run can use. At this batch a card is already ~2 orders of
 * magnitude oversubscribed against its resident warps: a 4x larger batch measured at +2.8%,
 * and auto-vs-pinned across three GPUs bought ~1%. It also bounds host-side readout, which
 * does not scale with VRAM at all.
 *
 * DEVICE-INDEPENDENT ON PURPOSE. It used to scale with the card's total VRAM off a reference
 * card's historical cap (49,152 at 12.0 GiB), which meant `--k-cap auto` resolved differently
 * per machine and a cross-GPU A/B had to pin an explicit cap to compare anything. That scaling
 * had also stopped doing ANYTHING: the floor below it exceeded the reference cap 5.3x, so the
 * proportional term could not win until ~68 GB of VRAM and every card we run on got this
 * constant anyway. A library's default batch should not depend on which GPU it woke up on.
 *
 * Revisit only with a throughput measurement showing the cap binding on something that is not
 * already occupancy-bound. Numbers: docs/findings/CROSS_GPU_FINDINGS.md. */
constexpr int32_t kBatchCap = 262144;

/* Clause count at or above which the warp-wide WAVE propagate beats the scalar one. Measured, see
 * gpap_dispatch_wave below for the whole calibration table and why the variable is M, not batch. */
constexpr int32_t kWaveMinClauses = 2048;

/* Trail headroom past the exact bound — one wavefront. See gpap_trail_cap for why that is the
 * unit. Keep in step with WAVE_W (cpp/tools/build_kernels.py). */
constexpr int32_t kTrailSlack = 32;
}  // namespace

/* NOTE — there is deliberately NO formula-bytes deduction here, and no gpap_formula_mem_bytes().
 * There used to be one, and it was a trap: gpap_mem_available() reports memory that is free
 * RIGHT NOW, which is already after the formula's occurrence lists and packed init have been
 * uploaded. Subtracting an estimate of them again double-counts and shrinks K for nothing. Every
 * in-tree caller had to remember to pass deduct_formula=False to avoid it, which is not a
 * default anyone should have to defend against. Size the batch against live free memory; the
 * formula is already paid for. */

/* THE trail sizing rule — one definition, called by the engine's own ensure_state() as well as
 * by every sizing consumer, so the allocation and the model cannot disagree.
 *
 * BOUND = num_vars, and it is TIGHT (measured on both kernels; docs/findings/INSIGHTS.md). The trail holds literals made TRUE and not yet propagated,
 * and every push is gated on the assignment BITMAP flipping that literal's bit from 0 to 1: the
 * scalar paths test-then-set inside one program, the WAVE paths take the `atomic_or` RETURN, so
 * for a given literal exactly one lane can ever observe the transition. Pushing a literal whose
 * negation is already set is a CONTRADICTION and conflicts the row instead of pushing, and l and
 * ~l always share a bitmap word (2v / 2v+1) — so a VARIABLE contributes at most ONE trail entry
 * per launch, whatever the decision batch B or the assumption width L. Neither is a parameter
 * here for that reason: both used to be, and each cost something. L made the cap grow and never
 * shrink, so a formula with many construction units inflated every later chunk permanently; B
 * survived as a floor the engine ignored while this model still honoured it, charging a caller
 * who asked for a wide decision batch for a trail that was never allocated. Max depth reaches
 * exactly N on a trajectory that decides every variable, and stays well under it on conflicted
 * rows.
 *
 * SLACK = one WAVEFRONT. The wave propagate and the warp-wide decision scan append a whole
 * wavefront of pushes in ONE masked store at `tail + prefix_sum`, and unlike the scalar path
 * (`elif tail < TRAIL_CAP`) a vectorised append has no bound test — so a wavefront is the
 * granularity in which a broken dedup invariant would overshoot, and one wavefront of headroom
 * keeps the first such step inside this row instead of in the next row's trail. It is 32 int32
 * = 128 bytes per row against 4*N + 4*M, i.e. nothing; the alternative (size at exactly N and
 * add the bound test to the hottest loop in the engine) buys no memory and costs a kernel
 * rebuild. Must not be smaller than WAVE_W in cpp/tools/build_kernels.py. */
extern "C" int32_t gpap_trail_cap(int32_t num_vars) {
    return num_vars + kTrailSlack;
}

/* `num_long` is the STATE clause count — clauses of length >= 3, `gpap_formula_info.num_long` —
 * because only those have a packed word per row. The parameter is NOT called `M`: `M` means the
 * total clause count everywhere else in this codebase, and the header's declaration and this
 * definition must not name the same thing differently. Binary clauses are implication edges in the
 * shared occ CSR and unit clauses live in the flush cache's bitmap: zero per-row bytes. Feeding
 * total num_clauses over-charges ~4-10x of the first term on circuit CNFs (65-82% of their
 * clauses are binary) and was exactly the kind of phantom charge the HISTORY note above bans. */
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
    (void)device;                 /* kept in the signature: the cap is device-INDEPENDENT now,
                                     and a caller should not have to care that it became so */
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

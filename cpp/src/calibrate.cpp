/* GPU memory budget: free-memory query and the conservative analytic K bound.
 *
 * Port of gpap/engine/calibrate.py, INCLUDING its deliberate conservatism. The model was
 * calibrated on the retired PyTorch propagate path, so it over-reserves for the packed engine;
 * every K it produced was validated spill-safe under the nvidia-smi kill-switch with THESE
 * numbers. Tightening it is a K change and must be re-validated the same way — this hardware
 * spills to host RAM instead of OOMing, so an optimistic model silently halves throughput.
 */
#include <cuda.h>

#include <algorithm>
#include <cstdint>

#include "gpap/gpap.h"

namespace {
constexpr double kAvgLitsPerClause = 4.0;
constexpr int64_t kWddmReserveBytes = 800000000ll;   /* WDDM/WSL2 driver overhead */
}  // namespace

extern "C" int64_t gpap_formula_mem_bytes(int32_t M, int32_t N, int64_t total_lits) {
    if (total_lits <= 0) total_lits = (int64_t)(kAvgLitsPerClause * M);
    return 8ll * M + 16ll * N + 8ll * total_lits;
}

static int64_t per_batch_mem_bytes(int32_t M, int32_t N, int32_t max_clause_size,
                                   int64_t total_lits) {
    if (total_lits <= 0) total_lits = (int64_t)(kAvgLitsPerClause * M);
    const int64_t sat_b = max_clause_size <= 127 ? 1 : (max_clause_size <= 32767 ? 2 : 4);
    const int64_t stat = (sat_b + 4) * M + 2ll * N;
    const int64_t apply_peak = 8ll * N;
    const int64_t prop_peak = 1000ll * total_lits / std::max(N, 1);
    return std::max<int64_t>(1, stat + std::max(apply_peak, prop_peak) + 3);
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

extern "C" int32_t gpap_compute_k_safe(int32_t M, int32_t N, int64_t bytes, int32_t cap,
                                       int32_t max_clause_len, int64_t total_lits) {
    if (bytes <= 0) return 1;
    const int64_t pb = per_batch_mem_bytes(M, N, max_clause_len, total_lits);
    int64_t k = bytes / pb;
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

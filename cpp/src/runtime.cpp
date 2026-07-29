/* Driver-API plumbing: error strings, the AOT kernel registry, launches, device buffers. */
#include "gpap_internal.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "generated/gpap_kernels.h"

namespace gpap {

static thread_local char g_error[1024] = "";

void set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
}

const char *get_error() { return g_error; }

const PackedSplit kPackedSplits[2] = {{16, 65535, 65535}, {8, 0xFFFFFF, 255}};

/* ------------------------------------------------------------------ kernels */

KernelCache::~KernelCache() {
    for (CUmodule m : modules_) cuModuleUnload(m);
}

gpap_status KernelCache::get(const std::string &key, Kernel *out) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        *out = it->second;
        return GPAP_OK;
    }
    for (int32_t i = 0; i < gpap_kernel_count; ++i) {
        const gpap_kernel_desc &d = gpap_kernel_table[i];
        if (key != d.key) continue;
        CUmodule mod = nullptr;
        GPAP_CU(cuModuleLoadData(&mod, d.cubin));
        Kernel k;
        k.shared = d.shared;
        k.threads = d.threads;
        CUresult rc = cuModuleGetFunction(&k.fn, mod, d.entry);
        if (rc != CUDA_SUCCESS) {
            cuModuleUnload(mod);
            set_error("kernel '%s': entry '%s' not found in cubin", d.key, d.entry);
            return GPAP_ERR_KERNEL;
        }
        modules_.push_back(mod);
        cache_[key] = k;
        *out = k;
        return GPAP_OK;
    }
    set_error("no AOT kernel for '%s' (rebuild cpp/tools/build_kernels.py?)", key.c_str());
    return GPAP_ERR_KERNEL;
}

gpap_status launch(const Kernel &k, unsigned gx, unsigned gy, std::vector<void *> args) {
    if (gx == 0 || gy == 0) return GPAP_OK;             /* empty grid: nothing to do */
    /* Triton kernels always take a global-scratch and a profile-scratch pointer last; the
     * AOT build refuses any kernel that actually needs them, so both are null here. */
    static thread_local CUdeviceptr scratch[2] = {0, 0};
    args.push_back(&scratch[0]);
    args.push_back(&scratch[1]);
    GPAP_CU(cuLaunchKernel(k.fn, gx, gy, 1, k.threads, 1, 1, k.shared, nullptr, args.data(),
                           nullptr));
    return GPAP_OK;
}

/* ------------------------------------------------------------------ device memory */

gpap_status DevBuf::reserve(size_t n) {
    if (n <= bytes) return GPAP_OK;
    release();
    CUresult rc = cuMemAlloc(&ptr, n);
    if (rc != CUDA_SUCCESS) {
        const char *msg = nullptr;
        cuGetErrorString(rc, &msg);
        ptr = 0;
        set_error("cuMemAlloc(%zu bytes) failed: %s", n, msg ? msg : "unknown");
        return GPAP_ERR_MEMORY;
    }
    bytes = n;
    return GPAP_OK;
}

gpap_status DevBuf::zero() {
    if (ptr && bytes) GPAP_CU(cuMemsetD8(ptr, 0, bytes));
    return GPAP_OK;
}

void DevBuf::release() {
    if (ptr) cuMemFree(ptr);
    ptr = 0;
    bytes = 0;
}

}  // namespace gpap

extern "C" const char *gpap_last_error(void) { return gpap::get_error(); }
extern "C" const char *gpap_version(void) { return "0.1.0"; }

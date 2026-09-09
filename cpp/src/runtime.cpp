/* Driver-API plumbing: error strings, the AOT kernel registry, launches, device buffers. */
#include "gpap_internal.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>                 /* dladdr, for gpap_build_id */
#endif

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

KernelCache::~KernelCache() {
    for (CUmodule m : modules_) cuModuleUnload(m);
}

/* Compute capability of the current context's device as major*10+minor, or 0 if unavailable. */
static int current_device_cc() {
    CUdevice dev = 0;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) return 0;
    int major = 0, minor = 0;
    if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) !=
            CUDA_SUCCESS ||
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) !=
            CUDA_SUCCESS)
        return 0;
    return major * 10 + minor;
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
        /* Architecture-specific cubins, no ptx fallback: check before loading, since
         * cuModuleLoadData's own failure says nothing about how to fix it. */
        const int cc = current_device_cc();
        if (cc && (cc / 10 != d.arch / 10 || cc < d.arch)) {
            set_error("kernels were compiled for sm_%d but this GPU is sm_%d. Rebuild with:\n"
                      "    python cpp/tools/build_kernels.py --arch %d\n"
                      "    cmake --build build -j\n"
                      "(or just scripts/build_core.sh on the target machine)",
                      d.arch, cc, cc);
            return GPAP_ERR_KERNEL;
        }
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
    /* Triton kernels take two trailing scratch pointers; the AOT build refuses any that use them. */
    static thread_local CUdeviceptr scratch[2] = {0, 0};
    args.push_back(&scratch[0]);
    args.push_back(&scratch[1]);
    GPAP_CU(cuLaunchKernel(k.fn, gx, gy, 1, k.threads, 1, 1, k.shared, nullptr, args.data(),
                           nullptr));
    return GPAP_OK;
}

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

#ifndef GPAP_VERSION_STR
#define GPAP_VERSION_STR "0.0.0-dev"      /* built outside CMake */
#endif
extern "C" const char *gpap_version(void) { return GPAP_VERSION_STR; }

/* Fingerprint of the loaded library's own file: FNV-1a 64 (the question is "same build?"). */
extern "C" const char *gpap_build_id(void) {
    /* Built once into a function-local static: the magic-static rule makes initialization
     * thread-safe and publishes the value only when COMPLETE — a hand-rolled memo would hand a
     * concurrent caller the bare "<version>+" prefix, identical for every build. */
    static const std::string id = [] {
        std::string digest = "unknown";
#if defined(__linux__) || defined(__APPLE__)
        Dl_info info{};
        /* Any symbol IN this library resolves to its file; gpap_version is as good as any. */
        if (dladdr(reinterpret_cast<const void *>(&gpap_version), &info) && info.dli_fname) {
            if (FILE *fp = std::fopen(info.dli_fname, "rb")) {
                uint64_t h = 1469598103934665603ull;    /* FNV-1a 64 offset basis */
                std::vector<unsigned char> buf(1 << 16);  /* heap: 64 KB overflows a small stack */
                size_t n;
                while ((n = std::fread(buf.data(), 1, buf.size(), fp)) > 0)
                    for (size_t i = 0; i < n; ++i) h = (h ^ buf[i]) * 1099511628211ull;
                /* ferror: a short read ends the loop as EOF does, so a truncated read would
                 * hash a PREFIX and return it as a fingerprint. */
                const bool complete = std::ferror(fp) == 0;
                std::fclose(fp);
                if (complete) {
                    char hex[17];
                    std::snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)h);
                    digest = hex;
                }
            }
        }
#endif
        return std::string(GPAP_VERSION_STR) + "+" + digest;
    }();
    return id.c_str();
}

/* The CUDA plug of the GPU socket (gpu_api.hpp): the driver API and the Triton kernels compiled
 * ahead of time into the library. The only file that includes cuda.h, for its types; the driver
 * library itself is opened at run time (load_cuda). Calls run on the legacy default stream of
 * the device's primary context, so they interoperate with any CUDA runtime user in the
 * process. */
#include <cuda.h>
#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "generated/gpap_kernels.h"
#include "gpap_internal.hpp"

namespace gpap {
namespace {

/* ------------------------------------------------------------------ the driver library */

/* libcuda is opened at run time, as gpu_opencl.cpp opens libOpenCL, so libgpap loads on a
 * machine without the NVIDIA driver, where only this backend is missing. cuda.h maps some names
 * to versioned symbols (cuMemAlloc is cuMemAlloc_v2); the table entry and the dlsym name both
 * pass through that mapping, so each entry is the symbol the header declares. */
#define GPAP_CU_FUNCTIONS(X)                                                                   \
    X(cuInit) X(cuDeviceGet) X(cuDeviceGetAttribute) X(cuDeviceGetName)                        \
    X(cuDevicePrimaryCtxRetain) X(cuDevicePrimaryCtxRelease) X(cuCtxSetCurrent)                \
    X(cuGetErrorString) X(cuMemGetInfo) X(cuMemAlloc) X(cuMemFree) X(cuMemsetD8)               \
    X(cuMemcpyHtoD) X(cuMemcpyDtoH) X(cuMemcpyDtoD) X(cuModuleLoadData) X(cuModuleGetFunction) \
    X(cuModuleUnload) X(cuLaunchKernel)

struct CuApi {
#define X(name) decltype(&::name) name = nullptr;
    GPAP_CU_FUNCTIONS(X)
#undef X
};
CuApi g_cu;                              /* filled once by load_cuda, read-only after */

#define GPAP_CU_STR_(x) #x
#define GPAP_CU_STR(x) GPAP_CU_STR_(x)

/* Open the driver library, once per process. Every entry point of this plug calls it first;
 * $GPAP_CUDA_LIB names another library, as $GPAP_OPENCL_LIB does for OpenCL. */
gpap_status load_cuda() {
    static std::once_flag once;
    static std::string missing;          /* empty once the table is complete */
    std::call_once(once, [] {
        const char *env = std::getenv("GPAP_CUDA_LIB");
        const char *name = env && *env ? env : "libcuda.so.1";
        void *lib = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            missing = std::string("the cuda backend needs the NVIDIA driver and found no ") +
                      name + ". On a machine without it, GPAP_BACKEND=opencl runs the engine " +
                      "on an OpenCL device (docs/OPENCL.md)";
            return;
        }
#define X(fn)                                                                              \
    g_cu.fn = reinterpret_cast<decltype(g_cu.fn)>(dlsym(lib, GPAP_CU_STR(fn)));            \
    if (!g_cu.fn) {                                                                        \
        missing = std::string("the NVIDIA driver library has no ") + GPAP_CU_STR(fn) +     \
                  " (the driver is too old for this build)";                               \
        return;                                                                            \
    }
        GPAP_CU_FUNCTIONS(X)
#undef X
    });
    if (missing.empty()) return GPAP_OK;
    set_error("%s", missing.c_str());
    return GPAP_ERR_CUDA;
}

/* Every driver call goes through this: it records a readable message and returns. */
#define GPAP_CU(call)                                                                    \
    do {                                                                                 \
        CUresult _rc = (call);                                                           \
        if (_rc != CUDA_SUCCESS) {                                                        \
            const char *_msg = nullptr;                                                  \
            g_cu.cuGetErrorString(_rc, &_msg);                                                \
            ::gpap::set_error("%s: %s (%s:%d)", #call, _msg ? _msg : "unknown",          \
                              __FILE__, __LINE__);                                       \
            return GPAP_ERR_CUDA;                                                        \
        }                                                                                \
    } while (0)

/* Elements ONE PROGRAM covers along each axis of a work extent: the BLOCK constexpr of the
 * kernels that tile their extent (cpp/tools/build_kernels.py::kernel_specs). Every other kernel
 * is one program per row. A BLOCK changed there must change here. */
const struct {
    const char *key;
    int b0, b1;
} kBlocks[] = {
    {"bcast_rows", 1, 1024}, {"fill_i8", 1024, 1}, {"var_bits", 1, 256}, {"unpack_rows", 1, 1024}};

struct Kernel {
    CUfunction fn = nullptr;
    int shared = 0;
    int threads = 32;
    int b0 = 1, b1 = 1;
};

const gpap_kernel_desc *find_desc(const char *key) {
    for (int32_t i = 0; i < gpap_kernel_count; ++i)
        if (std::strcmp(key, gpap_kernel_table[i].key) == 0) return &gpap_kernel_table[i];
    return nullptr;
}

/* Compute capability of `dev` as major*10+minor, or 0 if unavailable. */
int device_cc(CUdevice dev) {
    int major = 0, minor = 0;
    if (g_cu.cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) !=
            CUDA_SUCCESS ||
        g_cu.cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) !=
            CUDA_SUCCESS)
        return 0;
    return major * 10 + minor;
}

class CudaGpu final : public Gpu {
  public:
    CudaGpu(CUdevice device, CUcontext ctx) : device_(device), ctx_(ctx) {
        char name[128] = "";
        if (g_cu.cuDeviceGetName(name, sizeof(name), device_) == CUDA_SUCCESS) name_ = name;
    }

    ~CudaGpu() override {
        g_cu.cuCtxSetCurrent(ctx_);
        for (CUmodule m : modules_) g_cu.cuModuleUnload(m);
        g_cu.cuDevicePrimaryCtxRelease(device_);
    }

    const char *backend() const override { return "cuda"; }
    const char *device_name() const override { return name_.c_str(); }

    gpap_status make_current() override {
        GPAP_CU(g_cu.cuCtxSetCurrent(ctx_));
        return GPAP_OK;
    }

    gpap_status mem_info(int64_t *free_bytes, int64_t *total_bytes) override {
        size_t free_b = 0, total_b = 0;
        GPAP_CU(g_cu.cuMemGetInfo(&free_b, &total_b));
        *free_bytes = (int64_t)free_b;
        *total_bytes = (int64_t)total_b;
        return GPAP_OK;
    }

    int64_t max_buffer_bytes() const override { return 0; }

    gpap_status alloc(size_t bytes, GpuMem *out) override {
        CUdeviceptr p = 0;
        CUresult rc = g_cu.cuMemAlloc(&p, bytes);
        if (rc != CUDA_SUCCESS) {
            const char *msg = nullptr;
            g_cu.cuGetErrorString(rc, &msg);
            set_error("cuMemAlloc(%zu bytes) failed: %s", bytes, msg ? msg : "unknown");
            return GPAP_ERR_MEMORY;
        }
        *out = (GpuMem)p;
        return GPAP_OK;
    }

    void free(GpuMem m) override {
        if (m) g_cu.cuMemFree((CUdeviceptr)m);
    }

    gpap_status zero(GpuMem m, size_t bytes) override {
        if (m && bytes) GPAP_CU(g_cu.cuMemsetD8((CUdeviceptr)m, 0, bytes));
        return GPAP_OK;
    }

    gpap_status upload(GpuMem dst, const void *src, size_t bytes) override {
        if (bytes) GPAP_CU(g_cu.cuMemcpyHtoD((CUdeviceptr)dst, src, bytes));
        return GPAP_OK;
    }

    gpap_status download(void *dst, GpuMem src, size_t offset, size_t bytes) override {
        if (bytes) GPAP_CU(g_cu.cuMemcpyDtoH(dst, (CUdeviceptr)src + offset, bytes));
        return GPAP_OK;
    }

    gpap_status copy(GpuMem dst, GpuMem src, size_t bytes) override {
        if (bytes) GPAP_CU(g_cu.cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, bytes));
        return GPAP_OK;
    }

    gpap_status adopt(uintptr_t foreign, GpuMem *out) override {
        *out = (GpuMem)foreign;          /* a CUdeviceptr of the same primary context */
        return GPAP_OK;
    }

    bool has_kernel(const char *key) const override { return find_desc(key) != nullptr; }

    gpap_status run(const char *key, int64_t n0, int64_t n1,
                    std::initializer_list<GpuArg> args) override {
        if (n0 <= 0 || n1 <= 0) return GPAP_OK;             /* empty extent: nothing to do */
        Kernel k;
        GPAP_TRY(get(key, &k));
        const unsigned gx = (unsigned)((n0 + k.b0 - 1) / k.b0);
        const unsigned gy = (unsigned)((n1 + k.b1 - 1) / k.b1);
        /* cuLaunchKernel takes a pointer to each argument's value. The three stores are sized
         * up front, so the addresses taken below stay valid. */
        std::vector<CUdeviceptr> ptrs;
        std::vector<int32_t> i32s;
        std::vector<int64_t> i64s;
        ptrs.reserve(args.size());
        i32s.reserve(args.size());
        i64s.reserve(args.size());
        std::vector<void *> argv;
        argv.reserve(args.size() + 2);
        for (const GpuArg &a : args) {
            if (a.kind == GpuArg::BUF) {
                ptrs.push_back((CUdeviceptr)a.mem + a.offset);
                argv.push_back(&ptrs.back());
            } else if (a.kind == GpuArg::I32) {
                i32s.push_back((int32_t)a.value);
                argv.push_back(&i32s.back());
            } else {
                i64s.push_back(a.value);
                argv.push_back(&i64s.back());
            }
        }
        /* Triton kernels take two trailing scratch pointers; the AOT build refuses any kernel
         * that uses them. */
        static thread_local CUdeviceptr scratch[2] = {0, 0};
        argv.push_back(&scratch[0]);
        argv.push_back(&scratch[1]);
        GPAP_CU(g_cu.cuLaunchKernel(k.fn, gx, gy, 1, k.threads, 1, 1, k.shared, nullptr,
                                    argv.data(), nullptr));
        return GPAP_OK;
    }

  private:
    /* Loads a cubin from the generated table on first use and caches its CUfunction. */
    gpap_status get(const char *key, Kernel *out) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            *out = it->second;
            return GPAP_OK;
        }
        const gpap_kernel_desc *d = find_desc(key);
        if (!d) {
            set_error("no AOT kernel for '%s' (rebuild cpp/tools/build_kernels.py?)", key);
            return GPAP_ERR_KERNEL;
        }
        /* Architecture-specific cubins, no ptx fallback: check before loading, since
         * cuModuleLoadData's own failure says nothing about how to fix it. */
        const int cc = device_cc(device_);
        if (cc && (cc / 10 != d->arch / 10 || cc < d->arch)) {
            set_error("kernels were compiled for sm_%d but this GPU is sm_%d. Rebuild with:\n"
                      "    python cpp/tools/build_kernels.py --arch %d\n"
                      "    cmake --build build -j\n"
                      "(or just scripts/build_core.sh on the target machine)",
                      d->arch, cc, cc);
            return GPAP_ERR_KERNEL;
        }
        CUmodule mod = nullptr;
        GPAP_CU(g_cu.cuModuleLoadData(&mod, d->cubin));
        Kernel k;
        k.shared = d->shared;
        k.threads = d->threads;
        CUresult rc = g_cu.cuModuleGetFunction(&k.fn, mod, d->entry);
        if (rc != CUDA_SUCCESS) {
            g_cu.cuModuleUnload(mod);
            set_error("kernel '%s': entry '%s' not found in cubin", d->key, d->entry);
            return GPAP_ERR_KERNEL;
        }
        for (const auto &b : kBlocks)
            if (std::strcmp(key, b.key) == 0) {
                k.b0 = b.b0;
                k.b1 = b.b1;
            }
        modules_.push_back(mod);
        cache_[key] = k;
        *out = k;
        return GPAP_OK;
    }

    CUdevice device_;
    CUcontext ctx_;                 /* primary context, retained */
    std::string name_;
    std::unordered_map<std::string, Kernel> cache_;
    std::vector<CUmodule> modules_;
};

gpap_status cuda_open(int device, Gpu **out) {
    GPAP_TRY(load_cuda());
    GPAP_CU(g_cu.cuInit(0));
    CUdevice dev = 0;
    if (g_cu.cuDeviceGet(&dev, device) != CUDA_SUCCESS) {
        set_error("cuDeviceGet(%d) failed", device);
        return GPAP_ERR_CUDA;
    }
    /* Primary context: the one the CUDA runtime uses, so mixing the two costs no switch. */
    CUcontext ctx = nullptr;
    if (g_cu.cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) {
        set_error("cuDevicePrimaryCtxRetain failed");
        return GPAP_ERR_CUDA;
    }
    g_cu.cuCtxSetCurrent(ctx);
    *out = new CudaGpu(dev, ctx);
    return GPAP_OK;
}

gpap_status cuda_mem_info(int device, int64_t *free_bytes, int64_t *total_bytes) {
    CUdevice dev;
    CUcontext ctx;
    GPAP_TRY(load_cuda());
    if (g_cu.cuInit(0) != CUDA_SUCCESS) return GPAP_ERR_CUDA;
    if (g_cu.cuDeviceGet(&dev, device) != CUDA_SUCCESS) return GPAP_ERR_CUDA;
    if (g_cu.cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) return GPAP_ERR_CUDA;
    g_cu.cuCtxSetCurrent(ctx);
    size_t free_b = 0, total_b = 0;
    CUresult rc = g_cu.cuMemGetInfo(&free_b, &total_b);
    g_cu.cuDevicePrimaryCtxRelease(dev);
    if (rc != CUDA_SUCCESS) return GPAP_ERR_CUDA;
    *free_bytes = (int64_t)free_b;
    *total_bytes = (int64_t)total_b;
    return GPAP_OK;
}

int32_t cuda_resident_warps(int device) {
    CUdevice dev;
    if (load_cuda() != GPAP_OK || g_cu.cuInit(0) != CUDA_SUCCESS) return 0;
    if (g_cu.cuDeviceGet(&dev, device) != CUDA_SUCCESS) return 0;
    int sms = 0, threads_per_sm = 0;
    g_cu.cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    g_cu.cuDeviceGetAttribute(&threads_per_sm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                              dev);
    return sms * (threads_per_sm / 32);
}

}  // namespace

const GpuDriver kCudaDriver = {"cuda", cuda_open, cuda_mem_info, cuda_resident_warps};

}  // namespace gpap

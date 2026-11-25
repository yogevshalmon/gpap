/* The GPU socket: every device operation the engine performs, behind one interface with one
 * plug per GPU library — gpu_cuda.cpp (the CUDA driver API and the Triton kernels, the reference
 * and the fast path) and gpu_opencl.cpp (OpenCL and cpp/kernels/bcp.cl). Nothing above this
 * header names a GPU library, so a new one is one more plug.
 *
 * A kernel is named by its KEY ("fixpoint.s16.w0", the names of cpp/tools/build_kernels.py) and
 * runs over a WORK EXTENT (n0, n1): rows, or rows x columns. Its arguments are listed in the
 * order of the Triton signature. How an extent becomes a launch grid, and how the arguments
 * reach the kernel, is the plug's business: the CUDA plug passes them as they are, the OpenCL
 * plug packs the scalars into a parameter buffer. Both run the same engine code. */
#ifndef GPAP_GPU_API_HPP
#define GPAP_GPU_API_HPP

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "gpap/gpap.h"

namespace gpap {

/* A device buffer, as the plug names it (a CUdeviceptr, a cl_mem). 0 is "no buffer". */
typedef uintptr_t GpuMem;

/* One kernel argument. A buffer with a byte offset is an INPUT VIEW: the kernel may read from
 * that offset on, and must not write through it (a plug is free to hand the kernel a copy). */
struct GpuArg {
    enum Kind { BUF, I32, I64 };
    Kind kind;
    GpuMem mem;
    size_t offset;
    int64_t value;
};
inline GpuArg buf(GpuMem m, size_t offset = 0) { return {GpuArg::BUF, m, offset, 0}; }
inline GpuArg i32(int32_t v) { return {GpuArg::I32, 0, 0, v}; }
inline GpuArg i64(int64_t v) { return {GpuArg::I64, 0, 0, v}; }

/* What a plug reports about the memory the engine may use on its device (engine_batch
 * sizes from this). Each field's zero is what a plug that knows nothing reports: no total, and
 * `source` TOTAL, so GPAP takes `free` for a guess and keeps a share of it. The exception is
 * `max_buffer`, where zero means the device limits no single buffer: a plug whose device does
 * must say so, or a buffer past it fails at the allocation. */
struct GpuMemory {
    /* What `free` is. */
    enum Source : int32_t {
        /* Not a measurement: the device total (OpenCL has no free-memory query), or a guess.
         * GPAP keeps a share of it, or a limit stands in for it. */
        TOTAL = 0,
        /* Measured free device memory (cuMemGetInfo). It already excludes what this device's
         * allocations hold, the engine's own included, so the engine counts back the rows it
         * will reuse. GPAP keeps a safety margin under it. */
        DEVICE = 1,
        /* A host's own count of what it still lets GPAP allocate, its other GPU work and its
         * own transients already deducted: exact, so GPAP keeps no margin under it. Like
         * DEVICE, it excludes what this engine already holds. */
        GRANT = 2,
    };
    int64_t total = 0;          /* device bytes; 0 = unknown */
    int64_t free = 0;           /* bytes for new allocations, read as `source` says */
    Source source = TOTAL;
    int64_t max_buffer = 0;     /* the largest single buffer; 0 = the device sets no limit */
};

/* One open device. Not thread-safe, like the engine that owns it. Every call that can fail
 * returns a gpap_status and leaves the message in gpap_last_error(). */
class Gpu {
  public:
    virtual ~Gpu() = default;

    virtual const char *backend() const = 0;        /* "cuda" | "opencl" */
    virtual const char *device_name() const = 0;
    /* Make this device the calling thread's target; every public entry point calls it first. */
    virtual gpap_status make_current() = 0;
    /* What memory the device can give the engine now (GpuMemory). */
    virtual gpap_status memory(GpuMemory *out) = 0;

    virtual gpap_status alloc(size_t bytes, GpuMem *out) = 0;
    virtual void free(GpuMem m) = 0;
    virtual gpap_status zero(GpuMem m, size_t bytes) = 0;
    virtual gpap_status upload(GpuMem dst, const void *src, size_t bytes) = 0;
    virtual gpap_status download(void *dst, GpuMem src, size_t offset, size_t bytes) = 0;
    virtual gpap_status copy(GpuMem dst, GpuMem src, size_t bytes) = 0;
    /* A device pointer the CALLER owns (the `_dev` entry points), as a buffer of this plug. */
    virtual gpap_status adopt(uintptr_t foreign, GpuMem *out) = 0;

    /* Can this device run the kernel? False for a key the plug does not ship, and for a WAVE
     * kernel on a device that cannot form its work group (the engine then dispatches scalar).
     * Running a missing key is GPAP_ERR_KERNEL with a message naming the plug. */
    virtual bool has_kernel(const char *key) const = 0;
    /* Run kernel `key` over the extent (n0, n1); an empty extent is a no-op. Asynchronous.
     * Every operation of a Gpu takes effect in call order: a run sees every earlier upload,
     * zero, copy and run; a download returns once `dst` holds what they left. */
    virtual gpap_status run(const char *key, int64_t n0, int64_t n1,
                            std::initializer_list<GpuArg> args) = 0;
};

/* One GPU library: what can be asked of it without an engine. */
struct GpuDriver {
    const char *name;
    gpap_status (*open)(int device, Gpu **out);
    /* Gpu::memory for a device no engine holds (gpap_mem_available). */
    gpap_status (*memory)(int device, GpuMemory *out);
    /* Warps the device keeps resident (the WAVE dispatch's small-batch term); 0 = unknown. */
    int32_t (*resident_warps)(int device);
};

/* The plug new engines get: $GPAP_BACKEND = cuda (the default) or opencl, read on every call
 * so one process can open engines on both. NULL, with the error set, for any other value. */
const GpuDriver *gpu_driver();

extern const GpuDriver kCudaDriver;      /* gpu_cuda.cpp */
extern const GpuDriver kOpenClDriver;    /* gpu_opencl.cpp */

}  // namespace gpap

#endif /* GPAP_GPU_API_HPP */

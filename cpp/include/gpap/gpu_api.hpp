/* The GPU socket: every device operation the engine performs, behind one interface with one
 * plug per GPU library — gpu_cuda.cpp (the CUDA driver API and the Triton kernels, the reference
 * and the fast path) and gpu_opencl.cpp (OpenCL and cpp/kernels/bcp.cl). Nothing above this
 * header names a GPU library, so a new one is one more plug.
 *
 * A kernel is named by its KEY ("fixpoint.s16.w0", the names of cpp/tools/build_kernels.py) and
 * runs over a WORK EXTENT (n0, n1): rows, or rows x columns. Its arguments are listed in the
 * order of the Triton signature. How an extent becomes a launch grid, and how the arguments
 * reach the kernel, is the plug's business: the CUDA plug passes them as they are, the OpenCL
 * plug packs the scalars into a parameter buffer. Both run the same engine code.
 *
 * The header is public so that a HOST can bring its own plug: implement Gpu over the host's GPU
 * layer and hand it to gpap::engine_create_on (below). A host plug runs the OpenCL kernels,
 * cpp/kernels/bcp.cl, by the launch rules in gpap/opencl_launch.hpp; cpp/examples/host_plug.cpp
 * is a plug that forwards to GPAP's own, and checks this contract. */
#ifndef GPAP_GPU_API_HPP
#define GPAP_GPU_API_HPP

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>

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

/* One open device. The contract a plug keeps, GPAP's own and a host's alike:
 *   * Calls come one at a time, from the thread that uses the engine owning the plug: a plug
 *     needs no locking of its own, unless it shares state with other plugs.
 *   * No method throws: they run under the extern "C" entry points. A call that can fail
 *     returns a gpap_status and sets the message with gpap::set_error (below), which
 *     gpap_last_error() then returns.
 *   * Every operation takes effect in call order (run, below), whatever the host's queue does.
 * The engine owns its plug and destroys it after freeing every buffer it allocated. */
class Gpu {
  public:
    virtual ~Gpu() = default;

    /* The engine reports it (gpap_engine_backend). GPAP's own plugs: "cuda", "opencl"; a host
     * plug may use any name. */
    virtual const char *backend() const = 0;
    virtual const char *device_name() const = 0;
    /* Make this device the calling thread's target; every public entry point calls it first. */
    virtual gpap_status make_current() = 0;
    /* Bytes free for new allocations and the device total. A library with no free-memory query
     * reports the total as free; the caller's reserve and fraction still apply (what that means
     * for a host's own budget: GitHub issue 72). */
    virtual gpap_status mem_info(int64_t *free_bytes, int64_t *total_bytes) = 0;
    /* True when mem_info's free figure is measured, so it already excludes what this device's
     * allocations hold; false when it is the total. engine_batch counts the engine's own row
     * buffers back only in the first case: in the second they were never taken out. */
    virtual bool free_is_measured() const = 0;
    /* The largest single buffer the device allows, or 0 if it sets no limit of its own. The
     * batch must keep the widest per-row buffer under it (engine_batch). */
    virtual int64_t max_buffer_bytes() const = 0;

    virtual gpap_status alloc(size_t bytes, GpuMem *out) = 0;
    /* May come while launches that use `m` are still queued: the plug keeps the memory alive
     * until they are done (OpenCL's reference counting does). */
    virtual void free(GpuMem m) = 0;
    virtual gpap_status zero(GpuMem m, size_t bytes) = 0;
    /* Has read all of `src` when it returns: the engine uploads from temporaries. */
    virtual gpap_status upload(GpuMem dst, const void *src, size_t bytes) = 0;
    virtual gpap_status download(void *dst, GpuMem src, size_t offset, size_t bytes) = 0;
    virtual gpap_status copy(GpuMem dst, GpuMem src, size_t bytes) = 0;
    /* A device pointer the CALLER owns (the `_dev` entry points), as a buffer of this plug. */
    virtual gpap_status adopt(uintptr_t foreign, GpuMem *out) = 0;

    /* Can this device run the kernel? False for a key the plug does not ship, and for a WAVE
     * kernel on a device that cannot form its work group (the engine then dispatches scalar).
     * Running a missing key is GPAP_ERR_KERNEL with a message naming the plug. Probing and
     * checking need the keys of opencl_launch.hpp's kSpecs; sampling also needs the
     * trajectory, which only the CUDA plug has. */
    virtual bool has_kernel(const char *key) const = 0;
    /* Run kernel `key` over the extent (n0, n1); an empty extent is a no-op. Asynchronous.
     * Every operation of a Gpu takes effect in call order: a run sees every earlier upload,
     * zero, copy and run; a download returns once `dst` holds what they left. */
    virtual gpap_status run(const char *key, int64_t n0, int64_t n1,
                            std::initializer_list<GpuArg> args) = 0;

    /* Warps the device keeps resident: the WAVE dispatch's small-batch term (dispatch_wave).
     * 0 = unknown, which keeps formulas under its clause threshold on the scalar kernel. */
    virtual int32_t resident_warps() const { return 0; }
};

/* An engine on a device the CALLER opened: gpap_engine_create with `gpu` in place of the plug
 * $GPAP_BACKEND would open (opts->device is ignored). The engine owns `gpu` from here on, and
 * frees it on any failure too, setting *out to NULL. A NULL f, gpu or out is GPAP_ERR_ARG. The
 * C++ wrapper (gpap.hpp) cannot hold such an engine: use the C API on it. */
gpap_status engine_create_on(const gpap_formula *f, const gpap_engine_options *opts,
                             std::unique_ptr<Gpu> gpu, gpap_engine **out);

/* GPAP's own plug for `backend` ("cuda" or "opencl"; NULL or "" = $GPAP_BACKEND, as
 * gpap_engine_create) on device ordinal `device`: for a host that wraps it, or hands it to
 * engine_create_on. GPAP_ERR_ARG, with the message set, for a backend this libgpap does not
 * have (a build without CUDA has opencl only). */
gpap_status open_gpu(const char *backend, int device, std::unique_ptr<Gpu> *out);

#if defined(__GNUC__) || defined(__clang__)
#define GPAP_PRINTF_LIKE(fmt, first) __attribute__((format(printf, fmt, first)))
#else
#define GPAP_PRINTF_LIKE(fmt, first)
#endif

/* Set the message gpap_last_error() returns on the calling thread (printf format): how a plug
 * that returns a failed status says why. */
void set_error(const char *fmt, ...) GPAP_PRINTF_LIKE(1, 2);

}  // namespace gpap

#endif /* GPAP_GPU_API_HPP */

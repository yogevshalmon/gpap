/* The OpenCL plug of the GPU socket (gpu_api.hpp): cpp/kernels/bcp.cl on any OpenCL device.
 *
 * No build dependency: the OpenCL library is opened at run time (dlopen), and the few types,
 * constants and entry points used are declared below, from the Khronos OpenCL 1.2 C API, whose
 * ABI is fixed. A machine without OpenCL builds and runs libgpap as before; asking for this
 * backend there is an error that says what is missing.
 *
 * What the plug owns, so that bcp.cl and the engine stay ignorant of each other:
 *   * the kernel list (kSpecs): which keys exist here and how a work extent becomes a launch;
 *   * the parameter block: scalars go into a small buffer, buffers follow in order (the
 *     layout is documented at the top of bcp.cl);
 *   * one context and one compiled program per device, shared by every engine in the process
 *     and kept until it exits — compiling takes seconds, creating an engine must not. */
#include "gpap_internal.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#define GPAP_HAVE_DLOPEN 1
#endif

extern "C" {
extern const char gpap_cl_source[];          /* cpp/kernels/bcp.cl, embedded by CMake */
extern const size_t gpap_cl_source_len;
}

namespace gpap {
namespace {

/* ------------------------------------------------------------------ the OpenCL C API we use */

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_uint cl_bool;
typedef cl_ulong cl_bitfield;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_bool CL_TRUE = 1;
constexpr cl_bitfield CL_DEVICE_TYPE_GPU = 1u << 2;
constexpr cl_bitfield CL_DEVICE_TYPE_ALL = 0xFFFFFFFFu;
constexpr cl_uint CL_DEVICE_TYPE = 0x1000;
constexpr cl_uint CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;
constexpr cl_uint CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010;
constexpr cl_uint CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_uint CL_DEVICE_VERSION = 0x102F;
constexpr cl_bitfield CL_MEM_READ_WRITE = 1u << 0;
constexpr cl_bitfield CL_MEM_READ_ONLY = 1u << 2;
constexpr cl_bitfield CL_MEM_COPY_HOST_PTR = 1u << 5;
constexpr cl_uint CL_MEM_SIZE = 0x1102;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;

#define GPAP_CL_FUNCTIONS(X)                                                                   \
    X(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id *, cl_uint *))                        \
    X(cl_int, clGetDeviceIDs, (cl_platform_id, cl_bitfield, cl_uint, cl_device_id *, cl_uint *)) \
    X(cl_int, clGetDeviceInfo, (cl_device_id, cl_uint, size_t, void *, size_t *))              \
    X(cl_context, clCreateContext,                                                             \
      (const intptr_t *, cl_uint, const cl_device_id *,                                        \
       void (*)(const char *, const void *, size_t, void *), void *, cl_int *))                \
    X(cl_int, clReleaseContext, (cl_context))                                                  \
    X(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id, cl_bitfield, cl_int *)) \
    X(cl_int, clReleaseCommandQueue, (cl_command_queue))                                       \
    X(cl_mem, clCreateBuffer, (cl_context, cl_bitfield, size_t, void *, cl_int *))             \
    X(cl_int, clReleaseMemObject, (cl_mem))                                                    \
    X(cl_int, clGetMemObjectInfo, (cl_mem, cl_uint, size_t, void *, size_t *))                 \
    X(cl_program, clCreateProgramWithSource,                                                   \
      (cl_context, cl_uint, const char **, const size_t *, cl_int *))                          \
    X(cl_int, clReleaseProgram, (cl_program))                                                  \
    X(cl_int, clBuildProgram,                                                                  \
      (cl_program, cl_uint, const cl_device_id *, const char *,                                \
       void (*)(cl_program, void *), void *))                                                  \
    X(cl_int, clGetProgramBuildInfo,                                                           \
      (cl_program, cl_device_id, cl_uint, size_t, void *, size_t *))                           \
    X(cl_kernel, clCreateKernel, (cl_program, const char *, cl_int *))                         \
    X(cl_int, clReleaseKernel, (cl_kernel))                                                    \
    X(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void *))                      \
    X(cl_int, clEnqueueNDRangeKernel,                                                          \
      (cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *,   \
       cl_uint, const cl_event *, cl_event *))                                                 \
    X(cl_int, clEnqueueReadBuffer,                                                             \
      (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *,   \
       cl_event *))                                                                            \
    X(cl_int, clEnqueueWriteBuffer,                                                            \
      (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint,               \
       const cl_event *, cl_event *))                                                          \
    X(cl_int, clEnqueueCopyBuffer,                                                             \
      (cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *,    \
       cl_event *))                                                                            \
    X(cl_int, clEnqueueFillBuffer,                                                             \
      (cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint,                \
       const cl_event *, cl_event *))

struct ClApi {
#define X(ret, name, args) ret (*name) args = nullptr;
    GPAP_CL_FUNCTIONS(X)
#undef X
};

const char *cl_error_name(cl_int rc) {
    switch (rc) {
    case -1: return "device not found";
    case -4: return "memory object allocation failure";
    case -5: return "out of resources";
    case -6: return "out of host memory";
    case -11: return "program build failure";
    case -30: return "invalid value";
    case -38: return "invalid memory object";
    case -46: return "invalid kernel name";
    case -51: return "invalid argument size";
    case -52: return "invalid kernel arguments";
    case -54: return "invalid work group size";
    case -61: return "invalid buffer size";
    case -1001: return "no OpenCL driver is registered (no ICD found)";
    default: return "see the OpenCL error codes";
    }
}

gpap_status cl_fail(const char *what, cl_int rc) {
    set_error("OpenCL: %s failed with %d (%s)", what, (int)rc, cl_error_name(rc));
    return GPAP_ERR_CUDA;               /* the status for "a GPU driver call failed" */
}

#define GPAP_CL(call)                                                                    \
    do {                                                                                 \
        cl_int _rc = (call);                                                             \
        if (_rc != CL_SUCCESS) return cl_fail(#call, _rc);                               \
    } while (0)

/* ------------------------------------------------------------------ the kernel list */

/* How a work extent (n0, n1) becomes an OpenCL launch. Always one-dimensional: an OpenCL built
 * over CUDA (PoCL's driver) limits every dimension but the first to 65,535 work groups, and a
 * batch has more rows than that. */
enum Shape {
    ROWS,    /* one work item per element of axis 0 */
    GRID,    /* one work item per (row, block of `chunk` columns), numbered row * blocks + block */
    GROUP,   /* a single work group, whatever the extent (the two reductions) */
    LANES    /* a work group of `chunk` items per element of axis 0, numbered row * chunk + lane */
};

constexpr int kWaveW = 32;               /* GPAP_WAVE_W in bcp.cl: the lanes of a WAVE row */

/* THE list of kernels this plug ships: bcp.cl defines `gpap_<key with . as _>` for each. A key
 * missing here is a kernel the OpenCL backend does not have (the trajectory).
 * A scalar fixpoint kernel (.w0) runs a row on one work item, a WAVE one (.w1) on a work group
 * of kWaveW. A row per work item, not per warp, because the scalar kernel is dispatched on
 * small formulas, where a warp per row loses (docs/findings/OPENCL_FINDINGS.md §9).
 * bcast_rows copies 4 columns per work item (measured: §8). */
const struct ClSpec {
    const char *key;
    Shape shape;
    int chunk;
} kSpecs[] = {
    {"fixpoint.s16.w0", ROWS, 1},             {"fixpoint.s8.w0", ROWS, 1},
    {"fixpoint_clean.s16.w0", ROWS, 1},       {"fixpoint_clean.s8.w0", ROWS, 1},
    {"fixpoint.s16.w1", LANES, kWaveW},       {"fixpoint.s8.w1", LANES, kWaveW},
    {"fixpoint_clean.s16.w1", LANES, kWaveW}, {"fixpoint_clean.s8.w1", LANES, kWaveW},
    {"queue.s16.w0", ROWS, 1},                {"queue.s8.w0", ROWS, 1},
    {"queue.s16.w1", LANES, kWaveW},          {"queue.s8.w1", LANES, kWaveW},
    {"bcast_rows", GRID, 4},                  {"fill_i8", ROWS, 1},
    {"complete_mask", ROWS, 1},               {"sound_rows.s16", ROWS, 1},
    {"sound_rows.s8", ROWS, 1},               {"var_bits", GRID, 64},
    {"unpack_rows", GRID, 64},                {"sum_i8_to", GROUP, 1},
    {"sum_i32_acc", GROUP, 1},
};

constexpr size_t kGroup = 64;            /* work-group size of the ROWS and GRID kernels (the
                                          * size made no difference: §8) */
constexpr size_t kReduceGroup = 256;     /* GPAP_GROUP_MAX in bcp.cl */

const ClSpec *find_spec(const char *key) {
    for (const ClSpec &s : kSpecs)
        if (std::strcmp(key, s.key) == 0) return &s;
    return nullptr;
}

/* ------------------------------------------------------------------ devices and programs */

struct ClDevice {
    cl_device_id id = nullptr;
    std::string name;                    /* "<device> (<OpenCL version string>)" */
    int64_t total_mem = 0;
    int64_t max_alloc = 0;
    size_t max_group = 1;
    /* built on first use, then shared by every engine on this device */
    cl_context ctx = nullptr;
    cl_program program = nullptr;
};

std::mutex g_mu;                         /* guards everything below */
ClApi g_cl;
bool g_loaded = false;
/* GPUs first, then every other device. Never destroyed: an engine freed while the process
 * exits still points into it. */
std::vector<ClDevice> &g_devices = *new std::vector<ClDevice>();

std::string device_string(cl_device_id id, cl_uint what) {
    size_t n = 0;
    if (g_cl.clGetDeviceInfo(id, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "";
    std::string s(n, '\0');
    if (g_cl.clGetDeviceInfo(id, what, n, &s[0], nullptr) != CL_SUCCESS) return "";
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

/* Open the OpenCL library and list its devices, once. Call with g_mu held. */
gpap_status load_devices() {
    if (g_loaded) return GPAP_OK;
#ifndef GPAP_HAVE_DLOPEN
    set_error("the opencl backend is not available on this platform");
    return GPAP_ERR_CUDA;
#else
    /* $GPAP_OPENCL_LIB names the library; otherwise the system's ICD loader. */
    const char *env = std::getenv("GPAP_OPENCL_LIB");
    void *lib = nullptr;
    if (env && *env) {
        lib = dlopen(env, RTLD_NOW | RTLD_LOCAL);
    } else {
        for (const char *name : {"libOpenCL.so.1", "libOpenCL.so"})
            if ((lib = dlopen(name, RTLD_NOW | RTLD_LOCAL)) != nullptr) break;
    }
    if (!lib) {
        set_error("the opencl backend needs an OpenCL library and found none (%s). Install the "
                  "OpenCL ICD loader and a driver, or point GPAP_OPENCL_LIB at libOpenCL.so "
                  "(docs/OPENCL.md)", env && *env ? env : "libOpenCL.so.1");
        return GPAP_ERR_CUDA;
    }
#define X(ret, name, args)                                                               \
    g_cl.name = reinterpret_cast<ret (*) args>(dlsym(lib, #name));                       \
    if (!g_cl.name) {                                                                    \
        set_error("the OpenCL library has no %s (OpenCL 1.2 is required)", #name);       \
        return GPAP_ERR_CUDA;                                                            \
    }
    GPAP_CL_FUNCTIONS(X)
#undef X

    cl_uint n_platforms = 0;
    cl_int rc = g_cl.clGetPlatformIDs(0, nullptr, &n_platforms);
    if (rc != CL_SUCCESS || n_platforms == 0) {
        set_error("OpenCL reports no platform (%d: %s). Is a driver (ICD) installed and "
                  "registered? `clinfo -l` lists what the loader sees (docs/OPENCL.md)",
                  (int)rc, cl_error_name(rc));
        return GPAP_ERR_CUDA;
    }
    std::vector<cl_platform_id> platforms(n_platforms);
    GPAP_CL(g_cl.clGetPlatformIDs(n_platforms, platforms.data(), nullptr));
    std::vector<ClDevice> gpus, others;
    for (cl_platform_id p : platforms) {
        cl_uint n = 0;
        if (g_cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &n) != CL_SUCCESS || n == 0)
            continue;                    /* a platform with no device is not an error */
        std::vector<cl_device_id> ids(n);
        if (g_cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, n, ids.data(), nullptr) != CL_SUCCESS)
            continue;
        for (cl_device_id id : ids) {
            ClDevice d;
            d.id = id;
            d.name = device_string(id, CL_DEVICE_NAME) + " (" +
                     device_string(id, CL_DEVICE_VERSION) + ")";
            cl_ulong mem = 0, max_alloc = 0;
            size_t group = 1;
            cl_bitfield type = 0;
            const bool known =
                g_cl.clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem,
                                     nullptr) == CL_SUCCESS &&
                g_cl.clGetDeviceInfo(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc),
                                     &max_alloc, nullptr) == CL_SUCCESS &&
                g_cl.clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(group), &group,
                                     nullptr) == CL_SUCCESS &&
                g_cl.clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(type), &type, nullptr) ==
                    CL_SUCCESS;
            if (!known || mem == 0) continue;   /* the batch is sized from these: no guessing */
            d.total_mem = (int64_t)mem;
            d.max_alloc = (int64_t)max_alloc;
            d.max_group = std::max<size_t>(group, 1);
            ((type & CL_DEVICE_TYPE_GPU) ? gpus : others).push_back(d);
        }
    }
    g_devices = gpus;
    g_devices.insert(g_devices.end(), others.begin(), others.end());
    if (g_devices.empty()) {
        set_error("OpenCL reports no device on any of its %u platform(s)", n_platforms);
        return GPAP_ERR_CUDA;
    }
    g_loaded = true;
    return GPAP_OK;
#endif
}

/* The device at `ordinal` (GPUs first). Call with g_mu held. */
gpap_status device_at(int ordinal, ClDevice **out) {
    GPAP_TRY(load_devices());
    if (ordinal < 0 || ordinal >= (int)g_devices.size()) {
        set_error("OpenCL device %d does not exist (%d device(s); 0 is %s)", ordinal,
                  (int)g_devices.size(), g_devices[0].name.c_str());
        return GPAP_ERR_ARG;
    }
    *out = &g_devices[(size_t)ordinal];
    return GPAP_OK;
}

/* The device's context and compiled program, built on first use. Call with g_mu held. */
gpap_status ensure_program(ClDevice *d) {
    if (d->program) return GPAP_OK;
    cl_int rc = CL_SUCCESS;
    cl_context ctx = g_cl.clCreateContext(nullptr, 1, &d->id, nullptr, nullptr, &rc);
    if (rc != CL_SUCCESS || !ctx) return cl_fail("clCreateContext", rc);
    const char *src = gpap_cl_source;
    const size_t len = gpap_cl_source_len;
    cl_program prog = g_cl.clCreateProgramWithSource(ctx, 1, &src, &len, &rc);
    if (rc != CL_SUCCESS || !prog) {
        g_cl.clReleaseContext(ctx);
        return cl_fail("clCreateProgramWithSource", rc);
    }
    rc = g_cl.clBuildProgram(prog, 1, &d->id, "", nullptr, nullptr);
    if (rc != CL_SUCCESS) {
        size_t n = 0;
        g_cl.clGetProgramBuildInfo(prog, d->id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        if (n) g_cl.clGetProgramBuildInfo(prog, d->id, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
        set_error("OpenCL: cpp/kernels/bcp.cl failed to compile on %s (%d). Build log:\n%.700s",
                  d->name.c_str(), (int)rc, log.c_str());
        g_cl.clReleaseProgram(prog);
        g_cl.clReleaseContext(ctx);
        return GPAP_ERR_KERNEL;
    }
    d->ctx = ctx;
    d->program = prog;
    return GPAP_OK;
}

/* ------------------------------------------------------------------ one open device */

class OpenClGpu final : public Gpu {
  public:
    OpenClGpu(const ClDevice *d, cl_command_queue queue, cl_mem null_buf)
        : d_(d), queue_(queue), null_buf_(null_buf), group_(std::min(kGroup, d->max_group)) {}

    ~OpenClGpu() override {
        for (auto &kv : kernels_) {
            for (auto &p : kv.second.params) g_cl.clReleaseMemObject(p.buf);
            g_cl.clReleaseKernel(kv.second.kernel);
        }
        g_cl.clReleaseMemObject(null_buf_);
        g_cl.clReleaseCommandQueue(queue_);
    }

    const char *backend() const override { return "opencl"; }
    const char *device_name() const override { return d_->name.c_str(); }
    gpap_status make_current() override { return GPAP_OK; }     /* OpenCL has no current device */

    gpap_status mem_info(int64_t *free_bytes, int64_t *total_bytes) override {
        *free_bytes = d_->total_mem;         /* OpenCL has no free-memory query */
        *total_bytes = d_->total_mem;
        return GPAP_OK;
    }

    bool free_is_measured() const override { return false; }     /* mem_info reports the total */

    int64_t max_buffer_bytes() const override { return d_->max_alloc; }

    gpap_status alloc(size_t bytes, GpuMem *out) override {
        cl_int rc = CL_SUCCESS;
        cl_mem m = g_cl.clCreateBuffer(d_->ctx, CL_MEM_READ_WRITE, bytes, nullptr, &rc);
        if (rc != CL_SUCCESS || !m) {
            set_error("OpenCL: clCreateBuffer(%zu bytes) failed with %d (%s); this device "
                      "allows %lld bytes in one buffer. Use a smaller batch",
                      bytes, (int)rc, cl_error_name(rc), (long long)d_->max_alloc);
            return GPAP_ERR_MEMORY;
        }
        *out = (GpuMem)m;
        return GPAP_OK;
    }

    void free(GpuMem m) override {
        if (m) g_cl.clReleaseMemObject((cl_mem)m);
    }

    gpap_status zero(GpuMem m, size_t bytes) override {
        const unsigned char z = 0;
        if (m && bytes)
            GPAP_CL(g_cl.clEnqueueFillBuffer(queue_, (cl_mem)m, &z, 1, 0, bytes, 0, nullptr,
                                             nullptr));
        return GPAP_OK;
    }

    gpap_status upload(GpuMem dst, const void *src, size_t bytes) override {
        if (bytes)
            GPAP_CL(g_cl.clEnqueueWriteBuffer(queue_, (cl_mem)dst, CL_TRUE, 0, bytes, src, 0,
                                              nullptr, nullptr));
        return GPAP_OK;
    }

    gpap_status download(void *dst, GpuMem src, size_t offset, size_t bytes) override {
        if (bytes)
            GPAP_CL(g_cl.clEnqueueReadBuffer(queue_, (cl_mem)src, CL_TRUE, offset, bytes, dst, 0,
                                             nullptr, nullptr));
        return GPAP_OK;
    }

    gpap_status copy(GpuMem dst, GpuMem src, size_t bytes) override {
        if (bytes)
            GPAP_CL(g_cl.clEnqueueCopyBuffer(queue_, (cl_mem)src, (cl_mem)dst, 0, 0, bytes, 0,
                                             nullptr, nullptr));
        return GPAP_OK;
    }

    gpap_status adopt(uintptr_t, GpuMem *) override {
        set_error("the opencl backend takes host arrays: a device pointer (the `_dev` entry "
                  "points) is a CUDA address and means nothing to OpenCL");
        return GPAP_ERR_ARG;
    }

    /* A WAVE kernel needs work groups of kWaveW items: a device that cannot form one does not
     * have it, and the engine then dispatches scalar. */
    bool has_kernel(const char *key) const override {
        const ClSpec *s = find_spec(key);
        return s && !(s->shape == LANES && (size_t)s->chunk > d_->max_group);
    }

    gpap_status run(const char *key, int64_t n0, int64_t n1,
                    std::initializer_list<GpuArg> args) override {
        if (n0 <= 0 || n1 <= 0) return GPAP_OK;             /* empty extent: nothing to do */
        Bound *k = nullptr;
        GPAP_TRY(bind(key, &k));

        /* The parameter block, then the buffers in order (the layout at the top of bcp.cl). */
        const Shape shape = k->spec->shape;
        const int chunk = k->spec->chunk;
        std::vector<int64_t> prm = {n0, n1, chunk};
        for (const GpuArg &a : args)
            if (a.kind != GpuArg::BUF) prm.push_back(a.value);
        cl_mem prm_buf = nullptr;
        GPAP_TRY(params(k, prm, &prm_buf));
        GPAP_CL(g_cl.clSetKernelArg(k->kernel, 0, sizeof(cl_mem), &prm_buf));
        std::vector<cl_mem> views;           /* copies standing in for offset arguments */
        cl_uint slot = 1;
        gpap_status st = GPAP_OK;
        for (const GpuArg &a : args) {
            if (a.kind != GpuArg::BUF) continue;
            cl_mem m = a.mem ? (cl_mem)a.mem : null_buf_;
            if (a.offset) {
                st = view(m, a.offset, &m);
                if (st != GPAP_OK) break;
                views.push_back(m);
            }
            const cl_int rc = g_cl.clSetKernelArg(k->kernel, slot++, sizeof(cl_mem), &m);
            if (rc != CL_SUCCESS) {
                st = cl_fail("clSetKernelArg", rc);
                break;
            }
        }
        if (st == GPAP_OK) {
            size_t global = 1, local = group_;
            if (shape == LANES) {                   /* a group per row */
                local = (size_t)chunk;
                global = (size_t)n0 * local;
            } else if (shape == ROWS) {
                global = round_up((size_t)n0, local);
            } else if (shape == GRID) {
                const size_t blocks = ((size_t)n1 + chunk - 1) / chunk;
                global = round_up((size_t)n0 * blocks, local);
            } else {
                local = global = std::min(kReduceGroup, d_->max_group);
            }
            const cl_int rc = g_cl.clEnqueueNDRangeKernel(queue_, k->kernel, 1, nullptr, &global,
                                                          &local, 0, nullptr, nullptr);
            if (rc != CL_SUCCESS) {
                set_error("OpenCL: launching kernel '%s' failed with %d (%s)", key, (int)rc,
                          cl_error_name(rc));
                st = GPAP_ERR_CUDA;
            }
        }
        /* The queued launch keeps its own reference to every buffer it was given. */
        for (cl_mem m : views) g_cl.clReleaseMemObject(m);
        return st;
    }

  private:
    struct Params {
        std::vector<int64_t> values;
        cl_mem buf;
    };
    struct Bound {
        const ClSpec *spec = nullptr;
        cl_kernel kernel = nullptr;
        std::vector<Params> params;         /* the few parameter blocks this kernel runs with */
    };

    static size_t round_up(size_t n, size_t to) { return (n + to - 1) / to * to; }

    gpap_status bind(const char *key, Bound **out) {
        auto it = kernels_.find(key);
        if (it != kernels_.end()) {
            *out = &it->second;
            return GPAP_OK;
        }
        const ClSpec *spec = find_spec(key);
        if (!spec) {
            set_error("kernel '%s' is not part of the opencl backend, which ships the fixpoint "
                      "kernels and the support kernels (no sampling trajectory). Use "
                      "GPAP_BACKEND=cuda for it", key);
            return GPAP_ERR_KERNEL;
        }
        if (!has_kernel(key)) {
            set_error("kernel '%s' needs work groups of %d items and %s allows %zu. Use the "
                      "scalar kernel (GPAP_KERNEL=scalar) on this device", key, spec->chunk,
                      d_->name.c_str(), d_->max_group);
            return GPAP_ERR_KERNEL;
        }
        std::string fn = std::string("gpap_") + key;
        std::replace(fn.begin(), fn.end(), '.', '_');
        cl_int rc = CL_SUCCESS;
        cl_kernel kernel = g_cl.clCreateKernel(d_->program, fn.c_str(), &rc);
        if (rc != CL_SUCCESS || !kernel) {
            set_error("OpenCL: kernel '%s' (function %s) is missing from bcp.cl: %d (%s)", key,
                      fn.c_str(), (int)rc, cl_error_name(rc));
            return GPAP_ERR_KERNEL;
        }
        Bound &b = kernels_[key];
        b.spec = spec;
        b.kernel = kernel;
        *out = &b;
        return GPAP_OK;
    }

    /* A device buffer holding `values`. A sweep launches the same kernel with the same few
     * parameter blocks over and over, so they are kept; each is written once, at creation, and
     * never again — a queued launch may still be reading it. */
    gpap_status params(Bound *k, const std::vector<int64_t> &values, cl_mem *out) {
        for (const Params &p : k->params)
            if (p.values == values) {
                *out = p.buf;
                return GPAP_OK;
            }
        cl_int rc = CL_SUCCESS;
        cl_mem m = g_cl.clCreateBuffer(d_->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       values.size() * sizeof(int64_t),
                                       const_cast<int64_t *>(values.data()), &rc);
        if (rc != CL_SUCCESS || !m) return cl_fail("clCreateBuffer(parameter block)", rc);
        if (k->params.size() >= 8) {         /* the oldest; a queued launch keeps its reference */
            g_cl.clReleaseMemObject(k->params.front().buf);
            k->params.erase(k->params.begin());
        }
        k->params.push_back({values, m});
        *out = m;
        return GPAP_OK;
    }

    /* A buffer holding `src` from `offset` on: what a kernel sees for an offset argument. A
     * copy, not a sub-buffer, whose offset would have to meet a device-specific alignment. */
    gpap_status view(cl_mem src, size_t offset, cl_mem *out) {
        size_t size = 0;
        GPAP_CL(g_cl.clGetMemObjectInfo(src, CL_MEM_SIZE, sizeof(size), &size, nullptr));
        if (offset >= size) {
            set_error("OpenCL: an argument offset of %zu is past its %zu-byte buffer", offset,
                      size);
            return GPAP_ERR_ARG;
        }
        cl_int rc = CL_SUCCESS;
        cl_mem m = g_cl.clCreateBuffer(d_->ctx, CL_MEM_READ_WRITE, size - offset, nullptr, &rc);
        if (rc != CL_SUCCESS || !m) return cl_fail("clCreateBuffer(argument view)", rc);
        rc = g_cl.clEnqueueCopyBuffer(queue_, src, m, offset, 0, size - offset, 0, nullptr,
                                      nullptr);
        if (rc != CL_SUCCESS) {
            g_cl.clReleaseMemObject(m);
            return cl_fail("clEnqueueCopyBuffer(argument view)", rc);
        }
        *out = m;
        return GPAP_OK;
    }

    const ClDevice *d_;
    cl_command_queue queue_;                /* in order: a launch sees every earlier one */
    cl_mem null_buf_;                       /* what a kernel gets for a "no buffer" argument */
    const size_t group_;                    /* work items per group, ROWS and GRID kernels */
    std::unordered_map<std::string, Bound> kernels_;
};

gpap_status opencl_open(int device, Gpu **out) {
    std::lock_guard<std::mutex> lock(g_mu);
    ClDevice *d = nullptr;
    GPAP_TRY(device_at(device, &d));
    GPAP_TRY(ensure_program(d));
    cl_int rc = CL_SUCCESS;
    cl_command_queue queue = g_cl.clCreateCommandQueue(d->ctx, d->id, 0, &rc);
    if (rc != CL_SUCCESS || !queue) return cl_fail("clCreateCommandQueue", rc);
    cl_mem null_buf = g_cl.clCreateBuffer(d->ctx, CL_MEM_READ_WRITE, 16, nullptr, &rc);
    if (rc != CL_SUCCESS || !null_buf) {
        g_cl.clReleaseCommandQueue(queue);
        return cl_fail("clCreateBuffer", rc);
    }
    *out = new OpenClGpu(d, queue, null_buf);
    return GPAP_OK;
}

gpap_status opencl_mem_info(int device, int64_t *free_bytes, int64_t *total_bytes) {
    std::lock_guard<std::mutex> lock(g_mu);
    ClDevice *d = nullptr;
    GPAP_TRY(device_at(device, &d));
    *free_bytes = d->total_mem;             /* OpenCL has no free-memory query */
    *total_bytes = d->total_mem;
    return GPAP_OK;
}

/* OpenCL cannot say how many warps a device keeps resident. The WAVE dispatch reads this for
 * formulas below its clause threshold only, so those always run scalar here. */
int32_t opencl_resident_warps(int) { return 0; }

}  // namespace

const GpuDriver kOpenClDriver = {"opencl", opencl_open, opencl_mem_info, opencl_resident_warps};

}  // namespace gpap

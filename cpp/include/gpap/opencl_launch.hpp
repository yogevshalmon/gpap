/* How cpp/kernels/bcp.cl is launched: THE launch rules of GPAP's OpenCL kernels, shared by
 * GPAP's own OpenCL plug (cpp/src/gpu_opencl.cpp) and any host plug that runs bcp.cl through
 * its own GPU layer (gpu_api.hpp). Header-only, and no OpenCL dependency.
 *
 * To run a kernel KEY over a work extent (n0, n1) with Gpu::run's argument list, a plug:
 *   1. calls the kernel function function_name(key): "gpap_" + key with '.' as '_';
 *   2. passes as argument 0 a read-only buffer holding param_block(spec, n0, n1, args): the
 *      extent, the spec's chunk, then every scalar argument in order, all as int64 (the layout
 *      at the top of bcp.cl);
 *   3. passes every buffer argument, in order, as arguments 1, 2, ...: a "no buffer" (0) as any
 *      small buffer, and a buffer with an offset as a copy of it from that offset on (an input
 *      view, gpu_api.hpp);
 *   4. enqueues a one-dimensional range of launch(spec, n0, n1, max_group).global work items in
 *      groups of .local, where max_group is the device's largest work group.
 * A plug built against this header follows a change to any of these. */
#ifndef GPAP_OPENCL_LAUNCH_HPP
#define GPAP_OPENCL_LAUNCH_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include "gpap/gpu_api.hpp"

namespace gpap {
namespace opencl {

/* How a work extent (n0, n1) becomes a launch. Always one-dimensional: an OpenCL built over
 * CUDA (PoCL's driver) limits every dimension but the first to 65,535 work groups, and a batch
 * has more rows than that. */
enum Shape {
    ROWS,    /* one work item per element of axis 0 */
    GRID,    /* one work item per (row, block of `chunk` columns), numbered row * blocks + block */
    GROUP,   /* a single work group, whatever the extent (the two reductions) */
    LANES    /* a work group of `chunk` items per element of axis 0, numbered row * chunk + lane */
};

constexpr int kWaveW = 32;               /* GPAP_WAVE_W in bcp.cl: the lanes of a WAVE row */

struct Spec {
    const char *key;
    Shape shape;
    int chunk;
};

/* THE list of kernels bcp.cl defines, as `gpap_<key with . as _>`. A key missing here is a
 * kernel the OpenCL backend does not have (the trajectory).
 * A scalar fixpoint kernel (.w0) runs a row on one work item, a WAVE one (.w1) on a work group
 * of kWaveW. A row per work item, not per warp, because the scalar kernel is dispatched on
 * small formulas, where a warp per row loses (docs/findings/OPENCL_FINDINGS.md §9).
 * bcast_rows copies 4 columns per work item (measured: §8). */
inline constexpr Spec kSpecs[] = {
    {"fixpoint.s16.w0", ROWS, 1},             {"fixpoint.s8.w0", ROWS, 1},
    {"fixpoint_clean.s16.w0", ROWS, 1},       {"fixpoint_clean.s8.w0", ROWS, 1},
    {"fixpoint.s16.w1", LANES, kWaveW},       {"fixpoint.s8.w1", LANES, kWaveW},
    {"fixpoint_clean.s16.w1", LANES, kWaveW}, {"fixpoint_clean.s8.w1", LANES, kWaveW},
    {"bcast_rows", GRID, 4},                  {"fill_i8", ROWS, 1},
    {"complete_mask", ROWS, 1},               {"sound_rows.s16", ROWS, 1},
    {"sound_rows.s8", ROWS, 1},               {"var_bits", GRID, 64},
    {"unpack_rows", GRID, 64},                {"sum_i8_to", GROUP, 1},
    {"sum_i32_acc", GROUP, 1},
};

constexpr size_t kGroup = 64;            /* work-group size of the ROWS and GRID kernels (the
                                          * size made no difference: §8) */
constexpr size_t kReduceGroup = 256;     /* GPAP_GROUP_MAX in bcp.cl */

/* The spec of `key`, or NULL for a key bcp.cl does not define. */
inline const Spec *find_spec(const char *key) {
    for (const Spec &s : kSpecs)
        if (std::strcmp(key, s.key) == 0) return &s;
    return nullptr;
}

/* Can a device whose largest work group is `max_group` run the kernel? A WAVE kernel needs a
 * work group of kWaveW items; a device that cannot form one dispatches scalar instead. */
inline bool fits(const Spec &s, size_t max_group) {
    return !(s.shape == LANES && (size_t)s.chunk > max_group);
}

/* The kernel function bcp.cl defines for `key`. */
inline std::string function_name(const char *key) {
    std::string fn = std::string("gpap_") + key;
    std::replace(fn.begin(), fn.end(), '.', '_');
    return fn;
}

/* Argument 0 of every kernel: the extent, the chunk, then the scalars of `args` in order. */
inline std::vector<int64_t> param_block(const Spec &s, int64_t n0, int64_t n1,
                                        std::initializer_list<GpuArg> args) {
    std::vector<int64_t> values = {n0, n1, s.chunk};
    for (const GpuArg &a : args)
        if (a.kind != GpuArg::BUF) values.push_back(a.value);
    return values;
}

struct Launch {
    size_t global;   /* work items, a multiple of `local` */
    size_t local;    /* work items per group */
};

/* The one-dimensional launch of a non-empty extent (n0, n1 >= 1). */
inline Launch launch(const Spec &s, int64_t n0, int64_t n1, size_t max_group) {
    const auto round_up = [](size_t n, size_t to) { return (n + to - 1) / to * to; };
    const size_t group = std::min(kGroup, max_group);
    switch (s.shape) {
    case LANES:                                    /* a group per row */
        return {(size_t)n0 * (size_t)s.chunk, (size_t)s.chunk};
    case ROWS:
        return {round_up((size_t)n0, group), group};
    case GRID: {
        const size_t blocks = ((size_t)n1 + s.chunk - 1) / s.chunk;
        return {round_up((size_t)n0 * blocks, group), group};
    }
    case GROUP:
    default: {
        const size_t reduce = std::min(kReduceGroup, max_group);
        return {reduce, reduce};
    }
    }
}

}  // namespace opencl
}  // namespace gpap

#endif /* GPAP_OPENCL_LAUNCH_HPP */

/* BCPEngine — K-parallel packed BCP: state ownership, kernel launches, readout.
 *
 * Port of gpap/engine/bcp_engine.py, and now the ONLY implementation — the Python engine is a
 * facade over this. All row state is allocated with the driver API and owned here, so the
 * library has no PyTorch dependency and C++ consumers get the same engine Python gets.
 * Performance is a TIE with the retired Python engine (geomean 1.013x, identical model sets):
 * same AOT-vs-JIT cubins, bit-identical SASS. An earlier "owning the memory is worth 1.2-1.35x"
 * reading was retracted — it was a device-memory PLACEMENT effect, see
 * docs/findings/CPP_BACKEND_FINDINGS.md §4 and the state-arena follow-up in
 * docs/plans/CPP_CORE_PLAN.md §8.
 *
 * State layout, unchanged from the Python engine:
 *   packed  (K, M)  int32  low SHIFT bits = sat_count, high bits = modular literal sum
 *   abits   (K, NW) int32  bit l = literal l is TRUE (l and l^1 share a word)
 *   active / conflict (K,) int8
 * The post-units-flush state is computed ONCE at construction and broadcast-restored by
 * reset(K) — no per-chunk unit flush.
 */
#include "gpap_internal.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using gpap::DevBuf;
using gpap::Kernel;
using gpap::launch;
using gpap::set_error;

namespace {

/* Decision-batch floor used when sizing the trail. See trail_cap_for(). */
constexpr int32_t kDefaultBFloor = 64;
constexpr size_t kReadoutSliceBytes = 256u << 20;   /* bound a (rows, 2N) readout transient */

inline unsigned ceil_div(int64_t a, int64_t b) { return (unsigned)((a + b - 1) / b); }

std::string traj_key(int32_t shift, int wave, int vscan, int soft) {
    return "trajectory.s" + std::to_string(shift) + ".w" + std::to_string(wave) + ".v" +
           std::to_string(vscan) + ".f" + std::to_string(soft);
}

/* Accumulate one (K,) int32 per-launch counter into its int64 device accumulator (no sync). */
gpap_status accumulate(gpap_engine *e, DevBuf &src, DevBuf &acc, int32_t n) {
    Kernel k;
    GPAP_TRY(e->kernels.get("sum_i32_acc", &k));
    return launch(k, 1, 1, {&src.ptr, &n, &acc.ptr});
}

gpap_status fill_flags(gpap_engine *e, DevBuf &buf, int32_t value, int32_t n) {
    Kernel k;
    GPAP_TRY(e->kernels.get("fill_i8", &k));
    return launch(k, ceil_div(n, 1024), 1, {&buf.ptr, &value, &n});
}

gpap_status broadcast_rows(gpap_engine *e, DevBuf &dst, DevBuf &src, int32_t n_cols, int32_t K) {
    Kernel k;
    GPAP_TRY(e->kernels.get("bcast_rows", &k));
    return launch(k, (unsigned)K, ceil_div(n_cols, 1024), {&dst.ptr, &src.ptr, &n_cols});
}

/* Upload `n` bytes into a scratch buffer, growing it as needed. */
gpap_status stage(DevBuf &buf, const void *host, size_t n) {
    GPAP_TRY(buf.reserve(n));
    if (n) GPAP_CU(cuMemcpyHtoD(buf.ptr, host, n));
    return GPAP_OK;
}

/* Materialize the ascending row indices of a selection (host + device). */
gpap_status ensure_rows(gpap_engine *e, int32_t slot) {
    Selection &s = e->slots[slot];
    if (s.rows_valid) return GPAP_OK;
    std::vector<int8_t> mask((size_t)e->K);
    if (e->K) GPAP_CU(cuMemcpyDtoH(mask.data(), s.mask.ptr, (size_t)e->K));
    s.host_rows.clear();
    s.host_rows.reserve((size_t)s.count);
    for (int32_t i = 0; i < e->K; ++i)
        if (mask[(size_t)i]) s.host_rows.push_back(i);
    s.count = (int32_t)s.host_rows.size();
    GPAP_TRY(stage(s.rows, s.host_rows.data(), s.host_rows.size() * sizeof(int32_t)));
    s.rows_valid = true;
    return GPAP_OK;
}

gpap_status use_ctx(gpap_engine *e) {
    GPAP_CU(cuCtxSetCurrent(e->ctx));
    return GPAP_OK;
}

}  // namespace

gpap_engine::~gpap_engine() {
    if (ctx) {
        cuCtxSetCurrent(ctx);
        for (auto *b : pair_masks) delete b;
        pair_masks.clear();
    }
}

/* ------------------------------------------------------------------ construction */

static gpap_status build_flush_cache(gpap_engine *e, const gpap_formula *f) {
    /* One fixpoint launch at K=1 seeded with the formula's own unit literals; the result is
     * cached and broadcast-restored by every reset(K). An empty clause (or contradictory
     * units) conflicts the flush itself — reset() then marks every row conflicted. */
    GPAP_TRY(gpap_engine_reset(e, 1));
    GPAP_TRY(broadcast_rows(e, e->packed, e->packed_init, e->num_clauses, 1));
    GPAP_TRY(e->abits.zero());
    GPAP_TRY(fill_flags(e, e->active, e->has_empty_clause ? 0 : 1, 1));
    GPAP_TRY(fill_flags(e, e->conflict, e->has_empty_clause ? 1 : 0, 1));

    if (!e->has_empty_clause && !f->unit_lits.empty()) {
        std::vector<int64_t> units(f->unit_lits.begin(), f->unit_lits.end());
        GPAP_TRY(gpap_engine_apply(e, units.data(), (int32_t)units.size()));
    }
    GPAP_TRY(e->packed_flushed.reserve((size_t)e->num_clauses * sizeof(int32_t)));
    GPAP_TRY(e->abits_flushed.reserve((size_t)e->nw * sizeof(int32_t)));
    GPAP_CU(cuMemcpyDtoD(e->packed_flushed.ptr, e->packed.ptr,
                         (size_t)e->num_clauses * sizeof(int32_t)));
    GPAP_CU(cuMemcpyDtoD(e->abits_flushed.ptr, e->abits.ptr, (size_t)e->nw * sizeof(int32_t)));
    int8_t conflict0 = 0;
    GPAP_CU(cuMemcpyDtoH(&conflict0, e->conflict.ptr, 1));
    e->flush_conflict = conflict0 ? 1 : 0;
    e->flush_ready = 1;

    gpap_counters_reset(e);
    return GPAP_OK;
}

extern "C" gpap_status gpap_engine_create(const gpap_formula *f, const gpap_engine_options *opts,
                                          gpap_engine **out) {
    if (!f || !out) return GPAP_ERR_ARG;
    *out = nullptr;
    /* All-zero, i.e. exactly what a C caller writing `gpap_engine_options o = {0}` gets — the
     * header documents both spellings as "the defaults" and they must not differ. This field
     * used to be -1 here, so passing NULL silently turned the warp-wide decision scan OFF while
     * `{0}` left it on (worth 12-19% on wave instances, same picks either way). */
    const gpap_engine_options defaults = {0, GPAP_MODE_SCALAR, 0, 0};
    const gpap_engine_options o = opts ? *opts : defaults;
    const int device = o.device;
    GPAP_CU(cuInit(0));
    gpap_engine *e = new gpap_engine();
    e->device_ordinal = device;
    gpap_status st = GPAP_OK;
    do {
        CUresult rc = cuDeviceGet(&e->device, device);
        if (rc != CUDA_SUCCESS) { set_error("cuDeviceGet(%d) failed", device); st = GPAP_ERR_CUDA; break; }
        /* PRIMARY context: the one the CUDA runtime (hence PyTorch) uses, so a host process
         * mixing the two never pays a context switch. */
        rc = cuDevicePrimaryCtxRetain(&e->ctx, e->device);
        if (rc != CUDA_SUCCESS) { set_error("cuDevicePrimaryCtxRetain failed"); st = GPAP_ERR_CUDA; break; }
        cuCtxSetCurrent(e->ctx);

        if (f->shift == 0) {   /* formula shape fits no packed split (see gpap_packed_envelope) */
            gpap_packed_envelope(f->num_vars, f->max_clause_len, nullptr);   /* sets the message */
            st = GPAP_ERR_ENVELOPE;
            break;
        }
        if (o.force_shift) {   /* pin a split (A/B testing); must still be non-overflowing */
            const int32_t fs = o.force_shift;
            const gpap::PackedSplit *sp = nullptr;
            for (const auto &cand : gpap::kPackedSplits)
                if (cand.shift == fs) sp = &cand;
            if (!sp || 2ll * f->num_vars > sp->max_two_n || f->max_clause_len > sp->max_len) {
                set_error("force_shift=%d does not fit this formula (2N=%lld, max len=%d)", fs,
                          2ll * f->num_vars, f->max_clause_len);
                st = GPAP_ERR_ENVELOPE;
                break;
            }
        }
        e->num_vars = f->num_vars;
        e->num_clauses = f->num_clauses;
        e->shift = o.force_shift ? o.force_shift : f->shift;
        e->total_lits = f->total_lits;
        e->max_clause_len = f->max_clause_len;
        e->has_empty_clause = f->has_empty_clause;
        e->mode = o.mode;
        /* 0 = default, 1 = on, -1 = off. The zero value MUST mean the default: a C caller
         * writing `gpap_engine_options o = {0}` would otherwise silently lose the warp-wide
         * decision scan (worth 12-19% on wave instances) while the header promises otherwise. */
        e->vscan = o.vscan < 0 ? 0 : 1;
        e->nw = std::max(1, (2 * f->num_vars + 31) / 32);
        e->trail_cap = f->num_vars + kDefaultBFloor + 8;

        st = stage(e->occ_off, f->occ_offsets.data(), f->occ_offsets.size() * sizeof(int32_t));
        if (st != GPAP_OK) break;
        st = stage(e->occ_cls, f->occ_clauses.data(), f->occ_clauses.size() * sizeof(int32_t));
        if (st != GPAP_OK) break;
        if (e->shift == f->shift) {
            st = stage(e->packed_init, f->packed_init.data(),
                       f->packed_init.size() * sizeof(int32_t));
        } else {   /* pinned to the other split: re-pack from clause_sizes + lit_sum */
            std::vector<int32_t> pk((size_t)f->num_clauses);
            const uint32_t mask = (uint32_t)((1ull << (32 - e->shift)) - 1);
            for (int32_t c = 0; c < f->num_clauses; ++c)
                pk[(size_t)c] = (int32_t)((uint32_t)f->clause_sizes[(size_t)c] |
                                          (((uint32_t)f->lit_sum[(size_t)c] & mask) << e->shift));
            st = stage(e->packed_init, pk.data(), pk.size() * sizeof(int32_t));
        }
        if (st != GPAP_OK) break;
        st = e->acc_ops.reserve(8);
        if (st == GPAP_OK) st = e->acc_props.reserve(8);
        if (st == GPAP_OK) st = e->acc_decs.reserve(8);
        if (st == GPAP_OK) st = e->scratch_i32.reserve(64);
        if (st != GPAP_OK) break;
        st = build_flush_cache(e, f);
    } while (0);

    if (st != GPAP_OK) {
        gpap_engine_free(e);
        return st;
    }
    *out = e;
    return GPAP_OK;
}

extern "C" void gpap_engine_free(gpap_engine *e) {
    if (!e) return;
    CUdevice dev = e->device;
    bool had_ctx = e->ctx != nullptr;
    delete e;                       /* releases every DevBuf + the kernel modules */
    if (had_ctx) cuDevicePrimaryCtxRelease(dev);
}

extern "C" gpap_status gpap_engine_set_mode(gpap_engine *e, gpap_mode mode) {
    if (!e) return GPAP_ERR_ARG;
    e->mode = mode;
    return GPAP_OK;
}

extern "C" gpap_mode gpap_engine_mode(const gpap_engine *e) { return e ? e->mode : GPAP_MODE_SCALAR; }
extern "C" int32_t gpap_engine_shift(const gpap_engine *e) { return e ? e->shift : 0; }
extern "C" int32_t gpap_engine_k(const gpap_engine *e) { return e ? e->K : 0; }
extern "C" int32_t gpap_engine_num_vars(const gpap_engine *e) { return e ? e->num_vars : 0; }
extern "C" int32_t gpap_engine_num_clauses(const gpap_engine *e) {
    return e ? e->num_clauses : 0;
}
extern "C" int gpap_engine_flush_conflict(const gpap_engine *e) { return e ? e->flush_conflict : 0; }

/* ------------------------------------------------------------------ reset */

/* Trail slots one row can ever need.
 *
 * The trail holds literals that have been made TRUE and not yet propagated. EVERY push is
 * guarded by the assignment bitmap: the scalar seed loop skips a literal whose bit is already
 * set and conflicts the row when the opposite bit is, and the wave path uses the `atomic_or`
 * RETURN value for the same two tests. Since l and ~l share a word, a variable can contribute
 * AT MOST ONE trail entry before the row conflicts. So N + a little slack is sufficient, for
 * any number of decisions and any assumption-set width.
 *
 * This is worth stating precisely because getting it wrong is expensive and SILENT: the cap
 * used to be `num_vars + max(L, 64) + 8` where L was the width of whatever was last applied,
 * and it only ever grows. The construction-time units flush applies L = (number of unit
 * clauses), so a formula with many units permanently inflated every later chunk —
 * 10.sk_1_46 (1,121 units, N=1,494) ran at trail_cap 2,623 instead of 1,566, costing +27%
 * of per-row device memory for the life of the engine, on every chunk, for nothing. */
static inline int32_t trail_cap_for(int32_t num_vars, int32_t B) {
    return num_vars + std::max(B, kDefaultBFloor) + 8;
}

static gpap_status ensure_state(gpap_engine *e, int32_t K, int32_t B) {
    const int32_t cap = trail_cap_for(e->num_vars, B);
    if (cap > e->trail_cap) e->trail_cap = cap;
    GPAP_TRY(e->packed.reserve((size_t)K * e->num_clauses * sizeof(int32_t)));
    GPAP_TRY(e->abits.reserve((size_t)K * e->nw * sizeof(int32_t)));
    GPAP_TRY(e->active.reserve((size_t)K));
    GPAP_TRY(e->conflict.reserve((size_t)K));
    GPAP_TRY(e->trail.reserve((size_t)K * e->trail_cap * sizeof(int32_t)));
    for (DevBuf *b : {&e->nprops, &e->nops, &e->ndecs, &e->nrounds})
        GPAP_TRY(b->reserve((size_t)K * sizeof(int32_t)));
    for (int i = 0; i < GPAP_SLOTS; ++i) GPAP_TRY(e->slots[i].mask.reserve((size_t)K));
    return GPAP_OK;
}

extern "C" gpap_status gpap_engine_reset(gpap_engine *e, int32_t K) {
    if (!e || K <= 0) return GPAP_ERR_ARG;
    GPAP_TRY(use_ctx(e));
    GPAP_TRY(ensure_state(e, K, kDefaultBFloor));
    e->K = K;
    for (int i = 0; i < GPAP_SLOTS; ++i) { e->slots[i].rows_valid = false; e->slots[i].count = 0; }
    if (!e->flush_ready) return GPAP_OK;                 /* construction path: caller seeds */
    GPAP_TRY(broadcast_rows(e, e->packed, e->packed_flushed, e->num_clauses, K));
    GPAP_TRY(broadcast_rows(e, e->abits, e->abits_flushed, e->nw, K));
    GPAP_TRY(fill_flags(e, e->active, e->flush_conflict ? 0 : 1, K));
    GPAP_TRY(fill_flags(e, e->conflict, e->flush_conflict ? 1 : 0, K));
    return GPAP_OK;
}

extern "C" gpap_status gpap_engine_recache_flush(gpap_engine *e) {
    if (!e) return GPAP_ERR_ARG;
    GPAP_TRY(use_ctx(e));
    GPAP_CU(cuMemcpyDtoD(e->packed_flushed.ptr, e->packed.ptr,
                         (size_t)e->num_clauses * sizeof(int32_t)));
    GPAP_CU(cuMemcpyDtoD(e->abits_flushed.ptr, e->abits.ptr, (size_t)e->nw * sizeof(int32_t)));
    int8_t c0 = 0;
    GPAP_CU(cuMemcpyDtoH(&c0, e->conflict.ptr, 1));
    e->flush_conflict = c0 ? 1 : 0;
    return GPAP_OK;
}

/* ------------------------------------------------------------------ entry point 1 */

extern "C" gpap_status gpap_engine_apply_dev(gpap_engine *e, uintptr_t decisions_dev, int32_t L) {
    if (!e || L < 0 || (L > 0 && !decisions_dev)) return GPAP_ERR_ARG;
    if (L == 0) return GPAP_OK;                     /* no assumptions: nothing to propagate */
    GPAP_TRY(use_ctx(e));
    /* kDefaultBFloor, NOT L: the trail is bounded by the variable count, not by how many
     * literals were assumed (dedup — see trail_cap_for). Passing L here made a wide apply
     * permanently inflate the per-row footprint. */
    GPAP_TRY(ensure_state(e, e->K, kDefaultBFloor));
    /* Inactive rows are left unwritten by the kernel; stale values would corrupt the sums. */
    GPAP_TRY(e->nprops.zero());
    GPAP_TRY(e->nops.zero());
    GPAP_TRY(e->ndecs.zero());

    Kernel k;
    const std::string key = "fixpoint.s" + std::to_string(e->shift) + ".w" +
                            std::to_string(e->mode == GPAP_MODE_WAVE ? 1 : 0);
    GPAP_TRY(e->kernels.get(key, &k));
    CUdeviceptr dec = (CUdeviceptr)decisions_dev;
    int32_t M = e->num_clauses, NW = e->nw, B = L, cap = e->trail_cap;
    GPAP_TRY(launch(k, (unsigned)e->K, 1,
                    {&e->packed.ptr, &e->abits.ptr, &e->active.ptr, &e->conflict.ptr, &dec,
                     &e->occ_off.ptr, &e->occ_cls.ptr, &e->trail.ptr, &e->nprops.ptr,
                     &e->nops.ptr, &e->ndecs.ptr, &M, &NW, &B, &cap}));
    e->launches++;
    GPAP_TRY(accumulate(e, e->nops, e->acc_ops, e->K));
    GPAP_TRY(accumulate(e, e->nprops, e->acc_props, e->K));
    GPAP_TRY(accumulate(e, e->ndecs, e->acc_decs, e->K));
    for (int i = 0; i < GPAP_SLOTS; ++i) e->slots[i].rows_valid = false;
    return GPAP_OK;
}

extern "C" gpap_status gpap_engine_apply(gpap_engine *e, const int64_t *decisions_host,
                                         int32_t L) {
    if (!e || !decisions_host || L <= 0) return GPAP_ERR_ARG;
    GPAP_TRY(use_ctx(e));
    GPAP_TRY(stage(e->staging_a, decisions_host, (size_t)e->K * L * sizeof(int64_t)));
    return gpap_engine_apply_dev(e, (uintptr_t)e->staging_a.ptr, L);
}

/* ------------------------------------------------------------------ entry point 2 */

extern "C" gpap_status gpap_engine_trajectory_dev(gpap_engine *e, uintptr_t order_dev, int32_t P,
                                                  int32_t n_support, int32_t B, int64_t seed,
                                                  int soft, int32_t *max_rounds_out) {
    if (!e || P < 0 || B <= 0 || (P > 0 && !order_dev)) return GPAP_ERR_ARG;
    GPAP_TRY(use_ctx(e));
    if (P == 0) {
        /* Nothing to decide (a formula whose variables are all determined, e.g. after unit
         * propagation): every active row is already a complete model. */
        if (max_rounds_out) *max_rounds_out = 0;
        for (int i = 0; i < GPAP_SLOTS; ++i) e->slots[i].rows_valid = false;
        return GPAP_OK;
    }
    GPAP_TRY(ensure_state(e, e->K, kDefaultBFloor));
    GPAP_TRY(e->nprops.zero());
    GPAP_TRY(e->nops.zero());
    GPAP_TRY(e->ndecs.zero());
    GPAP_TRY(e->nrounds.zero());

    Kernel k;
    GPAP_TRY(e->kernels.get(traj_key(e->shift, e->mode == GPAP_MODE_WAVE ? 1 : 0,
                                     e->vscan ? 1 : 0, soft ? 1 : 0), &k));
    CUdeviceptr order = (CUdeviceptr)order_dev;
    int32_t M = e->num_clauses, NW = e->nw, cap = e->trail_cap, max_rounds = 1 << 30;
    int64_t seed64 = seed;
    GPAP_TRY(launch(k, (unsigned)e->K, 1,
                    {&e->packed.ptr, &e->abits.ptr, &e->active.ptr, &e->conflict.ptr, &order,
                     &e->occ_off.ptr, &e->occ_cls.ptr, &e->trail.ptr, &e->nrounds.ptr,
                     &e->nops.ptr, &e->nprops.ptr, &e->ndecs.ptr, &M, &NW, &P, &n_support, &B,
                     &cap, &max_rounds, &seed64}));
    e->launches++;
    GPAP_TRY(accumulate(e, e->nops, e->acc_ops, e->K));
    GPAP_TRY(accumulate(e, e->nprops, e->acc_props, e->K));
    GPAP_TRY(accumulate(e, e->ndecs, e->acc_decs, e->K));
    for (int i = 0; i < GPAP_SLOTS; ++i) e->slots[i].rows_valid = false;

    if (max_rounds_out) {                                   /* one sync, as run_trajectory does */
        *max_rounds_out = 0;
        if (e->K > 0) {
            std::vector<int32_t> rounds((size_t)e->K);
            GPAP_CU(cuMemcpyDtoH(rounds.data(), e->nrounds.ptr, (size_t)e->K * sizeof(int32_t)));
            *max_rounds_out = *std::max_element(rounds.begin(), rounds.end());
        }
    }
    return GPAP_OK;
}

extern "C" gpap_status gpap_engine_trajectory(gpap_engine *e, const int32_t *order_host, int32_t P,
                                              int32_t n_support, int32_t B, int64_t seed, int soft,
                                              int32_t *max_rounds_out) {
    if (!e || !order_host) return GPAP_ERR_ARG;
    GPAP_TRY(use_ctx(e));
    GPAP_TRY(stage(e->staging_a, order_host, (size_t)P * sizeof(int32_t)));
    return gpap_engine_trajectory_dev(e, (uintptr_t)e->staging_a.ptr, P, n_support, B, seed, soft,
                                      max_rounds_out);
}

/* Selections and readout live in readout.cpp; these helpers are shared with it. */
namespace gpap {
gpap_status engine_use_ctx(gpap_engine *e) { return use_ctx(e); }
gpap_status engine_stage(DevBuf &buf, const void *host, size_t n) { return stage(buf, host, n); }
gpap_status engine_ensure_rows(gpap_engine *e, int32_t slot) { return ensure_rows(e, slot); }
size_t engine_readout_slice_bytes() { return kReadoutSliceBytes; }
unsigned engine_ceil_div(int64_t a, int64_t b) { return ceil_div(a, b); }
}  // namespace gpap

/* ------------------------------------------------------------------ counters + info */

extern "C" gpap_status gpap_counters_get(gpap_engine *e, gpap_counters *out) {
    if (!e || !out) return GPAP_ERR_ARG;
    GPAP_TRY(use_ctx(e));
    int64_t v[3] = {0, 0, 0};
    GPAP_CU(cuMemcpyDtoH(&v[0], e->acc_ops.ptr, sizeof(int64_t)));
    GPAP_CU(cuMemcpyDtoH(&v[1], e->acc_props.ptr, sizeof(int64_t)));
    GPAP_CU(cuMemcpyDtoH(&v[2], e->acc_decs.ptr, sizeof(int64_t)));
    out->ops = v[0];
    out->props = v[1];
    out->decs = v[2];
    out->launches = e->launches;
    return GPAP_OK;
}

extern "C" void gpap_counters_reset(gpap_engine *e) {
    if (!e) return;
    cuCtxSetCurrent(e->ctx);
    e->acc_ops.zero();
    e->acc_props.zero();
    e->acc_decs.zero();
    e->launches = 0;
}

/* The K-parallel packed BCP engine: state ownership, kernel launches, readout. The only
 * implementation — the Python engine is a facade over it — and all row state is owned here
 * (docs/findings/CPP_BACKEND_FINDINGS.md). State layout, where M_long counts clauses of length
 * >= 3 (binary clauses are implication edges in the occ list with no per-row state, units live
 * in the flush cache — see formula.cpp):
 *   packed  (K, M_long) int32  low SHIFT bits = sat_count, high bits = modular literal sum
 *   abits   (K, NW)     int32  bit l = literal l is TRUE (l and l^1 share a word)
 *   active / conflict (K,) int8
 * The post-units-flush state is computed once at construction and broadcast-restored by
 * reset(K) — no per-chunk unit flush. */
#include "gpap_internal.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using gpap::DevBuf;
using gpap::Kernel;
using gpap::launch;
using gpap::set_error;

namespace {

constexpr size_t kReadoutSliceBytes = 256u << 20;   /* bound a (rows, 2N) readout transient */

inline unsigned ceil_div(int64_t a, int64_t b) { return (unsigned)((a + b - 1) / b); }

/* VSCAN is read only under `WAVE and not SOFT`, so the other pairs compile once at v0 — asking
 * for v1 there asks for an unshipped cubin. Mirrors build_kernels.vscan_is_read. */
/* The kernel choice for K rows: an explicit $GPAP_KERNEL pin first, else the dispatch rule.
 * One definition — gpap_engine_reset and gpap_engine_set_mode(AUTO) both call it. */
gpap_mode resolve_mode(gpap_engine *e, int32_t K) {
    if (e->env_mode != GPAP_MODE_AUTO) return e->env_mode;
    return gpap_dispatch_wave(e->num_clauses, K, e->device_ordinal) ? GPAP_MODE_WAVE
                                                                    : GPAP_MODE_SCALAR;
}

std::string traj_key(int32_t shift, int wave, int vscan, int soft) {
    const int v = (wave && !soft) ? (vscan ? 1 : 0) : 0;
    return "trajectory.s" + std::to_string(shift) + ".w" + std::to_string(wave) + ".v" +
           std::to_string(v) + ".f" + std::to_string(soft);
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
    if (n_cols <= 0 || K <= 0) return GPAP_OK;   /* num_long == 0 is legal (pure <=2-CNF) */
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
    if (!s.selected) {           /* never selected into since the reset: EMPTY, not the mask's
                                    uninitialized bytes compacted into rows */
        s.host_rows.clear();
        s.count = 0;
        s.rows_valid = true;
        return GPAP_OK;
    }
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

static gpap_status build_flush_cache(gpap_engine *e, const gpap_formula *f) {
    /* One K=1 fixpoint launch over the formula's own units, cached and broadcast-restored by
     * every reset(K); contradictory units conflict the flush itself. */
    GPAP_TRY(gpap_engine_reset(e, 1));
    GPAP_TRY(broadcast_rows(e, e->packed, e->packed_init, e->num_long, 1));
    GPAP_TRY(e->abits.zero());
    GPAP_TRY(fill_flags(e, e->active, e->has_empty_clause ? 0 : 1, 1));
    GPAP_TRY(fill_flags(e, e->conflict, e->has_empty_clause ? 1 : 0, 1));

    if (!e->has_empty_clause && !f->unit_lits.empty()) {
        std::vector<int64_t> units(f->unit_lits.begin(), f->unit_lits.end());
        GPAP_TRY(gpap_engine_apply(e, units.data(), (int32_t)units.size()));
    }
    if (e->num_long) {
        GPAP_TRY(e->packed_flushed.reserve((size_t)e->num_long * sizeof(int32_t)));
        GPAP_CU(cuMemcpyDtoD(e->packed_flushed.ptr, e->packed.ptr,
                             (size_t)e->num_long * sizeof(int32_t)));
    }
    GPAP_TRY(e->abits_flushed.reserve((size_t)e->nw * sizeof(int32_t)));
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
    /* All-zero, exactly what `gpap_engine_options o = {0}` gets: the header calls both
     * spellings "the defaults" and they must not differ. */
    const gpap_engine_options defaults = {0, GPAP_MODE_AUTO, 0, 0};
    const gpap_engine_options o = opts ? *opts : defaults;
    const int device = o.device;
    GPAP_CU(cuInit(0));
    gpap_engine *e = new gpap_engine();
    e->device_ordinal = device;
    gpap_status st = GPAP_OK;
    do {
        CUresult rc = cuDeviceGet(&e->device, device);
        if (rc != CUDA_SUCCESS) { set_error("cuDeviceGet(%d) failed", device); st = GPAP_ERR_CUDA; break; }
        /* Primary context: the one the CUDA runtime uses, so mixing the two costs no switch. */
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
        e->num_long = f->num_long;
        e->shift = o.force_shift ? o.force_shift : f->shift;
        e->total_lits = f->total_lits;
        e->max_clause_len = f->max_clause_len;
        e->has_empty_clause = f->has_empty_clause;
        /* AUTO is an input value only: reset(K) re-resolves `mode` before any launch. */
        e->auto_mode = (o.mode == GPAP_MODE_AUTO) ? 1 : 0;
        e->mode = e->auto_mode ? GPAP_MODE_SCALAR : o.mode;
        /* 0 = default, 1 = on, -1 = off; the zero must mean the default, or `{0}` loses it. */
        e->vscan = o.vscan < 0 ? 0 : 1;
        const char *ce = std::getenv("GPAP_CLEAN");
        e->clean_enabled = (ce && ce[0] == '0') ? 0 : 1;
        /* GPAP_KERNEL=scalar|wave forces the propagate kernel for an A/B. Read in the core so it
         * reaches every consumer, and applied where AUTO resolves, so an explicit pin wins. */
        const char *ke = std::getenv("GPAP_KERNEL");
        e->env_mode = (ke && std::strcmp(ke, "scalar") == 0)  ? GPAP_MODE_SCALAR
                      : (ke && std::strcmp(ke, "wave") == 0)  ? GPAP_MODE_WAVE
                                                              : GPAP_MODE_AUTO;
        e->nw = std::max(1, (2 * f->num_vars + 31) / 32);
        e->trail_cap = gpap_trail_cap(f->num_vars);   /* THE rule, in calibrate.cpp */

        st = stage(e->occ_off, f->occ_offsets.data(), f->occ_offsets.size() * sizeof(int32_t));
        if (st != GPAP_OK) break;
        st = stage(e->occ_mid, f->occ_mid.data(), f->occ_mid.size() * sizeof(int32_t));
        if (st != GPAP_OK) break;
        st = stage(e->occ_cls, f->occ_clauses.data(), f->occ_clauses.size() * sizeof(int32_t));
        if (st != GPAP_OK) break;
        if (e->shift == f->shift) {
            st = stage(e->packed_init, f->packed_init.data(),
                       f->packed_init.size() * sizeof(int32_t));
        } else {   /* pinned to the other split: re-pack the LONG clauses from sizes + sums */
            std::vector<int32_t> pk;
            gpap::pack_long_init(f, e->shift, pk);
            st = stage(e->packed_init, pk.data(), pk.size() * sizeof(int32_t));
        }
        if (st != GPAP_OK) break;
        /* Soundness pairs: binaries plus every unit u as (u, u), since none carries a word. */
        {
            std::vector<int32_t> pairs;
            pairs.reserve(f->bin_pairs.size() + 2 * f->unit_lits.size());
            pairs.insert(pairs.end(), f->bin_pairs.begin(), f->bin_pairs.end());
            for (int32_t u : f->unit_lits) {
                pairs.push_back(u);
                pairs.push_back(u);
            }
            e->n_pairs = (int32_t)(pairs.size() / 2);
            st = stage(e->pair_check, pairs.data(), pairs.size() * sizeof(int32_t));
            if (st != GPAP_OK) break;
            e->pair_host = std::move(pairs);   /* the SAME list, not a second copy */
        }
        /* Host side of gpap_sat_rows' original-order synthesis (the engine outlives f). */
        e->num_bin = f->num_bin;
        e->sat_src_host = f->sat_src;
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
    /* Pinning clears auto; asking for auto hands the choice back to reset(K). The sampler pins
     * its chunk's kernel so a short final reset cannot switch mid-measurement. */
    e->auto_mode = (mode == GPAP_MODE_AUTO) ? 1 : 0;
    if (!e->auto_mode) e->mode = mode;
    else if (e->K > 0) e->mode = resolve_mode(e, e->K);   /* the env pin survives, per resolve_mode */
    return GPAP_OK;
}

extern "C" gpap_mode gpap_engine_mode(const gpap_engine *e) { return e ? e->mode : GPAP_MODE_SCALAR; }
extern "C" int32_t gpap_engine_auto(const gpap_engine *e) {
    /* auto_mode alone is not the answer: $GPAP_KERNEL pins through it. */
    return (e && e->auto_mode && e->env_mode == GPAP_MODE_AUTO) ? 1 : 0;
}
extern "C" int32_t gpap_engine_shift(const gpap_engine *e) { return e ? e->shift : 0; }
extern "C" int32_t gpap_engine_k(const gpap_engine *e) { return e ? e->K : 0; }
extern "C" int32_t gpap_engine_num_vars(const gpap_engine *e) { return e ? e->num_vars : 0; }
extern "C" int32_t gpap_engine_num_clauses(const gpap_engine *e) {
    return e ? e->num_clauses : 0;
}
/* Engine-side: it copies what it needs at create time, so a C consumer may free the formula. */
extern "C" int32_t gpap_engine_num_long(const gpap_engine *e) { return e ? e->num_long : 0; }
extern "C" int gpap_engine_flush_conflict(const gpap_engine *e) { return e ? e->flush_conflict : 0; }

/* ------------------------------------------------------------------ reset */

static gpap_status ensure_state(gpap_engine *e, int32_t K) {
    /* gpap_trail_cap (calibrate.cpp) is the one definition, shared with the sizing model. */
    const int32_t cap = gpap_trail_cap(e->num_vars);
    if (cap > e->trail_cap) e->trail_cap = cap;
    GPAP_TRY(e->packed.reserve((size_t)K * e->num_long * sizeof(int32_t)));
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
    /* A grow is a realloc, so the packed rows are gone and no longer the baseline — dropped on
     * FAILURE too, or the next reset takes the lite path over an uninitialized buffer. */
    const size_t packed_cap = e->packed.bytes;
    const gpap_status alloc = ensure_state(e, K);
    if (alloc != GPAP_OK || e->packed.bytes != packed_cap) e->packed_clean_rows = 0;
    if (alloc != GPAP_OK) return alloc;
    e->K = K;
    /* Resolve the kernel for THIS K: cheap, and the one place every entry point passes through. */
    if (e->auto_mode) e->mode = resolve_mode(e, K);
    for (int i = 0; i < GPAP_SLOTS; ++i) {
        e->slots[i].rows_valid = false;
        e->slots[i].count = 0;
        e->slots[i].selected = false;    /* a new chunk selects nothing until asked to */
    }
    if (!e->flush_ready) return GPAP_OK;                 /* construction path: caller seeds */
    if (e->clean_enabled && e->packed_clean_rows >= K) {
        /* RESET-LITE: the previous launch was CLEAN-mode (or nothing dirtied since the last
         * broadcast), so the packed rows already ARE the baseline and the K*M broadcast — most
         * of a probe chunk's wall — is skipped. Gated on clean_enabled, so GPAP_CLEAN=0 is exact
         * legacy behaviour (docs/findings/CLEAN_MODE_FINDINGS.md). */
        e->resets_lite++;
    } else {
        GPAP_TRY(broadcast_rows(e, e->packed, e->packed_flushed, e->num_long, K));
        if (K > e->packed_clean_rows) e->packed_clean_rows = K;
        e->resets_full++;
    }
    GPAP_TRY(broadcast_rows(e, e->abits, e->abits_flushed, e->nw, K));
    e->abits_diverged = 0;             /* abits are the baseline again, same as packed */
    GPAP_TRY(fill_flags(e, e->active, e->flush_conflict ? 0 : 1, K));
    GPAP_TRY(fill_flags(e, e->conflict, e->flush_conflict ? 1 : 0, K));
    return GPAP_OK;
}

extern "C" gpap_status gpap_engine_recache_flush(gpap_engine *e) {
    if (!e) return GPAP_ERR_ARG;
    if (e->abits_diverged) {
        /* Refused, not silently wrong: after a CLEAN launch packed holds the baseline and abits
         * the fixpoint, so caching that pair yields a baseline whose abits claim literals true
         * packed never decremented. */
        set_error("gpap_engine_recache_flush: the last launch was gpap_engine_apply_clean, which "
                  "restored the packed state to the baseline while the assignment bitmap kept the "
                  "fixpoint — caching that pair would corrupt the flush cache. Re-run the "
                  "assumption with gpap_engine_apply (the plain kernel) and recache that.");
        return GPAP_ERR_ARG;
    }
    GPAP_TRY(use_ctx(e));
    /* Not atomic: two device copies update the two cached buffers, so a failure between them
     * leaves an inconsistent pair. On anything but GPAP_OK, DISCARD THE ENGINE. */
    e->packed_clean_rows = 0;                       /* the baseline itself is moving */
    if (e->num_long)
        GPAP_CU(cuMemcpyDtoD(e->packed_flushed.ptr, e->packed.ptr,
                             (size_t)e->num_long * sizeof(int32_t)));
    GPAP_CU(cuMemcpyDtoD(e->abits_flushed.ptr, e->abits.ptr, (size_t)e->nw * sizeof(int32_t)));
    int8_t c0 = 0;
    GPAP_CU(cuMemcpyDtoH(&c0, e->conflict.ptr, 1));
    e->flush_conflict = c0 ? 1 : 0;
    return GPAP_OK;
}

/* ------------------------------------------------------------------ entry point 1 */

static gpap_status apply_dev_common(gpap_engine *e, uintptr_t decisions_dev, int32_t L,
                                    int clean) {
    if (!e || L < 0 || (L > 0 && !decisions_dev)) return GPAP_ERR_ARG;
    /* The per-launch counters describe THE LAST CALL and reset() does not clear them, so a
     * zero-width apply must — which is why this sits above the L == 0 return. */
    GPAP_TRY(use_ctx(e));
    GPAP_TRY(e->nprops.zero());
    GPAP_TRY(e->nops.zero());
    GPAP_TRY(e->ndecs.zero());
    if (L == 0) return GPAP_OK;                     /* no assumptions: nothing to propagate */
    /* Same guard the packed readouts carry: after a CLEAN launch packed is the BASELINE while
     * the bitmap holds the fixpoint, so propagating again would corrupt counts. */
    GPAP_TRY(gpap::engine_reject_if_packed_is_baseline(e, "gpap_engine_apply"));
    /* No width argument: the trail is bounded by the variable count, not the assumed width
     * (dedup — see gpap_trail_cap). */
    GPAP_TRY(ensure_state(e, e->K));

    /* Dirty from here; a completed CLEAN launch puts the claim back after the last GPAP_TRY. */
    const int32_t clean_rows_prev = e->packed_clean_rows;
    e->packed_clean_rows = 0;

    Kernel k;
    const std::string key = std::string(clean ? "fixpoint_clean.s" : "fixpoint.s") +
                            std::to_string(e->shift) + ".w" +
                            std::to_string(e->mode == GPAP_MODE_WAVE ? 1 : 0);
    GPAP_TRY(e->kernels.get(key, &k));
    CUdeviceptr dec = (CUdeviceptr)decisions_dev;
    int32_t M = e->num_long, NW = e->nw, B = L, cap = e->trail_cap;
    std::vector<void *> args = {&e->packed.ptr, &e->abits.ptr, &e->active.ptr, &e->conflict.ptr,
                                &dec, &e->occ_off.ptr, &e->occ_mid.ptr, &e->occ_cls.ptr};
    if (clean) args.push_back(&e->packed_flushed.ptr);   /* init_ptr: what the tail restores */
    for (void *p : {(void *)&e->trail.ptr, (void *)&e->nprops.ptr, (void *)&e->nops.ptr,
                    (void *)&e->ndecs.ptr, (void *)&M, (void *)&NW, (void *)&B, (void *)&cap})
        args.push_back(p);
    GPAP_TRY(launch(k, (unsigned)e->K, 1, args));
    e->launches++;
    /* Record what the launch did BEFORE anything that can fail: the accumulate() calls below can
     * return GPAP_ERR_CUDA (a deferred async fault), and an unrecorded divergence would leave the
     * packed readouts taking BASELINE for fixpoint. */
    for (int i = 0; i < GPAP_SLOTS; ++i) e->slots[i].rows_valid = false;
    /* CLEAN leaves packed at the baseline and abits at the fixpoint; plain leaves both there. */
    e->abits_diverged = clean ? 1 : 0;
    GPAP_TRY(accumulate(e, e->nops, e->acc_ops, e->K));
    GPAP_TRY(accumulate(e, e->nprops, e->acc_props, e->K));
    GPAP_TRY(accumulate(e, e->ndecs, e->acc_decs, e->K));
    /* Last: this one PERMITS the next reset to skip the broadcast, so it holds only if the
     * restore tail ran to completion. */
    if (clean) e->packed_clean_rows = clean_rows_prev;
    return GPAP_OK;
}

extern "C" gpap_status gpap_engine_apply_dev(gpap_engine *e, uintptr_t decisions_dev, int32_t L) {
    return apply_dev_common(e, decisions_dev, L, 0);
}

extern "C" gpap_status gpap_engine_apply(gpap_engine *e, const int64_t *decisions_host,
                                         int32_t L) {
    if (!e || L < 0 || (L > 0 && !decisions_host)) return GPAP_ERR_ARG;
    /* L == 0 is legal and a no-op ("is the formula itself conflicting?"). Handled in
     * apply_dev_common, which is where the counters are zeroed. */
    if (L == 0) return apply_dev_common(e, 0, 0, 0);
    GPAP_TRY(use_ctx(e));
    GPAP_TRY(stage(e->staging_a, decisions_host, (size_t)e->K * L * sizeof(int64_t)));
    return gpap_engine_apply_dev(e, (uintptr_t)e->staging_a.ptr, L);
}

extern "C" gpap_status gpap_engine_apply_clean(gpap_engine *e, const int64_t *decisions_host,
                                               int32_t L) {
    if (!e || L < 0 || (L > 0 && !decisions_host)) return GPAP_ERR_ARG;
    if (L == 0) return apply_dev_common(e, 0, 0, 1);                         /* as above */
    if (!e->clean_enabled) return gpap_engine_apply(e, decisions_host, L);   /* GPAP_CLEAN=0 */
    GPAP_TRY(use_ctx(e));
    GPAP_TRY(stage(e->staging_a, decisions_host, (size_t)e->K * L * sizeof(int64_t)));
    return apply_dev_common(e, (uintptr_t)e->staging_a.ptr, L, 1);
}

/* ------------------------------------------------------------------ entry point 2 */

extern "C" gpap_status gpap_engine_trajectory_dev(gpap_engine *e, uintptr_t order_dev, int32_t P,
                                                  int32_t n_support, int32_t B, int64_t seed,
                                                  int soft, int32_t *max_rounds_out) {
    if (!e || P < 0 || B <= 0 || (P > 0 && !order_dev)) return GPAP_ERR_ARG;
    /* n_support ends decision phase 1 and the kernel walks `order` to it, so n_support > P
     * reads past the caller's buffer. O(1) to refuse. */
    if (n_support < 0 || n_support > P) {
        set_error("gpap_engine_trajectory: n_support=%d must lie in [0, P=%d] — it is the end of "
                  "decision phase 1, so a larger value reads past `order`", n_support, P);
        return GPAP_ERR_ARG;
    }
    GPAP_TRY(use_ctx(e));
    /* Above the P == 0 return, as in apply_dev_common: skipping it reports the last launch. */
    GPAP_TRY(e->nprops.zero());
    GPAP_TRY(e->nops.zero());
    GPAP_TRY(e->ndecs.zero());
    GPAP_TRY(e->nrounds.zero());
    if (P == 0) {
        /* Nothing to decide (every variable already determined): each active row is a model. */
        if (max_rounds_out) *max_rounds_out = 0;
        for (int i = 0; i < GPAP_SLOTS; ++i) e->slots[i].rows_valid = false;
        return GPAP_OK;
    }
    /* Entry point 2 needs the same guard — worse here, since the line below would record the
     * pair as consistent and hide it. */
    GPAP_TRY(gpap::engine_reject_if_packed_is_baseline(e, "gpap_engine_trajectory"));
    GPAP_TRY(ensure_state(e, e->K));
    e->packed_clean_rows = 0;       /* a trajectory dirties without a restorable record */
    e->abits_diverged = 0;          /* ...but it leaves packed and abits CONSISTENT */

    Kernel k;
    GPAP_TRY(e->kernels.get(traj_key(e->shift, e->mode == GPAP_MODE_WAVE ? 1 : 0,
                                     e->vscan ? 1 : 0, soft ? 1 : 0), &k));
    CUdeviceptr order = (CUdeviceptr)order_dev;
    int32_t M = e->num_long, NW = e->nw, cap = e->trail_cap, max_rounds = 1 << 30;
    int64_t seed64 = seed;
    GPAP_TRY(launch(k, (unsigned)e->K, 1,
                    {&e->packed.ptr, &e->abits.ptr, &e->active.ptr, &e->conflict.ptr, &order,
                     &e->occ_off.ptr, &e->occ_mid.ptr, &e->occ_cls.ptr, &e->trail.ptr,
                     &e->nrounds.ptr,
                     &e->nops.ptr, &e->nprops.ptr, &e->ndecs.ptr, &M, &NW, &P, &n_support, &B,
                     &cap, &max_rounds, &seed64}));
    e->launches++;
    /* Invalidated BEFORE the fallible accumulates: a surviving selection describes the past. */
    for (int i = 0; i < GPAP_SLOTS; ++i) e->slots[i].rows_valid = false;
    GPAP_TRY(accumulate(e, e->nops, e->acc_ops, e->K));
    GPAP_TRY(accumulate(e, e->nprops, e->acc_props, e->K));
    GPAP_TRY(accumulate(e, e->ndecs, e->acc_decs, e->K));

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
    if (!e || !order_host || P < 0) return GPAP_ERR_ARG;
    /* Range-check the order here, where the array is host-side and walked anyway: a v past N
     * writes into a LATER ROW's bitmap, so row A's bad input changes row B's answer. */
    for (int32_t i = 0; i < P; ++i)
        if (order_host[i] < 0 || order_host[i] >= e->num_vars) {
            set_error("gpap_engine_trajectory: order[%d] = %d is not a variable index in "
                      "[0, %d)", i, order_host[i], e->num_vars);
            return GPAP_ERR_ARG;
        }
    GPAP_TRY(use_ctx(e));
    GPAP_TRY(stage(e->staging_a, order_host, (size_t)P * sizeof(int32_t)));
    return gpap_engine_trajectory_dev(e, (uintptr_t)e->staging_a.ptr, P, n_support, B, seed, soft,
                                      max_rounds_out);
}

/* Selections and readout live in readout.cpp; these helpers are shared with it. */
namespace gpap {
gpap_status engine_use_ctx(gpap_engine *e) { return use_ctx(e); }

/* Refuse a packed-derived readout after a CLEAN launch: it would answer from the restored
 * baseline. gpap_packed_rows is unguarded; the identity gate checks the restore with it. */
gpap_status engine_reject_if_packed_is_baseline(gpap_engine *e, const char *what) {
    if (!e || !e->abits_diverged) return GPAP_OK;
    set_error("%s: the last launch was gpap_engine_apply_clean, which restored the packed state "
              "to the baseline — this readout would describe the baseline, not the fixpoint. Use "
              "gpap_engine_apply (the plain kernel) when you need the packed state afterwards.",
              what);
    return GPAP_ERR_ARG;
}
gpap_status engine_stage(DevBuf &buf, const void *host, size_t n) { return stage(buf, host, n); }
gpap_status engine_ensure_rows(gpap_engine *e, int32_t slot) { return ensure_rows(e, slot); }
size_t engine_readout_slice_bytes() { return kReadoutSliceBytes; }
unsigned engine_ceil_div(int64_t a, int64_t b) { return ceil_div(a, b); }
}  // namespace gpap

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

extern "C" void gpap_engine_reset_stats(const gpap_engine *e, int64_t *full_out,
                                        int64_t *lite_out) {
    if (full_out) *full_out = e ? e->resets_full : 0;
    if (lite_out) *lite_out = e ? e->resets_lite : 0;
}

extern "C" void gpap_counters_reset(gpap_engine *e) {
    if (!e) return;
    cuCtxSetCurrent(e->ctx);
    e->acc_ops.zero();
    e->acc_props.zero();
    e->acc_decs.zero();
    e->launches = 0;
}

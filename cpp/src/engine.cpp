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
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using gpap::buf;
using gpap::DevBuf;
using gpap::i32;
using gpap::i64;
using gpap::set_error;

namespace {

constexpr size_t kReadoutSliceBytes = 256u << 20;   /* bound a (rows, 2N) readout transient */

std::string fixpoint_key(const char *kernel, int32_t shift, int wave) {
    return std::string(kernel) + ".s" + std::to_string(shift) + ".w" + std::to_string(wave);
}

/* The kernel choice for K rows: an explicit $GPAP_KERNEL pin first, else the dispatch rule.
 * gpap_engine_reset and gpap_engine_set_mode(AUTO) both call it. */
gpap_mode resolve_mode(gpap_engine *e, int32_t K) {
    if (e->env_mode != GPAP_MODE_AUTO) return e->env_mode;
    return gpap::dispatch_mode(e, K);
}

/* VSCAN is read only under `WAVE and not SOFT`, so the other pairs compile once at v0 — asking
 * for v1 there asks for an unshipped cubin. Mirrors build_kernels.vscan_is_read. */
std::string traj_key(int32_t shift, int wave, int vscan, int soft) {
    const int v = (wave && !soft) ? (vscan ? 1 : 0) : 0;
    return "trajectory.s" + std::to_string(shift) + ".w" + std::to_string(wave) + ".v" +
           std::to_string(v) + ".f" + std::to_string(soft);
}

/* Accumulate one (K,) int32 per-launch counter into its int64 device accumulator (no sync). */
gpap_status accumulate(gpap_engine *e, DevBuf &src, DevBuf &acc, int32_t n) {
    return e->gpu->run("sum_i32_acc", 1, 1, {buf(src), i32(n), buf(acc)});
}

gpap_status fill_flags(gpap_engine *e, DevBuf &flags, int32_t value, int32_t n) {
    return e->gpu->run("fill_i8", n, 1, {buf(flags), i32(value), i32(n)});
}

gpap_status broadcast_rows(gpap_engine *e, DevBuf &dst, DevBuf &src, int32_t n_cols, int32_t K) {
    if (n_cols <= 0 || K <= 0) return GPAP_OK;   /* num_long == 0 is legal (pure <=2-CNF) */
    return e->gpu->run("bcast_rows", K, n_cols, {buf(dst), buf(src), i32(n_cols)});
}

/* Upload `n` bytes into a scratch buffer, growing it as needed. */
gpap_status stage(gpap_engine *e, DevBuf &dst, const void *host, size_t n) {
    GPAP_TRY(dst.reserve(e->mem, n));
    if (n) GPAP_TRY(e->gpu->upload(dst.ptr, host, n));
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
    if (e->K) GPAP_TRY(e->gpu->download(mask.data(), s.mask.ptr, 0, (size_t)e->K));
    s.host_rows.clear();
    s.host_rows.reserve((size_t)s.count);
    for (int32_t i = 0; i < e->K; ++i)
        if (mask[(size_t)i]) s.host_rows.push_back(i);
    s.count = (int32_t)s.host_rows.size();
    GPAP_TRY(stage(e, s.rows, s.host_rows.data(), s.host_rows.size() * sizeof(int32_t)));
    s.rows_valid = true;
    return GPAP_OK;
}

gpap_status use_ctx(gpap_engine *e) { return e->gpu->make_current(); }

/* A queued call's launches put every row back to the baseline it copied at begin, so until it
 * ends nothing else may change the rows or the kernel: each such entry point refuses by name. */
gpap_status refuse_while_queued(const gpap_engine *e, const char *what) {
    if (!e->queued_call) return GPAP_OK;
    set_error("%s: a queued call holds the engine's rows (gpap::QueuedCall); nothing may change "
              "them or the kernel until it ends", what);
    return GPAP_ERR_ARG;
}

}  // namespace

gpap_engine::~gpap_engine() {
    if (gpu) gpu->make_current();      /* the buffers below free themselves on this device */
    for (auto *b : pair_masks) delete b;
    pair_masks.clear();
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
    gpap::Gpu *g = e->gpu.get();
    if (e->num_long) {
        GPAP_TRY(e->packed_flushed.reserve(e->mem, (size_t)e->num_long * sizeof(int32_t)));
        GPAP_TRY(g->copy(e->packed_flushed.ptr, e->packed.ptr,
                         (size_t)e->num_long * sizeof(int32_t)));
    }
    GPAP_TRY(e->abits_flushed.reserve(e->mem, (size_t)e->nw * sizeof(int32_t)));
    GPAP_TRY(g->copy(e->abits_flushed.ptr, e->abits.ptr, (size_t)e->nw * sizeof(int32_t)));
    int8_t conflict0 = 0;
    GPAP_TRY(g->download(&conflict0, e->conflict.ptr, 0, 1));
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
    const gpap::GpuDriver *driver = gpap::gpu_driver();     /* $GPAP_BACKEND */
    if (!driver) return GPAP_ERR_ARG;
    gpap::Gpu *g = nullptr;
    GPAP_TRY(driver->open(device, &g));
    gpap_engine *e = new gpap_engine();
    e->gpu.reset(g);
    e->driver = driver;
    e->device_ordinal = device;
    e->mem.gpu = g;
    e->mem.account = gpap::mem_account(driver, device);
    gpap_status st = GPAP_OK;
    do {
        /* The device's total first: a limit given as a share of it applies from the first
         * buffer, which is this formula's. */
        gpap::GpuMemory m;
        st = use_ctx(e);
        if (st == GPAP_OK) st = g->memory(&m);
        if (st != GPAP_OK) break;
        gpap::mem_note_total(e->mem.account, m.total);
        e->mem_source = m.source;

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
        /* GPAP_QUEUE=0 keeps every sweep in chunks, one set per row per launch: the sweep as
         * it ran before the queue, for A/Bs (QueuedCall::begin). */
        const char *qe = std::getenv("GPAP_QUEUE");
        e->queue_enabled = (qe && qe[0] == '0') ? 0 : 1;
        /* GPAP_KERNEL=scalar|wave forces the propagate kernel for an A/B. Read in the core so it
         * reaches every consumer, and applied where AUTO resolves, so an explicit pin wins. */
        const char *ke = std::getenv("GPAP_KERNEL");
        e->env_mode = (ke && std::strcmp(ke, "scalar") == 0)  ? GPAP_MODE_SCALAR
                      : (ke && std::strcmp(ke, "wave") == 0)  ? GPAP_MODE_WAVE
                                                              : GPAP_MODE_AUTO;
        e->nw = std::max(1, (2 * f->num_vars + 31) / 32);
        e->trail_cap = gpap_trail_cap(f->num_vars);   /* THE rule, in calibrate.cpp */

        st = stage(e, e->occ_off, f->occ_offsets.data(),
                   f->occ_offsets.size() * sizeof(int32_t));
        if (st != GPAP_OK) break;
        st = stage(e, e->occ_mid, f->occ_mid.data(), f->occ_mid.size() * sizeof(int32_t));
        if (st != GPAP_OK) break;
        st = stage(e, e->occ_cls, f->occ_clauses.data(),
                   f->occ_clauses.size() * sizeof(int32_t));
        if (st != GPAP_OK) break;
        if (e->shift == f->shift) {
            st = stage(e, e->packed_init, f->packed_init.data(),
                       f->packed_init.size() * sizeof(int32_t));
        } else {   /* pinned to the other split: re-pack the LONG clauses from sizes + sums */
            std::vector<int32_t> pk;
            gpap::pack_long_init(f, e->shift, pk);
            st = stage(e, e->packed_init, pk.data(), pk.size() * sizeof(int32_t));
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
            st = stage(e, e->pair_check, pairs.data(), pairs.size() * sizeof(int32_t));
            if (st != GPAP_OK) break;
            e->pair_host = std::move(pairs);   /* the SAME list, not a second copy */
        }
        /* Host side of gpap_sat_rows' original-order synthesis (the engine outlives f). */
        e->num_bin = f->num_bin;
        e->sat_src_host = f->sat_src;
        st = e->acc_ops.reserve(e->mem, 8);
        if (st == GPAP_OK) st = e->acc_props.reserve(e->mem, 8);
        if (st == GPAP_OK) st = e->acc_decs.reserve(e->mem, 8);
        if (st == GPAP_OK) st = e->scratch_i32.reserve(e->mem, 64);
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
    delete e;                       /* releases every DevBuf, then the device (member order) */
}

extern "C" gpap_status gpap_engine_set_mode(gpap_engine *e, gpap_mode mode) {
    if (!e) return GPAP_ERR_ARG;
    GPAP_TRY(refuse_while_queued(e, "gpap_engine_set_mode"));
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

extern "C" const char *gpap_backend(void) {
    const gpap::GpuDriver *d = gpap::gpu_driver();
    return d ? d->name : nullptr;
}
extern "C" const char *gpap_engine_backend(const gpap_engine *e) {
    return e ? e->gpu->backend() : "";
}
extern "C" const char *gpap_engine_device_name(const gpap_engine *e) {
    return e ? e->gpu->device_name() : "";
}

/* ------------------------------------------------------------------ reset */

/* Whether the batch is sized against an exact bound, a memory limit or a host's grant, which
 * counts every byte: the default rule's margins do not then cover a row buffer held past what
 * the sizing counted. */
static bool exact_sizing(const gpap_engine *e) {
    if (e->mem_source == gpap::GpuMemory::GRANT) return true;
    gpap::MemState acct;
    return e->mem.account && gpap::mem_state(e->mem.account, &acct) == GPAP_OK &&
           acct.limit > 0;
}

/* The row state for K rows. A buffer only grows, so a smaller batch keeps a bigger one's rows,
 * except with `fit`: then a buffer bigger than K rows is given back and taken at K's size. The
 * sizing credits the engine's own row state as reused by the next batch (engine_batch), which
 * holds for a bigger batch and, under an exact bound, only if a smaller one fits its buffers to
 * itself. Only a caller that re-initializes every row afterwards may pass `fit` (a reset, a
 * queued call's start): apply and trajectory run on rows whose contents a refit would lose. */
static gpap_status ensure_state(gpap_engine *e, int32_t K, bool fit = false) {
    /* gpap_trail_cap (calibrate.cpp) is the one definition, shared with the sizing model. */
    const int32_t cap = gpap_trail_cap(e->num_vars);
    if (cap > e->trail_cap) e->trail_cap = cap;
    auto size = [&](DevBuf &b, size_t n) {
        if (fit && b.bytes > n) b.release();
        return b.reserve(e->mem, n);
    };
    GPAP_TRY(size(e->packed, (size_t)K * e->num_long * sizeof(int32_t)));
    GPAP_TRY(size(e->abits, (size_t)K * e->nw * sizeof(int32_t)));
    GPAP_TRY(size(e->active, (size_t)K));
    GPAP_TRY(size(e->conflict, (size_t)K));
    GPAP_TRY(size(e->trail, (size_t)K * e->trail_cap * sizeof(int32_t)));
    for (DevBuf *b : {&e->nprops, &e->nops, &e->ndecs, &e->nrounds})
        GPAP_TRY(size(*b, (size_t)K * sizeof(int32_t)));
    for (int i = 0; i < GPAP_SLOTS; ++i) GPAP_TRY(size(e->slots[i].mask, (size_t)K));
    return GPAP_OK;
}

/* After a refused grow: ensure_state frees each buffer before it grows it, so a refusal part-way
 * (a memory limit, the device) leaves some of the old rows' buffers gone. Holding K rows over
 * them would send the next launch or readout through freed memory, and on CUDA an illegal
 * address loses the context: the engine holds no rows instead, as after a trim, until a reset
 * succeeds. Returns `st`. */
static gpap_status drop_rows(gpap_engine *e, gpap_status st) {
    e->K = 0;
    e->packed_clean_rows = 0;
    for (Selection &s : e->slots) {
        s.rows_valid = false;
        s.count = 0;
        s.selected = false;
    }
    return st;
}

/* Bytes the engine holds in the row state ensure_state grows, which its next batch reuses: a
 * buffer grows in place and frees itself before it allocates, so a new batch never needs these
 * bytes and its own at once, and under an exact bound a smaller batch refits them to itself
 * (ensure_state's `fit`), so it does not keep them either. Not staging_a: a caller whose matrix is narrower than the last one's
 * leaves part of it held. Keep the list in step with ensure_state. */
static int64_t row_buffers_held(const gpap_engine *e) {
    int64_t n = 0;
    for (const DevBuf *b : {&e->packed, &e->abits, &e->active, &e->conflict, &e->trail,
                            &e->nprops, &e->nops, &e->ndecs, &e->nrounds})
        n += (int64_t)b->bytes;
    for (const Selection &s : e->slots) n += (int64_t)s.mask.bytes;
    return n;
}

/* The row state (ensure_state's list) and the call buffers that grow with a batch go; the
 * formula, its flush cache and the pair masks stay. The next reset grows the rows back from
 * nothing, so it broadcasts the baseline in full (packed_clean_rows is dropped here). */
extern "C" gpap_status gpap_engine_trim(gpap_engine *e) {
    if (!e) return GPAP_ERR_ARG;
    GPAP_TRY(refuse_while_queued(e, "gpap_engine_trim"));
    GPAP_TRY(use_ctx(e));
    for (DevBuf *b : {&e->packed, &e->abits, &e->active, &e->conflict, &e->trail, &e->nprops,
                      &e->nops, &e->ndecs, &e->nrounds, &e->staging_a, &e->staging_b})
        b->release();
    for (Selection &s : e->slots) {
        s.mask.release();
        s.rows.release();
        s.host_rows.clear();
        s.count = 0;
        s.rows_valid = false;
        s.selected = false;
    }
    e->K = 0;
    e->packed_clean_rows = 0;
    e->abits_diverged = 0;
    return GPAP_OK;
}

extern "C" int64_t gpap_engine_mem_held(const gpap_engine *e, gpap_mem_kind kind) {
    if (!e) return 0;
    return kind == GPAP_MEM_FORMULA ? e->mem.formula
           : kind == GPAP_MEM_ROWS  ? e->mem.rows
                                    : e->mem.held();
}
extern "C" gpap_bound gpap_engine_batch_bound(const gpap_engine *e) {
    return e ? e->bound : GPAP_BOUND_NONE;
}

extern "C" gpap_status gpap_engine_reset(gpap_engine *e, int32_t K) {
    if (!e || K <= 0) return GPAP_ERR_ARG;
    GPAP_TRY(refuse_while_queued(e, "gpap_engine_reset"));
    GPAP_TRY(use_ctx(e));
    /* A grow is a realloc, so the packed rows are gone and no longer the baseline — dropped on
     * FAILURE too, or the next reset takes the lite path over an uninitialized buffer. */
    const size_t packed_cap = e->packed.bytes;
    const gpap_status alloc = ensure_state(e, K, exact_sizing(e));
    if (alloc != GPAP_OK || e->packed.bytes != packed_cap) e->packed_clean_rows = 0;
    if (alloc != GPAP_OK) return drop_rows(e, alloc);
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
    e->abits_diverged = 0;            /* abits are the baseline again, same as packed */
    GPAP_TRY(fill_flags(e, e->active, e->flush_conflict ? 0 : 1, K));
    GPAP_TRY(fill_flags(e, e->conflict, e->flush_conflict ? 1 : 0, K));
    return GPAP_OK;
}

extern "C" gpap_status gpap_engine_recache_flush(gpap_engine *e) {
    if (!e) return GPAP_ERR_ARG;
    GPAP_TRY(refuse_while_queued(e, "gpap_engine_recache_flush"));
    if (e->K <= 0) {     /* trimmed, or a grow was refused: row 0 is not there to cache */
        set_error("gpap_engine_recache_flush: the engine holds no rows (after gpap_engine_trim, or "
                  "a refused reset): reset it and apply what you want cached first");
        return GPAP_ERR_ARG;
    }
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
        GPAP_TRY(e->gpu->copy(e->packed_flushed.ptr, e->packed.ptr,
                              (size_t)e->num_long * sizeof(int32_t)));
    GPAP_TRY(e->gpu->copy(e->abits_flushed.ptr, e->abits.ptr, (size_t)e->nw * sizeof(int32_t)));
    int8_t c0 = 0;
    GPAP_TRY(e->gpu->download(&c0, e->conflict.ptr, 0, 1));
    e->flush_conflict = c0 ? 1 : 0;
    return GPAP_OK;
}

/* ------------------------------------------------------------------ entry point 1 */

static gpap_status apply_common(gpap_engine *e, gpap::GpuMem decisions, int32_t L, int clean) {
    if (!e || L < 0 || (L > 0 && !decisions)) return GPAP_ERR_ARG;
    GPAP_TRY(refuse_while_queued(e, "gpap_engine_apply"));
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

    const std::string key = fixpoint_key(clean ? "fixpoint_clean" : "fixpoint", e->shift,
                                         e->mode == GPAP_MODE_WAVE ? 1 : 0);
    const int32_t M = e->num_long, NW = e->nw, B = L, cap = e->trail_cap;
    if (clean)        /* init_ptr (packed_flushed) is what the tail restores */
        GPAP_TRY(e->gpu->run(key.c_str(), e->K, 1,
                             {buf(e->packed), buf(e->abits), buf(e->active), buf(e->conflict),
                              buf(decisions), buf(e->occ_off), buf(e->occ_mid), buf(e->occ_cls),
                              buf(e->packed_flushed), buf(e->trail), buf(e->nprops), buf(e->nops),
                              buf(e->ndecs), i32(M), i32(NW), i32(B), i32(cap)}));
    else
        GPAP_TRY(e->gpu->run(key.c_str(), e->K, 1,
                             {buf(e->packed), buf(e->abits), buf(e->active), buf(e->conflict),
                              buf(decisions), buf(e->occ_off), buf(e->occ_mid), buf(e->occ_cls),
                              buf(e->trail), buf(e->nprops), buf(e->nops), buf(e->ndecs), i32(M),
                              i32(NW), i32(B), i32(cap)}));
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
    if (!e || L < 0 || (L > 0 && !decisions_dev)) return GPAP_ERR_ARG;
    gpap::GpuMem dec = 0;
    if (L > 0) GPAP_TRY(e->gpu->adopt(decisions_dev, &dec));
    return apply_common(e, dec, L, 0);
}

extern "C" gpap_status gpap_engine_apply(gpap_engine *e, const int64_t *decisions_host,
                                         int32_t L) {
    if (!e || L < 0 || (L > 0 && !decisions_host)) return GPAP_ERR_ARG;
    /* L == 0 is legal and a no-op ("is the formula itself conflicting?"). Handled in
     * apply_common, which is where the counters are zeroed. */
    if (L == 0) return apply_common(e, 0, 0, 0);
    GPAP_TRY(use_ctx(e));
    GPAP_TRY(stage(e, e->staging_a, decisions_host, (size_t)e->K * L * sizeof(int64_t)));
    return apply_common(e, e->staging_a.ptr, L, 0);
}

extern "C" gpap_status gpap_engine_apply_clean(gpap_engine *e, const int64_t *decisions_host,
                                               int32_t L) {
    if (!e || L < 0 || (L > 0 && !decisions_host)) return GPAP_ERR_ARG;
    if (L == 0) return apply_common(e, 0, 0, 1);                             /* as above */
    if (!e->clean_enabled) return gpap_engine_apply(e, decisions_host, L);   /* GPAP_CLEAN=0 */
    GPAP_TRY(use_ctx(e));
    GPAP_TRY(stage(e, e->staging_a, decisions_host, (size_t)e->K * L * sizeof(int64_t)));
    return apply_common(e, e->staging_a.ptr, L, 1);
}

/* ------------------------------------------------------------------ entry point 2 */

static gpap_status trajectory_common(gpap_engine *e, gpap::GpuMem order, int32_t P,
                                     int32_t n_support, int32_t B, int64_t seed, int soft,
                                     int32_t *max_rounds_out) {
    if (!e || P < 0 || B <= 0 || (P > 0 && !order)) return GPAP_ERR_ARG;
    GPAP_TRY(refuse_while_queued(e, "gpap_engine_trajectory"));
    /* n_support ends decision phase 1 and the kernel walks `order` to it, so n_support > P
     * reads past the caller's buffer. O(1) to refuse. */
    if (n_support < 0 || n_support > P) {
        set_error("gpap_engine_trajectory: n_support=%d must lie in [0, P=%d] — it is the end of "
                  "decision phase 1, so a larger value reads past `order`", n_support, P);
        return GPAP_ERR_ARG;
    }
    GPAP_TRY(use_ctx(e));
    /* Above the P == 0 return, as in apply_common: skipping it reports the last launch. */
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

    const std::string key = traj_key(e->shift, e->mode == GPAP_MODE_WAVE ? 1 : 0,
                                     e->vscan ? 1 : 0, soft ? 1 : 0);
    const int32_t M = e->num_long, NW = e->nw, cap = e->trail_cap, max_rounds = 1 << 30;
    GPAP_TRY(e->gpu->run(key.c_str(), e->K, 1,
                         {buf(e->packed), buf(e->abits), buf(e->active), buf(e->conflict),
                          buf(order), buf(e->occ_off), buf(e->occ_mid), buf(e->occ_cls),
                          buf(e->trail), buf(e->nrounds), buf(e->nops), buf(e->nprops),
                          buf(e->ndecs), i32(M), i32(NW), i32(P), i32(n_support), i32(B), i32(cap),
                          i32(max_rounds), i64(seed)}));
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
            GPAP_TRY(e->gpu->download(rounds.data(), e->nrounds.ptr, 0,
                                      (size_t)e->K * sizeof(int32_t)));
            *max_rounds_out = *std::max_element(rounds.begin(), rounds.end());
        }
    }
    return GPAP_OK;
}

extern "C" gpap_status gpap_engine_trajectory_dev(gpap_engine *e, uintptr_t order_dev, int32_t P,
                                                  int32_t n_support, int32_t B, int64_t seed,
                                                  int soft, int32_t *max_rounds_out) {
    if (!e || P < 0 || B <= 0 || (P > 0 && !order_dev)) return GPAP_ERR_ARG;
    gpap::GpuMem order = 0;
    if (P > 0) GPAP_TRY(e->gpu->adopt(order_dev, &order));
    return trajectory_common(e, order, P, n_support, B, seed, soft, max_rounds_out);
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
    GPAP_TRY(stage(e, e->staging_a, order_host, (size_t)P * sizeof(int32_t)));
    return trajectory_common(e, e->staging_a.ptr, P, n_support, B, seed, soft, max_rounds_out);
}

/* ------------------------------------------------------------------ the queue */

namespace {

std::string queue_key(int32_t shift, int wave) {
    return "queue.s" + std::to_string(shift) + ".w" + std::to_string(wave);
}

}  // namespace

namespace gpap {

gpap_status QueuedCall::begin(int32_t requested, int64_t n_sets, int64_t width,
                              int64_t first_block) {
    gpap_engine *e = e_;
    if (!e || rows_ > 0) {
        set_error("QueuedCall::begin: %s", !e ? "engine must not be NULL" : "already begun");
        return GPAP_ERR_ARG;
    }
    GPAP_TRY(refuse_while_queued(e, "QueuedCall::begin"));   /* one call holds the rows */
    first_ = std::min<int64_t>(std::max<int64_t>(first_block, 1), kQueueMaxJobs);
    /* A queued row restores itself with CLEAN mode's restore, so $GPAP_CLEAN=0, the escape
     * hatch from that restore, keeps the calls in chunks on the plain kernel too. */
    if (!e->queue_enabled || !e->clean_enabled || !e->flush_ready) return GPAP_OK;
    if (e->flush_conflict || e->has_empty_clause) return GPAP_OK;
    int32_t resident = e->driver->resident_warps(e->device_ordinal);
    if (resident <= 0) resident = kQueueRowsUnknown;
    /* The rows that fit with this call's launch buffers taken out of the budget. In bytes, not
     * rows: the batch is capped by gpap_default_k_cap, so rows taken off it say nothing of
     * memory. A pin is held to it too. A queued row is the engine's state and its sums: no
     * per-row decision matrix. Where the launch buffers leave no row, the call runs in chunks,
     * which need none of them. A declined call keeps the bound its chunk sizing recorded. */
    const gpap_bound chunk_bound = e->bound;
    const int64_t fit = engine_batch(e, kQueueRowBytes, 0, queue_call_bytes(n_sets, width));
    if (fit <= 0) {
        e->bound = chunk_bound;
        return GPAP_OK;
    }
    int64_t rows = requested > 0 ? std::min<int64_t>(requested, fit) : fit;
    if (requested > 0 && requested <= fit) e->bound = GPAP_BOUND_PIN;
    if (rows > resident) e->bound = GPAP_BOUND_USEFUL;   /* rows past it would take no job */
    rows = std::max<int64_t>(std::min<int64_t>(rows, resident), 1);
    if (n_sets <= rows) {
        e->bound = chunk_bound;
        return GPAP_OK;
    }
    /* WAVE unless pinned: the rows are counted in warps, and a WAVE row is one warp on both
     * backends. A SCALAR row is a warp with one working lane (CUDA) or one work item, so a
     * thirty-second of a warp (OpenCL). */
    const gpap_mode mode = !e->auto_mode                     ? e->mode
                           : e->env_mode != GPAP_MODE_AUTO ? e->env_mode
                                                           : GPAP_MODE_WAVE;
    if (!e->gpu->has_kernel(queue_key(e->shift, mode == GPAP_MODE_WAVE ? 1 : 0).c_str())) {
        e->bound = chunk_bound;
        return GPAP_OK;
    }

    GPAP_TRY(use_ctx(e));
    /* A grow is a realloc: those rows are no longer the baseline, on failure too. */
    const size_t packed_cap = e->packed.bytes;
    const gpap_status alloc = ensure_state(e, (int32_t)rows, exact_sizing(e));
    if (alloc != GPAP_OK || e->packed.bytes != packed_cap) e->packed_clean_rows = 0;
    if (alloc != GPAP_OK) return drop_rows(e, alloc);
    for (int i = 0; i < GPAP_SLOTS; ++i) {
        e->slots[i].rows_valid = false;
        e->slots[i].count = 0;
        e->slots[i].selected = false;
    }
    /* Every row at the baseline, as a reset leaves it: the packed words unless the claim
     * already covers these rows, the bitmap always. */
    if (e->packed_clean_rows < rows) {
        GPAP_TRY(broadcast_rows(e, e->packed, e->packed_flushed, e->num_long, (int32_t)rows));
        e->packed_clean_rows = (int32_t)rows;
        e->resets_full++;
    } else {
        e->resets_lite++;
    }
    GPAP_TRY(broadcast_rows(e, e->abits, e->abits_flushed, e->nw, (int32_t)rows));
    e->abits_diverged = 0;
    /* The rows' flags and per-row counters describe no job: leave them as a reset does. */
    GPAP_TRY(fill_flags(e, e->active, 1, (int32_t)rows));
    GPAP_TRY(fill_flags(e, e->conflict, 0, (int32_t)rows));
    GPAP_TRY(e->nprops.zero());
    GPAP_TRY(e->nops.zero());
    GPAP_TRY(e->ndecs.zero());
    e->K = (int32_t)rows;
    e->mode = mode;               /* the queue's kernel, until the call ends */
    e->queued_call = 1;
    rows_ = (int32_t)rows;
    return GPAP_OK;
}

int64_t QueuedCall::next_block(double seconds_left) {
    if (last_ <= 0) return last_ = first_;
    if (last_ > first_ && took_ >= 0.05) return last_;
    if (8.0 * took_ >= seconds_left) return last_;
    return last_ = std::min<int64_t>(last_ * 8, kQueueMaxJobs);
}

gpap_status QueuedCall::launch(const int32_t *jobs, int64_t n, int32_t L, int8_t *conflict_out,
                               int32_t *implied_out) {
    gpap_engine *e = e_;
    if (rows_ <= 0 || !jobs || !conflict_out || n <= 0 || n > kQueueMaxJobs || L <= 0 ||
        e->K <= 0) {
        set_error("QueuedCall::launch: bad arguments (n=%lld of at most %lld, L=%d, rows=%d)",
                  (long long)n, (long long)kQueueMaxJobs, L, rows_);
        return GPAP_ERR_ARG;
    }
    const auto t0 = std::chrono::steady_clock::now();
    GPAP_TRY(use_ctx(e));
    const int32_t rows = e->K;
    gpap::Gpu *g = e->gpu.get();
    GPAP_TRY(stage(e, e->queue_dec, jobs, (size_t)n * (size_t)L * sizeof(int32_t)));
    GPAP_TRY(e->queue_conf.reserve(e->mem, (size_t)n));
    GPAP_TRY(e->queue_impl.reserve(e->mem, (size_t)n * sizeof(int32_t)));
    GPAP_TRY(e->queue_sums.reserve(e->mem, (size_t)rows * kQueueRowBytes));
    GPAP_TRY(e->queue_next.reserve(e->mem, 16));
    GPAP_TRY(e->queue_next.zero());

    /* Dirty until the readbacks below show the launch ran to its end: a launch that died
     * half-way leaves rows mid-job, so the next call must copy them, and until a reset a packed
     * readout or recache_flush is refused rather than served from them (abits_diverged). */
    const int32_t clean_prev = e->packed_clean_rows;
    e->packed_clean_rows = 0;
    e->abits_diverged = 1;
    const std::string key = queue_key(e->shift, e->mode == GPAP_MODE_WAVE ? 1 : 0);
    const int32_t M = e->num_long, NW = e->nw, cap = e->trail_cap;
    GPAP_TRY(g->run(key.c_str(), rows, 1,
                    {buf(e->packed), buf(e->abits), buf(e->queue_dec), buf(e->occ_off),
                     buf(e->occ_mid), buf(e->occ_cls), buf(e->packed_flushed),
                     buf(e->abits_flushed), buf(e->trail), buf(e->queue_next),
                     buf(e->queue_conf), buf(e->queue_impl), buf(e->queue_sums), i32(M),
                     i32(NW), i32(L), i32(cap), i32((int32_t)n)}));
    e->launches++;
    e->queue_launches++;
    e->queue_jobs += n;
    e->queue_mode = e->mode;
    GPAP_TRY(g->download(conflict_out, e->queue_conf.ptr, 0, (size_t)n));
    if (implied_out)
        GPAP_TRY(g->download(implied_out, e->queue_impl.ptr, 0, (size_t)n * sizeof(int32_t)));
    std::vector<int64_t> sums((size_t)rows * 3);
    GPAP_TRY(g->download(sums.data(), e->queue_sums.ptr, 0, sums.size() * sizeof(int64_t)));
    for (int32_t r = 0; r < rows; ++r) {
        e->host_ops += sums[(size_t)r * 3];
        e->host_props += sums[(size_t)r * 3 + 1];
        e->host_decs += sums[(size_t)r * 3 + 2];
    }
    e->packed_clean_rows = clean_prev;
    e->abits_diverged = 0;
    last_ = n;                    /* next_block grows from what ran, which a cut makes smaller */
    took_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return GPAP_OK;
}

QueuedCall::~QueuedCall() {
    gpap_engine *e = e_;
    if (rows_ <= 0 || !e) return;
    e->queued_call = 0;
    if (!e->gpu) return;
    e->gpu->make_current();
    for (DevBuf *b : {&e->queue_dec, &e->queue_next, &e->queue_conf, &e->queue_impl,
                      &e->queue_sums})
        b->release();
    if (e->auto_mode && e->K > 0) e->mode = resolve_mode(e, e->K);   /* as gpap_engine_reset */
}

}  // namespace gpap

extern "C" void gpap_engine_queue_stats(const gpap_engine *e, int64_t *launches_out,
                                        int64_t *jobs_out, gpap_mode *mode_out) {
    if (launches_out) *launches_out = e ? e->queue_launches : 0;
    if (jobs_out) *jobs_out = e ? e->queue_jobs : 0;
    if (mode_out) *mode_out = e ? e->queue_mode : GPAP_MODE_AUTO;
}

/* Selections and readout live in readout.cpp; these helpers are shared with it. */
namespace gpap {
gpap_status engine_use_ctx(gpap_engine *e) { return use_ctx(e); }

gpap_mode dispatch_mode(gpap_engine *e, int32_t K) {
    /* A device that cannot form a WAVE work group runs SCALAR; a pin is honoured instead, and
     * fails at the launch with a message that says so. */
    if (!e->gpu->has_kernel(fixpoint_key("fixpoint", e->shift, 1).c_str())) return GPAP_MODE_SCALAR;
    return dispatch_wave(e->driver, e->num_clauses, K, e->device_ordinal) ? GPAP_MODE_WAVE
                                                                          : GPAP_MODE_SCALAR;
}

gpap_status engine_require_trajectory(gpap_engine *e) {
    const std::string key = traj_key(e->shift, 0, 0, 0);
    if (e->gpu->has_kernel(key.c_str())) return GPAP_OK;
    set_error("kernel '%s' is not part of the %s backend, which cannot sample. Use "
              "GPAP_BACKEND=cuda for it", key.c_str(), e->gpu->backend());
    return GPAP_ERR_KERNEL;
}

/* What an exact bound (a limit, a host's grant) counts besides the rows and the caller's own
 * buffers: what a call may still allocate once it is sized, which the MEMORY bound's margins
 * cover instead (so its batch is unchanged). Keep in step with what readout.cpp, the sampler and
 * the plugs grow after sizing: tests/engine/test_mem_limit.py runs every entry point under a
 * limit just above one row and fails on a byte this misses. */
static void exact_terms(const gpap_engine *e, int64_t *per_row, int64_t *fixed) {
    *per_row = 4ll * GPAP_SLOTS        /* each slot's selected-row list (Selection::rows) */
               + 1                     /* soundness flags (gpap_check_sound, staging_b) */
               + 4;                    /* an OpenCL plug's copy of a row list from an offset */
    *fixed = 32ll * 2 * e->num_vars    /* the smallest readout slice, 32 rows of 2N (staging_b) */
             + 8ll * e->num_vars       /* column lists and the trajectory order (staging_a) */
             + 2ll * 4 * e->nw         /* the sampler's two pair masks */
             + 4096;                   /* an OpenCL plug's parameter blocks */
}

/* Why no row fits, in the terms of the bound that left none. */
static void no_row_message(const gpap_engine *e, const SizeRequest &r, const Sizing &s,
                           const MemState &acct) {
    char a[32], b[32], c[32], d[32];
    const char *dev = mem_account_name(e->mem.account);
    if (s.bound == GPAP_BOUND_BUFFER) {
        set_error("not even one row fits on %s: one row's widest buffer is %s, past the %s the "
                  "device allows in one buffer", dev, fmt_bytes(r.widest, a),
                  fmt_bytes(r.max_buffer, b));
        return;
    }
    const int64_t fixed = s.bound == GPAP_BOUND_MEMORY ? r.fixed : r.fixed + r.exact_fixed;
    /* The default budget's reserve and fraction are mem_budget's (calibrate.cpp), not repeated
     * here: say what it is a share of. */
    const char *memory = e->mem_source == GpuMemory::DEVICE
                             ? "the default budget, a share of the device's free memory"
                             : "the default budget, a share of the device's total memory (its "
                               "free memory cannot be measured)";
    if (s.bound == GPAP_BOUND_LIMIT) {
        char f[32];
        set_error("not even one row fits on %s: a row of this formula needs %s, and the memory "
                  "limit of %s (%s) leaves %s beside the %s GPAP holds there and the %s this "
                  "call takes once. Raise the limit, or trim or free another engine",
                  dev, fmt_bytes(s.per_row, a), fmt_bytes(acct.limit, b),
                  acct.source == GPAP_LIMIT_ENV ? "GPAP_MEM_LIMIT" : "gpap_set_mem_limit",
                  fmt_bytes(s.left, c), fmt_bytes(std::max<int64_t>(r.held_other, 0), d),
                  fmt_bytes(fixed, f));
        return;
    }
    set_error("not even one row fits on %s: a row of this formula needs %s, and %s leaves %s "
              "beside the %s this call takes once%s", dev, fmt_bytes(s.per_row, a),
              s.bound == GPAP_BOUND_GRANT ? "the host's grant" : memory, fmt_bytes(s.left, b),
              fmt_bytes(fixed, c),
              s.bound == GPAP_BOUND_MEMORY ? " (GPAP_VRAM_BUDGET_MIB caps it, where set)" : "");
}

int32_t engine_batch(gpap_engine *e, int64_t extra_row_bytes, int32_t cap,
                     int64_t reserve_bytes) {
    e->bound = GPAP_BOUND_NONE;
    GpuMemory m;
    e->sizing_status = use_ctx(e);
    if (e->sizing_status == GPAP_OK) e->sizing_status = e->gpu->memory(&m);
    if (e->sizing_status != GPAP_OK) return 0;                   /* the plug said why */
    MemState acct;
    e->sizing_status = mem_state(e->mem.account, &acct);
    if (e->sizing_status != GPAP_OK) return 0;                   /* a malformed limit */
    /* The rows this call reuses: a grow frees them before it allocates, so a new batch never
     * needs these bytes and its own at once. Counted back where `free` excludes them (DEVICE,
     * GRANT) and against the ledger for a limit, never on an unmeasured total, which never took
     * them out: the batch is then a function of the formula and the card, not of the engine's
     * call history. */
    const int64_t own = row_buffers_held(e);
    SizeRequest r;
    r.total = m.total;
    r.limit = acct.limit;
    r.held_other = (e->mem.account ? acct.held : e->mem.held()) - own;
    r.per_row = gpap_row_bytes(e->num_long, e->num_vars, e->trail_cap) + extra_row_bytes;
    r.fixed = reserve_bytes;
    exact_terms(e, &r.exact_per_row, &r.exact_fixed);
    r.cap = cap == 0 ? gpap_default_k_cap(e->device_ordinal) : cap;
    r.max_buffer = m.max_buffer;
    /* Each per-row buffer holds the batch's rows, so the widest one meets the device's
     * one-buffer limit first: the engine's own (packed words, bitmap, trail) or the caller's. */
    r.widest = std::max<int64_t>(
        {4ll * e->num_long, 4ll * e->nw, 4ll * e->trail_cap, extra_row_bytes, 1});
    switch (m.source) {
        case GpuMemory::DEVICE: r.memory = mem_budget(m.free + own, kMemFraction); break;
        case GpuMemory::GRANT: r.grant = m.free + own; break;
        default:
            /* A total stands in for free memory, unless a limit does: less what GPAP holds there
             * besides the rows this call reuses, which CUDA's measured free memory already
             * leaves out. Two engines then never both plan on the whole device. */
            if (r.limit <= 0)
                r.memory = mem_budget(std::max<int64_t>(m.free - r.held_other, 0), kMemFraction);
            break;
    }
    const Sizing s = size_rows(r);
    e->bound = s.bound;
    if (s.rows <= 0) {
        no_row_message(e, r, s, acct);
        e->sizing_status = GPAP_ERR_MEMORY;
        return 0;
    }
    return (int32_t)s.rows;
}

int64_t engine_transient_room(gpap_engine *e, int64_t reuse) {
    MemState acct;
    if (e->mem.account && mem_state(e->mem.account, &acct) == GPAP_OK && acct.limit > 0)
        return std::max<int64_t>(acct.limit - acct.held + reuse, 0);
    GpuMemory m;
    if (e->mem_source == GpuMemory::GRANT && e->gpu->memory(&m) == GPAP_OK)
        return std::max<int64_t>(m.free + reuse, 0);
    return INT64_MAX;
}

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
gpap_status engine_stage(gpap_engine *e, DevBuf &dst, const void *host, size_t n) {
    return stage(e, dst, host, n);
}
gpap_status engine_ensure_rows(gpap_engine *e, int32_t slot) { return ensure_rows(e, slot); }
size_t engine_readout_slice_bytes() { return kReadoutSliceBytes; }
}  // namespace gpap

extern "C" gpap_status gpap_counters_get(gpap_engine *e, gpap_counters *out) {
    if (!e || !out) return GPAP_ERR_ARG;
    GPAP_TRY(use_ctx(e));
    int64_t v[3] = {0, 0, 0};
    GPAP_TRY(e->gpu->download(&v[0], e->acc_ops.ptr, 0, sizeof(int64_t)));
    GPAP_TRY(e->gpu->download(&v[1], e->acc_props.ptr, 0, sizeof(int64_t)));
    GPAP_TRY(e->gpu->download(&v[2], e->acc_decs.ptr, 0, sizeof(int64_t)));
    out->ops = v[0] + e->host_ops;          /* the queued launches' sums live on the host */
    out->props = v[1] + e->host_props;
    out->decs = v[2] + e->host_decs;
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
    e->gpu->make_current();
    e->acc_ops.zero();
    e->acc_props.zero();
    e->acc_decs.zero();
    e->host_ops = e->host_props = e->host_decs = 0;
    e->launches = 0;
}

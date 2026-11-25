/* Batch sizing: how many rows of engine state fit a memory budget, and how wide a batch the card
 * should run. The one implementation — Python calls these through the ABI instead of mirroring
 * the arithmetic, because two copies of a memory model drift silently and the symptom is a spill,
 * not an error. The model is the ALLOCATION LIST of gpap_engine's per-chunk buffers, verified
 * against the driver's free-memory deltas: if the engine allocates something, add it; if not, do
 * not. */
#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "gpap_internal.hpp"

namespace {
/* Absolute slack between what the driver reports free and what we then allocate: `free` is a
 * snapshot of somebody else's bookkeeping, and a paging driver or another process can take memory
 * back underneath us (`fraction` is the proportional margin). Flat on every platform on purpose:
 * making it conditional changes gpap_compute_k_safe's input, i.e. K, unvalidated. */
constexpr int64_t kFreeVramReserveBytes = 800000000ll;

/* The batch cap: one number, the same on every GPU. A SATURATION limit, not a safety one — the
 * analytic bound in gpap_compute_k_safe protects memory, and at this batch a card is already
 * heavily oversubscribed against its resident warps. Device-independent on purpose, or `--k-cap
 * auto` resolves differently per machine. Revisit only with a measurement showing it bind off
 * the occupancy plateau: docs/findings/CROSS_GPU_FINDINGS.md. */
constexpr int32_t kBatchCap = 262144;

/* Clause count at or above which the warp-wide WAVE propagate beats the scalar one; measured,
 * docs/findings/KERNEL_AB_FINDINGS.md. See gpap_dispatch_wave for why the variable is M. */
constexpr int32_t kWaveMinClauses = 2048;

/* Trail headroom past the exact bound — one wavefront; see gpap_trail_cap for why that is the
 * unit. Keep in step with WAVE_W (cpp/tools/build_kernels.py). */
constexpr int32_t kTrailSlack = 32;

/* GPAP_VRAM_BUDGET_MIB: an upper bound, in MiB, on what gpap_mem_available may report. Unset (or
 * <= 0) is the default and changes nothing. It exists for CROSS-CARD REPRODUCTION: K is
 * memory-bound on large formulas, so the same run on a larger card sizes a larger batch, and a
 * head-to-head whose rival is pinned to a recorded batch would then be measuring two different
 * grants. Capping the budget to the recorded card's makes both sides identical on any card at
 * least that large. It is a CEILING, never a floor: a smaller card still reports less.
 *
 * ONE CAP COVERS BOTH SIDES because this function is the only budget either side has: the rival
 * harnesses size their batch through eval/rivals/rival_common.rival_vram_budget, which calls
 * straight into here. Set the cap once and the comparison is capped, not half of it. */
/* The grammar is deliberately narrow and is mirrored exactly by gpap.engine.calibrate: optional
 * surrounding blanks, an optional '+', then digits and nothing else. strtoll's own rule — take
 * the longest numeric PREFIX — is what a shared parser must not have here: "1e3" would cap the
 * engine to 1 MiB while Python's int() rejected the same string and left the rival uncapped, so
 * the two sides of a fairness comparison would be capped apart by a typo. Anything outside the
 * grammar, and anything that would overflow the shift, means NO CAP rather than a small one. */
int64_t env_budget_cap() {
    const char *e = std::getenv("GPAP_VRAM_BUDGET_MIB");
    if (!e) return 0;
    while (*e == ' ' || *e == '\t') ++e;
    if (*e == '+') ++e;
    if (!*e) return 0;
    int64_t mib = 0;
    for (; *e >= '0' && *e <= '9'; ++e) {
        if (mib > (INT64_MAX >> 20) / 10) return 0;   /* far past any real budget: no cap */
        mib = mib * 10 + (*e - '0');
    }
    while (*e == ' ' || *e == '\t') ++e;
    if (*e) return 0;                                 /* trailing junk: no cap, never a prefix */
    if (mib <= 0 || mib > (INT64_MAX >> 20)) return 0;
    return mib << 20;
}
}  // namespace

/* There is deliberately no formula-bytes deduction here: gpap_mem_available() reports memory
 * free right now, i.e. after the occ lists and packed init were uploaded — already paid for. */

/* The trail sizing rule — one definition, called by the engine's own ensure_state() as well as
 * by every sizing consumer, so the allocation and the model cannot disagree.
 *
 * BOUND = num_vars, and it is tight. Every push is gated on the assignment bitmap flipping that
 * literal's bit from 0 to 1 (the scalar paths test-then-set within one program, the wave paths
 * take the `atomic_or` return), so exactly one lane can observe the transition for a literal;
 * pushing a literal whose negation is already set is a contradiction and conflicts the row
 * instead; and l and ~l share a bitmap word. So a VARIABLE contributes at most one trail entry
 * per launch, whatever the decision batch B or the assumption width L — which is why neither is
 * a parameter here.
 *
 * SLACK = one WAVEFRONT. The wave propagate and the warp-wide decision scan append a whole
 * wavefront of pushes in one masked store at `tail + prefix_sum`, and unlike the scalar path
 * (`elif tail < TRAIL_CAP`) a vectorised append has no bound test — so a wavefront is the
 * granularity by which a broken dedup invariant would overshoot, and one of headroom keeps that
 * step inside this row. Must not be smaller than WAVE_W in cpp/tools/build_kernels.py. */
extern "C" int32_t gpap_trail_cap(int32_t num_vars) {
    return num_vars + kTrailSlack;
}

/* `num_long` is the STATE clause count — length >= 3 — because only those carry a packed word
 * per row; binary clauses are implication edges in the occ CSR and units live in the flush
 * cache's bitmap, at zero per-row bytes. It is deliberately not called `M`, which means the
 * total clause count everywhere else here. */
extern "C" int64_t gpap_row_bytes(int32_t num_long, int32_t N, int32_t trail_cap) {
    if (trail_cap <= 0) trail_cap = gpap_trail_cap(N);
    const int64_t words = (2ll * N + 31) / 32;          /* assignment bitmap, int32 words */
    return 4ll * num_long                               /* packed clause word (K, num_long) */
           + 4ll * words                                /* assignment bitmap    (K, ceil2N/32)*/
           + 4ll * trail_cap                            /* trail                (K, cap)    */
           + 2                                          /* active + conflict                */
           + 16                                         /* nprops/nops/ndecs/nrounds        */
           + GPAP_SLOTS;                                /* one selection mask per slot      */
}

/* The one budget rule: reserve, fraction, cap. gpap_mem_available applies it to the device named
 * by an ordinal; an engine applies it to the device it already holds. */
int64_t gpap::mem_budget(int64_t free_bytes, double fraction) {
    const int64_t usable = std::max<int64_t>(0, free_bytes - kFreeVramReserveBytes);
    const int64_t budget = (int64_t)(usable * fraction);
    const int64_t cap = env_budget_cap();
    return cap > 0 ? std::min(budget, cap) : budget;
}

namespace {
/* The account of `device` on the backend $GPAP_BACKEND names, the device's total noted for a
 * share; NULL, the error set, for an unknown backend. `m` gets the device's report, or stays
 * empty (source TOTAL, total 0) where the device cannot be asked. */
gpap::MemAccount *account_of(int device, gpap::GpuMemory *m, bool *reported) {
    const gpap::GpuDriver *driver = gpap::gpu_driver();
    if (!driver) return nullptr;
    gpap::MemAccount *a = gpap::mem_account(driver, device);
    *m = gpap::GpuMemory{};
    *reported = driver->memory(device, m) == GPAP_OK;
    if (*reported) gpap::mem_note_total(a, m->total);
    return a;
}
}  // namespace

/* The budget a NEW engine's rows would get, before its own formula: the batch rule's bounds
 * (memory.cpp) with nothing reused and one byte a row, so its answer is bytes. Without a limit
 * that is mem_budget alone, as it always was. */
extern "C" int64_t gpap_mem_available(int device, double fraction) {
    gpap::GpuMemory m;
    bool reported = false;
    gpap::MemAccount *a = account_of(device, &m, &reported);
    if (!a || !reported) return 0;
    gpap::MemState s;
    if (gpap::mem_state(a, &s) != GPAP_OK) return 0;
    gpap::SizeRequest r;
    r.total = m.total;
    r.limit = s.limit;
    r.held_other = s.held;
    if (m.source == gpap::GpuMemory::GRANT) r.grant = m.free;
    else if (m.source == gpap::GpuMemory::DEVICE || s.limit <= 0)
        r.memory = gpap::mem_budget(m.free, fraction);
    return gpap::size_rows(r).left;
}

extern "C" gpap_status gpap_set_mem_limit(int device, const char *spec) {
    gpap::MemSpec s;                                   /* NULL: UNSET, follow GPAP_MEM_LIMIT */
    if (spec) GPAP_TRY(gpap::parse_mem_spec(spec, "memory limit ", &s));
    gpap::GpuMemory m;
    bool reported = false;
    gpap::MemAccount *a = account_of(device, &m, &reported);
    if (!a) return GPAP_ERR_ARG;
    gpap::mem_set_limit(a, s);
    return GPAP_OK;
}

extern "C" gpap_status gpap_parse_mem_limit(const char *spec, int64_t total_bytes,
                                            int64_t *bytes_out) {
    gpap::MemSpec s;
    GPAP_TRY(gpap::parse_mem_spec(spec, "memory limit ", &s));
    if (bytes_out) *bytes_out = gpap::mem_spec_bytes(s, total_bytes);
    return GPAP_OK;
}

namespace {
/* An account's state for the getters below. A limit that cannot be resolved reads -1, never
 * "none": every engine on the device would fail, and the getters must not say otherwise. The
 * counts stand either way. */
gpap::MemState state_of(int device) {
    gpap::GpuMemory m;
    bool reported = false;
    gpap::MemState s;
    gpap::MemAccount *a = account_of(device, &m, &reported);
    if (a && gpap::mem_state(a, &s) != GPAP_OK) {
        s.limit = -1;
        s.source = GPAP_LIMIT_NONE;
    }
    return s;
}
}  // namespace

extern "C" int64_t gpap_mem_limit(int device) { return state_of(device).limit; }
extern "C" gpap_limit_source gpap_mem_limit_source(int device) { return state_of(device).source; }
extern "C" int64_t gpap_mem_held(int device) { return state_of(device).held; }
extern "C" int64_t gpap_mem_peak(int device) { return state_of(device).peak; }
extern "C" void gpap_mem_reset_peak(int device) {
    const gpap::GpuDriver *driver = gpap::gpu_driver();
    if (driver) gpap::mem_reset_peak(gpap::mem_account(driver, device));
}

extern "C" int32_t gpap_default_k_cap(int device) {
    (void)device;                 /* kept in the signature: the cap is device-independent, and a
                                     caller should not have to care that it became so */
    return kBatchCap;
}

/* Rows of `per_row` bytes that fit `bytes`, at most `cap` (<= 0: no cap), never fewer than one:
 * gpap_compute_k_safe's division, for a caller sizing an explicit K. An engine's own batch is
 * size_rows (memory.cpp), which reports 0 where no row fits instead. */
int32_t gpap::rows_fitting(int64_t bytes, int64_t per_row, int32_t cap) {
    int64_t k = bytes > 0 ? bytes / std::max<int64_t>(per_row, 1) : 0;
    if (cap > 0) k = std::min<int64_t>(k, cap);
    return (int32_t)std::min<int64_t>(std::max<int64_t>(k, 1), INT32_MAX);
}

extern "C" int32_t gpap_compute_k_safe(int32_t num_long, int32_t N, int64_t bytes, int32_t cap) {
    return gpap::rows_fitting(bytes, gpap_row_bytes(num_long, N, gpap_trail_cap(N)), cap);
}

extern "C" int32_t gpap_resident_warps(int device) {
    const gpap::GpuDriver *driver = gpap::gpu_driver();
    return driver ? driver->resident_warps(device) : 0;
}

/* Which propagate kernel should a chunk of `batch` rows of THIS formula run? 1 = WAVE, 0 = SCALAR.
 * One definition of the rule: gpap::Engine::plan_for and Python's calibrate.dispatch_mode call
 * this instead of repeating the arithmetic. Both variants share the same state and reach the same
 * fixpoints (BCP is confluent), so this is a pure launch-flag choice — only speed changes.
 *
 * The rule keys off M, the CLAUSE COUNT, never the batch alone: M decides whether a warp-wide
 * propagate has work to spread, since WAVE hands one row 32 lanes and a row with few clauses
 * cannot fill them, while SCALAR keeps far more rows in flight. Do not reintroduce a batch-only
 * rule: the old `batch < 30 * resident_warps` was a proxy for M (K ~ budget/4M, so "huge batch"
 * and "small M" were the same rows) that a later sizing change broke, misclassifying every
 * mid-size formula sitting at the batch cap. A/B at fixed K: docs/findings/KERNEL_AB_FINDINGS.md.
 * The batch term survives only for the regime M cannot speak for: a batch too small to fill even
 * one warp per SM, where SCALAR's extra rows do not exist to be had. */
int32_t gpap::dispatch_wave(const gpap::GpuDriver *driver, int32_t num_clauses, int32_t batch,
                            int device) {
    if (!driver) return 0;
    if (num_clauses >= kWaveMinClauses) return 1;
    /* Only the small-M path needs the device query, so the common case costs nothing. A
     * library that cannot answer it (0) keeps small formulas on SCALAR. */
    const int32_t resident = driver->resident_warps(device);
    return (resident > 0 && batch <= resident) ? 1 : 0;
}

/* The rule for the backend a NEW engine would get ($GPAP_BACKEND). An engine asks its own. */
extern "C" int32_t gpap_dispatch_wave(int32_t num_clauses, int32_t batch, int device) {
    return gpap::dispatch_wave(gpap::gpu_driver(), num_clauses, batch, device);
}

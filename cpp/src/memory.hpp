/* Device memory: what GPAP holds on a device, the limit a user sets on it, and the batch rule
 * that sizes a call under both (docs/ENGINE.md §7). One home, so an entry point cannot size past
 * what the ledger counts.
 *
 * THE LEDGER. Every device buffer the engine allocates is a DevBuf, and DevBuf::reserve is the
 * only place it allocates: it counts the bytes here before it asks the plug and uncounts them
 * when it frees. So what an engine holds, and what all of a process's engines hold on one
 * device, are counts, not estimates. A plug allocates nothing through it but its own short
 * launch transients (an OpenCL argument view, a parameter block), which the batch rule leaves
 * room for instead (exact_terms in engine.cpp). */
#ifndef GPAP_MEMORY_HPP
#define GPAP_MEMORY_HPP

#include <cstdint>

#include "gpap/gpap.h"
#include "gpu_api.hpp"

namespace gpap {

/* ------------------------------------------------------------------ the ledger */

/* What GPAP's engines hold on one device of one GPU library, and the limit set on it. One per
 * (library, ordinal): every engine opens its own Gpu, so a plug cannot be the key. Created on
 * first use and never destroyed, so a buffer freed at process exit still finds its account. */
struct MemAccount;
MemAccount *mem_account(const GpuDriver *driver, int device);
/* Remember the device's total, which a share ("75%") is taken of. */
void mem_note_total(MemAccount *a, int64_t total);

/* What a buffer holds, which decides the count it goes on. FORMULA: what the engine builds once
 * and keeps until it is freed (the formula's lists, its flush cache, pair masks, the
 * counters). ROWS: everything that grows with a batch or a call. Two counts, not one, because
 * the two have different sizes and lifetimes: gpap_engine_trim gives back every row byte and
 * keeps every formula byte, and a later split of the engine into a resident formula and row
 * pools moves each count to its own object. ROWS is the default, so a new buffer is a row
 * buffer unless it says otherwise, and a trim that misses it shows. */
enum class MemKind : uint8_t { ROWS = 0, FORMULA = 1 };

/* What one engine's buffers are counted on: its counts by kind, and its device's account. An
 * engine on a host's plug has no account: the host keeps that count (GpuMemory::GRANT). */
struct MemHolder {
    Gpu *gpu = nullptr;
    MemAccount *account = nullptr;
    int64_t formula = 0;        /* bytes in FORMULA buffers */
    int64_t rows = 0;           /* bytes in ROWS buffers */
    int64_t held() const { return formula + rows; }
};

/* Count `bytes` of `kind` the holder is about to allocate. Refused with GPAP_ERR_MEMORY, the
 * message set, when that would take its device past the limit in force (checked and added
 * under the account's lock, so two threads cannot both pass), or, where free memory is
 * measured, leave less than half the reserve free (the sizing keeps all of it). GPAP_ERR_ARG
 * for a malformed GPAP_MEM_LIMIT or GPAP_MEM_RESERVE, which are read on every call. */
gpap_status mem_take(MemHolder &h, MemKind kind, int64_t bytes);
/* Uncount bytes the holder freed, or did not get after mem_take. */
void mem_give(MemHolder &h, MemKind kind, int64_t bytes);
/* The plug allocated what mem_take counted: only now does it reach the device's peak, so a
 * refused or failed allocation never shows as memory GPAP held. */
void mem_got(MemHolder &h);

/* ------------------------------------------------------------------ the limit */

/* A parsed limit. UNSET follows GPAP_MEM_LIMIT; NONE is "no limit", even if it is set. */
struct MemSpec {
    enum Kind { UNSET, NONE, BYTES, SHARE };
    Kind kind = UNSET;
    int64_t bytes = 0;          /* BYTES */
    double percent = 0;         /* SHARE: in (0, 100], of the device's total */
};
/* "8G", "512M", "1.5G", "8GiB", "8GB", "100000B" (binary units: 1G = 1024^3 bytes; case
 * ignored; a blank may separate number and unit), "75%", or "none". Nothing else: no bare
 * number, no zero, no exponent, no trailing text. GPAP_ERR_ARG with `label` and the reason in
 * the message ("GPAP_MEM_LIMIT=" for the variable, "memory limit " for the API). One grammar
 * for the limit and the reserve. */
gpap_status parse_mem_spec(const char *spec, const char *label, MemSpec *out);
/* The bytes of `s` on a device of `total` bytes: 0 for NONE, and for a share of an unknown
 * total (a caller that must know says so: mem_state). */
int64_t mem_spec_bytes(const MemSpec &s, int64_t total);

/* ------------------------------------------------------------------ the reserve */

/* What GPAP leaves free on a device by default, for what it cannot count: other processes and
 * the desktop, the caller's own tensors, kernel code loaded at a first launch. The default
 * budget is free memory less the reserve, and on a measured device an allocation that would
 * leave less than half of it free is refused, because a driver that pages device memory to the
 * host (WSL2) does not refuse it: it slows down instead. The measurement that set it:
 * docs/findings/MEMORY_FINDINGS.md §9. */
constexpr int64_t kDefaultMemReserve = 1ll << 30;

/* An account now, its limit and reserve resolved: the API's setting, else the variable
 * (GPAP_MEM_LIMIT, GPAP_MEM_RESERVE), else no limit and the default reserve. */
struct MemState {
    int64_t held = 0, peak = 0, total = 0;
    int64_t limit = 0;          /* 0 = none */
    gpap_limit_source source = GPAP_LIMIT_NONE;
    int64_t reserve = 0;        /* bytes to leave free */
};
/* GPAP_ERR_ARG for a malformed GPAP_MEM_LIMIT or GPAP_MEM_RESERVE, or a share of a device whose
 * total is unknown. */
gpap_status mem_state(MemAccount *a, MemState *out);
void mem_set_limit(MemAccount *a, const MemSpec &s);
void mem_set_reserve(MemAccount *a, const MemSpec &s);
void mem_reset_peak(MemAccount *a);
/* "cuda device 0", for messages. */
const char *mem_account_name(const MemAccount *a);

/* "6.0 GiB": `buf` must hold 32 bytes. */
const char *fmt_bytes(int64_t bytes, char *buf);

/* ------------------------------------------------------------------ the batch rule */

/* What one call is sized from. Every byte figure is device bytes. */
struct SizeRequest {
    int64_t total = 0;          /* the device total, 0 = unknown */
    /* The MEMORY bound's budget: the default rule (mem_budget: free memory less the reserve, or
     * a share of an unmeasured total, under GPAP_VRAM_BUDGET_MIB), with the buffers the call
     * reuses counted back on a measured device. < 0: the bound does not apply (a host's grant,
     * or a limit standing in for an unmeasured total). */
    int64_t memory = -1;
    /* The GRANT bound: what a host's plug still grants, the reused rows counted back. < 0: none. */
    int64_t grant = -1;
    int64_t limit = 0;          /* the LIMIT bound: the limit in force, 0 = none */
    /* What GPAP holds on the device besides the rows this call reuses: the formula, other
     * buffers, other engines. The LIMIT bound subtracts it. */
    int64_t held_other = 0;
    /* Everything the call allocates, per row and once, on every bound: the row state, the
     * caller's own buffers (the job queue's), and what the call may still allocate once it is
     * sized (exact_terms in engine.cpp). */
    int64_t per_row = 1;
    int64_t fixed = 0;
    /* One-off bytes only the MEMORY bound counts: the readout slice the default budget sets
     * aside (engine_batch). A limit and a grant keep the 32-row minimum `fixed` counts. */
    int64_t memory_fixed = 0;
    int32_t cap = 0;            /* the batch cap; <= 0: none */
    int64_t max_buffer = 0;     /* the device's one-buffer limit; 0: none */
    int64_t widest = 1;         /* the widest per-row buffer, which max_buffer bounds */
};

struct Sizing {
    int64_t rows = 0;           /* 0: not even one row fits */
    gpap_bound bound = GPAP_BOUND_NONE;
    int64_t left = 0;           /* bytes the binding memory bound left for rows */
    int64_t per_row = 0;        /* the per-row bytes that bound counted */
};

/* THE batch rule. Each memory bound that applies gives rows = (its bytes - the call's fixed
 * bytes) / per-row bytes; the batch is the smallest, then the batch cap, then the one-buffer
 * limit. The bound reported is the one that set it; at a tie the cap, then LIMIT, GRANT and
 * MEMORY, so "CAP" means more memory would not raise the batch. Pure: no device, no ledger. */
Sizing size_rows(const SizeRequest &r);

}  // namespace gpap

#endif /* GPAP_MEMORY_HPP */

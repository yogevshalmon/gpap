/* Device memory: what GPAP holds on a device (docs/ENGINE.md §7).
 *
 * THE LEDGER. Every device buffer the engine allocates is a DevBuf, and DevBuf::reserve is the
 * only place it allocates: it counts the bytes here before it asks the plug and uncounts them
 * when it frees. So what an engine holds, and what all of a process's engines hold on one
 * device, are counts, not estimates. A plug allocates nothing through it but its own short
 * launch transients (an OpenCL argument view, a parameter block). */
#ifndef GPAP_MEMORY_HPP
#define GPAP_MEMORY_HPP

#include <cstdint>

#include "gpap/gpap.h"
#include "gpu_api.hpp"

namespace gpap {

/* ------------------------------------------------------------------ the ledger */

/* What GPAP's engines hold on one device of one GPU library. One per (library, ordinal): every
 * engine opens its own Gpu, so a plug cannot be the key. Created on first use and never
 * destroyed, so a buffer freed at process exit still finds its account. */
struct MemAccount;
MemAccount *mem_account(const GpuDriver *driver, int device);

/* What a buffer holds, which decides the count it goes on. FORMULA: what the engine builds once
 * and keeps until it is freed (the formula's lists, its flush cache, pair masks, the
 * counters). ROWS: everything that grows with a batch or a call. Two counts, not one, because
 * the two have different sizes and lifetimes: a later split of the engine into a resident
 * formula and row pools moves each count to its own object. ROWS is the default, so a new
 * buffer is a row buffer unless it says otherwise. */
enum class MemKind : uint8_t { ROWS = 0, FORMULA = 1 };

/* What one engine's buffers are counted on: its counts by kind, and its device's account. */
struct MemHolder {
    Gpu *gpu = nullptr;
    MemAccount *account = nullptr;
    int64_t formula = 0;        /* bytes in FORMULA buffers */
    int64_t rows = 0;           /* bytes in ROWS buffers */
    int64_t held() const { return formula + rows; }
};

/* Count `bytes` of `kind` the holder is about to allocate, under the account's lock. */
gpap_status mem_take(MemHolder &h, MemKind kind, int64_t bytes);
/* Uncount bytes the holder freed, or did not get after mem_take. */
void mem_give(MemHolder &h, MemKind kind, int64_t bytes);
/* The plug allocated what mem_take counted: only now does it reach the device's peak, so a
 * failed allocation never shows as memory GPAP held. */
void mem_got(MemHolder &h);

/* An account now. */
struct MemState {
    int64_t held = 0, peak = 0;
};
gpap_status mem_state(MemAccount *a, MemState *out);
void mem_reset_peak(MemAccount *a);

}  // namespace gpap

#endif /* GPAP_MEMORY_HPP */

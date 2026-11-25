/* The ledger (memory.hpp). */
#include "memory.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <utility>

namespace gpap {

struct MemAccount {
    std::mutex mu;
    int64_t held = 0, peak = 0;
};

namespace {

/* Never destroyed (see mem_account): a static map would be torn down at exit before the last
 * engine a caller leaked frees its buffers into it. */
std::mutex &table_mu() {
    static std::mutex *mu = new std::mutex();
    return *mu;
}
std::map<std::pair<const GpuDriver *, int>, MemAccount *> &table() {
    static auto *t = new std::map<std::pair<const GpuDriver *, int>, MemAccount *>();
    return *t;
}

}  // namespace

MemAccount *mem_account(const GpuDriver *driver, int device) {
    std::lock_guard<std::mutex> lock(table_mu());
    MemAccount *&a = table()[{driver, device}];
    if (!a) a = new MemAccount();
    return a;
}

gpap_status mem_take(MemHolder &h, MemKind kind, int64_t bytes) {
    if (MemAccount *a = h.account) {
        std::lock_guard<std::mutex> lock(a->mu);
        a->held += bytes;
    }
    (kind == MemKind::FORMULA ? h.formula : h.rows) += bytes;
    return GPAP_OK;
}

void mem_got(MemHolder &h) {
    if (MemAccount *a = h.account) {
        std::lock_guard<std::mutex> lock(a->mu);
        a->peak = std::max(a->peak, a->held);
    }
}

void mem_give(MemHolder &h, MemKind kind, int64_t bytes) {
    if (MemAccount *a = h.account) {
        std::lock_guard<std::mutex> lock(a->mu);
        a->held -= bytes;
    }
    (kind == MemKind::FORMULA ? h.formula : h.rows) -= bytes;
}

gpap_status mem_state(MemAccount *a, MemState *out) {
    *out = MemState{};
    if (!a) return GPAP_OK;
    std::lock_guard<std::mutex> lock(a->mu);
    out->held = a->held;
    out->peak = a->peak;
    return GPAP_OK;
}

void mem_reset_peak(MemAccount *a) {
    if (!a) return;
    std::lock_guard<std::mutex> lock(a->mu);
    a->peak = a->held;
}

}  // namespace gpap

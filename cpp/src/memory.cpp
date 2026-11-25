/* The ledger, the limit and the batch rule (memory.hpp). */
#include "memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "gpap_internal.hpp"

namespace gpap {

struct MemAccount {
    std::mutex mu;
    int64_t held = 0, peak = 0, total = 0;
    MemSpec api;                /* UNSET: follow GPAP_MEM_LIMIT */
    MemSpec reserve;            /* UNSET: follow GPAP_MEM_RESERVE, else kDefaultMemReserve */
    std::string name;           /* "cuda device 0" */
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

bool blank(const char *s) {
    for (; *s; ++s)
        if (*s != ' ' && *s != '\t') return false;
    return true;
}

char lower(char c) { return (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c; }

/* The limit in force, under a->mu. */
gpap_status resolve(MemAccount *a, int64_t *limit, gpap_limit_source *source) {
    MemSpec s = a->api;
    gpap_limit_source src = GPAP_LIMIT_API;
    if (s.kind == MemSpec::UNSET) {
        /* Read on every call, as GPAP_BACKEND and GPAP_VRAM_BUDGET_MIB are; unset or blank is
         * no limit. A malformed value is an error, never "no limit": one parser reads it, so
         * there is no second reader it could disagree with. */
        const char *env = std::getenv("GPAP_MEM_LIMIT");
        if (!env || blank(env)) {
            *limit = 0;
            *source = GPAP_LIMIT_NONE;
            return GPAP_OK;
        }
        GPAP_TRY(parse_mem_spec(env, "GPAP_MEM_LIMIT=", &s));
        src = GPAP_LIMIT_ENV;
    }
    if (s.kind == MemSpec::SHARE && a->total <= 0) {
        set_error("a memory limit of %g%% on %s needs the device's total memory, which it does "
                  "not report: give the limit in bytes (8G, 512M)", s.percent, a->name.c_str());
        return GPAP_ERR_ARG;
    }
    *limit = mem_spec_bytes(s, a->total);
    if (s.kind == MemSpec::SHARE && *limit < 1) {      /* a share must be a limit, never none */
        set_error("a memory limit of %g%% of %s is less than one byte", s.percent,
                  a->name.c_str());
        return GPAP_ERR_ARG;
    }
    *source = src;
    return GPAP_OK;
}

/* The reserve in force, under a->mu. "none" is no reserve. */
gpap_status resolve_reserve(MemAccount *a, int64_t *reserve) {
    MemSpec s = a->reserve;
    if (s.kind == MemSpec::UNSET) {
        const char *env = std::getenv("GPAP_MEM_RESERVE");
        if (!env || blank(env)) {
            *reserve = kDefaultMemReserve;
            return GPAP_OK;
        }
        GPAP_TRY(parse_mem_spec(env, "GPAP_MEM_RESERVE=", &s));
    }
    if (s.kind == MemSpec::SHARE && a->total <= 0) {
        set_error("a memory reserve of %g%% on %s needs the device's total memory, which it does "
                  "not report: give the reserve in bytes (1G, 512M)", s.percent,
                  a->name.c_str());
        return GPAP_ERR_ARG;
    }
    *reserve = mem_spec_bytes(s, a->total);
    return GPAP_OK;
}

const char *source_name(gpap_limit_source s) {
    return s == GPAP_LIMIT_ENV ? "GPAP_MEM_LIMIT" : "gpap_set_mem_limit";
}

}  // namespace

MemAccount *mem_account(const GpuDriver *driver, int device) {
    std::lock_guard<std::mutex> lock(table_mu());
    MemAccount *&a = table()[{driver, device}];
    if (!a) {
        a = new MemAccount();
        a->name = std::string(driver ? driver->name : "gpu") + " device " + std::to_string(device);
    }
    return a;
}

void mem_note_total(MemAccount *a, int64_t total) {
    if (!a || total <= 0) return;
    std::lock_guard<std::mutex> lock(a->mu);
    a->total = total;
}

const char *mem_account_name(const MemAccount *a) { return a ? a->name.c_str() : "a host's plug"; }

gpap_status mem_take(MemHolder &h, MemKind kind, int64_t bytes) {
    if (MemAccount *a = h.account) {
        std::lock_guard<std::mutex> lock(a->mu);
        int64_t limit = 0;
        gpap_limit_source source = GPAP_LIMIT_NONE;
        GPAP_TRY(resolve(a, &limit, &source));
        if (limit > 0 && a->held + bytes > limit) {
            char l[32], held[32], need[32];
            set_error("the memory limit on %s is %s (%s): GPAP holds %s there, and a buffer of "
                      "%s would pass it. Raise the limit, or free an engine",
                      a->name.c_str(), fmt_bytes(limit, l), source_name(source),
                      fmt_bytes(a->held, held), fmt_bytes(bytes, need));
            return GPAP_ERR_MEMORY;
        }
        /* The reserve, kept where free memory is measured: a driver that pages device memory
         * to the host (WSL2) grants an allocation past the card instead of failing it, and the
         * run slows down with no error. A pinned batch is the one a sizing does not bound. The
         * sizing keeps all of the reserve; this refuses only past half of it, so what the
         * sizing cannot see (allocation rounding, kernel code a first launch loads, another
         * process) does not fail a call it fit. Two threads that both pass here before either
         * allocates can still take more. */
        GpuMemory m;
        if (h.gpu && h.gpu->memory(&m) == GPAP_OK && m.source == GpuMemory::DEVICE) {
            int64_t reserve = 0;
            GPAP_TRY(resolve_reserve(a, &reserve));
            if (m.free - bytes < reserve / 2) {
                char r[32], fr[32], need[32];
                set_error("GPAP leaves %s free on %s (the memory reserve: GPAP_MEM_RESERVE, "
                          "gpap_set_mem_reserve): %s is free, and a buffer of %s would leave "
                          "less than half of that. Use a smaller batch, free an engine, or "
                          "lower the reserve",
                          fmt_bytes(reserve, r), a->name.c_str(), fmt_bytes(m.free, fr),
                          fmt_bytes(bytes, need));
                return GPAP_ERR_MEMORY;
            }
        }
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

gpap_status parse_mem_spec(const char *spec, const char *label, MemSpec *out) {
    *out = MemSpec{};
    if (!spec) {
        set_error("%sNULL: a memory limit must be a string", label);
        return GPAP_ERR_ARG;
    }
    auto fail = [&](const char *why) {
        set_error("%s'%s': %s", label, spec, why);
        return GPAP_ERR_ARG;
    };
    const char *p = spec;
    while (*p == ' ' || *p == '\t') ++p;
    if (lower(p[0]) == 'n' && lower(p[1]) == 'o' && lower(p[2]) == 'n' && lower(p[3]) == 'e' &&
        blank(p + 4)) {
        out->kind = MemSpec::NONE;
        return GPAP_OK;
    }
    if (*p < '0' || *p > '9')
        return fail("not a memory size: give a size with its unit (8G, 512M), a share of the "
                    "device (75%), or 'none'");
    /* The digits by hand, not strtod: strtod takes exponents, hex and the C locale's decimal
     * point, and none of those is part of the grammar. */
    long double v = 0;
    for (; *p >= '0' && *p <= '9'; ++p) {
        v = v * 10 + (*p - '0');
        if (v > 1e24L) return fail("too large");
    }
    if (*p == '.') {
        ++p;
        if (*p < '0' || *p > '9') return fail("a decimal point needs digits after it");
        long double scale = 0.1L;
        for (; *p >= '0' && *p <= '9'; ++p, scale /= 10) v += (*p - '0') * scale;
    }
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '%') {
        ++p;
        if (!blank(p)) return fail("unexpected text after the '%'");
        if (!(v > 0) || v > 100) return fail("a share must be more than 0% and at most 100%");
        out->kind = MemSpec::SHARE;
        out->percent = (double)v;
        return GPAP_OK;
    }
    int shift = -1;
    switch (lower(*p)) {
        case 'b': shift = 0; break;
        case 'k': shift = 10; break;
        case 'm': shift = 20; break;
        case 'g': shift = 30; break;
        case 't': shift = 40; break;
        default: break;
    }
    if (shift < 0)
        return fail("a size needs its unit: 8G, 512M, 100000B (binary: 1G = 1024^3 bytes), or a "
                    "share such as 75%");
    ++p;
    if (shift > 0) {                     /* K, KB, KiB, Ki: every spelling is binary */
        if (lower(*p) == 'i') ++p;
        if (lower(*p) == 'b') ++p;
    }
    if (!blank(p)) return fail("unexpected text after the unit");
    const long double bytes = std::floor(v * std::ldexp(1.0L, shift));
    if (bytes > (long double)(INT64_MAX / 2)) return fail("too large");
    if (bytes < 1) return fail("must be more than zero");
    out->kind = MemSpec::BYTES;
    out->bytes = (int64_t)bytes;
    return GPAP_OK;
}

int64_t mem_spec_bytes(const MemSpec &s, int64_t total) {
    switch (s.kind) {
        case MemSpec::BYTES: return s.bytes;
        case MemSpec::SHARE:
            return total > 0 ? (int64_t)std::floor((long double)total * s.percent / 100) : 0;
        default: return 0;
    }
}

gpap_status mem_state(MemAccount *a, MemState *out) {
    *out = MemState{};
    if (!a) return GPAP_OK;
    std::lock_guard<std::mutex> lock(a->mu);
    out->held = a->held;
    out->peak = a->peak;
    out->total = a->total;
    GPAP_TRY(resolve(a, &out->limit, &out->source));
    return resolve_reserve(a, &out->reserve);
}

void mem_set_limit(MemAccount *a, const MemSpec &s) {
    if (!a) return;
    std::lock_guard<std::mutex> lock(a->mu);
    a->api = s;
}

void mem_set_reserve(MemAccount *a, const MemSpec &s) {
    if (!a) return;
    std::lock_guard<std::mutex> lock(a->mu);
    a->reserve = s;
}

void mem_reset_peak(MemAccount *a) {
    if (!a) return;
    std::lock_guard<std::mutex> lock(a->mu);
    a->peak = a->held;
}

const char *fmt_bytes(int64_t bytes, char *buf) {
    const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = (double)bytes;
    int u = 0;
    while (u < 4 && (v >= 1024 || v <= -1024)) {
        v /= 1024;
        ++u;
    }
    if (u == 0) std::snprintf(buf, 32, "%lld B", (long long)bytes);
    else std::snprintf(buf, 32, "%.1f %s", v, units[u]);
    return buf;
}

Sizing size_rows(const SizeRequest &r) {
    Sizing best;
    best.rows = INT64_MAX;
    /* Strict `<`: the first bound to reach a count keeps it, so the order below is the tie. */
    auto consider = [&](gpap_bound bound, int64_t bytes, int64_t fixed, int64_t per_row) {
        per_row = std::max<int64_t>(per_row, 1);
        const int64_t left = std::max<int64_t>(bytes - fixed, 0);
        const int64_t rows = left / per_row;
        if (rows < best.rows) best = {rows, bound, left, per_row};
    };
    if (r.limit > 0) {
        /* A limit above the device is the device: an unknown total is the limit alone. */
        const int64_t ceiling = r.total > 0 ? std::min(r.limit, r.total) : r.limit;
        consider(GPAP_BOUND_LIMIT, ceiling - r.held_other, r.fixed, r.per_row);
    }
    if (r.grant >= 0) consider(GPAP_BOUND_GRANT, r.grant, r.fixed, r.per_row);
    if (r.memory >= 0) consider(GPAP_BOUND_MEMORY, r.memory, r.fixed, r.per_row);
    if (best.rows == INT64_MAX) return Sizing{};      /* no bound applied: nothing to size from */
    if (r.cap > 0 && best.rows >= r.cap) {
        best.rows = r.cap;
        best.bound = GPAP_BOUND_CAP;
    }
    /* Each per-row buffer holds `rows` rows, so the widest one bounds them; 0 means one row's
     * widest buffer is already past the device's limit. */
    if (r.max_buffer > 0 && best.rows > 0) {
        const int64_t fit = r.max_buffer / std::max<int64_t>(r.widest, 1);
        if (fit < best.rows) {
            best.rows = fit;
            best.bound = GPAP_BOUND_BUFFER;
        }
    }
    best.rows = std::min<int64_t>(best.rows, INT32_MAX);
    return best;
}

}  // namespace gpap

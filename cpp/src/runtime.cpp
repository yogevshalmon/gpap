/* Error strings, the backend choice, device buffers, version and build id. */
#include "gpap_internal.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>                 /* dladdr, for gpap_build_id */
#endif

namespace gpap {

static thread_local char g_error[1024] = "";

void set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
}

const char *get_error() { return g_error; }

const PackedSplit kPackedSplits[2] = {{16, 65535, 65535}, {8, 0xFFFFFF, 255}};

/* $GPAP_BACKEND picks the plug. Read on every call, never cached: a process may open engines on
 * both, which is how the two are compared on one device. */
const GpuDriver *gpu_driver() {
    const char *b = std::getenv("GPAP_BACKEND");
    if (!b || !*b || std::strcmp(b, "cuda") == 0) return &kCudaDriver;
    if (std::strcmp(b, "opencl") == 0) return &kOpenClDriver;
    set_error("GPAP_BACKEND='%s' is not a backend: use cuda (the default) or opencl", b);
    return nullptr;
}

/* Counted before the plug is asked, so a refusal costs no allocation, and uncounted if the plug
 * then fails: the ledger is never ahead of the device for longer than one call. */
gpap_status DevBuf::reserve(MemHolder &h, size_t n) {
    if (n <= bytes) return GPAP_OK;
    release();
    GPAP_TRY(mem_take(h, kind, (int64_t)n));
    const gpap_status st = h.gpu->alloc(n, &ptr);
    if (st != GPAP_OK) {
        mem_give(h, kind, (int64_t)n);
        ptr = 0;
        return st;
    }
    mem_got(h);
    owner = &h;
    bytes = n;
    return GPAP_OK;
}

gpap_status DevBuf::zero() {
    if (ptr && bytes) return owner->gpu->zero(ptr, bytes);
    return GPAP_OK;
}

void DevBuf::release() {
    if (ptr) {
        owner->gpu->free(ptr);
        mem_give(*owner, kind, (int64_t)bytes);
    }
    ptr = 0;
    bytes = 0;
}

}  // namespace gpap

extern "C" const char *gpap_last_error(void) { return gpap::get_error(); }

#ifndef GPAP_VERSION_STR
#define GPAP_VERSION_STR "0.0.0-dev"      /* built outside CMake */
#endif
extern "C" const char *gpap_version(void) { return GPAP_VERSION_STR; }

/* Fingerprint of the loaded library's own file: FNV-1a 64 (the question is "same build?"). */
extern "C" const char *gpap_build_id(void) {
    /* Built once into a function-local static: the magic-static rule makes initialization
     * thread-safe and publishes the value only when COMPLETE — a hand-rolled memo would hand a
     * concurrent caller the bare "<version>+" prefix, identical for every build. */
    static const std::string id = [] {
        std::string digest = "unknown";
#if defined(__linux__) || defined(__APPLE__)
        Dl_info info{};
        /* Any symbol IN this library resolves to its file; gpap_version is as good as any. */
        if (dladdr(reinterpret_cast<const void *>(&gpap_version), &info) && info.dli_fname) {
            if (FILE *fp = std::fopen(info.dli_fname, "rb")) {
                uint64_t h = 1469598103934665603ull;    /* FNV-1a 64 offset basis */
                std::vector<unsigned char> buf(1 << 16);  /* heap: 64 KB overflows a small stack */
                size_t n;
                while ((n = std::fread(buf.data(), 1, buf.size(), fp)) > 0)
                    for (size_t i = 0; i < n; ++i) h = (h ^ buf[i]) * 1099511628211ull;
                /* ferror: a short read ends the loop as EOF does, so a truncated read would
                 * hash a PREFIX and return it as a fingerprint. */
                const bool complete = std::ferror(fp) == 0;
                std::fclose(fp);
                if (complete) {
                    char hex[17];
                    std::snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)h);
                    digest = hex;
                }
            }
        }
#endif
        return std::string(GPAP_VERSION_STR) + "+" + digest;
    }();
    return id.c_str();
}

/* A driver for tests/engine/test_size_rows.py: the batch rule (gpap::size_rows, cpp/src/memory.cpp)
 * over requests read from stdin, one per line, its answers on stdout. The rule is pure, so this
 * exercises every bound with no device, the GRANT a host's own plug reports included, which no
 * plug on this tree does.
 *
 * In:  total memory grant limit held_other per_row fixed cap max_buffer widest
 * Out: rows bound left per_row */
#include <cstdio>

#include "memory.hpp"

int main() {
    long long v[10];
    while (std::scanf("%lld %lld %lld %lld %lld %lld %lld %lld %lld %lld", &v[0], &v[1], &v[2],
                      &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9]) == 10) {
        gpap::SizeRequest r;
        r.total = v[0];
        r.memory = v[1];
        r.grant = v[2];
        r.limit = v[3];
        r.held_other = v[4];
        r.per_row = v[5];
        r.fixed = v[6];
        r.cap = (int)v[7];
        r.max_buffer = v[8];
        r.widest = v[9];
        const gpap::Sizing s = gpap::size_rows(r);
        std::printf("%lld %d %lld %lld\n", (long long)s.rows, (int)s.bound, (long long)s.left,
                    (long long)s.per_row);
    }
    return 0;
}

/* GPAP's BCP kernels in OpenCL C: the fixpoint kernels of bcp.py (scalar and WAVE, plain and
 * CLEAN) and the support kernels of helpers.py, for a GPU library that cannot load Triton's
 * binaries.
 *
 * BUILD INPUT: CMake embeds this file in libgpap as a string; the OpenCL plug
 * (cpp/src/gpu_opencl.cpp) compiles it on the device at first use.
 *
 * bcp.py and helpers.py are the REFERENCE. Every function here names the Triton function it
 * translates and keeps its contract (the docstring there); a change of behaviour there must be
 * mirrored here, a speed change need not be. Propagation reaches the same fixpoint in any
 * order, so the two must agree exactly on conflict flags, and on the assignment and counters
 * of every row without a conflict (tests/engine/test_opencl_backend.py).
 *
 * What differs from the Triton source, and why:
 *   * A SCALAR KERNEL (.w0) GIVES A ROW ONE WORK ITEM, where Triton gives it a 32-thread
 *     program: no lanes, no lane races and no atomics.
 *   * A WAVE KERNEL (.w1) GIVES A ROW ONE WORK GROUP of GPAP_WAVE_W work items, and declares
 *     that size, so no other launch is accepted. Triton's lanes talk through warp shuffles,
 *     which OpenCL C does not have; here they share a few words of local memory and meet at
 *     barriers (gpap_wave_row).
 *   * SCALARS TRAVEL IN A BUFFER. A kernel's first argument is its parameter block `prm` of
 *     64-bit slots, because some hosts pass buffers only:
 *         prm[0], prm[1]   the work extent (n0, n1): rows, or rows x columns
 *         prm[2]           columns one work item covers (the row-x-column kernels; the others
 *                          ignore it)
 *         prm[3 + i]       the i-th scalar of the Triton signature
 *     The buffers of the Triton signature follow, in its order.
 *   * EVERY LAUNCH IS ONE-DIMENSIONAL. A row-x-column kernel numbers its work items
 *     row * blocks + block (GPAP_GRID_ITEM), because a second launch dimension is limited to
 *     65,535 groups on implementations built over CUDA, and a batch has more rows than that.
 *   * THE LAUNCH IS ROUNDED UP to whole work groups, so every kernel checks its ids against
 *     the extent before touching memory.
 *   * WORDS ARE UNSIGNED. The packed word's sum field wraps off the top by design; that is
 *     defined for `uint` and undefined for `int` in C.
 *
 * Plain OpenCL C 1.2: no vendor extension and no host macro, so the same file runs under
 * libgpap's OpenCL plug, under another host's loader and on a CPU implementation. Not ported:
 * the trajectory (sampling) kernel. */

#define GPAP_PRM_N0 0
#define GPAP_PRM_N1 1
#define GPAP_PRM_CHUNK 2
#define GPAP_PRM_ARG 3

/* Most work items one reduction group may have (gpap_sum_i8_to, gpap_sum_i32_acc). */
#define GPAP_GROUP_MAX 256

/* A row-x-column kernel's work item: its ROW and its first column COL0, from the flat id. The
 * columns are cut into blocks of prm[2]; item id = row * blocks + block. */
#define GPAP_GRID_ITEM(prm, ROW, COL0)                                                        \
    const long _blocks = (prm[GPAP_PRM_N1] + prm[GPAP_PRM_CHUNK] - 1) / prm[GPAP_PRM_CHUNK]; \
    const long ROW = (long)get_global_id(0) / _blocks;                                        \
    const long COL0 = ((long)get_global_id(0) % _blocks) * prm[GPAP_PRM_CHUNK];

/* The trail cursor and the two results _propagate carries: Triton returns them as a tuple. */
typedef struct {
    int head, tail, conflict, nops;
} gpap_trail;

/* The forced-literal push both halves of _propagate share: literal `forced` must become TRUE.
 * One load of its bitmap word answers already-true (nothing to do), contradiction (conflict)
 * and free (set the bit, queue the literal). */
void gpap_push(__global uint *ab_row, __global int *trail_row, const int forced, gpap_trail *t,
               const int TRAIL_CAP)
{
    const uint w = ab_row[forced >> 5];
    const uint bit = 1u << (forced & 31);
    if ((w & bit) != 0u) return;
    if ((w & (1u << ((forced ^ 1) & 31))) != 0u) {
        t->conflict = 1;
    } else if (t->tail < TRAIL_CAP) {
        ab_row[forced >> 5] = w | bit;
        trail_row[t->tail] = forced;
        t->tail += 1;
    } else {
        t->conflict = 1;          /* unreachable: a variable enters the trail once, cap >= N */
    }
}

/* bcp.py::_propagate — pop literals off the trail and BCP to fixpoint (or conflict) on the
 * packed row. Each occ[lit] segment is [BINARY | LONG], split at occ_mid[lit]; a binary
 * entry's payload IS the forced literal, a long entry's is the clause index. */
void gpap_propagate(__global uint *st_row, __global uint *ab_row, __global int *trail_row,
                    __global const int *occ_off, __global const int *occ_mid,
                    __global const int *occ_cls, gpap_trail *t, const int TRAIL_CAP,
                    const int SHIFT)
{
    const uint SAT_MASK = (1u << SHIFT) - 1u;
    const uint SUM_MASK = (1u << (32 - SHIFT)) - 1u;
    while (t->head < t->tail && t->conflict == 0) {
        const int lit = trail_row[t->head];
        t->head += 1;
        const uint neg = (uint)(lit ^ 1);
        const int start = occ_off[lit];
        const int mid = occ_mid[lit];
        const int end = occ_off[lit + 1];
        t->nops += end - start;
        int o = start;
        while (o < mid && t->conflict == 0) {   /* binary prefix: payload IS the forced lit */
            gpap_push(ab_row, trail_row, occ_cls[o] & 0x7FFFFFFF, t, TRAIL_CAP);
            o += 1;
        }
        o = mid;
        while (o < end && t->conflict == 0) {   /* long suffix: packed clause words */
            const int c = occ_cls[o];
            const uint st = st_row[c] - 1u - (neg << SHIFT);
            st_row[c] = st;
            const uint sat = st & SAT_MASK;
            if (sat == 0u)
                t->conflict = 1;
            else if (sat == 1u)                 /* unit: remaining literal IS the sum field */
                gpap_push(ab_row, trail_row, (int)((st >> SHIFT) & SUM_MASK), t, TRAIL_CAP);
            o += 1;
        }
    }
}

/* The seed loop of bcp.py::_seed_and_propagate: assign each decision literal of row k and queue
 * it. Leaves the cursor at tail = the decisions assigned, conflict = two seeds contradict. */
void gpap_seed(__global uint *ab_row, __global int *trail_row, __global const long *dec,
               const long k, const int B, gpap_trail *t)
{
    t->head = 0;
    t->tail = 0;
    t->conflict = 0;
    t->nops = 0;
    for (int j = 0; j < B; ++j) {
        const long lit64 = dec[k * (long)B + j];
        if (lit64 >= 0 && t->conflict == 0) {
            const int lit = (int)lit64;
            const uint w = ab_row[lit >> 5];        /* lit and lit^1 share this word */
            if ((w & (1u << ((lit ^ 1) & 31))) != 0u) {
                t->conflict = 1;                    /* contradicts existing / earlier seed */
            } else if ((w & (1u << (lit & 31))) == 0u) {
                ab_row[lit >> 5] = w | (1u << (lit & 31));
                trail_row[t->tail] = lit;
                t->tail += 1;
            }
        }
    }
}

/* The last lines of bcp.py::_seed_and_propagate: the row's flags and its three counters. */
void gpap_row_results(const long k, __global char *active, __global char *conflict,
                      __global int *nprops, __global int *nops, __global int *ndecs,
                      const int tail, const int conflicted, const int n_ops, const int decided)
{
    if (conflicted != 0) {
        conflict[k] = 1;
        active[k] = 0;
    }
    nprops[k] = tail;
    nops[k] = n_ops;
    ndecs[k] = decided;
}

/* bcp.py::_clean_packed, on one work item — the CLEAN tail: rewrite every packed word this row
 * can have touched back to the baseline `init`. Packed writes only happen in the occurrence
 * walk of literals popped from the trail, so the long sub-ranges of trail[0..tail) cover them,
 * on surviving and conflicted rows alike. A row whose walk exceeds M/8 copies the whole row. */
void gpap_clean_packed(__global uint *st_row, __global const uint *init,
                       __global const int *trail_row, __global const int *occ_off,
                       __global const int *occ_mid, __global const int *occ_cls,
                       const int tail, const long M)
{
    long total = 0;
    for (int t = 0; t < tail; ++t) {
        const int lit = trail_row[t];
        total += occ_off[lit + 1] - occ_mid[lit];
    }
    if (total > M / 8) {
        for (long j = 0; j < M; ++j) st_row[j] = init[j];
        return;
    }
    for (int t = 0; t < tail; ++t) {
        const int lit = trail_row[t];
        const int end = occ_off[lit + 1];
        for (int o = occ_mid[lit]; o < end; ++o) {
            const int c = occ_cls[o];
            st_row[c] = init[c];
        }
    }
}

/* bcp.py::_bcp_fixpoint_kernel and _bcp_fixpoint_clean_kernel, WAVE = 0: seed row k, propagate
 * to fixpoint, write the flags and counters; with CLEAN, then restore the packed row (`init`
 * is read only then). One work item per row. */
void gpap_scalar_row(__global const long *prm, __global uint *st, __global uint *ab,
                     __global char *active, __global char *conflict, __global const long *dec,
                     __global const int *occ_off, __global const int *occ_mid,
                     __global const int *occ_cls, __global const uint *init,
                     __global int *trail, __global int *nprops, __global int *nops,
                     __global int *ndecs, const int SHIFT, const int CLEAN)
{
    const long k = (long)get_global_id(0);
    if (k >= prm[GPAP_PRM_N0]) return;
    const long M = prm[GPAP_PRM_ARG + 0];
    const long NW = prm[GPAP_PRM_ARG + 1];
    const int B = (int)prm[GPAP_PRM_ARG + 2];
    const int TRAIL_CAP = (int)prm[GPAP_PRM_ARG + 3];
    if (active[k] == 0) return;
    __global uint *st_row = st + k * M;
    __global uint *ab_row = ab + k * NW;
    __global int *trail_row = trail + k * (long)TRAIL_CAP;

    gpap_trail t;
    gpap_seed(ab_row, trail_row, dec, k, B, &t);
    const int decided = t.tail;                     /* seeds assigned so far ARE the decisions */
    gpap_propagate(st_row, ab_row, trail_row, occ_off, occ_mid, occ_cls, &t, TRAIL_CAP, SHIFT);
    gpap_row_results(k, active, conflict, nprops, nops, ndecs, t.tail, t.conflict, t.nops,
                     decided);
    if (CLEAN)
        gpap_clean_packed(st_row, init, trail_row, occ_off, occ_mid, occ_cls, t.tail, M);
}

/* ------------------------------------------------------------------ the WAVE kernels */

/* Lanes per row: a .w1 kernel runs one work group of exactly this many items per row (item id
 * = row * GPAP_WAVE_W + lane), and declares it (reqd_work_group_size), so a launch with any
 * other group size is refused instead of mixing rows. A power of two. WAVE_W in
 * cpp/tools/build_kernels.py is the Triton kernels' value. */
#define GPAP_WAVE_W 32

/* The two words of row state the lanes share, in local memory. */
#define GPAP_SH_TAIL 0
#define GPAP_SH_CONFLICT 1

/* One occurrence entry of a wave: the packed-word atomic, then the bitmap atomic and the trail
 * append for a literal it forces. Returns 1 when this lane found a conflict. */
int gpap_wave_entry(__global uint *st_row, __global uint *ab_row, __global int *trail_row,
                    __global const int *occ_cls, volatile __local int *sh, const int o,
                    const uint neg, const int TRAIL_CAP, const int SHIFT)
{
    const uint SAT_MASK = (1u << SHIFT) - 1u;
    const uint SUM_MASK = (1u << (32 - SHIFT)) - 1u;
    const int pay = occ_cls[o];
    int forced = -1;
    if (pay < 0) {                                  /* tagged: a binary entry IS the forced lit */
        forced = pay & 0x7FFFFFFF;
    } else {
        const uint old = atomic_sub((volatile __global uint *)(st_row + pay), 1u + (neg << SHIFT));
        const uint oldsat = old & SAT_MASK;
        if (oldsat == 1u) return 1;                 /* this lane emptied the clause */
        if (oldsat == 2u)                           /* ... or made it unit */
            forced = (int)(((old >> SHIFT) - neg) & SUM_MASK);
    }
    if (forced >= 0) {
        const uint bit = 1u << (forced & 31);
        const uint oldw = atomic_or((volatile __global uint *)(ab_row + (forced >> 5)), bit);
        if ((oldw & (1u << ((forced ^ 1) & 31))) != 0u) return 1;   /* both polarities pushed */
        if ((oldw & bit) == 0u) {
            const int pos = atomic_inc(sh + GPAP_SH_TAIL);
            if (pos >= TRAIL_CAP) return 1;         /* unreachable: as in gpap_push */
            trail_row[pos] = forced;
        }
    }
    return 0;
}

/* bcp.py::_bcp_fixpoint_kernel and _bcp_fixpoint_clean_kernel, WAVE = 1 (_wave_propagate and
 * the lanes of _clean_packed): the GPAP_WAVE_W work items of one group propagate one row.
 *
 * A wave pops up to W trail literals and spreads the union of their occurrence segments over
 * the lanes: entry t of the flattened union belongs to lane t mod W, as in the Triton kernel.
 * Both state updates are one atomic whose RETURN VALUE is the detection logic, exactly as
 * there: `atomic_sub` on the packed word (old count 1 = this lane emptied the clause, old
 * count 2 = it made it unit and the forced literal is (old_sum - neg) & SUM_MASK), `atomic_or`
 * on the bitmap word (old bit l = duplicate, old bit l^1 = contradiction).
 *
 * What replaces Triton's warp shuffles:
 *   * the trail's tail and the conflict flag live in local memory (`sh`). A lane appends a
 *     forced literal at `atomic_inc(tail)`, so the trail's order within a wave is whatever the
 *     lanes' race gives; fixpoints are confluent, so no surviving row can tell;
 *   * a wave has two barriers. Before the first, lane j publishes the segment of frontier
 *     literal j (`seg_*`); between them every lane takes its entries of the flattened
 *     frontier; after the second, every lane reads the same tail and conflict flag, so all of
 *     them take the same branch at the top of the loop. That is what makes a barrier inside
 *     the loop legal.
 * A lane that finds a conflict stops; the others finish the wave, on a row whose state is
 * unspecified from then on. Only lane 0 seeds the row and writes its results.
 *
 * `occ_mid` is read by the CLEAN tail only: the propagate reads the binary/long split off the
 * payload's tag bit. */

/* bcp.py::_wave_propagate — the waves of one row, from a seeded trail to its fixpoint or a
 * conflict, with the frontier flattened over the lanes as there. Every lane enters
 * straight from a barrier, with the tail and the conflict flag in `sh`, and leaves straight
 * from one. Gives each lane the trail depth, the conflict flag and the entries walked. */
void gpap_wave_propagate(__global uint *st_row, __global uint *ab_row, __global int *trail_row,
                         __global const int *occ_off, __global const int *occ_cls,
                         volatile __local int *sh, __local int *seg_s, __local int *seg_e,
                         __local uint *seg_neg, const int TRAIL_CAP, const int SHIFT,
                         int *tail_out, int *conflicted_out, int *n_ops_out)
{
    const int W = GPAP_WAVE_W;
    const int lane = (int)get_local_id(0);
    int head = 0, n_ops = 0, tail = 0, conflicted = 0;
    for (;;) {
        tail = sh[GPAP_SH_TAIL];
        conflicted = sh[GPAP_SH_CONFLICT];
        if (head >= tail || conflicted != 0) break;
        const int f = tail - head < W ? tail - head : W;
        if (lane < f) {                             /* publish frontier literal `lane` */
            const int lit = trail_row[head + lane];
            seg_s[lane] = occ_off[lit];
            seg_e[lane] = occ_off[lit + 1];
            seg_neg[lane] = (uint)(lit ^ 1);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* This lane owns entries lane, lane + W, ... of the flattened frontier, and finds each
         * by walking the published segments in local memory. In round r every lane that has
         * an entry is then at the same instruction, so their loads and atomics go out
         * together and a wave costs one memory round trip per round, as in the Triton kernel.
         * Walking the segments one after the other costs one per segment, which was most of
         * this kernel's distance from CUDA (docs/findings/OPENCL_FINDINGS.md §10). */
        int base = 0;                               /* flattened entries before segment j */
        int mine = 0;                               /* this lane found a conflict */
        int j = 0;
        int s = seg_s[0];
        int e = seg_e[0];
        for (int t = lane;; t += W) {
            while (j < f && t >= base + (e - s)) {
                base += e - s;
                ++j;
                if (j < f) {
                    s = seg_s[j];
                    e = seg_e[j];
                }
            }
            if (j >= f) break;                      /* past the last entry: base is the count */
            if (mine == 0)
                mine = gpap_wave_entry(st_row, ab_row, trail_row, occ_cls, sh, s + (t - base),
                                       seg_neg[j], TRAIL_CAP, SHIFT);
        }
        if (mine != 0) atomic_xchg(sh + GPAP_SH_CONFLICT, 1);
        head += f;
        n_ops += base;                              /* base is now the wave's entry count */
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    if (tail > TRAIL_CAP) tail = TRAIL_CAP;         /* the unreachable overflow, kept in bounds */
    *tail_out = tail;
    *conflicted_out = conflicted;
    *n_ops_out = n_ops;
}

/* bcp.py::_clean_packed on the lanes of a row: lane i owns trail literals i, i + W, ... Two
 * lanes may store the same baseline word to one clause, which is harmless. The walk's size is
 * summed across the lanes first (in seg_s), for the M/8 guard. Contains one barrier. */
void gpap_wave_clean(__global uint *st_row, __global const uint *init,
                     __global const int *trail_row, __global const int *occ_off,
                     __global const int *occ_mid, __global const int *occ_cls,
                     __local int *seg_s, const int tail, const long M)
{
    const int W = GPAP_WAVE_W;
    const int lane = (int)get_local_id(0);
    int total = 0;
    for (int t = lane; t < tail; t += W) {
        const int lit = trail_row[t];
        total += occ_off[lit + 1] - occ_mid[lit];
    }
    seg_s[lane] = total;
    barrier(CLK_LOCAL_MEM_FENCE);
    long walk = 0;
    for (int j = 0; j < W; ++j) walk += seg_s[j];
    if (walk > M / 8) {
        for (long j = lane; j < M; j += W) st_row[j] = init[j];
    } else {
        for (int t = lane; t < tail; t += W) {
            const int lit = trail_row[t];
            const int end = occ_off[lit + 1];
            for (int o = occ_mid[lit]; o < end; ++o) {
                const int c = occ_cls[o];
                st_row[c] = init[c];
            }
        }
    }
}

/* One WAVE row of the fixpoint kernels: lane 0 seeds it, the lanes propagate it, lane 0 writes
 * its results, and with CLEAN the lanes restore its packed words. */
void gpap_wave_row(__global const long *prm, __global uint *st, __global uint *ab,
                   __global char *active, __global char *conflict, __global const long *dec,
                   __global const int *occ_off, __global const int *occ_mid,
                   __global const int *occ_cls, __global const uint *init,
                   __global int *trail, __global int *nprops, __global int *nops,
                   __global int *ndecs, volatile __local int *sh, __local int *seg_s,
                   __local int *seg_e, __local uint *seg_neg, const int SHIFT, const int CLEAN)
{
    const int W = GPAP_WAVE_W;
    const int lane = (int)get_local_id(0);
    const long k = (long)get_global_id(0) / W;
    /* Both early returns are taken by a whole group or by none: every lane reads active[k]
     * before the seed barrier, and lane 0 writes it only after that barrier. */
    if (k >= prm[GPAP_PRM_N0]) return;
    const long M = prm[GPAP_PRM_ARG + 0];
    const long NW = prm[GPAP_PRM_ARG + 1];
    const int B = (int)prm[GPAP_PRM_ARG + 2];
    const int TRAIL_CAP = (int)prm[GPAP_PRM_ARG + 3];
    if (active[k] == 0) return;
    __global uint *st_row = st + k * M;
    __global uint *ab_row = ab + k * NW;
    __global int *trail_row = trail + k * (long)TRAIL_CAP;

    int decided = 0;
    if (lane == 0) {
        gpap_trail t;
        gpap_seed(ab_row, trail_row, dec, k, B, &t);
        sh[GPAP_SH_TAIL] = t.tail;
        sh[GPAP_SH_CONFLICT] = t.conflict;
        decided = t.tail;                           /* seeds assigned so far ARE the decisions */
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int tail = 0, conflicted = 0, n_ops = 0;
    gpap_wave_propagate(st_row, ab_row, trail_row, occ_off, occ_cls, sh, seg_s, seg_e, seg_neg,
                        TRAIL_CAP, SHIFT, &tail, &conflicted, &n_ops);
    if (lane == 0)
        gpap_row_results(k, active, conflict, nprops, nops, ndecs, tail, conflicted, n_ops,
                         decided);

    if (CLEAN)
        gpap_wave_clean(st_row, init, trail_row, occ_off, occ_mid, occ_cls, seg_s, tail, M);
}

/* The kernels. Extent: (K). Scalars: M, NW, B, TRAIL_CAP. A CLEAN kernel takes `init`, the
 * (M,) baseline row its tail restores, after occ_cls. */
#define GPAP_FIXPOINT_KERNEL(NAME, SHIFT)                                                     \
    __kernel void NAME(__global const long *prm, __global uint *st, __global uint *ab,       \
                       __global char *active, __global char *conflict,                       \
                       __global const long *dec, __global const int *occ_off,                \
                       __global const int *occ_mid, __global const int *occ_cls,             \
                       __global int *trail, __global int *nprops, __global int *nops,        \
                       __global int *ndecs)                                                  \
    {                                                                                         \
        gpap_scalar_row(prm, st, ab, active, conflict, dec, occ_off, occ_mid, occ_cls,       \
                        (__global const uint *)0, trail, nprops, nops, ndecs, SHIFT, 0);     \
    }

#define GPAP_FIXPOINT_CLEAN_KERNEL(NAME, SHIFT)                                               \
    __kernel void NAME(__global const long *prm, __global uint *st, __global uint *ab,       \
                       __global char *active, __global char *conflict,                       \
                       __global const long *dec, __global const int *occ_off,                \
                       __global const int *occ_mid, __global const int *occ_cls,             \
                       __global const uint *init, __global int *trail,                       \
                       __global int *nprops, __global int *nops, __global int *ndecs)        \
    {                                                                                         \
        gpap_scalar_row(prm, st, ab, active, conflict, dec, occ_off, occ_mid, occ_cls,       \
                        init, trail, nprops, nops, ndecs, SHIFT, 1);                         \
    }

/* Local memory must be declared in the kernel function itself. */
#define GPAP_WAVE_LOCALS                                                                      \
    volatile __local int sh[2];                                                               \
    __local int seg_s[GPAP_WAVE_W];                                                           \
    __local int seg_e[GPAP_WAVE_W];                                                           \
    __local uint seg_neg[GPAP_WAVE_W];

#define GPAP_WAVE_GROUP __attribute__((reqd_work_group_size(GPAP_WAVE_W, 1, 1)))

#define GPAP_WAVE_KERNEL(NAME, SHIFT)                                                         \
    __kernel GPAP_WAVE_GROUP                                                                  \
    void NAME(__global const long *prm, __global uint *st, __global uint *ab,                \
                       __global char *active, __global char *conflict,                       \
                       __global const long *dec, __global const int *occ_off,                \
                       __global const int *occ_mid, __global const int *occ_cls,             \
                       __global int *trail, __global int *nprops, __global int *nops,        \
                       __global int *ndecs)                                                  \
    {                                                                                         \
        GPAP_WAVE_LOCALS                                                                      \
        gpap_wave_row(prm, st, ab, active, conflict, dec, occ_off, occ_mid, occ_cls,         \
                      (__global const uint *)0, trail, nprops, nops, ndecs, sh, seg_s,       \
                      seg_e, seg_neg, SHIFT, 0);                                             \
    }

#define GPAP_WAVE_CLEAN_KERNEL(NAME, SHIFT)                                                   \
    __kernel GPAP_WAVE_GROUP                                                                  \
    void NAME(__global const long *prm, __global uint *st, __global uint *ab,                \
                       __global char *active, __global char *conflict,                       \
                       __global const long *dec, __global const int *occ_off,                \
                       __global const int *occ_mid, __global const int *occ_cls,             \
                       __global const uint *init, __global int *trail,                       \
                       __global int *nprops, __global int *nops, __global int *ndecs)        \
    {                                                                                         \
        GPAP_WAVE_LOCALS                                                                      \
        gpap_wave_row(prm, st, ab, active, conflict, dec, occ_off, occ_mid, occ_cls,         \
                      init, trail, nprops, nops, ndecs, sh, seg_s, seg_e, seg_neg, SHIFT,    \
                      1);                                                                    \
    }

GPAP_FIXPOINT_KERNEL(gpap_fixpoint_s16_w0, 16)
GPAP_FIXPOINT_KERNEL(gpap_fixpoint_s8_w0, 8)
GPAP_FIXPOINT_CLEAN_KERNEL(gpap_fixpoint_clean_s16_w0, 16)
GPAP_FIXPOINT_CLEAN_KERNEL(gpap_fixpoint_clean_s8_w0, 8)
GPAP_WAVE_KERNEL(gpap_fixpoint_s16_w1, 16)
GPAP_WAVE_KERNEL(gpap_fixpoint_s8_w1, 8)
GPAP_WAVE_CLEAN_KERNEL(gpap_fixpoint_clean_s16_w1, 16)
GPAP_WAVE_CLEAN_KERNEL(gpap_fixpoint_clean_s8_w1, 8)

/* ------------------------------------------------------------------ support kernels */

/* helpers.py::bcast_rows — dst (K, n_cols) int32 <- src (n_cols,) broadcast over rows.
 * Extent: (K, n_cols). Scalars: n_cols. */
__kernel void gpap_bcast_rows(__global const long *prm, __global int *dst,
                              __global const int *src)
{
    GPAP_GRID_ITEM(prm, r, c0)
    const long n_cols = prm[GPAP_PRM_N1];
    if (r >= prm[GPAP_PRM_N0]) return;
    const long c1 = c0 + prm[GPAP_PRM_CHUNK] < n_cols ? c0 + prm[GPAP_PRM_CHUNK] : n_cols;
    for (long c = c0; c < c1; ++c) dst[r * n_cols + c] = src[c];
}

/* helpers.py::fill_i8 — dst (n,) int8 <- value. Extent: (n). Scalars: value, n. */
__kernel void gpap_fill_i8(__global const long *prm, __global char *dst)
{
    const long i = (long)get_global_id(0);
    if (i >= prm[GPAP_PRM_N0]) return;
    dst[i] = (char)prm[GPAP_PRM_ARG + 0];
}

/* helpers.py::complete_mask — per row: out = active AND no free variable among the pair mask.
 * A pair (2v, 2v+1) is free iff both bits are 0: the even bit of ~(w | w >> 1), which `pm`
 * selects. Extent: (K). Scalars: NW. */
__kernel void gpap_complete_mask(__global const long *prm, __global const uint *ab,
                                 __global const uint *pm, __global const char *act,
                                 __global char *out)
{
    const long k = (long)get_global_id(0);
    if (k >= prm[GPAP_PRM_N0]) return;
    const long NW = prm[GPAP_PRM_ARG + 0];
    int sel = act[k] != 0;
    for (long i = 0; sel && i < NW; ++i) {
        const uint w = ab[k * NW + i];
        if (((~(w | (w >> 1))) & pm[i]) != 0u) sel = 0;
    }
    out[k] = (char)sel;
}

/* helpers.py::sound_rows — per selected row i: bad[i] = 1 iff some clause is EMPTIED. Long
 * clauses read their packed word's count, binary and unit clauses the bitmap through the
 * (n_pairs, 2) pair list. Extent: (n_rows). Scalars: M, NW, n_pairs. */
#define GPAP_SOUND_ROWS_KERNEL(NAME, SHIFT)                                                   \
    __kernel void NAME(__global const long *prm, __global const uint *st,                    \
                       __global const uint *ab, __global const int *rows,                    \
                       __global const int *pairs, __global char *bad)                        \
    {                                                                                         \
        const long i = (long)get_global_id(0);                                                \
        if (i >= prm[GPAP_PRM_N0]) return;                                                    \
        const long M = prm[GPAP_PRM_ARG + 0];                                                 \
        const long NW = prm[GPAP_PRM_ARG + 1];                                                \
        const long n_pairs = prm[GPAP_PRM_ARG + 2];                                           \
        const long r = (long)rows[i];                                                         \
        const uint SAT_MASK = (1u << SHIFT) - 1u;                                             \
        int emptied = 0;                                                                      \
        for (long c = 0; c < M; ++c)                                                          \
            if ((st[r * M + c] & SAT_MASK) == 0u) emptied = 1;                                \
        for (long j = 0; j < n_pairs; ++j) {                                                  \
            const int na = pairs[2 * j] ^ 1;                                                  \
            const int nb = pairs[2 * j + 1] ^ 1;                                              \
            const uint fa = (ab[r * NW + (na >> 5)] >> (na & 31)) & 1u;                       \
            const uint fb = (ab[r * NW + (nb >> 5)] >> (nb & 31)) & 1u;                       \
            if ((fa & fb) != 0u) emptied = 1;                                                 \
        }                                                                                     \
        bad[i] = (char)emptied;                                                               \
    }

GPAP_SOUND_ROWS_KERNEL(gpap_sound_rows_s16, 16)
GPAP_SOUND_ROWS_KERNEL(gpap_sound_rows_s8, 8)

/* helpers.py::var_bits — out (n_rows, n_cols) int8 = 1 iff variable cols[j] is TRUE in row
 * rows[i]. Extent: (n_rows, n_cols). Scalars: n_cols, NW. */
__kernel void gpap_var_bits(__global const long *prm, __global const uint *ab,
                            __global const int *rows, __global const int *cols,
                            __global char *out)
{
    GPAP_GRID_ITEM(prm, i, c0)
    const long n_cols = prm[GPAP_PRM_N1];
    if (i >= prm[GPAP_PRM_N0]) return;
    const long c1 = c0 + prm[GPAP_PRM_CHUNK] < n_cols ? c0 + prm[GPAP_PRM_CHUNK] : n_cols;
    const long NW = prm[GPAP_PRM_ARG + 1];
    const long r = (long)rows[i];
    for (long c = c0; c < c1; ++c) {
        const int v = cols[c];
        out[i * n_cols + c] = (char)((ab[r * NW + (v >> 4)] >> ((v & 15) * 2)) & 1u);
    }
}

/* helpers.py::unpack_rows — out (n_rows, 2N) int8 = the byte-per-literal assignment of the
 * selected rows. Extent: (n_rows, 2N). Scalars: two_n, NW. */
__kernel void gpap_unpack_rows(__global const long *prm, __global const uint *ab,
                               __global const int *rows, __global char *out)
{
    GPAP_GRID_ITEM(prm, i, l0)
    const long two_n = prm[GPAP_PRM_N1];
    if (i >= prm[GPAP_PRM_N0]) return;
    const long l1 = l0 + prm[GPAP_PRM_CHUNK] < two_n ? l0 + prm[GPAP_PRM_CHUNK] : two_n;
    const long NW = prm[GPAP_PRM_ARG + 1];
    const long r = (long)rows[i];
    for (long l = l0; l < l1; ++l)
        out[i * two_n + l] = (char)((ab[r * NW + (l >> 5)] >> (l & 31)) & 1u);
}

/* helpers.py::sum_i8_to — out[0] (int32) = sum(src[0:n]). The first work group does it, with at
 * most GPAP_GROUP_MAX of its items, so any launch shape gives the right sum: each item sums a
 * stride of the array, item 0 adds the partial sums after the barrier. Scalars: n. */
__kernel void gpap_sum_i8_to(__global const long *prm, __global const char *src,
                             __global int *out)
{
    __local int part[GPAP_GROUP_MAX];
    if (get_group_id(0) != 0) return;               /* a whole group returns: no barrier split */
    const long n = prm[GPAP_PRM_ARG + 0];
    const int g = min((int)get_local_size(0), GPAP_GROUP_MAX);
    const int t = (int)get_local_id(0);
    int acc = 0;
    if (t < g) {
        for (long i = t; i < n; i += g) acc += (int)src[i];
        part[t] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
        int total = 0;
        for (int j = 0; j < g; ++j) total += part[j];
        out[0] = total;
    }
}

/* helpers.py::sum_i32_acc — acc[0] (int64) += sum(src[0:n]). The first work group, as above:
 * the per-launch work counters accumulate on the device, with no host sync. Scalars: n. */
__kernel void gpap_sum_i32_acc(__global const long *prm, __global const int *src,
                               __global long *acc)
{
    __local long part[GPAP_GROUP_MAX];
    if (get_group_id(0) != 0) return;
    const long n = prm[GPAP_PRM_ARG + 0];
    const int g = min((int)get_local_size(0), GPAP_GROUP_MAX);
    const int t = (int)get_local_id(0);
    long sum = 0;
    if (t < g) {
        for (long i = t; i < n; i += g) sum += (long)src[i];
        part[t] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
        long total = 0;
        for (int j = 0; j < g; ++j) total += part[j];
        acc[0] += total;
    }
}

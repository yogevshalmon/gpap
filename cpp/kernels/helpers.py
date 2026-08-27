"""Helper Triton kernels the C core needs around the two BCP kernels.

These exist so the whole toolchain stays **nvcc-free**: everything that runs on the GPU —
including the state broadcast, the completion test and the readout gathers — is Triton, AOT
compiled into the library by cpp/tools/build_kernels.py. They replace what the retired Python
engine expressed as torch ops (`copy_(expand(...))`, `any_free`, `var_true_bits`,
`_bits_to_bytes`, `sat_rows(...) > 0`, `_nops.sum()`).

Conventions shared with cpp/kernels/bcp.py:
  * assignment BITMAP: bit l of (K, NW) int32 words = literal l TRUE; l and l^1 share a word.
  * packed clause word: low SHIFT bits = sat_count, high bits = modular literal sum.
  * one program owns one row wherever a row is the unit of work.

NO SCALAR ATOMICS anywhere in here. Triton executes scalar code redundantly on every thread of
the program (the documented num_warps gotcha — see docs/findings/INSIGHTS.md), which is benign
for a scalar store of a uniform value but would multiply an atomic counter by the thread count.
Counts are therefore produced by a separate single-program reduction over a per-row flag array.
"""
import triton
import triton.language as tl


@triton.jit
def bcast_rows(dst_ptr, src_ptr, n_cols, BLOCK: tl.constexpr):
    """dst (K, n_cols) int32 <- src (n_cols,) int32 broadcast over rows. grid = (K, ceil(n_cols/BLOCK)).
    This is `reset(K)`: the post-units-flush row restored into every row of the chunk."""
    r = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_cols
    v = tl.load(src_ptr + offs, mask=m, other=0)
    tl.store(dst_ptr + r * n_cols + offs, v, mask=m)


@triton.jit
def fill_i8(dst_ptr, value, n, BLOCK: tl.constexpr):
    """dst (n,) int8 <- value. grid = (ceil(n/BLOCK),). (cuMemsetD8 covers 0/1 fills, but the
    active/conflict flags are set from a runtime flag, so one kernel keeps it on-stream.)"""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst_ptr + offs, value.to(tl.int8), mask=offs < n)


@triton.jit
def complete_mask(ab_ptr, pm_ptr, act_ptr, out_ptr, NW, BLOCK: tl.constexpr):
    """Per row: out = active AND no free variable among the pair mask. grid = (K,).

    A variable pair (2v, 2v+1) is free iff both bits are 0, i.e. the even bit of ~(w | w>>1);
    `pm_ptr` carries the even bit of every selected variable (padding pairs stay 0, which is
    why the mask is required rather than a raw even-bits constant). This is `active & ~any_free`
    from the torch engine, in one pass over the bitmap words."""
    k = tl.program_id(0).to(tl.int64)
    sel = tl.load(act_ptr + k) != 0
    if sel:
        offs = tl.arange(0, BLOCK)
        free = offs < 0                          # all-False vector
        i = 0
        while i < NW:
            m = (i + offs) < NW
            w = tl.load(ab_ptr + k * NW + i + offs, mask=m, other=-1)
            pm = tl.load(pm_ptr + i + offs, mask=m, other=0)
            free = free | (((~(w | (w >> 1))) & pm) != 0)
            i += BLOCK
        sel = tl.sum(free.to(tl.int32), 0) == 0
    tl.store(out_ptr + k, sel.to(tl.int8))


@triton.jit
def sound_rows(st_ptr, ab_ptr, rows_ptr, M, NW, pairs_ptr, n_pairs, bad_ptr,
               SHIFT: tl.constexpr, BLOCK: tl.constexpr):
    """Per selected row i: bad[i] = 1 iff SOME clause is EMPTIED (every literal falsified).
    grid = (n_rows,).

    LONG (len>=3) clauses have a packed word: emptied <=> sat_count == 0 (M = num_long here).
    Binary and unit clauses have NO per-row word — they are checked from the assignment BITMAP
    through `pairs_ptr`, (n_pairs, 2) literal-index pairs holding every binary clause plus each
    unit clause u as the degenerate pair (u, u): emptied <=> both literals' negations are set.
    At a FULLY-assigned row, non-falsified <=> TRUE, so "no clause emptied" <=> satisfied —
    the same soundness contract as before, now covering every clause kind independently of the
    propagate kernels' own bookkeeping."""
    r = tl.load(rows_ptr + tl.program_id(0)).to(tl.int64)
    SAT_MASK: tl.constexpr = (1 << SHIFT) - 1
    offs = tl.arange(0, BLOCK)
    bad = offs < 0
    i = 0
    while i < M:
        m = (i + offs) < M
        st = tl.load(st_ptr + r * M + i + offs, mask=m, other=1)
        bad = bad | ((st & SAT_MASK) == 0)
        i += BLOCK
    j = 0
    while j < n_pairs:
        m = (j + offs) < n_pairs
        a = tl.load(pairs_ptr + 2 * (j + offs), mask=m, other=0)
        b = tl.load(pairs_ptr + 2 * (j + offs) + 1, mask=m, other=0)
        wa = tl.load(ab_ptr + r * NW + ((a ^ 1) >> 5), mask=m, other=0)
        wb = tl.load(ab_ptr + r * NW + ((b ^ 1) >> 5), mask=m, other=0)
        fa = (wa >> ((a ^ 1) & 31)) & 1
        fb = (wb >> ((b ^ 1) & 31)) & 1
        bad = bad | (m & ((fa & fb) != 0))
        j += BLOCK
    tl.store(bad_ptr + tl.program_id(0), (tl.sum(bad.to(tl.int32), 0) > 0).to(tl.int8))


@triton.jit
def var_bits(ab_ptr, rows_ptr, cols_ptr, out_ptr, n_cols, NW, BLOCK: tl.constexpr):
    """out (n_rows, n_cols) int8 = 1 iff variable cols[j] is TRUE in row rows[i].
    grid = (n_rows, ceil(n_cols/BLOCK)). Direct bit gather — no (K, N) materialization."""
    i = tl.program_id(0).to(tl.int64)
    r = tl.load(rows_ptr + i).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_cols
    c = tl.load(cols_ptr + offs, mask=m, other=0)
    w = tl.load(ab_ptr + r * NW + (c >> 4), mask=m, other=0)
    tl.store(out_ptr + i * n_cols + offs, (((w >> ((c & 15) * 2)) & 1)).to(tl.int8), mask=m)


@triton.jit
def unpack_rows(ab_ptr, rows_ptr, out_ptr, two_n, NW, BLOCK: tl.constexpr):
    """out (n_rows, 2N) int8 = the historical byte-per-literal assignment for selected rows.
    grid = (n_rows, ceil(2N/BLOCK)). Contiguous writes (the strided variant measured 16x
    sector RMW amplification)."""
    i = tl.program_id(0).to(tl.int64)
    r = tl.load(rows_ptr + i).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < two_n
    w = tl.load(ab_ptr + r * NW + (offs >> 5), mask=m, other=0)
    tl.store(out_ptr + i * two_n + offs, ((w >> (offs & 31)) & 1).to(tl.int8), mask=m)


@triton.jit
def sum_i8_to(src_ptr, n, out_ptr, BLOCK: tl.constexpr):
    """out[0] (int32) = sum(src[0:n] as int8 flags). ONE program, grid = (1,)."""
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    i = 0
    while i < n:
        acc += tl.load(src_ptr + i + offs, mask=(i + offs) < n, other=0).to(tl.int32)
        i += BLOCK
    tl.store(out_ptr, tl.sum(acc, 0))


@triton.jit
def sum_i32_acc(src_ptr, n, acc_ptr, BLOCK: tl.constexpr):
    """acc[0] (int64) += sum(src[0:n] as int64). ONE program, grid = (1,): the per-launch work
    counters are (K,) int32 and must accumulate without a host sync (as the torch engine's
    0-d GPU accumulators did)."""
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.int64)
    i = 0
    while i < n:
        acc += tl.load(src_ptr + i + offs, mask=(i + offs) < n, other=0).to(tl.int64)
        i += BLOCK
    tl.store(acc_ptr, tl.load(acc_ptr) + tl.sum(acc, 0))

# WAVE shuffle-gather schedule — the O(W²) one-hot lane→segment machinery replaced by warp shuffles

> **Status:** settled — shipped 2026-08-27.
> **Measured:** 2026-08-27, RTX 4080 Laptop (12 GB, WSL2), on top of the binary-occ segregation.
> **Rule lives in:** `cpp/kernels/bcp.py` (the shuffle schedule), `WAVE_W = 32` in
> `cpp/tools/build_kernels.py`.
> **Record only:** the PTX instruction counts, and the rule to re-dump the PTX after any change
> to W — nothing else holds them.

Origin: whether the WAVE kernel's per-lane branching could be reduced further, via a 2-bit
assignment plane as in the UnitMarch-style encoding of Aistleitner's 2013 FMV thesis.

## The 2-bit idea is already the shipped design

The assignment bitmap is a 2-bit-per-variable encoding (bits `2v`/`2v+1` share a word; sign is
the `l ^ 1` XOR), and the wave inner loop has no data-dependent branch: detection is read from the
return values of the two atomics (`atomic_add` old count = conflict/unit, `atomic_or` old word =
dedup/contradiction). The thesis's 16-per-word packing axis is our K axis; its per-pass
whole-formula iteration is the clause scanning this engine avoids. A third "pending" bit-plane
replacing the trail was rejected: it trades an O(1) trail pop for an O(2N/32) scan per wave and
breaks the CLEAN restore's trail-driven damage bound. The right question was what the lanes
still pay to coordinate.

## What the lanes paid: shared-memory round-trips

The wave inner loop mapped each lane to its (segment, offset) with a W×W compare-reduce plus
three one-hot gathers (`excl`, `base`, `mneg`): four (W,W) reductions per 32-entry stride. Triton
lowers axis-1 reductions of (W,W) tensors through shared memory with barriers even at
`num_warps=1`. Measured at the PTX level (Triton 3.6.0, sm_89):

| primitive | shfl.sync | shared refs | barriers | ~instructions |
|---|---|---|---|---|
| one W×W one-hot gather | 0 | 19 | 3 | ~239 |
| `tl.gather` on a (W,) tensor | 1 | 0 | 0 | ~18 |

Whole-kernel PTX for `fixpoint.s16.w1` through the AOT path: one-hot schedule 10 shfl / 28 shared
refs / 6 barriers / 128 shared bytes; shuffle schedule 17 shfl / 0 / 0 / 0. The one-hot's cost was
invisible in the source, where "one tl.sum" reads cheap. Check what a reduction lowers to before
assuming register-level cost.

## The rewrite (`_wave_propagate` + `_clean_packed`)

1. Hoist `adj_v = s_v - (cum_v - len_v)` once per frontier chunk (occ offset of entry t in
   segment j is `adj_j + t`).
2. Segment search: branchless lower_bound over `cum_v`, log2(W) `tl.gather` probes, each one
   `shfl.sync.idx` (`_seg_lookup`, the one home of the in-range proof: an active lane has
   `t < T <= cum_{f-1}`, so `seg <= f-1`; inactive lanes saturate at `seg = W-1` and are masked;
   W must be a power of two, which `tl.arange` enforces at compile time).
3. `o_v = t_v + gather(adj_v, seg_v)`; `mneg_v = gather(neg_v, seg_v)`; the two per-stride
   conflict reductions merged into one.
4. The same rewrite in `_clean_packed`'s flattened restore loop, which both kernels share.

Net per stride: 4 shared-memory (W,W) passes + 6 barriers → 7 warp shuffles. Semantically
invisible: fixpoints, conflicts, models and counters unchanged.

## Gates

Count-terminated pinned-K identity (`--k-cap 20000`, b8, seed 5, `--n-time 100`): bit-identical
`n_models` and success on b03-c25 (1872), s1196-c25 (3469), b12-c25 (101), b04-c25 (3665). Full
pytest suite green through the AOT path. `gpap mine` b12-c50 emits the identical 9,890 mined
clauses in both arms. The later `_seg_lookup` extraction was gated the same way: PTX unchanged,
identity bit-identical, interleaved best-of-2 1.005× geomean.

## Measured (interleaved A/B, best-of-3, order alternated, `GPAP_LIB` per arm)

Same K (`--k-cap 20000`, tseitin, rare, b8, mine off, `--n-time 20000 --time-limit 40`):

| cell | one-hot sps | shuffle sps | ratio | one-hot impl/s | shuffle impl/s |
|---|---|---|---|---|---|
| b03-c25 | 138,179 | 317,413 | 2.297× | 1.04e9 | 2.41e9 |
| s1196-c25 | 13,407 | 21,706 | 1.619× | 7.46e8 | 1.21e9 |
| b12-c25 | 11.3 | 19.6 | 1.743× | 8.72e8 | 1.57e9 |
| b04-c25 | 21,344 | 34,275 | 1.606× | 6.52e8 | 1.05e9 |

**Geomean 1.796× at same K, kernels only.** The per-family impl/s plateau moves from ~1e9 to
1.05–2.41e9 on this card; every older "flat ~1e9" figure is the one-hot number. `gpap mine`
b12-c50 e2e 6.1 s → 5.2 s (1.18×) is a mixed number: `_clean_packed` is called outside the wave
dispatch, so the scalar clean cubin gains the restore win too.

## W=64 at num_warps=1 (ILP ×2): a consistent loss

The earlier "W>32 is 2–5× worse" result spread W over W/32 real warps. W=64 in one warp (2
elements/lane) became plausible once the schedule stopped being O(W²); measured (same protocol,
identity gate first): b03-c25 0.855×, s1196-c25 0.918×, b12-c25 0.951×, b04-c25 0.952×, geomean
0.918×. Register pressure doubles, a 64-wide gather on 32 lanes costs 2 shuffles + select, and
short frontiers idle more lanes. `WAVE_W` in `cpp/tools/build_kernels.py` stays 32, an AOT
bake-time constant.

## Gotchas

- Build-time Triton floor is ≥ 3.2 (`tl.gather`); runtime is still `libcuda` alone.
- The `while step >= 1` ladder is not unrolled by the Triton frontend (`visit_While` has no
  static-condition path); MLIR/LLVM unrolls the constant trip count. Verified: optimized TTIR has
  no `scf.while`/`tt.gather` and the PTX is 17 `shfl.sync`, 0 `.shared`, 0 `bar.sync`. The
  no-shared-memory cost is an optimizer outcome, not a structural guarantee: re-dump the PTX after
  any change to `W` or the loop body.
- The wave/scalar dispatch threshold (`M >= 2048`) was calibrated against the one-hot kernel.
  `_wave_propagate` got faster and scalar `_propagate` did not (the shared `_clean_packed` speeds
  both arms equally, so it cannot move the boundary), so the true crossover moved down and the
  M≈0.5–9k tie band under-dispatches wave. Re-derive it with the next `eval/profiling/kernel_ab.py`
  campaign.
- Unmeasured candidate: an `f == 1` fast path (one frontier literal, warp-uniform, every lane's
  `seg` is 0) would skip 5 probe gathers; it adds a branch to the innermost loop, so one
  interleaved A/B first.

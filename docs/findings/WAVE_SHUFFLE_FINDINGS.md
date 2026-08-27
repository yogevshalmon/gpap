# WAVE shuffle-gather schedule — the O(W²) one-hot lane→segment machinery replaced by warp shuffles

2026-08-27, RTX 4080 12GB (WSL2), branch `organize_data_and_expr`, on top of the binary-occ
segregation. Origin: user asked whether the WAVE kernel's per-lane "if cases" could be reduced
further (idea: a 2-bit "demo" assignment plane, per the UnitMarch-style encoding in
Aistleitner's 2013 FMV thesis, which we had reviewed before and not used).

## The 2-bit idea — already isomorphic to the shipped design

The thesis's lever is: encode assignments as 2-bit entities so state tests become bit algebra
instead of branches, and SIMD-pack 16 assignments per word. Both halves already exist here in
stronger form: the assignment bitmap IS a 2-bit-per-variable encoding (bits `2v`/`2v+1` share a
word; sign handling is the `l ^ 1` XOR), and the wave inner loop has NO data-dependent branch —
detection is read from the RETURN VALUES of the two atomics (`atomic_add` old count = conflict/
unit, `atomic_or` old word = dedup/contradiction). The thesis's 16-per-word packing axis is our
K axis (262k programs); its per-pass whole-formula iteration is the clause scanning this engine
exists to avoid. A third "pending" bit-plane replacing the trail was considered and rejected:
it trades an O(1) trail pop for an O(2N/32) scan per wave and breaks the CLEAN restore's
trail-driven damage bound. Nothing transferable remains — the correct reading of the question
was not "remove branches" but "what do the lanes still pay to coordinate".

## What the lanes actually paid: shared-memory round-trips, not divergence

The wave inner loop mapped each lane to its (segment, offset) with a W×W compare-reduce
(`seg = #{j: cum_j <= t}`) plus THREE one-hot gathers (`excl`, `base`, `mneg`) — four (W,W)
reductions per 32-entry stride. Triton lowers axis-1 reductions of (W,W) tensors through
SHARED MEMORY with barriers. Measured at the PTX level (Triton 3.6.0, sm_89, num_warps=1):

| primitive                    | shfl.sync | shared refs | barriers | ~instructions |
|------------------------------|-----------|-------------|----------|---------------|
| one W×W one-hot gather       | 0         | 19          | 3        | ~239          |
| `tl.gather` on a (W,) tensor | **1**     | **0**       | **0**    | ~18           |

`tl.gather` did not exist when the loop was written; we are on Triton 3.6.0 now. Whole-kernel
PTX for `fixpoint.s16.w1` through the AOT path:

| build            | shfl.sync | shared refs | barriers | shared bytes |
|------------------|-----------|-------------|----------|--------------|
| one-hot schedule | 10        | 28          | 6        | 128          |
| shuffle schedule | 17        | **0**       | **0**    | **0**        |

## The rewrite (`_wave_propagate` + `_clean_packed`)

1. **Hoist**: `adj_v = s_v - (cum_v - len_v)` once per frontier chunk (occ offset of entry t in
   segment j is `adj_j + t`) — `excl` and `base` were only ever used as this difference.
2. **Segment search**: branchless lower_bound over `cum_v`, log2(W) `tl.gather` probes (each ONE
   `shfl.sync.idx`). In-range by construction: an ACTIVE lane has `t < T <= cum_{f-1}`, so
   `seg <= f-1` and the classic sum-of-steps-(W−1) search suffices; every probe index is
   `<= W-2` (pos `<= W-2s` entering the step-s round). Duplicate `cum` values (empty segments)
   resolve past them exactly as the count-of-<= did. Inactive lanes saturate at `seg = W-1`,
   still in range, and are masked.
3. `o_v = t_v + gather(adj_v, seg_v)`; `mneg_v = gather(neg_v, seg_v)`.
4. Rider: the two per-stride conflict reductions merged into one
   (`conf = contra | (long & oldsat==1)`).
5. Same seg/gather rewrite in `_clean_packed`'s flattened restore loop.

Net per stride: 4 shared-memory (W,W) passes + 6 barriers → 7 warp shuffles. Semantically
invisible (same indices, computed cheaper): fixpoints, conflicts, models, counters unchanged.

## Gates

- Count-terminated pinned-K identity (`--k-cap 20000`, b8, seed 5, `--n-time 100`):
  **bit-identical `n_models` + success** on b03-c25 (1872), s1196-c25 (3469), b12-c25 (101),
  b04-c25 (3665).
- Full pytest suite green (`tl.gather` compiles through the AOT path; 33 cubins; mining
  kernel-invariance + clean-mode sabotage tests included).
- `gpap mine` b12-c50 emits the IDENTICAL 9,890 mined clauses in both arms.

## Measured (interleaved A/B, best-of-3, order alternated, same tree, GPAP_LIB per arm)

Same-K (`--k-cap 20000`, tseitin, rare, b8, mine off, `--n-time 20000 --time-limit 40`):

| cell      | one-hot sps | shuffle sps | ratio  | one-hot impl/s | shuffle impl/s |
|-----------|-------------|-------------|--------|----------------|----------------|
| b03-c25   | 138,179     | 317,413     | 2.297× | 1.04e9         | 2.41e9         |
| s1196-c25 | 13,407      | 21,706      | 1.619× | 7.46e8         | 1.21e9         |
| b12-c25   | 11.3        | 19.6        | 1.743× | 8.72e8         | 1.57e9         |
| b04-c25   | 21,344      | 34,275      | 1.606× | 6.52e8         | 1.05e9         |

**Geomean 1.796× at same K, kernels only.** The per-family impl/s plateau (~1e9 on this card)
moves to 1.05–2.41e9 — the "flat ~1e9" figure in older docs is the one-hot-schedule number.
`gpap mine` b12-c50 e2e wall: 6.1 s → 5.2 s (**1.18×**), best-of-2 — ⚠️ **this is NOT a
wave-only number**: `_clean_packed` is called outside the `if WAVE:` dispatch, so the shuffle
rewrite of the CLEAN restore speeds up the scalar clean cubin too. The 1.18× mixes the wave
propagate win with a restore win that both kernels get.

## W=64 at num_warps=1 (ILP ×2) — unlocked by cheap shuffles, measured separately

The earlier "W>32 is 2–5× WORSE" result spread W over W/32 REAL warps via `num_warps`
(cross-warp reduction sync); W=64 in ONE warp (2 elements/lane) was never tested and only
became plausible once the schedule stopped being O(W²).

Result (same protocol, best-of-2 interleaved, identity gate PASS first):

| cell      | W=32 sps | W=64 sps | ratio  |
|-----------|----------|----------|--------|
| b03-c25   | 286,540  | 245,066  | 0.855× |
| s1196-c25 | 21,183   | 19,454   | 0.918× |
| b12-c25   | 18.9     | 18.0     | 0.951× |
| b04-c25   | 33,939   | 32,320   | 0.952× |

**Geomean 0.918× — a consistent loss; W=32 stays** (2 elements/lane doubles register pressure,
a 64-wide gather on 32 lanes costs 2 shuffles + select, and frontiers shorter than 64 idle more
lanes; the extra memory-level parallelism does not pay on any cell). `WAVE_W` in
`cpp/tools/build_kernels.py` remains 32; W is an AOT-bake-time constant — the old runtime
`GPAP_WAVE_W` knob is gone.

## Gotchas

- **Build-time Triton floor moves to ≥ 3.2** (first `tl.gather`); runtime is unaffected — the
  shipped cubins carry the shuffles, `libcuda` alone still runs them.
- **The `while step >= 1` ladder is NOT unrolled by the frontend.** Triton's
  `visit_While` (3.6.0) has no static-condition path — it emits a real `scf.while` even though
  `step` is a constexpr; the full unroll is done by MLIR/LLVM on the constant trip count.
  Verified on this toolchain: optimized TTIR has zero `scf.while`/`tt.gather` left and the PTX
  is 17 `shfl.sync` with 0 `.shared` and 0 `bar.sync`. So the log2(W)-shuffles-no-shared-memory
  cost is an OPTIMIZER OUTCOME, not a structural guarantee — re-dump the PTX after any change to
  `W` or the loop body. Making it structural would need a `tl.static_range` over a `LOG2W`
  constexpr threaded through every spec in `kernel_specs()`; not worth the churn while the
  PTX check is one command.
- The one-hot's cost was invisible in Triton source — "one tl.sum" reads cheap. Check what a
  reduction LOWERS TO before assuming register-level cost: axis-1 (W,W) reductions round-trip
  shared memory even at num_warps=1.
- The wave/scalar dispatch threshold (`M >= 2048`, `cpp/src/calibrate.cpp`) was calibrated
  against the ONE-HOT wave kernel. `_wave_propagate` got faster and the scalar `_propagate` did
  not, so the true crossover moved DOWN and the M≈0.5–9k tie band now under-dispatches wave.
  Inputs to the rule did not change (no K/memory-model change) — re-derive the threshold with
  the next forced-kernel M-ladder / `eval/kernel_ab.py` campaign rather than guessing.
  ⚠️ Say "scalar **`_propagate`** is untouched", not "the scalar side": `_clean_packed` IS
  shared by both kernels and DID change. That distinction is what makes the crossover argument
  valid (the shared restore speeds up both arms equally, so it cannot move a dispatch boundary)
  while making the `gpap mine` figure above a mixed number.

## Post-review notes (2026-08-27, same day)

A review of the diff produced 13 findings; 11 held up on verification and were fixed here (the
build pin, the three wrong claims above, the stale evidence pointers, the shared `_seg_lookup`
helper so the in-range proof has ONE home). Two did not:

- **"W=48 would compile and silently give wrong segment indices" — FALSE.** `tl.arange` rejects
  any non-power-of-two size outright (*"arange's range must be a power of 2"*, verified), so the
  ladder's power-of-two precondition is enforced by the compiler before it can matter. The
  precondition is documented in `_seg_lookup` anyway.
- **`f == 1` scalar fast path — an unmeasured candidate, not a defect.** When the frontier holds
  one literal (the steady state once a decision group drains) every active lane's `seg` is 0, so
  5 probe gathers plus the `neg_v` gather are dead work ahead of the `occ_cls` load. `f` is
  warp-uniform, so `if f == 1: o_v = t_v + s_0; mneg_v = neg_0` would be divergence-free. Worth
  ONE interleaved A/B before it goes anywhere near the hot loop — it adds a branch to the
  engine's innermost loop to save work whose cost has not been isolated. Same shape in
  `_clean_packed`.

The `_seg_lookup` extraction was itself gated, since it touches the hot loop: PTX unchanged
(17 shfl / 0 shared / 0 barriers on `fixpoint.s16.w1`), identity bit-identical on all four
cells, and an interleaved best-of-2 against the pre-refactor shuffle build reads **1.005×
geomean** (0.984-1.023) — neutral, as a pure extraction should be. One dead `SAT_MASK` constant
in `_bcp_trajectory_kernel` (orphaned when `_risk` was extracted the session before) was removed
in the same pass; `ruff check cpp/ gpap/ tests/ eval/ scripts/` is clean.

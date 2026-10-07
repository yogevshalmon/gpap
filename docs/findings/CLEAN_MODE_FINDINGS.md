# CLEAN mode — the fixpoint launch that restores its own state

> **Status:** settled — shipped and gated 2026-08-21, contract enforced 2026-08-22. Supersedes
> the reset-restore plan (deleted; git history keeps it).
> **Measured:** 2026-08-21, RTX 4080 Laptop (12 GB, WSL2); confirmed same day on a Colab L4.
> **Rule lives in:** `cpp/src/engine.cpp` (`gpap_engine_apply_clean`, the reset-lite path),
> `cpp/src/probe.cpp`, `cpp/kernels/bcp.py`, `cpp/tools/build_kernels.py`, `gpap/engine/_core.py`.
> **Gated by:** `tests/engine/test_clean_mode.py`.
> **Record only:** the four soundness lemmas L1–L4 below. They are per-kernel-version facts with
> no other home; the test is their tripwire, not their statement.

Origin: [MINING_FINDINGS.md](MINING_FINDINGS.md) "The RESET is the bottleneck" (2026-08-19).

## The problem

`gpap_engine_reset(K)` broadcast-restores the post-units-flush cache, `K·(M + NW)` int32 words.
The batch is sized `K ≈ 0.80·free/row_bytes`, so every chunk rewrote ~0.80 of free VRAM:
~26 ms/chunk on the RTX 4080 Laptop, 50/70/84/94% of probe-chunk wall
on b09-c25 / b03-c25 / b12-c50 / 6s215rb0-c25, while the probe it preceded modified 0.02–2.5%
of those words. Probing was bounded by `budget/(4M)/26 ms` tuples/s however fast BCP got.

## The design

Arithmetic undo (re-walk the trail, add the subtractions back) was rejected: a conflicted row
stops at a watermark stored nowhere (scalar mid-occurrence-list, wave mid-frontier), and a wave
contradiction strays an assignment bit with no trail record. Rewriting from the baseline is
superset-safe, idempotent, and covers the stray bit (L3).

`gpap_engine_apply_clean` runs the ordinary fixpoint and then, in the same launch, each row
rewrites the packed words it could have touched back to the flush cache:

    for each l in trail[0 .. tail):  for each c in occ[l]:  packed[r][c] = packed_flushed[c]

Trail and `tail` are still in registers and the touched words L2-resident, so the
restore is hot-cache. A row whose occ-entry count exceeds `M/8` streams its whole packed row
from the baseline instead (scattered 4-byte stores pull 32-byte sectors, ~8× amplification), so a
deep-propagation row costs what the broadcast did, never more. The engine records those rows
as clean (a high-water mark) and `reset(K)` goes lite — flags and bitmap only, no `K·M`
broadcast — while `K` stays within it; growth past the high-water or any reallocation (reserve
loses contents) re-broadcasts. CLEAN is a `constexpr` of the fixpoint kernel: four new
specializations (`SHIFT × WAVE`), with the pre-existing cubins byte-identical after the rebuild.

## Soundness (plan §3.1, checked against `cpp/kernels/bcp.py` as of 2026-08-21)

- **L1 — the trail is complete and its length is the tail.** Every path that walks an occ list
  first popped the literal from the trail; entries are contiguous in `trail[0..tail)`; rows
  inactive at entry have a zero counter and restore nothing, which is correct.
- **L2 — packed damage is covered.** Packed words are written only inside the occ walk of popped
  literals, so damaged words ⊆ ∪_{l∈trail} occ[l] for surviving and conflicted rows alike; a
  mid-list or mid-stride stop shrinks the damage, never moves it outside the union.
- **L3 — assignment-bit damage is covered.** The seed loop and scalar propagate are
  check-then-set, so bit ⟺ trail entry. A wave contradiction strays a bit, but that bit shares the
  word (`l` and `l^1` are neighbours) and was set this launch — flush-fixpoint consistency means a
  literal surviving in a clause's sum has its opposite clear in the baseline — so rewriting
  `word(l)` for every trail `l` clears every stray.
- **L4 — over-restore is free.** Every store writes the baseline value, so stale trail entries,
  unprocessed tails and unreached words rewrite baseline over baseline; two trail literals sharing
  a clause race with identical values, no atomics needed. Hence robustness to every partial-stop
  mode the kernels have.

The lemmas are per-kernel-version facts: an edit to the push/claim/conflict paths of `bcp.py`
can invalidate them silently; the fuzz gate below is the tripwire. `tail` is the raw trail
length including seeds; do not restore `tail − ndecs`, which under-covers.

## The contract

- CLEAN is a per-launch mode, not the universal path. It restores the packed words only; the
  bitmap keeps the fixpoint. Anything that reads post-apply state — `gpap_sat_rows`,
  `gpap_check_sound`, `recache_flush`, `check_assumptions(assignments=)`, `gpap_assume_units` —
  must use plain `apply`; after a clean launch those calls are refused (`engine.abits_diverged`,
  set by a completed clean launch, cleared by `reset` or any plain apply or trajectory).
  Copying the state out first would be the broadcast again.
- `gpap_packed_rows` stays open: the debug readout the identity gate uses to prove the restore.
- Consumers: `probe_chunk` and flags-only `check_sets`. The sampler stays on the broadcast: its
  trail ratio (0.32–0.62, × ~8 sector amplification) is the losing side of the crossover, and its
  reset is not its bottleneck.
- `GPAP_CLEAN=0` is exact legacy and also disables reset-lite, so an A/B is uncontaminated;
  `gpap_engine_reset_stats` shows whether lite resets fired (a silent fallback measures
  1.00× and looks worthless).
- Both entry points check the row count in the binding: a short array is read past its end, and an
  index past 2N makes the kernel write into a later row.

## Measured

RTX 4080 Laptop, 2026-08-21, CPU busy: b12-c50 sync-bracketed phase split, **reset_full
17.9 ms + apply 3.0 ms = 20.9 ms legacy chunk vs apply_clean 4.3 ms + reset_lite 0.4 ms = 4.7 ms
clean chunk = 4.4× GPU-side** (the tail costs 1.3 ms); b03-c25 probe sweep **2.60×
end-to-end**, interleaved best-of-3, identical conflicts. b12 end-to-end showed only 1.27×:
20–30 ms/chunk of host work (enumeration + driver calls) then dominated. Forced-scalar
at large M regresses (b12 apply 126 → 166 ms) — irrelevant to dispatched runs, which pick scalar
only below M = 2048. Colab L4, same day, full seq family T=25/rare/tseitin: 19/19 rows both
arms, 0 unsound, 16/19 bit-identical model counts (the 3 differ by the mining-ceiling tune).
Data: `$GPAP_RESULTS/clean_mode_l4_20260821/`.

## The gate

`tests/engine/test_clean_mode.py`: state-identity fuzz against a plain-apply twin and the reference BCP
across both kernels × both splits, the deep-row stream arm, K shrink/grow, every invalidation,
probe/checker/mining A/B under the env, and `TestCleanModeGuards` with a positive control per
guard. Verified to fail against a broken tail (walk `tail-1`).

## Considered, not adopted

- A separate `trail_restore` kernel at the next reset: cold-cache scatter plus a state machine, both
  deleted by fusing.
- Smaller chunks (ratio-invariant); two-stream overlap (≤1.3× bound); fusing the broadcast into
  BCP (`K·M` reaches DRAM either way).
- Prefix-sharing enumeration: touches the core's lex-order contract.
- Copy-on-write / sparse row state: the asymptotic answer, a separate project.
- A persistent kernel looping over many tuples: fights chunk granularity and the WSL2 watchdog.

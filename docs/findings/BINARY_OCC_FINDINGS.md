# Binary clauses as implication edges — design, gates, measurements

> **Status:** settled — shipped 2026-08-27.
> **Measured:** 2026-08-27, RTX 4080 Laptop (12 GB, WSL2).
> **Rule lives in:** `cpp/src/formula.cpp` (builder segregation), `cpp/kernels/bcp.py` +
> `helpers.py` (all kernel variants, `sound_rows`), `cpp/src/engine.cpp` / `readout.cpp`
> (staging, sizing, readout synthesis), and the sizing flip in `calibrate` callers.
> **Gated by:** the two-checkout no-change gate, a pure-2-SAT propagation test.

## The change

Only clauses of length ≥ 3 carry a packed word and per-row state (`num_long`). A length-2 clause
`(~l | b)` is an implication edge in the occ CSR: it sits in `occ[l]` with payload `b`, in the
binary prefix of the segment (`occ_mid[l]` is the split), so falsifying `~l` forces `b` with the
one bitmap `atomic_or` the forced-literal push already does; its return value answers duplicate
and contradiction race-free, so a binary conflict is just a contradiction push. Unit clauses live
in the construction flush's bitmap (no occ entry, no word; the bitmap detects any later `~u`
push). Fixpoints, conflicts and model sets are unchanged because BCP fixpoints are confluent: the
fixpoint does not depend on the order or mechanism by which each clause's forcing is applied, so
a binary forced through an edge and one forced through a packed word reach the same assignment
or the same conflict.

Binary occ payloads also carry a sign-bit tag (`GPAP_OCC_BIN_TAG`; a literal index needs ≤ 24
bits and a long-clause index is non-negative), so WAVE classifies an entry from the value it
already loaded instead of gathering `occ_mid` per frontier; `occ_mid` still bounds the scalar
loop split, the CLEAN restore and the SOFT scan.

## The static prize

In-memory Tseitin builds after `unit_closure`, 15 circuit CNFs from c1355-c1 (M=2,435) to
b14-c50 (M=1,988,063): binaries are 59–82% of M, length ≥ 3 only 10–23%, binary occ entries
64–75%, and the row-bytes model predicts K ×2.17–2.34, bounded near 2.3 by the N terms (bitmap
and trail) that stay. Post-unit-closure formulas also carried hundreds to thousands of unit
clauses, each a word per row; gone too. ELS removes mostly binaries, so the pipeline formula's
long fraction is higher (b12-c50: M_eff 119,178 with ~37k long) and these ratios are
conservative.

## Gates

- Full non-slow suite 479 passed / 0 failed after the representation change and again after the
  sizing flip. Formula structures entry-for-entry equal to the rewritten numpy oracle. CLEAN-mode
  state-identity fuzz, both kernels × both splits. Mining invariance: identical mined counts
  across arms and kernels (b03-c25 2,676; b12-c50 9,890).
- Two-checkout no-change gate (`--portfolio b8`, `--mine off`, seed 5): b03-c25 24,512 == 24,512;
  s1196-c25 24,761 == 24,761; b12-c25 count-terminated 101 == 101, success identical to 17
  significant digits. Two protocol rules: a no-change-gate cell must be count-terminated (a
  wall-terminated cell compares speed: b12-c25 read 322 vs 393 that way), and after the sizing
  flip the cap must sit below both arms' analytic bound (`--k-cap 20000`), since `--k-cap 262144`
  no longer pins K when head sizes ~2× base.
- A test that proved the deep-row guard on a chain of 400 binaries passed vacuously after the
  change (zero packed words). It now uses ternary `(-v|-v|v+1)` links, and the binary chain
  became the explicit pure-2-SAT test (`num_long == 0`: stateless rows, bitmap-synthesized
  `sat_rows`, reset-lite still firing).

## Measured wins (interleaved A/Bs, best of 3, order alternated)

Same K (kernels only, pinned cap): b03-c25 71,380 → 102,032 sps (1.429×), s1196-c25 13,800 →
20,810 (1.508×), b12-c25 7.7 → 10.2 (1.325×) — **geomean 1.42× from the propagation path alone**.

Free K (kernels + sizing): b03-c25 1.250× (both arms cap-bound), s1196-c25 1.506× (K 51,724 →
106,478), b12-c25 1.346× (K 30,812 → 63,999) — **geomean 1.36×**. The K doubling adds little on
saturated instances (K is hygiene, not a lever); its value is on analytic-bound big-M cells,
smaller cards and the probe path: `gpap mine` b12-c50 9.5 s → 6.3 s, **1.52×**.

Defaults-on circuit sweep (ELS + mining auto + bandit + auto K, `--target rare`, seed 5,
`--n-time 20000 --time-limit 90`, best-of-2 ABBA, 18 cells): geomean 1.409× over 17 (seq-c25
1.455× n=11, seq-c50 1.257×, comb 1.349×, hwmcc-c25 1.384×; b15-c25 excluded at 0/0). `impl/s`
tracks `sps` within a few percent on nearly every cell, so the win is the engine, not sampler
heuristics; head holds 5.3e8–1.03e9 impl/s. K grew 2.0–2.9× where analytic-bound and four cells
crossed into the 262,144 cap.

The tag was measured, not assumed (two interleaved runs at `--k-cap 20000`, tagged vs
tag-reverted): s1196-c25 wave 1.587× vs 1.436× (tag worth 1.11×), b12-c25 wave 1.581× vs 1.350×
(1.17×); b03-c25 is scalar at that pin and not readable. Since the wave schedule became
shuffle-based ([WAVE_SHUFFLE_FINDINGS.md](WAVE_SHUFFLE_FINDINGS.md)) the alternative the tag
avoids costs one `shfl.sync`, not a shared-memory reduction; re-measure before re-litigating the
24-bit literal ceiling.

## Sizing validation

- K at auto budget: b03-c25 129,369 → 262,144 (now cap-bound); b12-c50 7,322 → 17,061 (2.33×,
  exactly the static prediction).
- Row-bytes model vs measured `cuMemGetInfo` delta at `reset(K)`: 1.00× on b03-c25, 1.04× on
  b12-c50 (driver-side rounding on multi-GB allocations, inside the 0.8-budget headroom).
- Kill-switch (this box spills, never OOMs): b12-c50 tseitin rare at auto K=28,225, 45 s, max
  10,243 / 12,282 MB, no spill, sound.
- The per-row balance flipped: on b12-c50 the trail term (4·(N+72) ≈ 337 KB/row) now dominates
  the packed words (147 KB/row). The trail's N bound is the soundness proof (`gpap_trail_cap`),
  not a knob.
- Dispatch still keys off total M, which did not move, so kernel choices are unchanged.

## Consequences elsewhere

- Mined width-2 clauses are K-free; keep-if-it-helps keeps more of them
  ([MINING_FINDINGS.md](MINING_FINDINGS.md)): 6s215rb0 gain 1.43 → 2.44, texasparsesysp3 flipped
  revert → keep.
- `sat_rows` / `check_sets(sat_out)` keep the `(n, M_total)` original-order contract by
  synthesizing binary/unit counts from the row bitmap host-side; `gpap_packed_rows` is
  `(n, num_long)`. `gpap_formula::sat_src` is the one clause → per-kind slot map (`>= 0` long
  index, `< 0` = `-(p+1)` into the merged pair list) that both readouts index.
- `sound_rows` checks long words plus a device `(n_pairs, 2)` list of binaries and units (a unit
  as `(u,u)`): emptied ⇔ both negations set. Same contract, all clause kinds, still independent
  of the propagate kernels' own bookkeeping. `gpap_sat_rows` skips the per-row bitmap copy when
  `n_pairs == 0`.
- SOFT (opt-in): binary risk is exact from the bitmap (satisfied 0 / would-empty 1e6 /
  would-force 4); its distribution changes and it remains Barbarik-rejected and off-portfolio.
- `nops` still counts occ entries processed; per-entry cost now differs by kind.

## Considered and not adopted

- Tagging binary entries in place instead of the `[bin | long]` split: the split lets
  `_clean_packed` walk only `[mid, end)`, so the CLEAN restore shrinks by the binary share, and
  payloads keep full int32 width.
- A device-kernel rewrite of `gpap_sat_rows(sat_count=True)` (2 bitmap copies per row, 1.60× on
  b03-c25 at 2,048 rows): no consumer outside `checker.py` and its tests.
- Removing the `gpap_engine_num_long` export: it is the only way to size a `gpap_packed_rows`
  buffer after `gpap_formula_free()`.

## Open

L4 campaign re-run on the new engine (the 20260827 campaign is the pre-binary baseline);
stage-0 binary-fraction sweep on ABC/sk_/EPFL/HWMCC post-ELS; forced-kernel M-ladder re-check
(`eval/profiling/kernel_ab.py`).

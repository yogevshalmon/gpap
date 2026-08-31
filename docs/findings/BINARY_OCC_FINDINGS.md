# Binary clauses as implication edges — design, gates, measurements

*2026-08-27, RTX 4080 12GB (WSL2 dev box). Change landed across `cpp/src/formula.cpp` (builder
segregation), `cpp/kernels/bcp.py` + `helpers.py` (all four kernel variants + `sound_rows`),
`cpp/src/engine.cpp` / `readout.cpp` (staging, sizing, readout synthesis), and the sizing flip
in `calibrate` callers. The rule lives in code; this file is the evidence.*

## The change

Only clauses of length ≥ 3 carry a packed word / per-row state (`num_long`). Length-2 clauses
are **implication edges in the occ CSR**: clause `(~l | b)` sits in `occ[l]` with payload `b`,
in the binary PREFIX of the segment (`occ_mid[l]` is the split), so falsifying `~l` forces `b`
with the ONE bitmap `atomic_or` the forced-literal push already uses — its return value answers
duplicate/contradiction race-free, so a binary conflict is just a contradiction push. Length-1
clauses live in the construction flush's bitmap (no occ entry, no word: the bitmap detects any
later `~u` push, observationally identical to the old per-word path). Fixpoints, conflicts and
model sets are unchanged — BCP fixpoints are confluent.

The tag-bit variant (marking binary entries in-place) was considered and rejected: the
`[bin | long]` split lets `_clean_packed` walk only `[mid, end)`, so the CLEAN restore shrinks
by the binary share too, and payloads keep full int32 width; in `_wave_propagate` the split
costs one extra gather of `mid_v` — at the time a W×W one-hot reduction, the same price as testing a tag bit. ⚠️ RE-PRICED 2026-08-27: the schedule is shuffle-based now, so that alternative costs ONE `shfl.sync`, not a shared-memory reduction — see the note at the tag measurement below.

## The static prize (in-memory Tseitin builds, post-`unit_closure`, seed 5)

| inst | N | M | %bin | %len≥3 | occ entries bin | K ratio (row-bytes model) |
|---|---|---|---|---|---|---|
| b03-c25 | 5,701 | 10,526 | 72.3 | 9.5 | 74.7% | 2.33× |
| b04-c25 | 28,626 | 63,612 | 78.1 | 15.3 | 74.1% | 2.34× |
| b09-c25 | 5,976 | 11,615 | 66.1 | 14.4 | 66.9% | 2.23× |
| b10-c25 | 7,851 | 18,735 | 80.1 | 17.0 | 73.2% | 2.34× |
| b11-c25 | 31,326 | 73,852 | 78.6 | 17.4 | 72.9% | 2.32× |
| b12-c25 | 42,051 | 95,803 | 74.6 | 16.9 | 70.2% | 2.30× |
| b13-c25 | 12,926 | 22,300 | 59.4 | 12.6 | 63.8% | 2.17× |
| b14-c25 | 413,451 | 974,774 | 77.6 | 17.9 | 72.1% | 2.30× |
| s386-c25 | 4,151 | 12,272 | 76.1 | 22.9 | 64.7% | 2.30× |
| s1196-c25 | 17,801 | 46,603 | 79.1 | 20.6 | 70.3% | 2.29× |
| b12-c50 | 84,101 | 204,828 | 77.9 | 18.0 | 71.5% | 2.33× |
| b14-c50 | 826,901 | 1,988,063 | 78.6 | 18.2 | 72.5% | 2.31× |
| c1355-c1 | 1,004 | 2,435 | 80.3 | 19.3 | 72.8% | 2.22× |
| c6288-c1 | 4,577 | 11,472 | 79.2 | 20.8 | 71.8% | 2.24× |
| c7552-c1 | 4,802 | 11,822 | 82.2 | 17.8 | 74.0% | 2.33× |

Binaries 59–82% of M; **len ≥ 3 is only 10–23% of M**; binary occ entries 64–75%; the K ratio is
bounded at ~2.3 by the N terms (bitmap + trail) that stay. Post-unit-closure formulas also carry
hundreds–thousands of unit clauses (each used to occupy a word per row) — gone too.
ELS *raises* the long fraction (it removes mostly binaries): on b12-c50 the pipeline formula is
M_eff=119,178 with ~37k long — the ratios above are conservative for the pipeline.
TODO stage-0 completeness sweep: ABC CNFs (2-/3-lit only), sk_, EPFL, HWMCC.

## Gates (all on the first run after each stage, except where noted)

- **Full non-slow suite: 479 passed / 0 failed** after the representation change, and again
  after the sizing flip.
- Formula structures vs the rewritten numpy oracle: entry-for-entry equality (occ, occ_mid,
  payloads, sizes, sums, counts).
- CLEAN-mode state-identity fuzz (both kernels × both splits): packed rows bit-exact vs
  baseline, flags/counters vs a plain twin, conflicts vs the reference BCP.
- Mining invariance: identical mined-clause counts across arms and kernels (b03-c25: 2,676;
  b12-c50: 9,890 — both arms, both reps).
- **Two-checkout no-change gate** (pinned `--k-cap 262144`, `--portfolio b8`, `--mine off`,
  seed 5; per-arm libgpap since the C++ changed): b03-c25 24,512 == 24,512; s1196-c25
  24,761 == 24,761; b12-c25 first read as a MISMATCH (322 vs 393) — that cell was
  **WALL-terminated** (40 s limit bound before the 20k target), so its count measured *speed*
  (the head arm's +22% models-per-wall was the first e2e sighting of the win). Count-terminated
  re-check (`--n-time 100`), twice per arm: **101 == 101 == 101 == 101**, success identical to
  17 significant digits. ⚠️ Protocol note: a no-change-gate cell must be COUNT-terminated;
  a wall-terminated cell compares speed, not determinism.

## Measured wins (interleaved A/Bs, best of 3, order alternated — SM clock swings make any
other method meaningless on this box)

**Same-K (kernels only; sizing not yet flipped, pinned k-cap):**

| cell | base sps | head sps | head/base |
|---|---|---|---|
| b03-c25 | 71,380 | 102,032 | 1.429× |
| s1196-c25 | 13,800 | 20,810 | 1.508× |
| b12-c25 | 7.7 | 10.2 | 1.325× |

**Geomean 1.42× from the propagation path alone.**

**Free-K e2e (kernels + K sizing):** b03-c25 1.250× (both arms cap-bound post-ELS — pure
kernel), s1196-c25 1.506× (K 51,724 → 106,478), b12-c25 1.346× (K 30,812 → 63,999) —
**geomean 1.36×**. The K doubling adds little *throughput* on these already-saturated
instances — consistent with the standing "K is hygiene, not a lever" finding — the value of
the K headroom is on analytic-bound big-M cells, smaller cards, and the probe path below.

**`gpap mine` e2e (the probe path: fixpoint + CLEAN restore + broadcasts):** b12-c50
9.5 s → 6.3 s = **1.52×**; b03-c25 1.7 s → 1.6 s (startup-dominated, no signal).

## e2e defaults-on circuit sweep (4080, 2026-08-27 — the "what does a user get" number)

Full shipping pipeline on BOTH arms (ELS + mining auto + config bandit + auto K), `--target
rare`, seed 5, `--n-time 20000 --time-limit 90`, best-of-2 ABBA-interleaved per cell. Circuits
only (the paper's focus): ITC'99/ISCAS-89 T=25/50 Tseitin + 3 ABC-encoded, ISCAS-85 T=1
(pin 1), HWMCC'17 T=25.

| cell | M_eff | m_long | K b→h | sps b | sps h | × | impl/s b→h | × |
|---|---|---|---|---|---|---|---|---|
| b03-c25 | 1,178 | 311 | cap→cap | 1.44M | 2.22M | 1.55 | 4.9e8→7.6e8 | 1.55 |
| b04-c25 | 29,850 | 9,497 | 52k→107k | 53.4k | 72.8k | 1.36 | 5.3e8→7.3e8 | 1.36 |
| b04-c25-abc | 22,325 | 7,434 | 69k→136k | 55.8k | 84.6k | 1.52 | 4.4e8→6.7e8 | 1.52 |
| b10-c25 | 10,419 | 3,161 | 154k→cap | 3,066 | 4,112 | 1.34 | 5.9e8→7.4e8 | 1.25 |
| b12-c25 | 35,478 | 10,847 | 44k→94k | 45.4k | 57.5k | 1.26 | 5.8e8→7.3e8 | 1.26 |
| b12-c25-abc | 24,002 | 7,888 | 65k→129k | 339 | 470 | 1.39 | 5.7e8→8.1e8 | 1.41 |
| b13-c25 | 276 | 83 | cap→cap | 7.70M | 12.80M | 1.66 | 6.2e8→1.03e9 | 1.66 |
| b14-c25 | 525,066 | 169,005 | 2,208→5,305 | 2.2 | 3.2 | 1.46 | 4.0e8→5.3e8 | 1.33 |
| b15-c25 | 482,368 | 150,338 | 2,462→6,032 | 0 | 0 | — | 4.0e8→5.8e8 | 1.46 |
| s386-c25 | 8,630 | 2,429 | 187k→cap | 37.1k | 54.7k | 1.48 | 4.5e8→6.6e8 | 1.48 |
| s1196-c25 | 29,846 | 9,230 | 53k→110k | 40.8k | 63.6k | 1.56 | 4.2e8→6.5e8 | 1.56 |
| s1196-c25-abc | 30,035 | 10,002 | 51k→101k | 35.5k | 52.2k | 1.47 | 4.3e8→6.3e8 | 1.47 |
| b04-c50 | 62,860 | 20,026 | 24k→50k | 22.8k | 28.4k | 1.25 | 4.9e8→6.2e8 | 1.25 |
| b12-c50 | 98,690 | 30,315 | 15k→33k | 10.0k | 12.7k | 1.27 | 5.6e8→7.3e8 | 1.30 |
| c1355-c1 | 1,456 | 470 | cap→cap | 694k | 964k | 1.39 | 6.0e8→8.6e8 | 1.43 |
| c7552-c1 | 6,834 | 2,100 | 229k→cap | 238k | 312k | 1.31 | 5.0e8→6.6e8 | 1.31 |
| 6s215rb0-c25 | 225,419 | 74,732 | 6,098→12,938 | 61.1 | 76.9 | 1.26 | 4.0e8→5.4e8 | 1.34 |
| texasparsesysp3-c25 | 628,293 | 207,611 | 1,391→4,011 | 132 | 201 | 1.52 | 4.3e8→5.3e8 | 1.23 |

Geomeans (b15 excluded, 0/0): seq-c25 **1.455×** (n=11) · seq-c50 1.257× · comb 1.349× ·
hwmcc-c25 1.384× · **ALL 1.409× (n=17)**. `impl/s` tracks `sps` within a few % on nearly every
cell — the win is the ENGINE, not sampler heuristics; head holds 5.3e8–1.03e9 impl/s. K: 2.0–2.9×
where analytic-bound (texasparsesysp3 2.88×), and four cells crossed INTO the 262,144 cap.

**Mining keep-gate under the new K model (the "retune" answer — the proxy retunes ITSELF,
`smp.K` on the rebuilt sampler already charges only num_long):**
- Most cells: kept on both arms at ~equal gain (b03 3.4×, b04 3.9×, s1196 4.7×, s386 90–101×).
- **6s215rb0: gain 1.43 → 2.44** — the same mined clauses pay MORE on the new engine.
- **texasparsesysp3: base REVERTED mining (gain 0.74), head KEPT it (gain 1.10)** — a real
  keep-decision flip: clauses that didn't pay under the old K charge now do.
- gain 0.00 rows = the no-baseline path (un-mined warmup completed nothing → keep
  unconditionally); b10 flipped keep→drop at gain ≈ 1.00 (threshold coin-flip, harmless).
No code retune needed; the constants (MINE_SECONDS, STAGE1 ceilings — already count-based and
post-CLEAN calibrated) stay.

## Stage-3 sizing validation

- K growth at auto budget (cap 262,144): b03-c25 129,369 → 262,144 (2.03×, now cap-bound);
  b12-c50 7,322 → 17,061 (**2.33×**, exactly the static prediction).
- Row-bytes model vs measured `cuMemGetInfo` delta at `reset(K)`: **1.00×** on b03-c25 (to the
  MB); 1.04× on b12-c50 — the 4% is driver-side rounding on multi-GB single allocations (the
  same buffer list lands exact on b03), inside the 0.8-budget headroom.
- Kill-switch (this box SPILLS, never OOMs): `gpap sample` b12-c50 tseitin rare at auto
  K=28,225 (post-ELS M_eff=119,178, wave, 8|24), 45 s: max 10,243 / 12,282 MB — no spill,
  `sound=True`.
- **The per-row balance flipped**: on b12-c50 the TRAIL term (4·(N+72) ≈ 337 KB/row) now
  dominates the packed words (147 KB/row). The trail's N bound is the soundness proof
  (`gpap_trail_cap`, cpp/src/calibrate.cpp), not a knob.
- Dispatch: the rule still keys off TOTAL M — its input did not move, so kernel choices are
  identical to pre-change, and the wave-vs-scalar tradeoff is about occ entries per trail
  literal, which segregation does not change. The full forced-kernel M-ladder re-check rides
  the next L4 `kernel_ab` run.

## Consequences elsewhere

- **Mined width-2 clauses are K-free** (they join the implication list): keep-if-it-helps now
  faces zero memory cost for them; expect it to keep more.
- `sat_rows` / `check_sets(sat_out)` keep their public `(n, M_total)` original-order contract by
  synthesizing binary/unit counts from the row bitmap host-side (identical values — packed words
  also count non-falsified literals). `gpap_packed_rows` is `(n, num_long)`.
- `sound_rows` checks long words + a device `(n_pairs, 2)` list (binaries + units as `(u,u)`):
  *emptied ⇔ both negations set* — same contract, all clause kinds, still independent of the
  propagate kernels' own bookkeeping.
- SOFT (opt-in): binary risk is now EXACT from the bitmap (satisfied → 0 / would-empty → 1e6 /
  would-force → 4) — sharper than the sat_count read, which cannot see satisfaction. Soft's
  output distribution changes; it remains Barbarik-rejected and off-portfolio.
- `nops` still counts occ entries processed (definition unchanged, cross-campaign comparable);
  per-entry COST differs between kinds.

## Review fixes applied (2026-08-27, post-landing)

- `gpap_sat_rows` no longer copies the assignment bitmap per row when the formula has no
  binary/unit clauses (`n_pairs == 0`) — that path is pure waste on a 3-CNF, and it runs over a
  whole chunk from `gpap_check_sets`. (The remaining 2-copies-per-row cost on mixed formulas is
  real — measured 1.60x on b03-c25 at 2,048 rows — but `sat_count=True` has no consumer outside
  `checker.py` and its tests; the device-kernel rewrite that `gpap_var_bits` already uses is
  the fix if that ever changes.)
- **`gpap::pack_long_init` + `gpap::is_long_clause`** (`gpap_internal.hpp`): ONE definition of
  "which clauses get a packed word" and of the packed layout. The builder and
  `gpap_engine_create`'s `force_shift` repack both call it, so a change to the split or to
  `kMinLongLen` cannot land in one and miss the other — the old duplicate was invisible on the
  default route and only surfaced under the split-A/B lever.
- `gpap_formula_occ` gained `mid_out` (it is `n_offsets - 1` entries, so it needs no size query);
  `gpap_formula_occ_mid` is gone. One fewer permanent ABI export and one fewer round trip.
- `Engine.num_long` in Python reads `gpap_formula_info.num_long` like its sibling properties.
  The `gpap_engine_num_long` **export stays** — it is symmetric with `gpap_engine_num_vars` /
  `num_clauses`, which exist because a C consumer may `gpap_formula_free()` and keep the engine,
  and is then the only way to size a `gpap_packed_rows` buffer.
- Engine create builds the soundness pair list with reserve-then-insert (was copy-construct then
  reallocate). `tests/reference_formula.py` drops the now-dead `cls_idx` and SLICES the one
  vectorised literal encoding instead of re-deriving it in a comprehension — an oracle with two
  encodings of its own scheme can be self-inconsistent and still "pass".

Then the three items first deferred as "larger than a cleanup" — all of them duplications this
same change had introduced, so none was inherited debt:

- **`gpap_formula::sat_src`**, THE clause -> per-kind slot map (`>= 0` long index, `< 0` =
  `-(p+1)` into the merged pair list). Built by ONE cursor walk where the ordering rule is
  written down; `gpap_formula_clauses` and `gpap_sat_rows` now index it instead of each
  re-deriving that rule with their own cursors. Two of the three engine host mirrors go with it
  (`clause_sizes_host`, `bin_pairs_host`, `unit_lits_host` -> `sat_src_host` + the pair list
  MOVED from the staging vector, so it is no longer copied at all). `to_dimacs` joins its
  inverse `lit_index_from_dimacs`, replacing three inline copies.
- **`_risk`** (`cpp/kernels/bcp.py`): the four SOFT loops collapse to one helper called once per
  polarity, so the weights and the binary/long split have a single definition and the two
  polarities cannot be scored on different scales.
- **`GPAP_OCC_BIN_TAG`**: binary occ payloads carry the sign bit (a literal index needs <= 24
  bits, a long clause index is non-negative, so it is free and unambiguous). WAVE now classifies
  an entry from the value it already loaded, dropping both the per-frontier `occ_mid` load and a
  fifth W x W one-hot reduction from the engine's hottest loop. `occ_mid` still bounds the
  scalar loop split, the CLEAN restore and the SOFT scan.
  ⚠️ **RE-PRICED 2026-08-27**: the wave schedule is shuffle-based now
  (docs/findings/WAVE_SHUFFLE_FINDINGS.md), so the alternative this tag avoids costs ONE
  `shfl.sync`, not a ~239-instruction shared-memory reduction — roughly 13x less than the price
  quoted below. The tag is still cheaper, and it is already paid for, but anyone re-examining
  whether it earns its 24-bit literal ceiling should re-measure rather than quote the numbers
  in the table below, which were taken against the one-hot schedule.

**The tag was measured, not assumed** — two full interleaved runs at an identical pin
(`--k-cap 20000`, same cells, same base control), tagged vs tag-reverted:

| cell | kernel | head/base tagged | head/base no-tag | tag worth |
|---|---|---|---|---|
| s1196-c25 | wave | 1.587x | 1.436x | **1.11x** |
| b12-c25 | wave | 1.581x | 1.350x | **1.17x** |
| b03-c25 | scalar | 1.396x | 1.038x | not readable |

Consistent with the ALU argument: the gather it removes is in the WAVE inner loop, and scalar
(which only gains a mask instruction) should be ~neutral. ⚠️ b03-c25 is NOT evidence either way
— at this pin it is scalar AND finishes its 20k target in ~0.2 s, so its rate rests on a couple
of chunks; the base control itself drifted +10% between the two runs (108,013 -> 118,978 sps),
which bounds what a cross-run ratio can resolve. The two wave cells moved by more than that
drift in the same direction, which is the signal.

⚠️ **GATE PROTOCOL, second trap**: after the sizing flip, `--k-cap 262144` no longer PINS K
between arms — the analytic bound itself moved, so head sizes ~2x base and the two overshoot an
`--n-time` target by different amounts (b12-c25 read 101 vs 102, s1196 216 vs 425). That is the
intended change, not a regression. A no-change gate must cap BELOW BOTH arms' analytic bounds
(`--k-cap 20000` here): so pinned, b12-c25/s1196-c25/b03-c25 are bit-identical in models AND
success to 8 significant digits.

## Gotchas that bit during landing

- `tests/test_clean_mode.py`'s deep-row-guard formula was a chain of 400 BINARIES — post-change
  it had zero packed words and passed VACUOUSLY. Ternary `(-v|-v|v+1)` links keep the exact
  implication semantics with a packed word (the duplicate literal decrements twice); the binary
  chain became the explicit pure-2-SAT test (`num_long == 0`: stateless rows, full chain
  propagation, bitmap-synthesized sat_rows, reset-lite still firing).
- A VRAM-model check that passes `trail_cap` where the python wrapper expects
  `decisions_per_round` double-charges the trail and reads 0.6× — the wrapper derives trail_cap
  itself.

## Open / next

- L4 campaign re-run on the new engine (user decision): the 20260827 campaign is the pre-binary
  baseline; every published rate and the kernel A/B table would move.
- Stage-0 completeness sweep of binary fractions (ABC/sk_/EPFL/HWMCC, post-ELS).
- Forced-kernel M-ladder re-check on the L4 (`eval/kernel_ab.py`).

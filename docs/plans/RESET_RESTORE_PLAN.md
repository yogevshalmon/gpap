# Reset-restore plan — kill the per-chunk state broadcast

**STATUS: IMPLEMENTED AND GATED (2026-08-21), as the FUSED variant in §9 — the restore is a
per-launch CLEAN mode inside the fixpoint kernel, not the separate pass §3 designed. Read
§§2-3 for the soundness groundwork, then §9 for what actually shipped and its measurements.
§3's separate `trail_restore` kernel and its state machine were SUPERSEDED and never built.** Revives the
design parked in
`docs/findings/MINING_FINDINGS.md` §"The RESET is the bottleneck" (2026-08-20) — the user has
pulled it off the shelf because probing/mining is a first-class workload and its wall is 41–94%
reset. This file adds what the findings sketch did not have: a **kernel-verified soundness
argument** (the sketch's claims are checked against `cpp/kernels/bcp.py` line by line, one of
them was load-bearing and subtle), the **eligibility state machine**, the **per-row cost guard**,
and the gates. Read the findings section first; numbers are not repeated here.

## 1. Problem, one paragraph

`gpap_engine_reset(K)` broadcast-restores the post-units-flush cache: `K·(M + NW)` int32 words
(`bcast_rows` ×2) + two K-byte fills. The batch is sized `K ≈ 0.80·free/row_bytes` with
`row_bytes ≈ 4M`, so **every chunk rewrites ~0.80·free VRAM regardless of instance** — measured
~26 ms/chunk on the RTX 4080 Laptop, 50/70/84/94% of probe-chunk wall on b09-c25 / b03-c25 /
b12-c50 / 6s215rb0-c25 — while the probe it precedes modifies **0.02–2.5%** of those words.
Probing throughput is bounded by `budget/(4M)/26 ms` tuples/s no matter how fast BCP gets.
Beneficiaries of fixing it: `gpap_probe_*` (mining, backbone, `implications`), `gpap_check_sets`
(both AssumptionChecker faces), `gpap_assume_units` — every reset↔single-apply consumer. The
sampler is exempt by design (§5, regime).

## 2. Arithmetic UNDO (backtracking the work) is REJECTED — now kernel-verified

The obvious backtrack — re-walk the trail and `atomic_add` back `1 + (neg << SHIFT)` per
occurrence — is arithmetically exact (the packed word is an abelian group; the same atomics that
made the forward races safe make the reverse safe). It dies on knowing WHICH subtractions
happened:

- **Scalar breaks mid-occurrence-list**: `while (o < end) & (conflict == 0)` (`bcp.py:81`) — a
  conflicted row stops at a (literal, offset) watermark that is stored nowhere.
- **Wave breaks mid-frontier, stride-granular**: all atomics of the current W-stride complete
  (conflict is folded at stride end, `bcp.py:175`), but the remaining `t0..T` entries of the
  frontier never run, and neither the stride position nor the frontier is recorded.
- **Wave leaves a stray assignment bit on contradiction**: the push is OR-then-check
  (`bcp.py:167-170`) — `tl.atomic_or` fires, THEN `contra_v` is derived from the old word and the
  lane is excluded from the trail (`keep = fm & ... & ~contra_v`). Bit set, no trail record. An
  undo that walks the trail can never clear it.

Recording watermarks means new stores in the exit path of BOTH kernels × 29 specializations, and
still leaves the stray bit. The findings' **rewrite-from-baseline** variant needs none of it:
superset-safe (no watermark), idempotent (no atomics), and the stray bit is provably co-located
with a trail bit (L3 below). It strictly dominates; this plan is that variant.

## 3. The design

For each row, rewrite exactly the words the row can have touched, from the flush cache:

    for each row r, for each l in trail_r[0 .. nprops_r), for each c in occ[l]:
        packed[r][c] = packed_flushed[c]
    (v2 also:  abits[r][word(l)] = abits_flushed[word(l)])

**v1 splits the state by risk**: `packed` — 96–98% of the traffic — goes trail-restore; `abits`
(NW/(M+NW) ≈ 2–4%) and the flag fills stay broadcast, which makes lemma L3 unnecessary in v1.
v2 moves `abits` onto the trail walk once the gates are green; the lemma is worked out below so
v2 is mechanical.

### 3.1 Soundness lemmas (each checked against `cpp/kernels/bcp.py` as of 2026-08-21)

- **L1 — the trail is complete and its length is `nprops`.** Every path that walks an occ list
  first popped the literal from the trail; entries live contiguously at `trail[0..tail)`; the
  fixpoint kernel stores `tail` into `nprops` at exit for every row active at entry
  (`bcp.py:234`; pushes at 217-219 seed / 96-98 scalar / 173-174 wave). Rows inactive at entry
  write nothing — and `apply` zeroes the counter arrays first, so their `nprops` reads 0 and the
  restore walks nothing, which is correct (they were never dirtied).
- **L2 — packed damage is covered**: packed words are written only inside the occ walk of popped
  literals ⇒ damaged words ⊆ ∪_{l∈trail} occ[l], for surviving AND conflicted rows — a mid-list
  or mid-stride stop only shrinks the damage, never moves it outside the union.
- **L3 — abits damage is covered (v2 only)**: the seed loop and scalar propagate are
  check-then-set — a bit is set in the same branch that pushes the trail, so bit ⟺ trail entry.
  Wave contra lanes stray a bit, but the contradicting bit lives in the SAME word (l and l^1 are
  bit-neighbours by the 2N indexing), and that bit was set THIS launch: flush-fixpoint
  consistency means a literal still surviving in a clause's sum has its opposite clear in the
  baseline bitmap, so a contradiction is always against a this-launch bit — a seed, a kept push,
  or (inductively) an earlier stray in the same word whose own cause bottoms out at a trail
  entry. Rewriting `word(l)` from baseline for every trail l therefore also clears every stray.
- **L4 — over-restore is free**: every store writes the baseline value, so covering too much
  (stale trail entries after an L=0 no-op apply, trail tails never processed, occ words never
  reached) rewrites baseline over baseline. Two trail literals sharing a clause race on the same
  word with IDENTICAL values — benign, no atomics needed. This is the property that makes the
  whole design robust to every partial-stop mode the kernels have.

### 3.2 Eligibility — a state machine inside the engine

Two fields: `baseline_rows` (leading rows currently equal to the flush cache) and
`dirty_rows` + `dirty_recoverable` (what the last apply touched, and whether the trail record
still describes it). Transitions:

| event | effect |
|---|---|
| broadcast reset(K) | `baseline_rows = max(baseline_rows, K)`; dirty cleared |
| restore reset(K) | restore `dirty_rows` rows; dirty cleared; then K ≤ baseline_rows ⇒ done, else fall back to broadcast (K grew past the clean high-water) |
| `apply` (L>0) after a reset | `dirty_rows = e->K`, `dirty_recoverable = true` |
| second `apply` without a reset | `dirty_recoverable = false`, `baseline_rows = 0` — apply re-zeroes the counters and the kernel rewrites the trail from 0, so the first apply's record is gone |
| `apply` L==0 | no-op (engine returns before launching), nothing changes |
| trajectory launch | `dirty_recoverable = false`, `baseline_rows = 0` (v1 keeps the sampler on broadcast, §5) |
| `recache_flush` | baseline MOVED ⇒ `baseline_rows = 0`, `dirty_recoverable = false` |
| any realloc of packed/abits/trail/nprops | `baseline_rows = 0`, `dirty_recoverable = false` — `DevBuf::reserve` grows by release+`cuMemAlloc`, CONTENT IS LOST (`runtime.cpp:101-113`). Detect by comparing `.bytes` around `ensure_state`. |
| flush not ready / `flush_conflict` | unchanged today's path (construction seeding; probe already refuses flush-conflict formulas) |

Two consequences worth naming: a reset;reset with no apply between becomes ~free (empty dirty
set — fills only), and the probe/check pattern (short LAST chunk, `reset(rows)` with shrinking
rows — `probe.cpp:187`, `check.cpp:101`) restores the previous chunk's `dirty_rows` and keeps
the high-water, so the whole sweep after chunk 1 runs restore-only.

### 3.3 The per-row cost guard (the findings' regime bound, enforced where the cost is exact)

Scattered 4-byte stores pull 32-byte sectors — up to 8× amplification against the broadcast's
streamed writes; the findings put the crossover at `mean_trail·mean_occ/M ≈ 1/8`. Probing sits
at 0.0002–0.084, trajectories at 0.32–0.62 — but per-ROW is where the truth is, and the restore
CTA can afford it: sum `occ_off[l+1]-occ_off[l]` over its own trail (≤ nprops loads, vectorized),
and if `AMP·Σ > M` **stream the whole packed row from baseline instead** (same CTA, coalesced
loop — a per-row bcast). Deep-propagation rows (b09-class) degrade to today's cost instead of
past it; the win is monotone by construction. `AMP = 8` initially; tune in the perf pass.

### 3.4 The kernel and the plumbing

- **One new helpers.py kernel, `trail_restore`** — no SHIFT/WAVE dependence ⇒ ONE specialization
  (bcp's 29 stay untouched; no propagate-kernel change at all in v1). Grid `(dirty_rows,)`,
  W=32 lanes; args `(packed, packed_flushed, trail, nprops, occ_off, occ_cls, M, TRAIL_CAP,
  THRESH)`. Scatter arm reuses the wave segment-flattening idiom; stream arm is a BLOCK-strided
  row copy. int64 row indexing like every row kernel. Respect the helpers.py header rule (no
  scalar atomics; row-private mutation ⇒ the num_warps=1 gotcha) — stores here are idempotent
  values, so redundant threads are benign, but say so in the kernel docstring.
- **The decision lives inside `gpap_engine_reset`** — the auto-dispatch lesson (three times
  burned): put the rule where the ABI is and every consumer gets it — probe, check, assume_units,
  and any C-ABI consumer, with ZERO caller changes.
- **No allocation, no K change**: trail and nprops already exist and are already charged in
  `gpap_row_bytes` ⇒ the K model, the kill-switch validation and `gpap_dispatch_wave` are all
  untouched. (Mode still resolves at reset exactly as today.)
- **`GPAP_RESET=broadcast|auto`** env escape hatch, read at engine create (mirrors
  `GPAP_KERNEL`); default `auto` = restore when eligible.
- **Observability**: additive ABI getter (e.g. `gpap_reset_stats(e, *n_broadcast, *n_restore)`).
  The perf gate MUST be able to see restores actually fired — a silent fallback-to-broadcast
  measures as exactly 1.00× and looks like "the optimization is worthless".

## 4. Gates — all green BEFORE any perf number is quoted

1. **State-identity fuzz (new test, the one that would catch any lemma being wrong)**: random
   small CNFs × {scalar, wave forced} × {16|16, 8|24 forced} × set mixes engineered to hit:
   surviving rows, seed contradictions, forced-both-polarities (the wave contra/stray case),
   mid-list conflicts, K shrink (short last chunk), K grow, L==0 apply, double apply (must fall
   back), `assume_units`/recache, flush-conflict formulas. After every reset-via-restore, DtoH
   compare `packed` (v2: + `abits`) **bit-exact** against a broadcast-reset twin engine.
2. **Mined-set identity**: `gpap mine` clause-set SHAs identical under `GPAP_RESET=auto` vs
   `broadcast`, both kernels forced, on the standing gate instances b03-c25 / s386-c25 / b09-c25
   (hashes recorded in MINING_FINDINGS).
3. **Checker invariance**: the existing `tests/test_assumption_checker.py` chunk-invariance +
   reference-BCP oracle suites pass with restore active (they exercise the exact
   reset↔apply alternation, including conflicted-row semantics), plus
   `test_mining_is_kernel_invariant` and the full pytest run.
4. AOT rebuild sanity: 29+1 kernels, `gpap info`, CI build-core job (helpers.py is part of the
   AOT list).

## 5. Perf protocol — DEFERRED until the box is free (CPU busy 2026-08-21)

- Phase-split (sync-stepped) chunk timings on the four findings instances → new reset ms/share.
- End-to-end `gpap mine --width 1 2 --pool all --seconds 10` A/B via the env var, INTERLEAVED
  ≥3 alternations in one process (this box's SM clock swings 615–1710 MHz; sequential
  before/after cannot resolve the number — standing measurement rule), both kernels.
- Expected floors, from the findings' bound (restore ≤ measured apply): b03-c25 ≥1.7×,
  b12-c50 ≥3.3×, 6s215rb0-c25 ≥10.3×, b09-c25 ~1.0× (honest — its guard rows will stream).
- Same A/B on a `gpap_check_sets` microbench (the checker face inherits the win).
- End state to verify: probe chunks become APPLY-bound, i.e. tuples/s finally scales with BCP
  speed (and with SMs across GPUs) instead of with `budget/(4M)/26 ms`.
- Results replace the "STATUS: designed, measured, NOT implemented" block in MINING_FINDINGS.

## 6. Non-goals and standing rejections (recorded in the findings; do not re-litigate)

- Smaller chunks (ratio-invariant), two-stream overlap (≤1.3× bound, both phases memory-bound),
  fusing the broadcast into BCP (K·M must reach DRAM either way).
- Prefix-sharing enumeration (one reset per shared tuple prefix): touches the core's
  lex-order/gate-comparability contract — only revisit if trail-restore under-delivers.
- Copy-on-write / sparse row state: the asymptotic answer (kills K·M materialization AND lifts
  K; SparseBCPEngine measured up to 31× in this regime) — a separate project; trail-restore
  gets most of the win at a fraction of the risk.
- The SAMPLER stays on broadcast in v1 (trajectory clears eligibility): its trail ratio
  0.32–0.62 × ~8 amplification is the losing side, and its reset is not its bottleneck. The
  per-row guard would protect it anyway; making that flip is a v3 question, gated like any
  sampler change (uniformity + no-change gates).

## 7. Risks

- **The lemmas are per-kernel-version facts.** Any future edit to the push/claim/conflict paths
  of `bcp.py` can invalidate L1–L3 silently; the state-identity fuzz (gate 1) is the tripwire
  and must stay in the default test run, cheap.
- Stale-trail over-restore is safe only because trail/nprops never shrink and reallocs clear
  eligibility — the realloc detection must cover ALL FOUR buffers (packed, abits, trail, nprops).
- Wave's trail push has no cap check (structurally unreachable, cap ≥ N+B) — restore trusts
  `nprops ≤ trail_cap`; assert it host-side in debug builds.
- `nprops` is the raw trail length INCLUDING seeds (`ndecs` are the seeds) — do not "fix" the
  restore to `nprops - ndecs`; that under-covers and is exactly the kind of plausible edit gate 1
  exists to catch.

## 8. Sequencing

1. `trail_restore` kernel + engine state machine + env + stats getter (one sitting).
2. Gate 1 fuzz test — the real work; write it to fail first against a deliberately broken
   restore (e.g. walk `nprops - ndecs`) to prove it has teeth.
3. Gates 2–4; AOT rebuild.
4. Perf pass when the box frees up; MINING_FINDINGS + CLAUDE.md updates.
5. v2 (abits on the trail walk, L3) only if the residual 2–4% matters after measurement.

## 9. REVISION (2026-08-21, same day): fuse the restore into the kernel — CLEAN mode

> **STATUS: IMPLEMENTED, GATES GREEN (2026-08-21, same session).** What shipped, exactly as
> below: `_bcp_fixpoint_clean_kernel` + `_clean_packed` (bcp.py, appended — the 29 pre-existing
> cubins verified BYTE-IDENTICAL after the rebuild), `gpap_engine_apply_clean` +
> `gpap_engine_reset_stats` + `gpap_packed_rows` (ABI), the reset-lite branch + clean-rows
> high-water in `gpap_engine_reset`, `GPAP_CLEAN=0` as EXACT-legacy escape hatch (it also
> disables reset-lite, so the A/B is uncontaminated), and the two call sites (probe_chunk;
> check_sets flags-only). Gates: `tests/test_clean_mode.py` (13 tests — state-identity fuzz
> against a plain-apply twin + the reference BCP across both kernels x both splits, the deep-row
> stream arm, K shrink/grow, every invalidation, probe/checker/mining A/B under the env) —
> verified to FAIL against a deliberately broken tail (walk `tail-1`); full suite 417 passed.
> Measured (RTX 4080 Laptop, CPU busy — treat as smoke): b03-c25 probe sweep END-TO-END 2.60x
> interleaved best-of-3, identical conflicts; b12-c50 sync-bracketed phase split
> **reset_full 17.9 ms + apply 3.0 ms = 20.9 ms legacy chunk vs apply_clean 4.3 ms +
> reset_lite 0.4 ms = 4.7 ms clean chunk = 4.4x GPU-side** (the tail costs 1.3 ms — the
> hot-cache claim held). Two follow-ups for the quiet-box perf pass: (1) b12 END-TO-END showed
> only 1.27x because ~20-30 ms/chunk of HOST work (enumeration + driver calls on a contended
> CPU) now dominates — re-measure quiet; if it persists, device-side tuple generation
> (anticipated in probe.cpp's header) is the next lever; (2) FORCED-scalar at large M is a
> regression (b12 apply 126 -> 166 ms): the bigger clean kernel hurts the serial propagate —
> irrelevant to dispatched runs (scalar is only picked at M < 2048, where trails are tiny), but
> a forced-kernel A/B must expect it.

User proposal, adopted after analysis: instead of restoring at the NEXT reset, each row's program
ends its own launch with **propagate → emit result → rewrite its own dirtied words to baseline**.
Same lemmas (L1/L2/L4 verbatim — the clean tail IS the trail walk, with `tail` still in a
register; L3 only when the tail also cleans abits), better physics, less machinery. This
supersedes §3's separate `trail_restore` kernel and most of §3.2; §§4-7 stand with the deltas
below.

### Why fused wins over restore-at-reset

- **Hot-cache restore.** The tail rewrites words the propagate loop just touched: trail and
  `tail` in registers, occ segments warm, touched packed words still L2-resident (~1,400
  resident rows × 1.4-8.6 KB touched = 2-12 MB, inside this card's L2). The separate kernel
  re-reads trail/nprops/occ from DRAM and pays the scattered-sector cost cold — §3.3's 8×
  amplification mostly applies to the COLD variant.
- **A chunk becomes ~one launch.** No broadcast, no restore launch; CLEAN mode self-initializes
  its flags at entry (each row stores active=1/conflict=0 before seeding — sound because launches
  serialize on the stream and the host reads flags between them).
- **§3.2's state machine mostly evaporates** — the subtlest part of the old plan ("exactly one
  apply", stale-trail rules, recache interaction). State is clean BY CONSTRUCTION at launch exit.
  What survives: first-touch broadcast, and re-broadcast on K growth / any realloc (reserve
  loses contents), tracked as a clean-rows high-water.
- **The §3.3 guard improves**: the `nops` register already holds the row's occ-entry count, so a
  deep-propagation row streams its whole packed row from baseline as its tail, decided in-kernel
  for free. AMP=8 initial, tuned in the perf pass.

### The two constraints that shape it

1. **CLEAN is a PER-LAUNCH MODE, not the universal path — "copy result" is the crux.** Flags +
   counters (probing, mining incl. the `derivable` launch) are tiny and already emitted: clean
   fits. But `check_assumptions(assignments=/sat_count=)`, the width-1 closure
   (the closure sweep, which reads every row's assignment bits) and `gpap_assume_units`
   (reset(1)+apply+**recache_flush** — the POINT is capturing post-apply state) all need the
   state to SURVIVE the launch; copying sat_count out first would write K·M words = the
   broadcast again. Those callers keep today's semantics untouched. `probe_chunk` (and any
   flags-only caller) opts in.
2. **It touches the hot kernels — contained by making CLEAN a constexpr.** Four NEW fixpoint
   specializations (SHIFT × WAVE × CLEAN=1), 29 → 33 cubins; trajectory untouched. The CLEAN=0
   binaries compile from the identical source path and must come out **bit-identical at the SASS
   level** — diff them in the build as the proof the existing paths are untouched (extends the
   AOT build's existing JIT-vs-AOT identity check).

### Staging

- **v1**: tail cleans `packed` only; keep a tiny per-chunk abits+flags broadcast (NW/(M+NW) ≈
  2-4% of today's traffic) so L3 stays out of the blast radius. Engine entry:
  `gpap_engine_apply(..., clean=1)` or a sibling ABI call; it must set `e->K` itself (no reset
  precedes it in steady state).
- **v2**: fold abits into the tail via L3 (strays share their word with a trail bit — argued in
  §3.1, gate-checked); drop the last per-chunk broadcast, self-init flags in-kernel.
- **Parked**: the persistent-kernel extreme (one program loops over many tuples, cleaning
  between). It fights MINE_SECONDS chunk granularity ("a chunk is the smallest abandonable
  unit") and multi-second kernels on this WSL2 box risk the display watchdog; chunked fused
  launches (~30-60 ms) keep nearly all the win.

### Gate deltas (§4 otherwise stands)

- State-identity fuzz gains the clean-mode assertion: **post-launch state == baseline
  bit-exact** (packed in v1; + abits in v2) on every mix in the matrix, both kernels, both
  splits — plus flags/counters equal to a non-clean twin's.
- SASS-identity of the CLEAN=0 cubins vs today's (the no-regression proof for every non-probe
  consumer).
- Perf expectation vs §5's floors: fused ≈ 1.3-1.7× apply per chunk (vs restore-at-reset's
  ≈ 1.7-2×) — 6s215rb0-c25 ~26.6 → ~1.5-2 ms/chunk, b03-c25 ~31.8 → ~13 ms; b09-c25 stays ~1.0×
  (guard rows stream). Same interleaved-A/B protocol, still deferred until the box is free.

### L4 validation (2026-08-21) — the gate §5 asked for, on a second GPU

Colab L4 (sm_89), full seq family at T=25/rare/tseitin, NEW vs `GPAP_CLEAN=0` on the same VM:
**19/19 rows both arms, 0 errors, 0 unsound; 16/19 BIT-IDENTICAL model counts** (the 3 that
differ are the mining-ceiling tune, not CLEAN — their mined clause sets differ). The full test
suite runs on the L4 too: 386 passed, 54 skipped, 1 failed — and that one
(`test_oversized_result_is_refused_not_spilled`) is a PRE-EXISTING portability bug unrelated to
this work: it hardcoded a 16 GiB "obviously too big" ask, which a 23.6 GiB L4 legitimately
accepts. Fixed to size the ask from the device's own budget.

Against the RECORDED 2026-08-05 L4 eval the same family is **2.30× faster (geomean, vacuous pins
excluded) with M identical on 19/19** — same formula, same problem, pure throughput. That figure
is CUMULATIVE over everything since (the dispatch fix flipped 8 instances scalar→wave, plus the
batch-model collapse and this work), NOT a CLEAN-mode attribution; the same-VM A/B above is what
attributes CLEAN, and on the sampler it is ~1.0× by construction.

Data: `$GPAP_RESULTS/clean_mode_l4_20260821/`.

### Review fixes (2026-08-22) — the contract is now ENFORCED, not commented

A code-review pass found that the asymmetry CLEAN mode creates (packed = baseline, abits =
fixpoint) was documented but not defended, and one legal ABI sequence corrupted the engine:

| | was | now |
|---|---|---|
| `recache_flush` after a clean launch | cached the INCONSISTENT pair — abits claiming literals TRUE whose clause decrements packed never recorded, poisoning every later launch, silently | **refused** with an error naming the fix (`engine.abits_diverged`) |
| `gpap_sat_rows` / `gpap_check_sound` after one | answered from the baseline: plausible values describing no launch that ran; `check_sound` pronounced on the wrong state | **refused**, same flag |
| `gpap_packed_rows` | — | deliberately still OPEN: the raw/debug readout, and what the identity gate uses to PROVE the restore |
| `Engine.apply` / `apply_clean` row count | unchecked — the C side stages K·L int64s, so a short array read past its end and the garbage became literal indices, and an index past 2N makes the kernel write into a LATER row | `ValueError` in the binding, both entry points |
| the header's contract | claimed the assignment readout sees the baseline — WRONG, v1 cleans packed only, so abits keep the fixpoint | states the asymmetry per-readout |
| tier-2 `gpap::Engine` | no `apply_clean` / `reset_stats` — C++ consumers could not opt in without dropping to the raw ABI | both wrappers added |

`abits_diverged` is the one new bit of state: set by a completed clean launch, cleared by
`reset` (which re-broadcasts abits) and by any plain apply or trajectory (which leave the two
halves consistent). Gated by `TestCleanModeGuards` in tests/test_clean_mode.py — each guard has
a positive control, so a guard that stopped firing fails the suite rather than passing quietly.

### Why not do §3 first as a stepping stone

The separate-kernel variant would build a kernel and a state machine that the fused version then
deletes, while the actual risk item — the lemma-backed clean logic and its fuzz gate — is
IDENTICAL in both. Do the gate work once, on the architecture we keep.

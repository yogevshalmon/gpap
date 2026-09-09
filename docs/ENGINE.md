# The GPAP engine — how it works

GPAP's engine runs Boolean Constraint Propagation (BCP) over K independent partial assignments
at once, entirely on the GPU. K ranges from a few thousand rows on the largest formulas to
262,144; each row is its own assumption set over the same formula, propagated to its own
fixpoint, and no clause is ever scanned. The solution sampler, entailed-clause mining and
batched assumption checking are all consumers of this one primitive.

The engine is one C++ library, `libgpap` ([`cpp/`](../cpp/)), with a C ABI and two faces: Python
through ctypes and C++ through `<gpap/gpap.hpp>`. The kernels are written in Triton
([`cpp/kernels/bcp.py`](../cpp/kernels/bcp.py)) and compiled ahead of time into the library, so
running it needs only the NVIDIA driver (`libcuda`). This page is for building on the engine or
judging its claims; using it needs none of it — see [PYTHON_API.md](PYTHON_API.md) and
[`cpp/README.md`](../cpp/README.md).

## 1. The formula on the GPU

**Literals are indices.** DIMACS `+v` is `2(v−1)` and `−v` is `2(v−1)+1`, so negation is `l ^ 1`
and a variable's two polarities are neighbours. `N` variables give `2N` literal indices.

**Negated occurrence lists.** `occ[l]` lists the clauses in which `~l` appears — exactly the
clauses that lose a literal when `l` becomes true. Setting `l` walks `occ[l]` and nothing else.
The lists are one CSR array shared read-only by every row.

**Clauses are segregated by length:**

| clause length | where it lives | per-row state | propagating it costs |
|---|---|---|---|
| 1 (unit) | asserted once, at construction, into the flush cache (§3) | none | nothing |
| 2 (binary) | an implication edge in `occ`: the entry for `(~l \| b)` in `occ[l]` carries `b` itself | none | one bitmap `atomic_or` (§2) |
| ≥ 3 (long) | a clause index in `occ`, plus one packed counter word per row | 4 bytes | one `atomic_add` on that word (§2) |

On circuit CNFs 59–82% of clauses are binary and only 10–23% are long, so per-row state is
`4·M_long` bytes rather than `4·M`. Within each `occ[l]` segment the binary entries come first
(`occ_mid[l]` marks the split).

## 2. Two words that detect their own events

Every BCP step must know, per touched clause, whether it just became empty (conflict) or unit
(a forced literal), and per forced literal whether it is already assigned or contradicted. Both
are encoded so that the old value of a single word update is the whole answer — each update is
one atomic, and its return value resolves any race between lanes of the WAVE kernel.

| state | one word holds | the update | what the old value says |
|---|---|---|---|
| long clause — `(K, M_long)` int32 | `sat_count` in the low `SHIFT` bits; the sum of the surviving literal indices mod 2^(32−SHIFT) above | `atomic_add(−(1 + (¬l ≪ SHIFT)))` | old count 1: this update emptied the clause (conflict); old count 2: it is now unit, and the forced literal is `(old_sum − ¬l) & MASK` |
| assignment — `(K, ⌈2N/32⌉)` int32 bitmap | bit `l` = literal `l` is true | `atomic_or(1 ≪ (l & 31))` | old bit `l`: duplicate, skip; old bit `l^1`: contradiction |

Why a sum: when `sat_count` reaches 1 exactly one literal survives, so the sum of surviving
indices is that literal. Every removal is exact modulo 2^F, and the fields never interfere: a
decrement only happens at `sat_count ≥ 1` (a row that reaches 0 has conflicted and stops), so
the `−1` never borrows into the sum, and the sum wraps off the top of the word.

```
 SHIFT = 16   (16|16 split: 2N ≤ 65,535, clause length ≤ 65,535)
   bit:  31                     16 15                      0
        +-------------------------+-------------------------+
        |  Σ surviving lits % 2^16 |        sat_count        |
        +-------------------------+-------------------------+

 SHIFT = 8    (8|24 "wide" split: 2N ≤ 16,777,215, clause length ≤ 255)
   bit:  31                                  8 7            0
        +--------------------------------------+--------------+
        |      Σ surviving lits % 2^24          |  sat_count   |
        +--------------------------------------+--------------+
```

The split is chosen per formula — 16|16 when it fits, else 8|24. A formula outside both
envelopes is rejected at construction; there is no slower fallback. Why a bitmap: `l` and `l^1`
share a 32-bit word, so one load (or one atomic) answers free / duplicate / contradiction
together, and the assignment is 8× smaller than a byte per literal.

## 3. Lifecycle

```mermaid
flowchart TD
    F["formula: clauses in DIMACS"] --> B["build (cpp/src/formula.cpp)<br/>occ CSR + occ_mid · packed init (M_long words) · unit list"]
    B --> FL["flush cache: ONE K=1 fixpoint over the unit clauses<br/>= the starting state of every row"]
    FL -->|"reset(K): broadcast the cached row<br/>(skipped when the rows are already clean, §6)"| S["live state, K rows<br/>packed (K, M_long) · bitmap (K, ⌈2N/32⌉) · trail · active/conflict flags · counters"]
    S --> E1["entry point 1 — apply_and_propagate(K×L decisions)<br/>one fixpoint launch"]
    S --> E2["entry point 2 — run_trajectory(order, n_support, B, seed)<br/>the whole decide+propagate loop, one launch per chunk"]
    E1 --> R
    E2 --> R
    R["readout — the selection API<br/>select_complete · check_sound · var_bits (on device: only counts and bits cross the bus)<br/>conflict · active · assignment (copies, off the hot path)"]
```

Construction runs the unit clauses to fixpoint once, at K=1; every row of every chunk starts
from that state, so `reset(K)` is a broadcast of one cached row, never a per-chunk propagation.

**Entry point 1 — batched assumption checking.** Each row receives its own assumed literals and
is propagated to fixpoint in one launch. `check_sets` wraps it for any number of sets (the chunk
loop is `cpp/src/check.cpp`, shared by both faces), and the probe (`cpp/src/probe.cpp`) feeds it
every k-subset of a literal pool under a work budget — the primitive behind failed-literal
detection and entailed-clause mining.

**Entry point 2 — fused trajectory sampling.** Each row runs a whole randomized
decide-and-propagate trajectory: take the next `B` free variables in a caller-supplied order,
give each a hash-derived polarity, propagate, repeat until the row is fully assigned (a model) or
conflicts. No host round-trips inside the loop (§5).

**Readout never moves row state.** The selection API computes which rows are complete and sound
on the device and hands back a count; `var_bits` then gathers only the columns asked for. The
worked example is in [`cpp/README.md`](../cpp/README.md), "Getting answers out: the selection
API". The tensor-shaped accessors (`assignment`, `conflict`, `active`) are copies for tests and
off-hot-path consumers.

## 4. Propagation: the scalar and WAVE kernels

Both kernels drain a per-row trail of literals. One Triton program owns one row, so rows never
race with each other; the only races are between lanes inside a WAVE program, and the atomics
of §2 resolve exactly those.

- **Scalar** pops one literal and walks its occurrences serially: the binary prefix pushes the
  payload literal; each long entry does `word −= 1 + (¬l ≪ SHIFT)`, count 0 is a conflict and
  count 1 yields the forced literal from the high field, which the bitmap classifies as
  duplicate, contradiction or push. No atomics, one memory request in flight per row — enough
  when the formula is tiny and rows are many.
- **WAVE** pops up to `W = 32` literals, flattens their occ segments (prefix sums + a
  warp-shuffle lower_bound; no shared memory), issues `W` scattered atomics per step, classifies
  each by its return value and compacts the new forced literals onto the trail. Two lanes forcing
  `l` and `¬l` serialize on the bitmap word; exactly one sees the other's bit.

The clause count decides which one runs (`gpap_dispatch_wave` in `cpp/src/calibrate.cpp`,
resolved inside every `reset`, so every consumer gets it): WAVE when `M ≥ 2048`, or when the
batch is no larger than the card's resident warps; scalar otherwise. Do not key the rule off the
batch: a batch-based rule was a proxy for `M` that broke when batch sizing changed.
`GPAP_KERNEL=scalar|wave` forces a kernel; it can only change speed, never results.

**Confluence.** A BCP fixpoint is unique, so the two kernels — which visit clauses in different
orders — reach bit-identical states on every row that does not conflict. A conflicted row stops
at the first emptied clause, and which clause that is depends on order; its partial state and
its implication counter are unspecified, while its `conflict` flag is exact.

## 5. The trajectory kernel

`_bcp_trajectory_kernel` fuses the sampler's loop: pick the next `B` free variables in `order`
(polarity = a stateless splitmix hash of seed, row and variable), propagate to fixpoint (§4),
repeat; a row ends on conflict, or with every variable assigned — a model by construction.
`order` is the decision priority — the independent support first, completion variables after —
and a decision group never straddles that boundary: phase 1 reaches a fixpoint over the support
before any completion variable is decided (straddling it once collapsed an instance's success
from 100% to 0%).

The selection API reads the models out, and `check_sound` re-verifies every clause of every
selected row on the device — binaries and units against the bitmap, long clauses against their
words — independently of the propagate kernels' own bookkeeping.

Two compile-time variants: **VSCAN** (WAVE only) screens `W` order entries per step with one
bitmap gather instead of walking them one by one — the same picks, 12–19% faster where it
applies. **SOFT** chooses the lower-risk polarity from the clauses each literal would touch; it is
opt-in (`--allow-soft`) because it biases the distribution enough for Barbarik to reject it.

## 6. CLEAN mode: launches that leave no trace

Probing and mining run `reset → apply → read the flags` millions of times, and the reset
broadcast (`K·M_long` words per chunk) was 50–94% of their wall while a probe dirtied at most a
few percent of the words. `apply_clean` runs the same fixpoint and then, inside the same launch,
rewrites the words it touched back to the baseline — the trail is still in registers and the
words still in L2 — so the engine knows those rows are clean and the next `reset` skips the
broadcast. The restore covers every word a row could have touched (`∪ occ[l]` over its trail),
which is why it is safe on conflicted rows that stopped mid-walk: over-restoring writes baseline
over baseline. Only the clause words are restored; the bitmap keeps the fixpoint, so readouts
that need the post-apply state (`sat_rows`, `check_sound`, `recache_flush`) refuse after a clean
launch — use plain `apply` for those. `GPAP_CLEAN=0` is the exact legacy behaviour, for A/Bs.
Design, lemmas and measurements: [findings/CLEAN_MODE_FINDINGS.md](findings/CLEAN_MODE_FINDINGS.md).

## 7. Batch size

Nobody names K. Every face sizes it from the formula and the GPU (`cpp/src/calibrate.cpp`):

```
row_bytes = 4·M_long  +  4·⌈2N/32⌉  +  4·trail_cap  +  flags and counters
K         = min( 0.80 · free VRAM / row_bytes ,  cap )          cap = 262,144
```

`trail_cap = N + 32` is a proof, not a tuning knob: every trail push is deduplicated against the
bitmap and `l`/`¬l` share a word, so a variable enters a row's trail at most once — the bound is
exactly `N`, whatever the decision batch `B` or the assumption width. The `+32` is one
wavefront: the vectorised append writes at `tail + prefix_sum` with no bound test, so a wavefront
is the granularity a broken invariant would overshoot by. There is deliberately no `B` term (one
used to charge callers for a trail that was never allocated), and the cap is one constant on
every GPU (a per-card cap made `auto` mean a different batch per machine). The model is verified
against measured VRAM (1.00–1.04×). Because `K × M_long` is roughly the memory budget however
large the formula, a chunk's wall is roughly constant, and models per chunk = K × success is the
throughput proxy the sampling pipeline uses to decide whether mined clauses paid.

The cap is a saturation limit, not a safety one: on the cards measured, 4× the batch bought
~3%. Overrides exist for measurement (`k_cap`, `pin_batch`) — pin them for any A/B.

## 8. Knobs

| | effect | default |
|---|---|---|
| `GPAP_KERNEL=scalar\|wave` | force the propagate kernel | auto (§4) |
| `GPAP_CLEAN=0` | disable CLEAN mode and the reset skip (exact legacy behaviour) | on |
| `vscan = -1` in `gpap_engine_options` | disable the warp-wide decision scan | on in WAVE |
| `GPAP_LIB=/path/libgpap.so` | which library the Python face loads | `build/`, then the installed copy |
| `WAVE_W` in `cpp/tools/build_kernels.py` | lane width, a power of two, baked at build time | 32 (both ways of raising it measured out) |

## 9. What is guaranteed

- **Two entry points** — `apply_and_propagate` (with `check_sets`, the probe and `apply_clean`
  over it) and `run_trajectory` — plus readout. Application code never touches row state. The
  sampling chunk loop over entry point 2 (chunks, the config bandit, stop valves, kept-row
  buckets, accounting) is also the core's — `gpap_sampler_*` in `cpp/src/sample.cpp` — so the
  Python and C++ sampler faces are marshalling over one implementation.
- **`conflict` and `active` are exact** on every row. A non-conflicted row holds the exact BCP
  fixpoint; a conflicted row's partial state is unspecified (§4).
- **Every emitted model is verifiable** by anyone: a full assignment satisfying every clause,
  checked on the device before it is returned and re-checkable on the host.
- **Validated** against a pure-Python reference BCP (`tests/reference_bcp.py`) and a reference
  formula builder (`tests/reference_formula.py`), in both kernels and both packed splits, plus a
  bit-identical model-count gate that re-runs on every kernel change.

## 10. Where the evidence lives

| decision | evidence |
|---|---|
| the sum-packed word — one `atomic_add` per clause update | [findings/INSIGHTS.md](findings/INSIGHTS.md), 2026-07-13 |
| the assignment bitmap — one `atomic_or` per push | INSIGHTS.md, 2026-07-14 |
| binary clauses as implication edges; only long clauses carry state | [findings/BINARY_OCC_FINDINGS.md](findings/BINARY_OCC_FINDINGS.md) |
| the WAVE propagate and its shuffle-based lane schedule | [findings/WAVE_SHUFFLE_FINDINGS.md](findings/WAVE_SHUFFLE_FINDINGS.md) |
| the `M ≥ 2048` dispatch rule; the wave/scalar ratio by formula size | [findings/KERNEL_AB_FINDINGS.md](findings/KERNEL_AB_FINDINGS.md), INSIGHTS.md 2026-08-18 |
| CLEAN mode | [findings/CLEAN_MODE_FINDINGS.md](findings/CLEAN_MODE_FINDINGS.md), [findings/MINING_FINDINGS.md](findings/MINING_FINDINGS.md) |
| the batch model, the cap, and why K is not a throughput lever | [findings/CROSS_GPU_FINDINGS.md](findings/CROSS_GPU_FINDINGS.md), INSIGHTS.md |
| the C++ core and the ahead-of-time kernel build | [findings/CPP_BACKEND_FINDINGS.md](findings/CPP_BACKEND_FINDINGS.md) |
| what was tried and lost — XOR word, 1-byte state, W > 32, in-kernel retry, clause renumbering | INSIGHTS.md, "Engine & memory model" |

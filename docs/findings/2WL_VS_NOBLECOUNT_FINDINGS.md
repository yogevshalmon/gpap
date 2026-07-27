# 2-Watched-Literals vs NobleCount — per-warp BCP head-to-head

**Question.** GPAP's propagate primitive is *NobleCount*: one packed `sat_count | Σ-surviving-lits`
word per clause, updated by a single op whose old value yields conflict/unit/forced-literal with
**no clause scan**. The classic CDCL alternative is **2-watched-literals (2-WL)**: visit a clause
only when a watched literal is falsified, then *scan* the clause for a replacement watch. On
CPU/CDCL, 2-WL is *the* canonical BCP speedup. Does that carry to the GPU at the **same per-row
scalar parallelisation** (one Triton program per row, serial trail — GPAP's "scalar" kernel), on
GPAP's workload (narrow Tseitin/ABC circuit clauses, forward-only sampling trajectories, no
backtracking)?

**Answer: no — 2-WL is 1.8–2.6× SLOWER, does ~1.9× more memory work, and needs ~4.9× more
per-row state.** NobleCount is the right primitive for this domain. The original GPAP thesis
("NobleCount is more parallel-friendly than 2-WL") is confirmed *and quantified*, with an extra
finding: on narrow clauses NobleCount also does **less total work**, not merely more-parallel work.

---

## Method (apples-to-apples, correctness-gated)

- **`gpap/engine/wl_engine.py`** — `WLBCPEngine`, a full 2-WL engine mirroring `BCPEngine`'s two
  entry points (`apply_and_propagate` / `run_trajectory`), the SAME assignment bitmap, and the
  SAME splitmix polarity hash. So with identical `order`/`seed`/`B`, a 2-WL trajectory produces
  **bit-identical models** to the scalar NobleCount trajectory (BCP fixpoints are confluent) —
  that is the correctness gate. Scalar only, one warp per row.
- **CPU work proxy** (`scratchpad/wl_work.py`) — simulate both schemes over identical trajectories,
  count memory touches, assert identical fixpoints. Isolates the *work* question from GPU effects.
- **Interleaved fixed-K wall-time A/B**, same process, warmup-dropped.

**Correctness: 192/192 bit-identical + externally-valid** (8 instances × B∈{1,4,8} × 8 seeds).

## Results (RTX 4080 laptop, scalar vs scalar, K as noted)

| instance   |   K  |   M   | maxW | NobleCount | 2-WL     | **2-WL slower** | work (2WL/Noble) | per-row mem |
|------------|-----:|------:|:----:|-----------:|---------:|:---------------:|:----------------:|:-----------:|
| b06-c5     | 4096 |   881 |  5   |  1.16 ms   |  2.13 ms |   **1.84×**     |     1.88×        |    4.9×     |
| b03-c10    | 4096 |  5801 |  5   |  4.89 ms   | 11.42 ms |   **2.33×**     |     1.89×        |    4.9×     |
| s386-c10   | 4096 |  5061 |  5   |  5.92 ms   | 15.08 ms |   **2.55×**     |     1.91×        |    4.9×     |
| b10-c10    | 4096 |  7871 |  6   |  8.55 ms   | 21.17 ms |   **2.48×**     |     1.94×        |    4.9×     |
| s1196-c5   | 4096 |  9381 |  5   | 14.22 ms   | 36.00 ms |   **2.53×**     |     2.05×        |    4.9×     |
| b12-c10    | 2048 | 43771 |  6   | 26.77 ms   | 61.12 ms |   **2.28×**     |     1.88×        |    4.9×     |
| b03-c25    | 2048 | 14501 |  5   |  8.99 ms   | 20.12 ms |   **2.24×**     |     1.90×        |    4.9×     |

The GPU `work` column (occ-entries vs visits+scan-reads) matches the CPU proxy's **~2×**
independently.

## Why 2-WL loses here (the analysis)

1. **Narrow clauses defeat the watched-literal advantage.** 2-WL's win on CPU comes from (a)
   skipping *satisfied* clauses and (b) not re-touching every literal. For width-2/3 Tseitin
   clauses the CPU proxy shows 2-WL visits only **1.08–1.19× fewer** clauses than NobleCount —
   the satisfaction-skip barely helps because clauses are too short. Meanwhile every visit pays a
   **replacement-watch scan** (read the other watch + scan the clause), so 2-WL's total memory
   traffic is **~2× MORE** than NobleCount's. NobleCount's sum-word makes each touch a *single*
   word op with no scan — and at `sat_count==1` the sum field **is** the forced literal (no list,
   no scan). The scan is exactly the cost 2-WL cannot avoid and NobleCount never pays.

2. **No backtracking removes 2-WL's other CPU advantage.** Counter-based BCP is penalised in CDCL
   by counter maintenance across *backtracks*. GPAP trajectories are forward-only (a conflicted
   row dies), so NobleCount never redoes work — deleting the scenario where 2-WL pulls ahead.

3. **~4.9× more per-row state → less occupancy.** NobleCount keeps one `(K,M)` word + a shared
   read-only occ CSR. 2-WL additionally needs clause→literal CSR (which NobleCount stores *none*
   of) plus **mutable per-row watch arrays** (`wpos (K,2M)`, `warr (K,T)`, `wcnt (K,2N)`). Fewer
   rows fit; K/occupancy shrinks.

4. **2-WL does not decompose into the WAVE kernel.** NobleCount's per-occurrence update is an
   independent `atomic_add` whose return value resolves the race — which is exactly what enables
   the warp-wide WAVE propagate (a further win at low K). 2-WL's per-visit *scan + watch-list
   mutation* is inherently sequential per clause with no such decomposition. So the real-world gap
   is **larger** than the scalar-vs-scalar 2× measured here: NobleCount also has a gear 2-WL can't.

5. **Parallel-unfriendliness shows up even at the compiler.** The mutable-watch-list machinery
   (swap-remove, migration, triple-nested data-dependent scalar loops) is fragile on the GPU. The
   2-WL kernel initially crashed intermittently — diagnosed to Triton defaulting to **4 warps**,
   running 128 threads redundantly that **raced on the row-private watch arrays** (torn reads →
   garbage node index → OOB). The fix is `num_warps=1`. NobleCount's uniform, branch-light,
   atomic-decomposable propagate has none of this surface. (A CPU mirror running the *exact* kernel
   trajectories proved the algorithm itself never overflows — the crash was purely the redundant-
   thread race, not a bug in the scheme.)

## Takeaway

At the same per-warp scalar parallelisation and on GPAP's narrow-clause, forward-only workload,
classic 2-WL is a **~2× pessimisation** of NobleCount on every axis that matters — total memory
work, per-row footprint, GPU-friendliness, and headroom for the WAVE speedup. This is the
empirical justification for building GPAP on the sum-trick counter rather than watched literals.
The CDCL intuition ("watched literals are always faster") is a *long-clause, backtracking-search*
result and does not transfer to this regime.

**Reproduce:** `scratchpad/wl_work.py` (CPU work proxy), `scratchpad/validate_wl.py --sweep`
(192/192 bit-identical gate), `scratchpad/time_wl.py` (wall-time A/B). Engine:
`gpap/engine/wl_engine.py` (experimental; scalar-only, `num_warps=1`).

# INSIGHTS — settled results index

One-paragraph summaries of every settled result in this project. The code that produced a negative
result may be deleted; the insight lives here. Entries are dated and newest-first within a section.
Do not re-litigate without new evidence.

Every file below opens with a status block: what it settles, when and on which card it was
measured, where the rule it justifies lives in the code, and what it is the only record of. Read
that block before quoting anything out of the file.

**Current evidence** — the rule is live and the numbers are the engine's:

| topic | file |
|---|---|
| binary clauses as edges (2026-08-27) | [BINARY_OCC_FINDINGS.md](BINARY_OCC_FINDINGS.md) |
| the WAVE shuffle lane schedule (2026-08-27) | [WAVE_SHUFFLE_FINDINGS.md](WAVE_SHUFFLE_FINDINGS.md) |
| CLEAN mode: design and lemmas | [CLEAN_MODE_FINDINGS.md](CLEAN_MODE_FINDINGS.md) |
| the C++ core, AOT kernel build | [CPP_BACKEND_FINDINGS.md](CPP_BACKEND_FINDINGS.md) |
| what a bigger GPU buys | [CROSS_GPU_FINDINGS.md](CROSS_GPU_FINDINGS.md) |
| entailed-clause mining and redundancy | [MINING_FINDINGS.md](MINING_FINDINGS.md) |
| uniformity: TV, Barbarik, the B knob | [UNIFORMITY_FINDINGS.md](UNIFORMITY_FINDINGS.md) |
| rival-comparison fairness, and protocol v1.0 | [RIVAL_FAIRNESS_FINDINGS.md](RIVAL_FAIRNESS_FINDINGS.md) |
| what `--out` cost, and what the console says (2026-08-27, 2026-09-12) | [CLI_DELIVERY_FINDINGS.md](CLI_DELIVERY_FINDINGS.md) |
| reproducing the campaign (2026-09-09) | [ARTIFACT_FINDINGS.md](ARTIFACT_FINDINGS.md) |

**Historical record** — kept so a settled question is not re-opened, and so a deleted subsystem
leaves a trace. **Quote no rate from any of these.**

| topic | file | why it is here |
|---|---|---|
| WAVE vs scalar by formula size (2026-08-23) | [KERNEL_AB_FINDINGS.md](KERNEL_AB_FINDINGS.md) | the `M ≥ 2048` rule is current; its numbers predate the shuffle schedule and must be re-measured |
| the 2026-08 L4 campaign | [L4_EVAL_FINDINGS.md](L4_EVAL_FINDINGS.md) | superseded by `l4_protocol_20260904`; its screens survive in `paper_tables.py` |
| the sampler's development log | [GPU_SAMPLING_FINDINGS.md](GPU_SAMPLING_FINDINGS.md) | pre-libgpap; the only record of seven retired configurations |
| GPU inprocessing for CDCL — retired | [ELS_INPROCESSING_FINDINGS.md](ELS_INPROCESSING_FINDINGS.md) | the only record of a deleted subsystem, and why it is not re-tried |
| porting to AMD / Intel / Apple | [PORTABILITY.md](PORTABILITY.md) | an analysis, not a measurement: nothing in it is implemented |

## Engine & memory model

**Three roots by lifetime, and one owner each, because a path rule that needs restating at every
site is not in the code (2026-09-13).** Every path bug here was one shape: a file computed a
managed directory itself and its copy of the rule drifted. Scratch had five conventions, the
results record four derivations (two of which named different directories on a symlinked
checkout, since an un-collapsed `..` resolves physically when OPENED and lexically when COMPARED),
and a Colab VM path was built from the LOCAL machine's `$GPAP_HOME`. Patching sites made it worse:
the explanation got copied to each one and three copies were wrong. Now `.gpap/` is disposable
machinery including ALL scratch (`gpap/paths.py`), `.experiments/` is a run's output and
`$GPAP_RESULTS` the record (`eval/paths.py`), nothing else names them, and shell ASKS
(`python3 -m gpap.paths home`) rather than reimplementing the rule in a second language — which is
what the divergences all were. Gated by tests/test_suite_hygiene.py — every string
constant through the AST (a path literal has no whitespace; prose about one does, which is what
separates them without a whitelist), every repo-root constant re-derived in all four spellings the
tree uses, every internal import resolved (a driver had imported a module from its pre-reorg path
for weeks inside an `except` that printed a one-line skip), and every shell ask guarded, because
asking a module for a path let an empty answer reach `rm -rf`. Behaviour by tests/test_paths.py.

**A phase belongs where it happens, and a printed number must mean one thing (2026-09-12).**
`gpap sample` read every line off the returned dict, so mining printed below the throughput it ran
before; `sample_clauses(progress=)` reports "formula" and "mined" as they happen and the CLI prints
from it (the library still never prints). Same pass: the active support is restated after mining,
which moves it both ways (the header's count is pre-mining, `--stats` is post-), `chunk_capped` no
longer marks a proved-UNSAT run as cut short, a success rate renders once
(`gpap/verdict.py::fmt_pct`) instead of once per printer, and run scratch moved to
`<repo>/.gpap/tmp` — which exposed that abc splits its script on whitespace, so any path with a
space in it wrote nothing. [CLI_DELIVERY_FINDINGS.md](CLI_DELIVERY_FINDINGS.md).

**The sampling chunk loop moved into the core (2026-09-01).** `cpp/src/sample.cpp`
(`gpap_sampler_*`) owns chunks, bandit, valves, buckets and accounting; both faces marshal over it.
Gated by tests/test_sample_core.py and tests/test_face_parity.py; pre-port seed streams re-baselined
([BENCHMARKING.md](../BENCHMARKING.md) §3).

**`{0}` on an options struct must equal every default; `GPAP_MODE_AUTO` is 0 (2026-08-31).** `mode`
defaulting to SCALAR cost raw-ABI callers 1.28–2.20×. `vscan` and `gpap_probe_options.same_var`
broke the same way, so a field is named for its opt-out: a rename is a compile error, a flipped
default is not.

**A soundness verdict needs evidence, and a self-test's wall needs a subtractor (2026-08-31).**
`sound_check` reports `"off"` unless `sound_rows_checked > 0`, and `verify_sound_s` is subtracted by
the pipeline, not only per config — a pinned config has no per-config wall.
[CLI_DELIVERY_FINDINGS.md](CLI_DELIVERY_FINDINGS.md).

**6 of 33 AOT cubins were byte-identical duplicates (2026-08-31).** `VSCAN` is read only under `if
WAVE and VSCAN and not SOFT`; trimmed in `build_kernels.vscan_is_read` and `engine.cpp::traj_key`.
Byte-identity is no gate (line info moves 24 of 33) — compare SHF_EXECINSTR sections. Gated by
tests/test_kernel_table.py.

**Key a cache on `gpap_build_id()`, never `gpap_version()` (2026-08-31).** The version literal
stayed "0.1.0" while the engine got 1.8× faster under it; the build id is an FNV-1a of the loaded
library's bytes, stamped into every row by `_core_provenance()`.

**WAVE's lane→segment schedule is warp-shuffle based: 1.80× geomean at the same K (2026-08-27).** A
`tl.gather` lower_bound replaced W×W one-hot reductions that lowered through shared memory. The
impl/s plateau moves to 1.05–2.41e9, superseding earlier wave rates; W=64 as ILP 0.918×.
[WAVE_SHUFFLE_FINDINGS.md](WAVE_SHUFFLE_FINDINGS.md).

**Binary clauses are implication edges; only len ≥ 3 carries per-row state (2026-08-27).** Same-K
kernels 1.42× geomean, e2e 1.36×, `gpap mine` on b12-c50 1.52×, K ×2.03–2.33, VRAM model 1.00×.
Mined width-2 clauses are K-free; earlier rates are not comparable.
[BINARY_OCC_FINDINGS.md](BINARY_OCC_FINDINGS.md).

**WAVE vs scalar on the L4, 141 cells (2026-08-23): 1.30× geomean** — 1.01× below 2k clauses, 1.33×
at 2k–20k, 1.48× at 20k–100k, 1.64× above 100k; `M >= 2048` picked the faster kernel on 77/93. The
crossover must be re-measured on the shuffle engine. [KERNEL_AB_FINDINGS.md](KERNEL_AB_FINDINGS.md).

**CLEAN mode: the probe launch restores its own rows and the per-chunk broadcast is gone
(2026-08-21).** b12-c50 20.9 → 4.7 ms per chunk (4.4×), b03-c25 sweep 2.60× e2e. Readouts of
post-apply state refuse after a clean launch; `GPAP_CLEAN=0` is exact legacy.
[CLEAN_MODE_FINDINGS.md](CLEAN_MODE_FINDINGS.md).

**A single 30 s run cannot say which build is faster — the bandit is a coin flip at that budget
(2026-08-18).** Mis-pick rate over 8 seeds: 2/8 and 1/8 at 30 s, 0/3 at 90 s. Pin `--portfolio`, or
compare `implications_per_s`.

**A default belongs at the ABI, not in the convenient wrapper (2026-08-19).** `Mode::Auto` lived in
the two wrappers, so every ABI consumer ran scalar: b03-c25 1.28×, 6s215rb0-c25 1.67×, b09-c25
1.76×, b12-c50 2.20× (1.65× e2e on `gpap mine`). `GPAP_MODE_AUTO` resolves inside
`gpap_engine_reset`.

**Turning that dispatch on exposed a kernel-dependent output, in the subsumption test
(2026-08-19).** Subsumption is tested against the baseline-reduced formula, not the input (b03-c25
mined 3,096 clauses under scalar vs 3,113 under wave); the `implications == 0` proxy caught 0 of 28
conflicts under wave. Gated by `test_mining_is_kernel_invariant`.

**The probe reset dominates probe wall and is a constant per chunk (2026-08-19).** The broadcast is
50–94% of a probe chunk and ~26 ms regardless of instance, while a probe modifies 0.02–8.4% of the
words. Shipped 2026-08-21 as CLEAN mode; the sampler stays on the broadcast.
[MINING_FINDINGS.md](MINING_FINDINGS.md).

**The wave/scalar dispatch keys off M, not the batch (2026-08-18).** `batch < 30× resident_warps`
was a proxy for M that the K collapse invalidated (0.727× on flipped rows). `gpap_dispatch_wave`
takes wave when `M >= 2048` or `batch <= resident_warps`: 1.044× overall, geomean 1.39× (4080) /
1.46× (L4).

**Wave and scalar differ on the implication counter because of conflicted rows, not a tally bug
(2026-08-18).** Over 5,000 assumption sets: 809 of 3,282 conflicted rows differ in `num_propagated`,
0 of 1,718 surviving rows differ in any field, `conflict` matches on all 5,000. Gate on models.

**The engine became libgpap, shared by Python and C++; performance is a tie (2026-07-28).** Kernels
unchanged, the AOT cubin bit-identical to the JIT one at SASS level: geomean 1.013× over
b03/b04/b06/b10/b12. Retracted en route: "owning the device memory is worth 1.2–1.35×" was a
placement effect. [CPP_BACKEND_FINDINGS.md](CPP_BACKEND_FINDINGS.md).

**Portability: AMD is a small port, Apple is a kernel reimplementation (2026-07-29).** Only 75 of
1273 lines of the core touch CUDA, the host surface is 18 driver calls, and all 29 kernel
specializations already cross-compile for gfx90a/gfx942/gfx1100. ROCm ≈ 1–2 weeks, Intel 2–4, Apple
4–8. [PORTABILITY.md](PORTABILITY.md).

**Runtime memory probes are unreliable; `expandable_segments` fixed K sizing (2026-06-30 → 07-01).**
Allocator fragmentation, not tensor size, forced the small K (6.5 GB reserved vs 2.5 GB live). Do
not resurrect the probe; that era's 4× floor was retired 2026-07-13.

**This WSL2 box spills to shared host RAM, it does not OOM (2026-05-31).** Oversizing K migrates
tensors over PCIe and throughput collapses silently — a dense run "succeeded" at 16.45 GB peak on a
12 GB card. Validate K changes under a kill-switch.

**Power-state gotcha (2026-07-07).** This laptop 4080 has shown a 40 W power cap under battery
profiles, uniformly halving all rates. Check `nvidia-smi -q -d POWER` and SM clocks before trusting
absolute numbers; A/B ratios within one state stay valid.

**The SM clock also moves within a run, so cross-run rates cannot resolve a few percent
(2026-07-29).** 615–1710 MHz while >90% busy on one box, one run; a 92-row paired re-run straddled
1.0 with no consistent sign on an identical engine path. Use interleaved same-session A/Bs.
[BENCHMARKING.md §1b](../BENCHMARKING.md).

**The PyTorch execution path was deleted; the engine is packed-kernel-only (2026-07-13).** CUDA and
triton are required, the A/B oracle is `tests/reference_bcp.py`, out-of-envelope formulas are
rejected, and `orac`/full-lookahead are gone. Validated by 11 model-set hashes reproduced
bit-identically at pinned K, seed 5.

**Final-engine suite re-runs (2026-07-14): 1.29× geomean e2e on Tseitin over the pre-bitmap engine;
ABC at rare targets is a per-instance lever.** ABC vs Tseitin is geomean 0.86× but bimodal —
1.8–5.7× on 7/11 non-vacuous, unlocking b14 (0 → 24 models/s) — against b13 1.000 → 0.000 success.

**Assignment bitmap: one `atomic_or` is dedup plus contradiction detection; +40–50% (2026-07-14).**
Bit l of (K, ceil(2N/32)) int32 words, l and l^1 sharing a word, 8× smaller (b14 670 MB → 84 MB).
Readout must not bit-unpack with strided slice-writes (16× sector-RMW). Its dispatch threshold is
superseded.

**Post-engine-work suite re-run (2026-07-14): e2e models/s geomean 1.07× non-vacuous, wins
concentrated.** b12 1.36×, b05-c25 1.74×, b11 0 → 8 models/s; the saturated scalar class reads ±7%.
On low-success instances report attempts/s and models/s separately.

**Warp-wide decision scan in the trajectory kernel: wave b12 1.30×, b04 1.07×, scalar b06 1.16×
(2026-07-14).** Picks are bit-identical by construction and the 11-hash gate passes in every mode.
WAVE-only today (`vscan=-1` disables); the scalar pick loop remains as the SOFT path.

**SUM-packed word: one `atomic_add` is the whole clause update (2026-07-13).** The high field holds
the sum of surviving literal indices mod 2^(32-SHIFT), so the update is `st -= 1 + (neg << SHIFT)`
and its return value detects unit/conflict race-free. It beat the atomic_cas loop everywhere: b14
1.21×.

**Clause renumbering for occ locality: negative — topological emission is already near-optimal
(2026-07-13).** The 2.0–2.1 sectors per ~2.8-clause occ walk are structural. Var-major first-touch
0%, barycentric 18–24% worse. Do not re-litigate without a layout that beats 2.0 sectors/walk
offline first.

**K-calibrator retune: the PyTorch-era floor bound K at ~1/3 of the packed engine's real fit
(2026-07-13).** K = min(analytic, cap): b04 4.4k→14.9k, b12 3.0k→10.4k, b14 280→958, kill-switch
validated. Shared-memory staging of the row state was measured out.

**Warp-wide WAVE propagate: 3.9× on K-starved big-M instances (2026-07-13).** Scalar is warp-uniform
and MLP-starved wherever VRAM caps K; wave pops ≤W trail literals at once. At equal K: b14 3.06×,
b12 1.31×, b03 0.91×. Its CAS loop, dedup and schedule are superseded above.

**Pure-PyTorch BCP is structurally latency-bound; the fused Triton kernel is the fix (2026-07-02).**
The round loop pays launch + sync + `torch.unique` per round (~3.3 ms at K=1024) and no tensor-land
rewrite removes it. One-program-per-trajectory kernels gave 3.2–5.3× e2e, 10–20× fused.

**Packed clause state, 16|16 (2026-07-03).** Fusing sat_count and the remaining-literal key into one
int32 halves the scattered-sector traffic the inner loop is bound on: geomean 1.27–1.31× e2e, no row
below 0.9×.

**WIDE 8|24 re-split (2026-07-08).** sat_count needs ≤8 bits and the key ceil(log2 2N), so the same
int32 covers N ≤ 8.4M — every instance that used to fall back to the unpacked kernel. Geomean 1.36×
on c50 unrolls, 16|16 controls unchanged. Duplicate-literal padding is provably unsound.

**MONO 1-byte state loses to WIDE — negative result, kernel deleted (2026-07-08).** The int8 state
buys 1.67× more trajectories per chunk, but recovering the unit literal by clause scan costs ~13%:
net 0.93× vs wide, losing 5/7. Re-derivable from git history.

**In-kernel retry-on-conflict measured out (2026-07-03).** Three retries tripled per-slot success
(b04 11 → 34%) and tripled wall with it: models/wall 0.81–0.97×. The loop also taxed the RETRIES=0
path ~10% — experimental paths get their own kernel.

**Engine bug class to watch: double-decrement on re-applied literals (2026-07-03).** Any entry point
that applies literals must skip those already assigned TRUE, or `occ[lit]` is decremented twice and
sat_count decays into garbage forced literals. Gated by a regression test.

## Sampler / portfolio

**The uniformity readout left the library; the claim lives in `eval/uniformity/` (2026-09-01, the
user's direction).** An honest uniformity number needs the whole solution set or conditional
sampling. The statistics are kept in `eval/uniformity/metrics.py`; `gpap/implications.py` went too.

**Barbarik, circuit-only with both controls (2026-08-28): GPAP as shipped is Accepted on 64/66
instances with zero Rejects — CMSGen 66/66, `soft` Rejected 64/65, 6/6 vs 6/6 at barbarik's own
stricter default.** Accept is asymmetric; the two parameter settings are never pooled. CM_test
rejects its own reference where `c ind` is not an independent support (spread 4–103.7×). Exact TV at
|S| 128–1920: `b16` 0.0398, `b8` 0.0571, CMSGen 0.1219, pipeline 0.1651 — a TV claim must name a
config, and pinning B=8 costs ~10%. [UNIFORMITY_FINDINGS.md](UNIFORMITY_FINDINGS.md) §4b–§4c.

**B (decisions per round) = 8 is the tensor-path throughput sweet spot; B ≥ 16 over-constrains and
collapses success.** B=1's success edge is free in-kernel, which is why B1-fused wins broadly. The
collapse was measured on the retired tensor path: in-kernel, yield is unchanged across B=1..2048.

**B is also an exactness knob: at B ≥ |support| the sampler is exactly uniform (2026-07-30).** One
round assigns the whole support a uniform polarity and conflicted trajectories are discarded, so
keeping survivors is rejection sampling — TV reaches the control floor (0.039 vs 0.040). Yield =
p_accept.

**The adaptive bandit is the pipeline's uniformity cost (2026-07-30, 600 instances, zero unsound).**
Restricted to [b8] the pipeline reads TV 0.0420 ≡ raw b8, so simplify + mining + projection cost
nothing; the bandit picked B=1 on 30/30, having no uniformity term. Do not delete B=1 — it is
Accepted.

**The |support| ≫ B regime, where real benchmarks live, gets better rather than worse
(2026-07-30).** Synthetic circuits at |support| 18–20 give GPAP TV median 0.0406 at every B ∈
{1,2,4,8} (floor 0.0399) against CMSGen 0.0766; B=1 at |sup|=13 reads 0.0394. Round count alone does
not drive skew.

**Uniformity vs CMSGen is regime-split — never quote the aggregate (2026-07-30).** Circuits (support
= primary inputs): TV 0.040 vs 0.087, GPAP at the exact-uniform floor. Dense random CNF with support
= every var: 0.441 vs 0.272, GPAP loses. Regime-limited to |S| ≤ 384 above.

**`soft` (risk-polarity) is Barbarik-rejected — 5/5 on the first fuzz set, 64/65 on the 2026-08-28
circuit re-derivation — so it left CIRCUIT_PORTFOLIO on 2026-07-30 and is opt-in via
`--allow-soft`.** Exact-layer TV median 0.694; the price is parked at geomean 0.410 over 19 seq
benches (b11 11,269 models → 0, b12 50,434 → 789).

**Config value is regime-dependent — keep a losing lever as a bandit option rather than discarding
it globally.** Soft lookahead drops success to 0% on the sk_ niche yet wins big on tight/deep
circuits: b03 T=25, 10% → 88%.

**Full-propagation lookahead raises success exactly as predicted but is net-negative on throughput**
(~3× propagations and K halved outweigh the feasibility gain), and the cheap 1-step
`_immediate_conflict` variant misses the deep cascades. Random polarity (CMSGen Alg. 1) is
near-optimal on the niche.

**`orac` removed from the circuit portfolio after 0/64 wins (2026-07-06).** Its documented niche
(ultra-shallow wide b14 at T=1) ended >3× behind boosted-K fused configs while its cold-start chunks
cost ~67 s across the suite. Re-audition candidate on A100-class large-K batching.

**Any deterministic decision heuristic in a K-parallel sampler needs forced stochasticity
(2026-07-04).** Pure risk-argmin polarity is row-deterministic when risks never tie, so all K
trajectories replicate one path (c1355: 70 distinct of 6,144). Fixed with ε-greedy on a hash bit,
ε=1/8.

**Bandit mechanics (2026-07-01 → 07-06).** Cold-start every config once before `n_target` can end
the run; report the best-measured config's steady-state rate, not the exploration blend; filter
candidates to ≥ best/20. UCB-style bonuses failed acceptance (0.52× on small budgets) — decaying
ε-greedy stays.

**Mining: three notions of redundancy, a clause does k jobs, and mined clauses are kept only if they
pay (2026-08-19 → 21).** On b03-c25 `subsumed` removes 1.4% of wide clauses, `derivable` 61%;
keeping derivable ones is +29% / +43% / +3% e2e on b03-c25 / s386-c25 / b09-c25, so `subsumed` is
the default. Keep gate: models per chunk = K × success; do not widen the pool past the support (582×
slower). [MINING_FINDINGS.md](MINING_FINDINGS.md).

**Entailed-clause mining is the lose-set killer.** L1 backbone units are the top lever (sk geomean
vs CMSGen 4.65× → 27.3×) and a 5 s budget captures the benefit. The stuck gate and the
`pairs="auto"` thresholds are gone (2026-08-20/21): mining runs above `MINE_MIN_CLAUSES` and is kept
only if it pays.

**ELS (equivalent-literal substitution) is a real sampling win: geomean 1.48× over plain simplify,
4.16× vs CMSGen, identical coverage, sound** iff reconstruction is class-consistent — a removed var
takes its rep in every model, and don't-care reps draw one bit per class.

**Unit-prop simplification is model-preserving and is what wins** (unlike the VE/BCE that CMSGen
disables): GPAP-plain is 0.76× CMSGen, GPAP-simplified 3.4×+.

## Circuit encoding (the ABC pass)

**The ABC encoder is a clean ~5.9× e2e sampling win — and it helps CMSGen almost as much
(2026-07-10).** `strash; zero; frames; dc2/fraig` plus Tseitin-of-the-AIG shrinks unrolled CNFs 5.1×
geomean, but CMSGen gains 2.96× so the ratio widens only 1.12×. Route is `write_aiger -s`; `zero` is
mandatory.

**Encode cost lives in the post-unroll fraig; the `fast` preset moves optimization pre-unroll
(2026-07-10).** `full` cost ~99 s on b15-c25; `fast` is ≤4 s but forgoes the cross-frame collapse
(b15 1.18× vs 2.17×; b12's 12–14× needs a post sweep). `auto` runs full below 50k estimated unrolled
ANDs.

**The miner and ABC overlap, but the unit count was the wrong proxy — overturned 2026-08-19.**
Mining is worth more on ABC CNFs than Tseitin ones (geomean 3.28× vs 2.05×): which units are found
matters more than how many. The provenance gate and the inline `MINE_MIN_CLAUSES=3000` are deleted.

**Uniformity measurement has exactly three tiers: exact where enumerable, distribution-free skew
detection elsewhere, Barbarik for certificates (2026-07-11).** The shipped readout was removed
2026-09-01; tiers 1 and 2 live in `eval/uniformity/metrics.py`. `l2_ratio = p̂₂·d > ~1.5` certifies
gross skew only (b06-c4's TV=0.17 reads 1.14).

**Uniform-like means instance-dependent skew (2026-07-11).** On most instances GPAP is
indistinguishable from uniform (b03-c2, |S|=256, p=0.79); on b06-c4/seed5 it is decisively skewed
(TV≈0.17) — and CMSGen shows the same signature on the identical CNF (TV=0.139). A property of the
BCP-completion family, not a bug.

**Benchmark degeneracy: vacuous output pins (2026-07-10).** At T=25 the last-cycle outputs of b02,
b05, b07 and s526 are constant over all input sequences, so pinning them is a tautology and the
instance is a free-hypercube throughput test. b01/b09 are near-vacuous. Exclude them from headline
comparisons.

**The rejection baseline: at default targets simulate-and-filter beats every CNF sampler; rare
targets flip it (2026-07-11, gpap/circuits/simulate.py).** Random pins are size-biased (p_accept ≥
0.14 on 12/19) and CPU rejection wins ~17/19, so headline comparisons use `--target rare`, where
amplification is 500–500,000×. `--sim-check` is the encoder-independent soundness gate.

**GPU gradient-descent rivals, run on our hardware, collapse where sampling gets hard
(2026-07-12).** HTS: GPAP wins 26× to 380,000× on circuit unrolls; at rare targets HTS finds 0–17
solutions in total. DEMOTIC on b03-c25-rare reaches 43.5 u/s against GPAP's 523k. Honest losses:
75-10-1-q and prod-20.

**Recurrent-CircuitSAT-Sampler, the gradient family's sequential member (2026-07-13): competitive on
its own easy targets, 100–320,000× behind at rare ones.** On their rejection-easy pins (p_accept
0.03…1.0) they win s386 (0.1×), s832 and b10. At rare fixed-T targets GPAP is 102× (b04) to 320,629×
(s1196); b03 is an honest tie.

**New circuit families (2026-07-12): EPFL, HWMCC'17 and big ITC'99.** Of 231 HWMCC single-property
AIGs ≤30k ANDs only 7 ever fire under random stimuli at T=25; the rest are vacuous pins. EPFL-rare
splits into GPAP-only wins (arbiter, max, mem_ctrl, square, priority, sin), hard-for-both arithmetic
cones (multiplier, div, hyp, log2, bar), and rejection-easy leftovers.

**Standard public CNF suite vs CMSGen at equal 60 s budgets (2026-07-12): geomean 65.4× steady /
32.1× e2e over 49 instances, 68.2× / 22.1× over 28 sk_ instances.** The losses are structured —
hash, modexp and tire are CMSGen-only, the service family loses e2e. A reference point, not the
headline.

## Measurement methodology (the traps that actually bit)

**A reproduction on another card can only flatter us, in two directions, and both are now
counted (2026-09-09; [ARTIFACT_FINDINGS.md](ARTIFACT_FINDINGS.md)).** All 92 pinned
Recurrent-CircuitSAT rungs sit at the reference L4's memory ceiling, so a card under 24 GB walks
them down and hands the rival less parallelism than the paper did; and 63 of 141 GPAP rows are
VRAM-bound below the batch cap, so a larger card grows OUR K while the rival stays pinned.
`GPAP_VRAM_BUDGET_MIB` (clamped once, in `gpap_mem_available`) holds the engine to the recorded
card's budget, and `compare_campaign.py` counts every walked-down grant.

**A gate that knows one shape of a leak is a gate for that shape (2026-09-13).** The third
absolute path to ship had no `/home/` in it: an agent's scratch directory with the home path
flattened into its NAME, inside three Barbarik error rows, and the packager's inline `/home/` check
passed it. The shapes now live in one script, `eval/artifact/check_leaks.sh`, proven to refuse
each on its own (tests/test_artifact_gate.py); literal names come from a denylist outside the
repository, because the script ships in the tarball it checks; and the double-blind copy is a
transform over the packager's tarball, `$GPAP_RESULTS/artifact/anonymize.sh`, that must still pass
the gate and the host suite. Running `pytest` inside the packaged tree for the first time found
six failures no checkout could show: no `.git`, and a card with no built core.
docs/findings/ARTIFACT_FINDINGS.md §9.

**An absolute path in a generated file is a leak with a long half-life (2026-09-12).** Two
generators, asked to record their provenance, printed the argument they were handed:
`slide_tables.py` put `/home/<user>/…` into a committed slide table, where it sat for a month (the
decks have since moved to `$GPAP_RESULTS/paper/slides/`), and `check_emission_symmetry.py` put one
into `emission_audit.txt`, which
`make_artifact.sh` ships as `artifact-data/paper/emission_audit.txt` — permanent once the archive
has a DOI. A header is provenance for a reader who does not have our disk, so the path belongs in
`$GPAP_RESULTS/…` form: `eval/paths.py::display_path` is the one definition, and `make_artifact.sh`
refuses to package a staged tree containing one. The frozen row JSONs were already clean.

**A GATE THAT CANNOT FAIL IS NOT A GATE (2026-09-10;
[ARTIFACT_FINDINGS.md](ARTIFACT_FINDINGS.md) §8).** Reviewing the artifact layer turned up the
same shape four times: a check whose absence read as a pass. A deterministic field missing on one
side was "not a mismatch"; a soundness verdict of `off` satisfied the safety tier though `off`
also means never-reached; a completion marker printed after a driver that had died; a pinned
rival cell counted as agreeing with the row it was copied from. None of it changed a published
number, and all of it would have let a truncated reproduction certify itself. The rule: state
what the tool verified, in the number it prints, and make silence fail.

**A reproduction is asked for verdicts, not rates (2026-09-09).** Rates are a property of the
card, so `compare_campaign.py` asserts only what must survive the hardware — the same formula per
cell, soundness re-derived on the reviewer's machine, and the same per-cell winner — and prints
the rate ratio as the card's speed factor. Demonstrated cross-card on the smoke tier: verdicts
identical, GPAP x1.07, CMSGen x1.85 (the host CPU), the rival x0.64 exactly where its grant was
walked down.

**What the result rows do not time is a per-cell CONSTANT, not a fraction (2026-09-09).**
Start-up, the ABC pass (0.31 ms per gate, a minute on b19), the equivalence check and the
soundness replay are unrecorded, and on the cheapest cells they are four times the recorded
wall. `estimate_wall.py` models overhead as constant + per-gate + fraction, because a
percentage-only model underestimates precisely the tiers made of many cheap cells.

**Protocol v1.0 frozen (2026-09-06; [EVAL_RUNBOOK.md](../EVAL_RUNBOOK.md) §0,
RIVAL_FAIRNESS_FINDINGS.md §17).** Publish only quantities each tool defines for itself, every tool
at its shipped default; a review may fix a bug or prose, never a metric, configuration, screen,
budget or cell set. Table 2's e2e column is removed by rule.

**Table 1 is read off Table 2's rows — three tools, one constraint (2026-09-03,
RIVAL_FAIRNESS_FINDINGS.md §14).** Each rival runs on its own published set under its own pins and
our rare pin; on every such cell GPAP takes the `abc:fast` CNF of that circuit and CMSGen the same
file.

**A five-reviewer adversarial pass found four defects in the completed campaign, all moving numbers
in our favour (2026-09-04, §16).** An under-granted DEMOTIC batch (4.73× → 2.99×), a glob that
dropped `s208.1`, a cold-start cell, and the wrong substituted circuit (1,150× → 759×).

**The T=25 campaign is complete (2026-09-06, §16d); the v1.1 tables (2026-09-07, §17b) are the ones
to quote.** Rivals at their shipped configuration, 1,200 s init budget, one `abc` variant: vs
Recurrent 21.5× over 20 cells (10.6× without its top four, on an L4), 173.2× rare; vs CMSGen 78.0×
rare. T=50 appendix: 1,616.3× / 71.7× and 8.5× / 119.3×. On DEMOTIC-rare quote 7/7, not the ratio;
rival failures at 1,200 s are `b17`, `b18`, `b22`.

**The first protocol campaign, `l4_protocol_20260903`, is superseded — do not quote its numbers
(2026-09-03/04, §15).** The corrected campaign is `l4_protocol_20260904`. Two ledger bugs it exposed
are gated: screens before buckets in Table 2, and zero models is a zero, not "unmeasured".

**A rival was scored "incapable" on our stopwatch (2026-09-03, §11).** Recurrent's `start_point`
probe was capped at 60 s against the run's 300 s; given a real budget b14 runs. The ratio must come
from the granted batch — b14 84.7 u/s at 3,000 against 7.3 at 256, a ~40× cell, not ~470×.
Initialization gets one budget equal to the run's timeout.

**A missing instance is a bug, not a gap — three ways a sweep lost instances silently
(2026-09-03).** A coverage cap bounded rows where the cost is rows × nv × 72 B (61 GB on b14);
`fetch_benchmarks.sh` installed b17–b22 where no driver looks; a `prepare_variant` failure
`continue`d. A refusal must be a row with its reason. The rival arm now encodes `abc:fast`,
superseding pre-2026-09-03 rival rows.

**Seven fairness audits of the GPU head-to-heads, and all seven had been flattering us
(2026-08-24/26, [RIVAL_FAIRNESS_FINDINGS.md](RIVAL_FAIRNESS_FINDINGS.md)).** (1) Only Recurrent is
timed while delivering, 0.7–1.4%. (2) A rival's batch is its score: grant 0.80 of free VRAM; HTS
6,369× → 603×. (3) Preprocessing is a median 20.7% of our e2e wall — 165.0× steady vs 96.7× e2e. (4)
Our pinned `--num_steps 10` taxed Recurrent 8.50×, DEMOTIC 2.61×. (5) CMSGen's count was floored to
`--statsevery`, 81 of 357 rows a 0 scored as our win, fixed with `--maxtime`. (6) The encoder cell
was an `argmax(ratio)`; one row per (family, encoder) now. (7) `wins`/`inst` used a different row
set than the geomean: screen first, then bucket.

**A bigger GPU is worth ~1.8×, not the VRAM ratio, and the batch is not the lever (2026-08-03).**
Config-pinned over 8 instances: A100-40GB 1.822×, L4 0.838× vs the RTX 4080 laptop, tracking SM
count not bandwidth; 4× the batch buys +2.8%. Traps: a free bandit picks differently per machine.
Open: b12 loses 2× success on the A100 between K=20k and 28k.
[CROSS_GPU_FINDINGS.md](CROSS_GPU_FINDINGS.md).

**Same-seed repeats measure timing noise only, not draw noise.** Portfolio changes shift RNG
consumption, so per-instance rates jump ±20–40% with identical walls. Use a 5+-seed fixed-config
median or a single-config A/B where model counts match bit-identically (`gpap bench --seeds 5 6 7`).
Sub-second walls reach ±69%.

**Benchmark GPU work against the solver's own in-C operation** (CaDiCaL's `process()`/`probe`),
never a naive pysat re-implementation — the latter inflated a GPU "speedup" 50–100×. Validate solve
speedups on kissat.

**Emission is not free: every GPAP-vs-CMSGen ratio published before 2026-07-27 charged CMSGen for
writing its samples and GPAP for nothing.** On 27.sk_3_32, 50k samples: 0.42 s generating vs 3.6 s
with `--samplefile` (ostream formatting, scaling with |vars|). The patch adds `--nowrite` and
`--statsevery`; emission costs CMSGen 5.3× geomean.

**Full re-measured suite, emission-free (2026-07-28, RTX 4080, 357 rows / 13 stages / 0 errors).**
Excluding vacuous pins and zero-search cells: seq-c25 rare 36.5×, epfl rare 6.4×, sk 46.2×; its
CMSGen counts predate `--maxtime`. Rare targets cost ~4× of the ratio, and the encoder
recommendation is per family (hwmcc abc 4–8.5×, comb 1.8–2.0×, seq 1.1–2.4×, epfl use tseitin).

**Unique-solutions metric: the distinct count belongs off the timing path.** Folding reconstruction
and dedup into the timed run understates throughput up to 10×, so it stays `collect=False` and
`unique_sps_est = raw × distinct_frac`. distinct_frac ≈ 1.0 except where the space is tiny (b06-c5
has 64 distinct solutions).

**Comparison-to-papers grounding (2026-07-07).** Same circuits, same solution space over primary
inputs, same T=25. Deviations documented: we pin all final-cycle POs, pin-1 for ISCAS-85, no c17.
Local CMSGen on our CNFs is ~2–3× faster than the papers' numbers, so quote both ratios.

**`--n-time` is a delivery target, not a guarantee in either direction — quote the delivered
count.** A run falls short only on `--time-limit` or the zero-progress stop (both marked `<`) and
overshoots to the end of a chunk. Since 2026-08-27 `--out` holds `min(-n, --max-samples, delivered)`
rows.

## Retired application directions (the killer-app rule: many independent sets + shallow propagation + no CPU incremental shortcut)

**Preprocessing/inprocessing-for-solve is retired.** A CPU incremental prober is 7–15× faster on
failed-literal probing and ELS-for-solve; GPU vivification is ~14× slower even fully vectorized;
clause addition is sound but portfolio-only (~1.2–1.3×). GPU-ELS finds 2.81× more equivalences than
CaDiCaL decompose, but reach is uncorrelated with solve speedup.
[ELS_INPROCESSING_FINDINGS.md](ELS_INPROCESSING_FINDINGS.md).

**Where GPAP genuinely wins:** sampling (this repo's flagship), cube-and-conquer cubing, and
one-shot UNSAT proving on high-k random near-threshold. CUD@SAT-style watched-literals-on-GPU is
known-negligible; the novelty is K independent assumption sets + incremental occ-list/sum-trick
propagation + the fused per-trajectory kernel.

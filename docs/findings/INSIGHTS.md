# INSIGHTS — settled results index

One-paragraph summaries of every settled (measured, often multi-day) result in this project.
Code that produced a **negative** result may be deleted; the insight lives here. Deep write-ups:
[GPU_SAMPLING_FINDINGS.md](GPU_SAMPLING_FINDINGS.md), [ELS_INPROCESSING_FINDINGS.md](ELS_INPROCESSING_FINDINGS.md).
Do **not** re-litigate these without new evidence (a new GPU class or a new regime counts;
"maybe it'll work this time" does not).

## Engine & memory model

**K calibration = fixed 4×-floor multiple; runtime memory probes are unreliable (2026-06-30 →
07-01).** The sampler's "mid-run RAM growth" that forced a small K was allocator FRAGMENTATION,
not tensor size: peak reserved 6.5 GB vs 2.5 GB live (62% waste) on s1196, growing ≈K^2.3.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set **before** torch initializes CUDA) makes
reserved track allocated (2.3× less) and linear in K. With that, K = min(4 × old-safe-floor,
analytic, cap) is measured-safe. Every probe/differencing variant FAILED: under
expandable_segments the peak-reserved read at construction is ~1.6× below production
(allocation-history dependent) and the frontier peak varies with the RNG draw → probes
under-measure → over-sized K → spill. Do not resurrect the probe.

**This WSL2 box SPILLS to shared host RAM, it does not OOM (2026-05-31).** Oversizing K makes
tensors migrate over PCIe (~12 GB/s vs ~200 GB/s VRAM) and throughput collapses silently — a
dense run "succeeded" at 16.45 GB peak on the 12 GB card. OOM-probing for max K is therefore
meaningless here; validate any K-sizing change with a real run under an nvidia-smi kill-switch
(`used > ~11 GB → pkill`).

**Power-state gotcha (2026-07-07).** This laptop 4080 has shown a 40 W power cap (default 80 W,
max 105 W) under battery/whisper profiles, uniformly halving all rates. Check
`nvidia-smi -q -d POWER` and SM clocks under load (>2.5 GHz boost) before trusting absolute
numbers; A/B ratios within one state remain valid.

**Pure-PyTorch BCP is structurally latency-bound; the fused Triton kernel is the fix
(2026-07-02).** The Python round loop pays launch+sync+`torch.unique` per BCP round (measured
~3.3 ms/round at K=1024, nearly pure overhead; 12.9× more trajectories cost only 3.9× wall).
Dedup fundamentally needs a sort in tensor-land, `repeat_interleave` lacks `out=`, and the
5.6× reserved-over-allocated peak is inherent to the per-round temporaries — so Tier-1
"optimize the tensor loop" cannot help. The answer was one-program-per-trajectory Triton
kernels (sequential trail, no atomics), 3.2–5.3× e2e immediately, 10–20× with the fused
trajectory kernel; also fixes memory (reserved tracks allocated).

**Packed clause state (16|16 sat|XOR, 2026-07-03).** The kernel inner loop is bandwidth-bound on
scattered clause-state sectors; fusing sat_count + the remaining-literal key into ONE int32
(XOR replaces the sum — never overflows) halves the traffic: geomean 1.27–1.31× e2e, no row
<0.9×. The `st = (st - 1) ^ (neg << SHIFT)` update is safe because sat ≥ 1 at every decrement
(sat==0 halts the row first), so the subtraction never borrows into the XOR half.

**WIDE 8|24 re-split (2026-07-08).** sat_count needs only ≤8 bits (bounded by max clause LENGTH,
5–6 on Tseitin unrolls); the XOR needs ceil(log2(2N)). Re-splitting the same int32 to 8|24
covers N ≤ 8.4M — every big instance that silently fell back to the unpacked kernel (e.g.
b12-c25, 2N = 67,920 = 3.6% over the 16-bit cap). Measured on c50 unrolls: geomean 1.36× over
auto (b04 1.52×, b05/b11 1.38×), sols/s ratio == implications/s ratio (pure BCP-traffic win);
16|16-eligible controls unchanged (0.99×). No dummy-literal padding is ever needed (the count
half carries per-clause init); duplicate-literal padding is provably UNSOUND (unit detection
depends on which literal remains).

**MONO 1-byte state loses to WIDE — negative result, kernel deleted (2026-07-08).** Pushing the
delta idea to its floor — (K,M) int8 `st = 127 - sat_count`, no index bits, unit literal
recovered by scanning the clause's literals via a cls→lits CSR — DOES buy more K_fused (1.67×
more trajectories/chunk on s1196-c50, as the 4×-smaller-state theory predicts) but the
clause-scan-at-unit-time reads a separate scattered array and per-trajectory cost rises ~13%:
net **0.93× vs wide** (loses 5/7, ties the 2 controls). Lesson: K_fused parallelism gains must
be weighed against added per-event compute; the O(1)-XOR-lookup wide kernel is the sweet spot.
(Was `TritonBCPMono` + `_bcp_trajectory_mono8_kernel` + tests/test_mono_kernel.py; re-derivable
from git history if a future GPU's bandwidth/compute balance shifts.)

**In-kernel retry-on-conflict measured out (2026-07-03).** 3 retries tripled per-slot success
exactly as predicted (b04 11→34%) but wall tripled with it — conflicts on success-bound
residuals die LATE, so a failed attempt costs a full trajectory and models/wall stayed
0.81–0.97×. Retry ≡ more chunks minus overhead; the residuals need LEARNING, not restarts.
Structural lesson: wrapping an experimental loop around the accepted hot kernel taxed the
RETRIES=0 path ~10% (register pressure) — experimental paths get their OWN kernel.

**Engine bug class to watch: double-decrement on re-applied literals (2026-07-03).**
`apply_assumptions_bulk` must skip literals ALREADY assigned TRUE (e.g. flush-implied):
`_bulk_propagate_lits` decrements occ[lit] unconditionally, so re-applying corrupts sat_count →
garbage forced literals → false conflicts. The Triton kernels always had this skip-dup guard;
the bulk path got it 2026-07-03 (+ regression test). Any new engine entry point needs the guard.

## Sampler / portfolio

**B (decisions per round) = 8 is the tensor-path throughput sweet spot; B≥16 over-constrains
and collapses success.** B=1's success edge is FREE in-kernel (fused configs), which is why
B1-fused wins broadly.

**Config value is REGIME-DEPENDENT — keep losing levers as bandit options, don't globally
discard.** Soft (risk-polarity) lookahead BACKFIRES on the sk_ niche (sat_count is a
non-falsified count, not a quality signal; a clause going 2→1 becomes a unit i.e. PROPAGATES —
the win mechanism; penalizing it drops success to 0%) yet is a BIG WIN on tight/deep circuits
(b03 T=25: 10%→88% success). The adaptive bandit gates it per instance.

**Full-propagation lookahead raises success exactly as predicted but is net-negative on
throughput** (~3× propagations + K halved > feasibility gain); the cheap 1-step
`_immediate_conflict` variant misses the deep cascades that matter. No instantaneous counter
encodes the propagation closure. Random polarity (CMSGen Alg. 1) is near-optimal on the niche.

**`orac` (PyTorch-engine config in the circuit portfolio) removed after 0/64 wins
(2026-07-06).** Its documented niche (ultra-shallow wide b14 T=1) ended >3× behind boosted-K
fused configs, while its cold-start chunks cost ~67 s across the suite. The `"orac"` lookahead
mode stays supported; re-audition candidate on A100-class large-K PyTorch batching.

**Any deterministic decision heuristic in a K-parallel sampler needs forced stochasticity
(2026-07-04).** Pure risk-argmin polarity (fsoft) is row-deterministic when risks never tie →
all K trajectories replicate ONE path (c1355: 70 distinct of 6,144; invisible to the
raw-throughput bandit). Fix: ε-greedy (hash bit, ε=1/8). The bandit's models/wall objective is
blind to diversity — a future hardening is scaling reward by a sampled distinct fraction.

**Bandit mechanics (2026-07-01 → 07-06).** Cold-start every config once BEFORE n_target can end
the run (else a big-K GPU measures only plan[0]); report the best-measured config's steady-state
rate (`best_rate_sps`), not the exploration blend; exploration candidates are
plausibility-filtered to ≥ best/20 (run noise is ~1.9×, 20× behind is dead, not noisy);
UCB-style exploration bonuses FAILED acceptance (diverting budget from a dominant winner costs
0.52× on small budgets) — plain decaying-ε-greedy stays.

**Entailed-clause mining (miner.py) is the lose-set killer.** L1 backbone units are the #1 lever
(sk geomean vs CMSGen 4.65×→27.3×; massive-backbone instances collapse to N~100 formulas);
5 s budget captures the full benefit; the pre-mining "stuck" gate is a FALLACY (pre-mining
success is low almost everywhere BECAUSE mining lifts it — gate must be POST-L1, two-stage);
tier COST is predictable from |sup| (pairs="auto" thresholds), tier VALUE is per-instance.
After any big GPU phase, `torch.cuda.empty_cache()` before constructing a sampler — reserved
-not-freed VRAM poisons the K calibration (measured: K collapsed to 1).

**ELS (equivalent-literal substitution) is a real sampling win: geomean 1.48× over plain
simplify, 4.16× vs CMSGen, identical coverage, sound** iff reconstruction is class-CONSISTENT
(a removed var = its rep in EVERY model; don't-care reps draw ONE bit per class — independent
randomization would emit invalid projections).

**Unit-prop simplification is model-preserving and is what wins** (unlike the VE/BCE CMSGen
disables). GPAP-plain = 0.76× CMSGen; GPAP-simplified = 3.4×+.

## Circuit encoding (the ABC pass)

**The ABC encoder is a clean ~5.9× e2e sampling win — and it helps CMSGen almost as much
(2026-07-10).** `strash; zero; frames; dc2/fraig` + Tseitin-of-the-AIG shrinks unrolled-circuit
CNFs 5.1× geomean (1.5× s1196 … 33× b13) over our Tseitin+unit_closure; with mining handled
(below) GPAP e2e improves 5.89× geomean with zero regressions, all sound. But the shrink is
encoder-side, not sampler-side: CMSGen gains 2.96× on the same CNFs, so the GPAP-vs-CMSGen ratio
only widens 1.12×. Route = `write_aiger -s` (positional frame-major CI/CO map, name-verified),
NOT `write_cnf` (~1.5× tighter, no input map). `zero` is mandatory (else the T=0 state becomes
free PIs and the solution space changes). Soundness is gated by projected-solution-SET equality
tests, both presets.

**Encode cost lives in the post-unroll fraig; the `fast` preset moves optimization pre-unroll
(2026-07-10).** `full` (post-unroll `dc2;dc2;fraig`) cost ~99 s on b15-c25 — the fraig/`&fraig`
sweep of a 160k-AND unroll is unbounded in practice (`&fraig` >300 s there). Optimizing the
sequential AIG FIRST (T× smaller) + one post `dc2` is ≤4 s everywhere measured and matches full
on b14 (1.47× vs 1.46×) but forgoes the cross-frame equivalence collapse (b15 1.18× vs 2.17×;
b12's 12–14× NEEDS a post sweep). `auto` resolves the tension: cross-frame-collapse instances
are small (b12 est 26k unrolled ANDs), so full-below-50k / fast-above keeps every measured win.
b15 e2e (encode+sample, n=100): 1m47s → 9.6s.

**The miner and ABC do overlapping work — gate mining on encoder provenance (2026-07-10).**
dc2/fraig structurally removes exactly the backbone the miner's L1 sweep harvests: 59–1007 units
learned on Tseitin CNFs vs 0–45 on the same circuits' ABC CNFs, while still paying the full 5 s
budget (b15: 4.8 s of a 9.4 s e2e wall for 45 units on 222k clauses; b10/b04 regressed 0.40×/
0.86× e2e because the post-ABC success dip also tripped the stuck gate into stage-2 tiers). The
CNF carries `c encoder abc` and `sample_file` defaults mining OFF there (`--mine on` forces).
The miner stays THE lever for Tseitin circuit CNFs and plain projected CNFs (service family).

**Uniformity measurement: exact where enumerable, distribution-free skew detection elsewhere,
Barbarik for certificates (2026-07-11).** The readout (`info["uniformity"]`, off the timing
path) has three tiers, because they are the only three that exist — and it is OPT-IN
(`--uniformity off` is the default: even off the timing path the readout costs an extra
sampling pass, ~seconds of wall on slow instances). `fast` = tier 2 (repeat-rate only, no
solver work — the cheap explainable number); `exact` = tier 1; the sensitive machinery is a
flag away. When `exact` is requested but enumeration is defeated (cap/timeout, or M > 200k
skips it), the line falls back to the repeat-rate readout with an `enum_status`-aware hint —
never re-suggesting `exact` to someone already running it. (1) exact: if the projected space enumerates within budget
(≤4096 sols / 2 s / M ≤ 200k), a REAL goodness-of-fit test — chi2 p (valid when n ≥ 5|S|; the
sf is a native incomplete-gamma, no scipy) + TV + an observed⊆enumerated soundness cross-check;
measured live: b03-c2, |S|=256, chi2 p=0.79 at 27 ms.
(2) fast/unknown-|S|: NO black-box statistic can certify uniformity (that is Barbarik's raison
d'être), but skew IS detectable distribution-free: p̂₂ = Σc(c-1)/(n(n-1)) estimates ||p||₂²
unbiasedly and every uniform distribution has ||p||₂² = 1/|S| ≤ 1/d, so l2_ratio = p̂₂·d > ~1.5
certifies non-uniformity for ANY |S|; collision-free runs report the 95% birthday floor on |S|.
KNOWN sensitivity gap (by design): b06-c4's TV=0.17 bias reads l2_ratio 1.14 → fast says OK,
exact says p=0.000 — gross-skew detector vs goodness-of-fit test.
(3) certificates on large spaces = conditional sampling = Barbarik (AAAI'19): eval/barbarik_gpap.py
monkeypatches meelgroup/barbarik's SolutionRetriever to call GPAP IN-PROCESS (one torch init, not
a subprocess per call; SPUR stays the trusted reference). Notes: their repo needs
`git submodule update --init` (interfaces/convert); barbarik.py's own SAMPLER_* table is stale —
drive `interfaces.cnf.cnf_test` directly with `SAMPLER_CUSTOM=7`. FIRST RESULTS: **Barbarik
Accept:1 on BOTH b03-c2 (70,471 samples) and their canonical blasted_case110 (70,596 samples)
at eta=0.9 eps=0.1 delta=0.05, GPAP B=8** — 2/2 certified.

**Uniform-LIKE means instance-dependent skew — and the readout sees it (2026-07-11).** On most
tested instances GPAP's distribution is indistinguishable from uniform (b03-c2 |S|=256 p=0.79;
b06-c4/seed1 |S|=64 p=0.47-0.83). On b06-c4/seed5 (|S|=96) it is decisively skewed: TV≈0.17,
p=0.000, replicated across sampler seeds — and **CMSGen shows the SAME signature on the identical
CNF (TV=0.139, p=0.000)**. The skew is a property of the BCP-completion family (solution weight ∝
attraction basin), not a GPAP bug, and it is target/instance-dependent (same circuit, different
pinned output pattern flips it). This is precisely the CMSGen "uniform-like" trade-off from
FMCAD'21; comparisons vs CMSGen at equal uniformity remain apples-to-apples, and the opt-in
readout now quantifies it per run instead of assuming it.

**Benchmark degeneracy: vacuous output pins (2026-07-10).** At T=25 the last-cycle outputs of
b02, b05, b07, s526 are CONSTANT over all input sequences ⇒ pinning them is a tautology and the
instance is a free-hypercube throughput test, not constrained sampling (ABC collapses the whole
cone — its huge "shrink" there is degeneracy, not merit). b01/b09 are near-vacuous (2 patterns).
Exclude from headline comparisons. (b12 is NOT vacuous: 5 patterns at 200 trials — small probes
undersample.)

**THE REJECTION BASELINE: at default targets, simulate-and-filter beats every CNF sampler —
rare targets flip it (2026-07-11, gpap/circuits/simulate.py, .experiments/simbase_*.json).**
The vacuous-pin finding was the tip of a general phenomenon, now measured (`--sim-baseline`:
random inputs + bit-parallel numpy simulation, 64 patterns/word, CPU single-thread; PERFECTLY
uniform by construction; rate = p_accept × sim-throughput).
- **Default (`--target random`) pins are size-biased**: the target is the image of one random
  input draw, i.e. sampled from the circuit's own output distribution ⇒ typically a
  HIGH-probability pattern. Measured over ITC'99+ISCAS c25 seed5: p_accept ≥ 0.14 on 12/19
  instances (b12 = 0.93 — "effectively vacuous" despite 5 observed patterns), and CPU rejection
  beats GPAP-on-4080 on ~17/19 instances, often 10–100× (b03: 9.0M/s vs 0.6M/s). GPAP
  amplification (success/p_accept) is only 1–16×. This equally undermines the rival GPU
  samplers' published setups (DEMOTIC-style reachable targets, pin-1 combinational ⇒
  p_accept ≥ ~0.5 by construction).
- **`--target rare`** (pin the rarest pattern in a 262k-sequence sim batch; observed ⇒ reachable
  ⇒ SAT; deterministic per seed) splits the suite into the real regimes:
  * **BCP-earns-its-keep** (amplification 500–500,000×): s1196 (p=1.6e-6, rejection 8/s vs GPAP
    38k/s), b13 (GPAP success ROSE 35%→100%, 4.5M/s vs 123/s), s386 (161k/s vs 1.1k/s), b12
    (17k/s vs 141/s), b04 (32k/s vs 5.4k/s — GPAP also got FASTER: the rare pattern's backbone
    is propagation food). b14 marginal (14 sols vs ~1/s).
  * **rejection-still-wins**: the degenerate-output family (b01/b02/b05/b06/b07/b09/s526 — rare
    pattern is still common) and the GPAP lose-set where the rare target collapses success
    (b10 15%→1%, s832 89%→0%, b11 7%→0% — trajectory conflicts, not propagation).
- **Predictor**: GPAP wins where propagation retains success at rare targets (backbone-rich
  cones); loses where success collapses with p_accept. Amplification is the per-instance
  diagnostic — it is now a bench column (`p_acc`/`rej/s`/`amp`).
- **Consequences**: (1) headline circuit-sampling comparisons MUST use `--target rare` (or
  report p_accept) — random-target wins are rejection-dominated for every CNF sampler, ours and
  CMSGen and the GPU rivals alike; (2) the rejection baseline reported is CPU numpy — a GPU
  simulator would shift it another 10–100× in rejection's favor, so the honest GPAP niche is
  strictly the low-p_accept regime; (3) `--sim-check` (simulate emitted samples through the
  ORIGINAL circuit) is now the encoder-independent soundness gate — 2,000/2,000 on b03-c25
  through both encoders.

**GPU gradient-descent rivals, run on OUR hardware: they collapse exactly where sampling gets
hard (2026-07-12, .experiments/rivals_hts_full.json, eval/rivals_hts.py +
eval/unroll_to_bench.py).** Same RTX 4080, same instance files, their own metric (unique
solutions / their model_runtime — which EXCLUDES their CPU model-build; GPAP's number includes
construction).
- **HTS** (arXiv 2502.08673, the newest of the Berkeley family, PyTorch): on circuit unrolls
  GPAP wins 26× (their own s15850a_3_2: 63k vs 2.5k u/s) to 380,000× (b13-c25 random); at RARE
  targets HTS finds 0–17 solutions total (b12/b13-rare: 0) while GPAP delivers 17k–4.5M u/s.
- **DEMOTIC** (ASP-DAC'25, works on .bench directly — no CNF): fed OUR exact task via an
  unrolled single-output miter (eval/unroll_to_bench.py, self-checked p(MITER=1) == the
  instance's p_accept): b03-c25-rare best 43.5 u/s vs GPAP 523k (≈12,000×) — and BELOW the CPU
  rejection baseline (620k/s). b13-c25-rare (p=3e-5): 0 solutions in 111 s.
- **Honest losses**: 75-10-1-q and prod-20 — the known jointly-constrained CNF lose-family —
  GPAP 0% success while HTS produces (6.4k u/s on 75-10); gradient methods survive where
  BCP-completion dies. And on tiny-|S| or-* instances the u/s metric is a SATURATION artifact
  (both samplers enumerate the space; the number measures time-to-saturation, not throughput).
- **Their published setups are rejection-easy by construction**: DEMOTIC's targets are a
  hard-coded per-name table pinning only the FIRST 1–5 outputs; single-output pins have
  p_accept ≥ 1/3 unless the bit is skewed (EPFL pin-1 measured: p ≥ 0.33 everywhere, dec.aig
  the lone 3.9e-3). The rare-target protocol is where CNF/BCP sampling separates from both
  gradient descent AND rejection.
- Not run: DiffSampler (JAX; older sibling of HTS — same group, HTS is their successor),
  QuickSampler (needs z3 build; CPU tool). Portability quirks that bit are documented in the
  two eval/ harness docstrings.

**New circuit families (2026-07-12): EPFL + HWMCC'17 + big ITC'99
(benchmarks/circuits/README.md for provenance/licenses; scripts/fetch_benchmarks.sh for full
sets).** HWMCC'17: of the 231 single-property AIGs ≤30k ANDs, only 7 have a property output
that ever fires under random stimuli at T=25 (bit-parallel sweep) — those 7, pinned at
`--target rare`, are genuine witness-sampling instances (p_accept 3e-4…6e-2); every other
instance is a vacuous pin. EPFL (combinational, T=1): pin-1 — the DEMOTIC-style setup — is
rejection-trivial; all-pins + rare is properly constrained (p 1e-3…<4e-7).
MEASURED (.experiments/{hwmcc,epfl}_rare.json, 4080, n=20k/30s):
- EPFL-rare splits three ways: (a) GPAP-wins-outright — rejection sees 0 hits in ~10M patterns
  while GPAP keeps high success: arbiter 87k/s@91%, max 458k/s, mem_ctrl 148k/s@89%, square
  63.8M/s@100%, priority (measured amp 17,151×), sin (amp 8,848×), int2float 681×, plus slow-
  but-alone adder 217/s and sqrt 8.2k/s; (b) HARD-FOR-BOTH — the classic SAT-hard arithmetic
  cones (multiplier, div, hyp, log2, bar): GPAP 0 sols AND rejection ~0 — the open-challenge
  set; (c) rejection-easy leftovers (router p=0.07, voter p=0.5 — their output bits are skewed,
  so even the rare side is common).
- HWMCC-rare is humbler: a SINGLE property bit can only be as rare as its own skew, so 5/7 sit
  at p≈3e-2…7e-2 where rejection wins (texasifetch 122k/s vs GPAP 51k/s); the two truly-rare
  ones (6s215rb0 p=3e-4, amba p=2e-4) are ALSO the 0.7–1M-clause giants where GPAP's rate is
  K-limited — 6s215rb0 ends dead even (64/s vs 58/s), amba both ~0. Witness-sampling on RTL
  needs either multi-bit pins or the big-GPU (A100) rerun.

**Standard public CNF-sampling suite vs CMSGen, equal 60s budgets (2026-07-12,
.experiments/{cnf_suite,sk}_vs_cmsgen.json).** The FMCAD'21/DiffSampler-era corpus (blasted /
or / tire / modexp / hash / 75-10 / 90-10 `.no_w` files, zenodo-4022395 lineage; + the sk_
Sketch/service family), n=10k, SAME per-instance wall cap both sides (cmsgen_sample now takes
`timeout=` = the bench --time-limit; partial samples streamed before the kill still count).
- **Headline: geomean 65.4× steady / 32.1× e2e over 49 both-produced instances (standard
  suite), 68.2× / 22.1× over 28 (sk_ suite)** — GPU 4080 vs single-core CPU.
- Family texture: or-* 60–380× (GPAP up to 40M samples/s), blasted 66–236×, 90-10 15–55×,
  Sketch-numeric 100–3,464×.
- **Honest losses, all structured**: hash + modexp + tire are CMSGen-only families (8
  instances; CMSGen itself only trickles ~1/s on modexp) — plus 7 "neither" (hash-4/6/8 and 4
  of the 75/90-q instances defeat both in 60s); service-family e2e losses persist
  (NotificationServiceImpl 0.27×, PhaseService 0.61×, ActivityService2 0.77×) — the known
  jointly-constrained regime. GPAP-only: 0.
- Within-family variance is real: modexp8-4-6 is the lone modexp GPAP cracks (mining + 60s
  grind ⇒ 121×) while its 6 siblings give 0 — don't cherry-pick either direction.
- Position for the paper: the CNF suite is a REFERENCE point (GPAP = GPU-parallel CMSGen, so
  the ~65× geomean is the parallelism dividend at equal quality), not the headline — circuits
  with rare targets remain the flagship domain.

## Measurement methodology (the traps that actually bit)

**Same-seed repeats measure TIMING noise only, not DRAW noise.** Portfolio-membership changes
shift RNG consumption → every chunk's trajectories change → per-instance rates jump ±20–40%
both directions with identical walls. On low-success instances, 8-chunk portfolio numbers have
±40% noise; a "consistent 0.81×" can be pure draw luck. Arbiter = multi-seed (5+) fixed-config
shootout comparing MEDIANS, or deterministic same-seed single-config A/B where model COUNTS
must match bit-identically; mine-first if the compared row mined (like-for-like formulas only).

**Benchmark GPU work against the solver's OWN in-C operation** (e.g. CaDiCaL's process()/probe),
never a naive pysat re-implementation — the latter inflated a GPU "speedup" 50–100×. Validate
solve speedups on kissat (SOTA), not just cadical.

**Unique-solutions metric: distinct COUNT off the timing path.** Competitors (DEMOTIC/HTS/
recurrent) report unique-solutions/s. Folding reconstruction+dedup into the timed run
understates throughput up to 10× (Python delivery, not sampling, is the bottleneck), so the
timed run stays collect=False and `distinct_projections` is computed on the separate coverage
sample; `unique_sps_est = raw × distinct_frac`. distinct_frac ≈ 1.0 on all non-tiny spaces
(measured 100.0% on the 18 paper-table instances); the count matters on tiny spaces (b06-c5:
only 64 distinct solutions EXIST — raw rate overstates distinct delivery ~2500×).

**Comparison-to-papers grounding (2026-07-07).** Same circuits (PI/PO/gate counts match),
same solution space (unique solutions live over primary inputs; `c ind` == their input space),
same T=25 protocol. Deviations documented: we pin ALL final-cycle POs (sequential; ≥ as
constrained ⇒ conservative), pin-1 for ISCAS-85, no c17. Their tables' M/N≈0.97 is their own
counting (no standard Tseitin gives clauses<vars); ours is standard (M/N≈2.4–3.1). **Local
CMSGen on OUR CNFs is ~2–3× faster than the papers' published CMSGen numbers** (DFF-var-sharing
gives ~3× fewer vars), so like-for-like write-ups must quote BOTH paper-CMSGen (with footnote)
and local-CMSGen ratios. Recurrent's "this work" row is variable-length 1–50 sampling on 8×V100;
DEMOTIC is single V100.

**`--n-time` is a delivery CONTRACT** ("when I say 200000 I want 200000"): the run samples until
n_time models are delivered; the only outs are `--time-limit` and the zero-progress stop, both
marked `<`. Delivered-blend rate < winner steady-state rate is exploration drag, not a bug.

## Retired application directions (all three axes of the killer-app rule must be green:
many independent sets + SHALLOW propagation + no CPU incremental shortcut)

**Preprocessing/inprocessing-for-solve is retired.** Failed-literal probing and ELS-for-solve:
a CPU incremental prober shares the base state and is 7–15× faster. Vivification: depth-limited
throughput + CPU vivify is incremental & budgeted; GPU ~14× slower even fully vectorized; the
speedup (when any) was CaDiCaL-only and died on kissat. Clause ADDITION (discovered binaries):
sound but portfolio-only (~1.2–1.3×), trajectory chaos, no robust single-config win. GPU-ELS
finds 2.81× more equivalences than CaDiCaL decompose (full BCP closure vs binary graph) but
reach is UNCORRELATED with solve speedup. See ELS_INPROCESSING_FINDINGS.md.

**Where GPAP genuinely wins:** SAMPLING (this repo's flagship), cube-and-conquer cubing
(CUBE_AND_CONQUER_FINDINGS in the pre-refactor tree / memory), and one-shot UNSAT proving on
high-k random near-threshold (memory). CUD@SAT-style
watched-literals-on-GPU is known-negligible; GPAP's novelty = K independent assumption sets +
incremental occ-list/sum-trick propagation + the fused per-trajectory kernel.

# Running the evaluation on a Colab L4 — results, and what the platform does to you

> **Reader:** contributors and researchers — a historical record of running the evaluation on a Colab L4. A dated research log, not a user guide.
>
> **Status:** SUPERSEDED as a campaign — it predates protocol v1.0
> ([../papers/tacas-27/PROTOCOL.md](../papers/tacas-27/PROTOCOL.md), frozen 2026-09-06). The paper's campaign is described in
> [../papers/tacas-27/README.md](../papers/tacas-27/README.md). **Quote no number from this file.**
> **Measured:** 2026-08-03/04, Colab NVIDIA L4 (24 GB, 58 SM, sm_89).
> **Rule lives in:** the screens and the zero ledger are `eval/artifact/paper_tables.py`
> (`zero_search`, `tiny_space`, `VACUOUS_PINS`), gated by `tests/experiments/test_paper_tables.py`.
> **Record only:** the price of the uniformity-safe default (§4).
>
> Three inputs have moved: the rival rows predate the parallelism-grant and step-count rules
> ([RIVAL_FAIRNESS_FINDINGS.md](RIVAL_FAIRNESS_FINDINGS.md) §2, §6); the CMSGen counts predate
> `--maxtime` (§7 there), so a "CMSGen=0" cell here is a floored count; and the engine has since
> gained the binary-edge representation and the shuffle WAVE schedule, so every absolute rate is
> old. Kept for what holds: the degeneracy screens and ledger rules (§2), hardware-independent
> rival win counts (§1), the propagation identity (§3), the price of the uniformity-safe default
> (§4), reproducibility (§5). The Colab platform rules that were §6 moved to
> `eval/colab/README.md`, beside the drivers they describe.

Across everything here — 357 benchmark cells and ~2,800 uniformity runs — there was **not one
unsound emission**. The audit also found four errors in what we had been quoting: vacuous pins
inside the sequential headline, a one-sided zero ledger, three instances that "sample" without
search (§2), and a propagation table whose columns were filtered independently (§3). All four are
now enforced in code rather than remembered.

## 1. The rivals: the ranking transfers, the magnitude does not

**Win counts are hardware-independent** — identical on the L4 and the 4080, instance by instance,
because both sides share the GPU. That is the claim worth making; the geomeans are not. At rare
pins the rival often finds ≈0 solutions, so one or two move the ratio by orders of magnitude, and
the rival side is one run where GPAP is a 3-seed median; DEMOTIC's batch sizes were identical on
both cards, and its speedups appear only where it finds almost nothing. (An earlier draft claimed
the margin *widens* on weaker hardware from a single `c432` row; on that instance DEMOTIC reads
469k u/s on the `raw` variant and 1,366k on the `abc` variant of the same circuit.) **The protocol
axis dominates the rival axis**: the same sampler on the same hardware read 4.4× on
Recurrent-CircuitSAT's pins and 635× on ours, which is why the `--target rare` protocol exists.

## 2. GPAP vs CMSGen: the counting corrections, and the CPU

**Vacuous pins were inside the headline.** `b02`, `b05`, `b07`, `s526` have constant last-cycle
outputs at T=25, so pinning them is a tautology and any sampler scores ~100%; they were sitting in
the sequential geomeans on both machines.

| stage | as quoted | no-vac | inflation |
|---|---:|---:|---:|
| seq-c25-rare (4080) | **47.7×** | **36.5×** | 1.31× |
| seq-c50-rare (4080) | 47.6× | 35.7× | 1.33× |
| seq-c25-rare (L4) | 58.2× | 39.1× | 1.49× |
| seq-c50-rare (L4) | 26.3× | 14.0× | 1.87× |

The inflation lands on the **rare** column — the one we tell people to quote — and is negligible
at random pins (1.04×). **Three instances also "sample" without sampling**: EPFL `ctrl`, `dec`,
`square` and the sk_ `tableBasedAddition` emit at 235–240M samples/s with zero decisions, zero
implications and success 1.0, the support being free after propagation — and the best-encoder rule
*actively selects* them, since they carry the family's highest ratio.

| stage | with zero-search cells | excluded | inflation |
|---|---:|---:|---:|
| epfl-rare | 29.0× | **9.4×** | 3.09× |
| epfl-random | 19.5× | **8.2×** | 2.38× |
| sk | 113.1× | **98.9×** | 1.14× |

Both screens are per row (`paper_tables.zero_search`), not by name list, so a new instance cannot
smuggle one back in.

**The zero ledger was one-sided, and that is the worse bug.** A CMSGen-zero row carries
`ratio = inf` and was branched on explicitly, while a GPAP-zero row (`0.0`) and a both-zero row
(`None`) are falsy and fell through every branch: **38** CMSGen-zero instances showed as wins while
the **3** where we produced nothing (`bar` ×2, `79.sk_4_40`) and the **11** where neither did were
invisible. Table 1 now carries `CMSGen=0`, `GPAP=0` and `both=0` columns that sum, with the
geomean's instances, to the instance count. Our zeros are structural: on `bar`, 0 models in ~600k
trajectories at `p_accept = 0.0` — a solution set that is a vanishing fraction of the assignment
space is what a sampler discarding conflicted trajectories cannot reach, while CMSGen's
backtracking search walks to it.

**A GPAP/CMSGen ratio is a property of the host CPU too.** Paired against the 4080, GPAP moved
0.962× (sk), 0.499× (seq-c25-rare) and 0.816× (comb-rare) while **CMSGen moved 0.411× / 0.447× /
0.482×**, its own counters putting the Colab host at **0.413×** the laptop's: the sk_ ratio going
46.2× → 98.9× is the baseline getting slower, not GPAP getting faster. `gpap bench` records `cpu`
in every JSON — quote it with the ratio.

## 3. Propagation rate: separating the engine from the sampler

The sample-rate ratios span two orders of magnitude, which invites "you picked your benchmarks".
GPAP's implications/s is near-constant across every family while CMSGen's — from its own
`--statsevery` counters — moves over half a decade on the same instances, and an identity connects
them: `samples/s ratio = (implications/s ratio) / (implications-per-sample ratio)`. The engine buys
a large, instance-independent factor; the algorithm spends part of it back by discarding conflicted
trajectories, and that rejection cost is what varies between families. **The trap: every column
must be computed over the same rows** — filtered independently, `bar` (propagates hard, emits
nothing) enters the implications ratio but not the work-per-sample ratio, and the identity breaks
by 3× on that family while every per-instance identity holds. The magnitudes are superseded by the
wave rewrite; the identity is not.

## 4. Uniformity, and what `soft` actually costs

72 generated instances whose entire solution set is enumerated by brute force, χ² over 3 seeds,
13 samplers, **0 unsound emissions**, with controls that calibrate the instrument (exact-uniform
flagged 0/72, known-TV-0.05 72/72). **`soft` is not a uniform sampler** (median TV 0.63 / 0.76
against a 0.040 floor), **B ≥ |support| is exactly uniform**, and **the regime split is real**:
`gpap:b8` beats CMSGen on circuits (0.0401 vs 0.0945) and loses on random k-CNF (0.40 vs 0.26), so
never quote a single uniformity number for GPAP
([UNIFORMITY_FINDINGS.md](UNIFORMITY_FINDINGS.md)).

**The price of excluding `soft`**: GPAP against itself on 19 sequential circuits (4 vacuous
excluded) is geomean **0.423×**, `soft` winning the bandit on only 7 of 15, replicated on the 4080
(0.410×, 7/15). On the comparison we publish — the whole matrix re-run with `--allow-soft`, one
flag changed, CMSGen between arms 1.004× and 0 of 357 paired cells differing in `M` or |support| —
the safe portfolio runs at **0.796×** and the headline moves **1.30×**, pooling regimes that differ:
up to **4.6×** at T=50 rare, **≈1.0×** on ISCAS-85, sk_ and HWMCC-random. The honest form of the
claim is difficulty, the safe arm's own success rate:

| safe-arm success rate | cells | safe/soft | `soft` won the bandit |
|---|---:|---:|---:|
| < 10⁻⁴ | 13 | **0.119×** | 5/13 |
| 10⁻⁴ – 10⁻² | 22 | **0.313×** | 12/22 |
| 10⁻² – 0.5 | 86 | 0.822× | 24/86 |
| > 0.5 | 196 | **0.988×** | 25/196 |

Monotone across 317 paired cells: the guarantee is nearly free where the sampler already works and
expensive only where it does not — precisely the regime `soft` wins. Capability differences below
success 10⁻³ are **draws, not evidence** (the zero-progress valve makes such instances bimodal and
fires both ways), except `b12` (0.072 with `soft` vs 7.1×10⁻⁵ without) and `b11`, which replicates
across two campaigns. A throughput bandit must not hand a rejected distribution to someone who
asked for a sampler.

## 5. Consistency re-check: are these numbers reproducible?

23 cells across four families were re-measured on a **fresh L4** with the protocol pinned
(`eval/colab/colab_recheck_l4.sh`) and compared field by field (`verify_artifact.py --recheck`) against
agreement stated in advance: deterministic fields exact — **23/23**; `succ` within 0.05 — within
0.01 everywhere; throughput reported, not asserted — geomean **1.032×**. At ~15 min this is the
right thing to run after any multi-VM sweep.

## 6. The platform: a Colab VM lives about an hour

Moved to `eval/colab/README.md` (2026-09-12), beside `colab_supervisor.py`, which encodes it: the
~1 h VM lifetime, the rule that results exist only once mirrored to local disk, and the four traps
(marker files not log greps; resume replaces error rows; never wait on an append-only mirror; one
poller per session). Operational knowledge about a rented machine, not a measurement of GPAP.

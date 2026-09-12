# Cross-GPU scaling — what a bigger GPU actually buys

> **Status:** settled.
> **Measured:** 2026-08-03 on three cards — RTX 4080 Laptop (12 GB, 58 SM, the reference box),
> Colab L4 (24 GB, 58 SM, sm_89), Colab A100-SXM4 (40 GB, 108 SM, sm_80).
> **Rule lives in:** `gpap/sampling/pipeline.py` (`SINGLE_PORTFOLIOS`), `cpp/src/calibrate.cpp`
> (the occupancy plateau), `eval/artifact/compare_campaign.py`.
> **Record only:** §4, b12's unexplained 2× success cliff on the A100 — open.

An A100 is worth ~1.8×, an L4 ~0.85×. Two measurement traps (§3) each gave a plausible wrong
answer first; both are guarded in the harness. How the Colab machines were driven is
`eval/colab/README.md`, beside the drivers themselves.

## 1. The result

`--portfolio b1` and `b8` (single-config, no bandit), `--k-cap 49152` pinned on every arm,
`--n-time 2000000 --time-limit 20`, seed 5, two reps with arm order alternated. Instances: 4
projected `sk_` CNFs (M_eff 232–1,534) + b03 / s1196 / b12 / b14 at T=25, tseitin, `--target rare`
(M_eff 5k / 45k / 67k / 917k).

| vs RTX 4080 laptop | models/s | work rate (assigns/s) | success rate |
|---|---:|---:|---:|
| L4 24 GB | 0.838× | 0.838× | 1.006 |
| A100 40 GB | 1.822× | 1.905× | 0.956 |

Per instance the A100 sits in a narrow 1.58–2.32× band across every size and both kinds. b14
(M_eff 917k) delivers 0/2/5 models in 20 s on the three cards, below resolution, and is excluded.
Success rate is machine-independent, exactly 1.000 on 13 of 14 cells: success is a property of
the formula and the config, so any cross-GPU difference in it is a confound, not a result.

The L4 is not an L40S stand-in: Colab offers no L40/L40S; the L4 has the laptop 4080's 58 SMs at
a lower clock and ~300 GB/s vs ~432 GB/s, and measures slower than the laptop. An L40S (48 GB,
142 SM, ~864 GB/s) exists only on the HPC cluster.

## 2. Where the speedup comes from

Not from the batch: on one A100 and one formula, the batch 262,144 → 1,048,576 (4×) with
everything else fixed bought +2.8%, at 8.8 GB of 41 GB used; 262,144 rows over ~6,912 resident
warps is already ~38× oversubscribed. Not from VRAM: `--k-cap` is `min(analytic, cap)` and the
analytic bound drops below the cap once M ≳ 20k (M=100k: 9,696 on the 4080, 37,440 on the A100),
so on every real circuit unroll the cap never binds and the analytic bound already scales with
free VRAM; where the cap does bind the GPU is saturated. The cap is hygiene, not a throughput
lever.

From the hardware, flatly: the work rate (assigns/s) is flat in M — sk_ 1.58–2.02, b03 1.90,
s1196 1.71–2.31, b12 1.91–2.05 — and lands on the A100's 1.86× SM-count ratio rather than its
~3.6× bandwidth ratio. The engine is occupancy/compute-bound, not bandwidth-bound, at every size
measured. Two earlier readings are retracted: that the cap was the cross-GPU lever, and that the
work rate rises with M (an artifact of the confounded data in §3).

## 3. The two traps

### 3a. A free bandit picks differently per machine

The config bandit selects on measured models/wall with no stability term. On b12-c25 the 4080
chose B=1 (succ 0.5315) and the A100 B=8 (succ 0.2653), consistently across reps, so the A100 did
2.16× the raw work and read out as a bogus 1.19×; pinned, the same instance gives 1.85×. It also
costs reproducibility on one machine: 27.sk_3_32 alternated between 491,806 and 581,412 models
across identical local runs, on both arms of a before/after. Fix, shipped: `--portfolio b8 | b1 |
s1` are single-config portfolios (`pipeline.SINGLE_PORTFOLIOS`); one entry means `sample()` never
enables the bandit. Use one for any cross-GPU or before/after run.

### 3b. Too small a budget measures one chunk

At `--n-time 20000` against a 262,144-row batch the small instances finish in a single chunk, so
the rate is first-chunk overhead: that run read A100 2.42× / L4 1.30×, ~30% off in the opposite
direction from 3a. Size `--n-time` so several chunks run, or cap with `--time-limit`; check
`n_models / (batch × success)` ≫ 1.

### 3c. Two more that bite across machines

The miner is time-budgeted (~5 s), so a faster GPU mines a different clause set and the formula
differs (b12 learned 7,096 / 6,300 / 6,299 units on 4080 / L4 / A100). And `gpap_sps` is the
all-chunk rate on a single-config run but the winning config's steady-state rate on a bandit run:
compare pinned against pinned.

## 4. Open: b12's 2× success cliff on the A100

On b12-c25 the A100 loses half its success rate, 0.530 → 0.265, stable at 0.2649–0.2653 across
seven K values. Ruled out: config (pinned `b1`/`b8`, both show it), the miner (the L4 learned an
almost identical unit count and kept 0.530), kernel variant (all arms `wave`, split 16|16), and
int32 `K*M` overflow, falsified by a K ladder: the drop happens between K=20,000 (succ 0.5305,
`K*M/2^31` = 0.63) and K=28,000 (0.2649, ratio 0.88) and does not worsen through K=49,152 (ratio
1.55, 0.2652). It is machine- and K-dependent: the L4 at K=27,760 gives 0.530. Soundness is
unaffected (`gpu_sound=True` throughout): half as many models, not wrong ones. Start at
`cpp/kernels/bcp.py`.

## 5. Reproducing

```bash
# ALWAYS: pin the config, pin the cap, give it several chunks, alternate arm order.
gpap bench 'benchmarks/sampling/*.cnf' --portfolio b1 --k-cap 49152 \
    --n-time 2000000 --time-limit 20 --seed 5 --output out.json
gpap bench 'benchmarks/circuits/bench/seq/b12.bench' --cycles 25 --encoder tseitin \
    --target rare --portfolio b1 --k-cap 49152 --n-time 2000000 --time-limit 20
```

Read back per row `engine.batch`, `engine.K`, `engine.k_cap`, `succ`, `assigns_per_s`. If `succ`
differs across machines for the same instance and config, something is confounded. Before/after
on one machine: `scripts/perf_ab_trees.py` plus a pinned-K, single-config, fixed-seed run whose
model count must be identical (docs/BENCHMARKING.md §3).

## 6. Running this on Colab from a terminal

Moved to `eval/colab/README.md` (2026-09-12), beside the drivers it describes: how to drive the
Colab CLI, what its 0.6.0 `colab exec` bug forces, and the bootstrap. It is operational knowledge
about a rented machine, not a measurement of GPAP — the distinction this directory keeps.

# Cross-GPU scaling — what a bigger GPU actually buys

Measured 2026-08-03 on three cards: **RTX 4080 Laptop** (12 GB, 58 SM, the reference box),
**Colab L4** (24 GB, 58 SM, sm_89) and **Colab A100-SXM4** (40 GB, 108 SM, sm_80). Driven from a
terminal through the Google Colab CLI (see §6).

The headline is short: **an A100 is worth ~1.8×, an L4 is worth ~0.85× — and getting to that
number took three attempts, because two independent measurement traps each produced a plausible
wrong answer first.** Both traps are now guarded in the harness; §3 is the part to read before
running any comparison.

---

## 1. The result

`--portfolio b1` and `b8` (single-config, no bandit), `--k-cap 49152` pinned on every arm,
`--n-time 2000000 --time-limit 20`, seed 5, two reps with the arm order alternated. Instances:
4 projected `sk_` CNFs (M_eff 232–1,534) + b03 / s1196 / b12 / b14 at T=25, tseitin, `--target rare`
(M_eff 5k / 45k / 67k / 917k).

| vs RTX 4080 laptop | **models/s** | work rate (assigns/s) | success rate |
|---|---:|---:|---:|
| **L4 24 GB** | **0.838×** | 0.838× | 1.006 |
| **A100 40 GB** | **1.822×** | 1.905× | 0.956 |

Per-instance the A100 sits in a narrow **1.58–2.32×** band across every size and both kinds
(CNF and circuit). b14 (M_eff 917k) delivers 0/2/5 models in 20 s on the three cards — below
measurement resolution, excluded from the geomeans.

**Success rate is machine-independent — exactly 1.000 on 13 of 14 cells.** That is the check that
the measurement is clean: success is a property of the formula and the config, so any cross-GPU
difference in it is a confound, not a result.

### The L4 is not an L40S stand-in

Colab offers no L40/L40S. Its Ada part is the **L4**: same 58 SMs as the laptop 4080, lower clock,
~300 GB/s vs ~432 GB/s. Measured **0.838× — slower than the laptop**. Only the A100/H100 arms on
Colab are worth paying for. An L40S (48 GB, 142 SM, ~864 GB/s) exists only on the HPC cluster.

---

## 2. Where the speedup comes from — and where it does not

**Not from the batch.** The one experiment in which K is the only variable: one A100, one formula,
`K_fused` 262,144 → 1,048,576 (4×), everything else fixed → **+2.8%**, at 8.8 GB of 41 GB used. At
262,144 rows over ~6,912 resident warps we are already ~38× oversubscribed; extra rows queue.

**Not from VRAM.** `--k-cap` is `min(analytic, cap)`, and the analytic VRAM bound drops below the
cap once M ≳ 20k (at M=100k: 9,696 on the 4080, 37,440 on the A100). Every real circuit unroll is
far past that, so the cap never binds there — and the analytic bound already scales with free VRAM
on its own. Where the cap *does* bind (small M), the GPU is already saturated. The
VRAM-proportional `--k-cap auto` shipped the same day is **hygiene, not a throughput lever**.

**From the hardware, flatly.** The work rate (assigns/s, the raw propagation rate) is **flat in M**
— sk_ 1.58–2.02, b03 1.90, s1196 1.71–2.31, b12 1.91–2.05 — and lands on the A100's **1.86×
SM-count** ratio rather than its ~3.6× bandwidth ratio. So the engine is **occupancy/compute-bound,
not bandwidth-bound**, at every size we can measure.

> **Two retractions.** An earlier draft claimed the cap was the cross-GPU lever ("a 48 GB card
> reproduces the laptop's numbers") — wrong, see above. A later draft then claimed the work rate
> *rises* with M and that we were bandwidth-leaning on big instances — also wrong; that came from
> the confounded data of §3 and does not survive a pinned-config run.

---

## 3. The two traps (read this before measuring anything)

Each produced a confident, plausible, wrong number.

### 3a. A free bandit picks differently per machine

`configs` is an adaptive bandit that selects on measured models/wall with **no stability term**. On
instances where two configs are close it picks differently per machine — and the comparison then
A/Bs the bandit instead of the hardware:

```
b12-c25   4080 → B=1, succ 0.5315        A100 → B=8, succ 0.2653
```

Consistent across reps on each machine. The A100 did 2.16× the raw work and converted it to models
at half the rate, reading out as a bogus **1.19×**. With the config pinned the same instance gives
**1.85×**, in line with everything else.

It also costs reproducibility *within* one machine: 27.sk_3_32 alternated between 491,806 and
581,412 models across repeated identical local runs — **both arms of a before/after A/B produced
both values**, so it is inherent, not a regression.

**Fix, shipped:** `--portfolio b8 | b1 | s1` are single-config portfolios
(`pipeline.SINGLE_PORTFOLIOS`). One entry ⇒ `sample()` never enables the bandit. Use one for any
cross-GPU or before/after run.

### 3b. Too small a budget measures one chunk

At `--n-time 20000` against a 262,144-row batch, the small instances finish in a **single chunk**,
so the reported rate is first-chunk overhead (kernel warmup, allocation), not steady state. That
run read **A100 2.42× / L4 1.30×** — off by ~30% in the *opposite* direction from trap 3a, which
is why it looked believable.

**Fix:** size `--n-time` so several chunks run (2,000,000 here), or cap with `--time-limit` and let
it run to the cap. Check `n_models / (batch × success)` ≫ 1.

### 3c. Two more that bite specifically across machines

* **The entailed-clause miner is TIME-budgeted** (~5 s). A faster GPU mines a different clause set
  in the same wall budget, so the formula itself differs: b12 learned 7,096 / 6,300 / 6,299 units
  on 4080 / L4 / A100. Effects were small here but it is a real cross-machine variable.
* **`gpap_sps` semantics** differ subtly between a single-config run (rate over all chunks,
  including the first) and a bandit run (the winning config's steady-state rate, which excludes
  the exploration chunk that paid warmup). Do not compare pinned numbers against bandit numbers;
  compare pinned against pinned.

---

## 4. OPEN — b12's 2× success cliff on the A100

Unexplained and worth chasing before a cluster run, because it costs 2× on exactly the large-M
instances a big GPU is bought for.

On b12-c25 the A100 loses half its success rate — **0.530 → 0.265** — and it is not noise:
0.2649–0.2653 across seven separate K values.

| ruled out | how |
|---|---|
| config | pinned `b1`/`b8`; both show it |
| the miner | the L4 learned an almost identical unit count (6,300 vs 6,299) and kept 0.530 |
| kernel variant | all arms dispatched `wave`, split `16\|16` |
| **int32 `K*M` overflow** | **falsified by K-ladder** (below) |

`2^31 / M_eff ≈ 31,800` sits right in the suspicious range, so the obvious hypothesis was an int32
element-index overflow in the `(K, M)` packed state. A ladder on one A100 kills it — the drop
happens **between K=20,000 and 28,000**, at `K*M ≈ 0.9 × 2^31`, and does **not** worsen at 1.55×:

```
K= 20,000  K*M/2^31=0.63  succ=0.5305        K= 34,000  ratio=1.07  succ=0.2651
K= 28,000  K*M/2^31=0.88  succ=0.2649        K= 40,000  ratio=1.26  succ=0.2651
K= 31,500  K*M/2^31=0.99  succ=0.2652        K= 49,152  ratio=1.55  succ=0.2652
```

It is machine- **and** K-dependent: the L4 at K=27,760 gives 0.530 while the A100 at K=28,000 gives
0.265. **Soundness is unaffected** (`gpu_sound=True` throughout) — we get half as many models, not
wrong ones. Start at `gpap/engine/kernels.py`.

---

## 5. Reproducing

```bash
# ALWAYS: pin the config, pin the cap, give it several chunks, alternate arm order.
gpap bench 'benchmarks/sampling/*.cnf' --portfolio b1 --k-cap 49152 \
    --n-time 2000000 --time-limit 20 --seed 5 --output out.json

# circuits
gpap bench 'benchmarks/circuits/bench/seq/b12.bench' --cycles 25 --encoder tseitin \
    --target rare --portfolio b1 --k-cap 49152 --n-time 2000000 --time-limit 20
```

Read back per row: `engine.batch` (what actually ran), `engine.K` / `K_fused` / `k_cap`,
`succ`, `assigns_per_s`. If `succ` differs across machines for the same instance+config,
something is confounded — stop and find it before believing the rates.

Before/after on one machine: `scripts/perf_ab_trees.py` (throughput) and `scripts/engine_ab.py`
(bit-identical model sets). Use both.

---

## 6. Running this on Colab from a terminal

`pip install google-colab-cli`, then ADC with all four scopes (`openid`, `cloud-platform`,
`userinfo.email`, `colaboratory` — it 403s without the last). `colab whoami` verifies in one shot.

**`colab exec` is broken** in google-colab-cli 0.6.0: it calls `jupyter_kernel_client.KernelClient`,
which upstream renamed to `JupyterKernelClient`. **Use `colab console`** — a raw-websocket tmux
shell with no dependency on that module.

The working pattern, which also survives disconnects:

```bash
printf 'nohup bash /content/job.sh > /content/job.log 2>&1 &\necho LAUNCHED\n' \
    | colab console -s <session>                       # fire and detach
# poll from a SEPARATE call:
printf 'grep -ac DONE_MARKER /content/job.log\n' | colab console -s <session>
```

Console gotchas, all hit the hard way: it **exits on piped EOF**, so a long synchronous command
returns nothing — always detach and poll. Long one-liners get **line-wrapped by tmux**, which
corrupts `&&` into `& &`; keep commands short and newline-separated. Output carries terminal
escapes — filter with `grep -a` and `sed 's/\x1b\[[0-9;]*[a-zA-Z]//g'`. Never run two pollers
against one session; a background `until` loop stole every command until it was stopped. To move
results back, `tar` them on the VM and `colab download` the archive.

Bootstrap is ~2 min: `apt-get install build-essential cmake zlib1g-dev`, deps, then
`scripts/build_core.sh`. libgpap AOT-builds cleanly on **sm_80 and sm_89** with no CUDA toolkit —
sm_89 is the L40S arch, so the HPC build path is de-risked. Select the package by **cwd +
PYTHONPATH**, not `pip install -e .`, when two trees must coexist. `colab stop -s <name>` when
done; the keep-alive is a *local* daemon with a hard 24 h cap.

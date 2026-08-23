# The price of the uniformity guarantee

GPU **NVIDIA L4**, CPU **Intel(R) Xeon(R) CPU @ 2.20GHz** — both arms, same machine type, same protocol, same budgets, same encoders. The only difference is `--allow-soft`, injected through `run_full_eval.sh`'s single `bench` call so both arms traverse identical code.

`soft` is Barbarik-rejected 5/5 against an exact oracle (median TV 0.63 vs a 0.040 detection floor), so the shipped sampler cannot select it. Ratios below are **safe / soft**: less than 1.0 means the uniformity-safe default is slower.

## Summary

* **Throughput, all paired cells: 0.846x** (n=329) — -15% against a sampler free to choose `soft`.
* **Throughput where the bandit actually chose `soft`: 0.451x** (n=62 of 64 such cells). The all-cells figure is diluted: on the other cells both arms run the same configuration and contribute exactly 1.0x.
* **Capability lost** (only the `soft` arm sampled at all): **2** cells: `b11`/abc:fast (seq-c50-rare, succ 5.4e-04), `b11`/tseitin (seq-c50-rare, succ 1.7e-04)
* **Capability gained** (only the safe arm sampled): **0** cells

  ⚠️ **2 of 2 capability differences sit below success 1e-03**, where the zero-progress stop makes the outcome bimodal — the run halts once every config plus a grace chunk has missed. Treat those as draws, not as evidence about `soft`: `b11`/abc:fast, `b11`/tseitin.

* **Effect on the published GPAP/CMSGen headline: 1.14x.** Allowing `soft` would move the reported ratios to 84.3x → **138.7x** (ITC'99 T=25 rare), 42.1x → **64.4x** (ITC'99 T=50 rare), 333.8x → **383.1x** (ITC'99 T=25 random), 305.7x → **468.5x** (ITC'99 T=50 random), 935.2x → **941.3x** (ISCAS-85 T=1 rare), 934.6x → **951.1x** (ISCAS-85 T=1 random), 187.4x → **193.5x** (HWMCC'17 T=25 rare), 407.5x → **394.1x** (HWMCC'17 T=50 rare), 274.8x → **291.7x** (HWMCC'17 T=25 random), 559.2x → **521.2x** (HWMCC'17 T=50 random), 33.5x → **41.2x** (EPFL T=1 rare), 78.3x → **78.5x** (EPFL T=1 random), 142.0x → **138.2x** (Sketch -- --). That is the size of the claim we are choosing not to make.

## Per family

| family | depth | targets | cells | safe/soft (all) | safe/soft (soft-won) | soft-won cells | cap. lost | headline safe | headline soft |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ITC'99 + ISCAS-89 sequential | T=25 | rare | 38 | 0.640x | 0.181x | 9 | 0 | 84.3x | 138.7x |
| ITC'99 + ISCAS-89 sequential | T=50 | rare | 38 | 0.716x | 0.355x | 12 | 2 | 42.1x | 64.4x |
| ITC'99 + ISCAS-89 sequential | T=25 | random | 38 | 0.904x | 0.578x | 6 | 0 | 333.8x | 383.1x |
| ITC'99 + ISCAS-89 sequential | T=50 | random | 38 | 0.760x | 0.336x | 8 | 0 | 305.7x | 468.5x |
| ISCAS-85 combinational | T=1 | rare | 20 | 0.974x | 0.865x | 3 | 0 | 935.2x | 941.3x |
| ISCAS-85 combinational | T=1 | random | 20 | 0.972x | 0.902x | 3 | 0 | 934.6x | 951.1x |
| HWMCC'17 property | T=25 | rare | 14 | 0.794x | 0.585x | 6 | 0 | 187.4x | 193.5x |
| HWMCC'17 property | T=50 | rare | 14 | 0.825x | 0.371x | 3 | 0 | 407.5x | 394.1x |
| HWMCC'17 property | T=25 | random | 14 | 0.956x | — | 0 | 0 | 274.8x | 291.7x |
| HWMCC'17 property | T=50 | random | 14 | 1.024x | — | 0 | 0 | 559.2x | 521.2x |
| EPFL combinational | T=1 | rare | 40 | 0.841x | 0.610x | 9 | 0 | 33.5x | 41.2x |
| EPFL combinational | T=1 | random | 40 | 0.948x | 0.947x | 5 | 0 | 78.3x | 78.5x |
| Sketch / sk_ projected CNF | -- | -- | 29 | 1.023x | — | 0 | 0 | 142.0x | 138.2x |

## Where the price is paid — by instance difficulty

Bucketed by the **safe** arm's success rate, which is the thing you could know before choosing a portfolio. The cost is not spread evenly; it is a function of how hard the instance is.

| safe-arm success rate | cells | safe/soft | `soft` won the bandit |
|---|---:|---:|---:|
| < 1e-4 | 15 | **0.539x** | 3/15 |
| 1e-4 .. 1e-2 | 14 | **0.352x** | 10/14 |
| 1e-2 .. 0.5 | 80 | **0.715x** | 29/80 |
| > 0.5 | 220 | **0.981x** | 20/220 |

This is the shape that matters for the parked design question: a bandit that fell back to `soft` **only** where every uniformity-safe config yields almost nothing would recover most of the loss while leaving the easy majority — where the safe portfolio is already within a few percent — untouched and uniform.

## Where it actually costs (bandit chose `soft`, worst 15 cells)

| stage | instance | encoder | safe samples/s | soft samples/s | safe/soft |
|---|---|---|---:|---:|---:|
| seq-c25-rare | `b11` | abc:fast | 1.04 | 298 | 0.003x |
| seq-c25-rare | `b11` | tseitin | 0.59 | 20 | 0.029x |
| seq-c50-rare | `b12` | abc:fast | 82 | 2,573 | 0.032x |
| seq-c50-random | `b10` | abc:fast | 2,498 | 76,353 | 0.033x |
| seq-c50-random | `b10` | tseitin | 2,851 | 54,760 | 0.052x |
| seq-c25-rare | `b12` | abc:fast | 788 | 13,215 | 0.060x |
| hwmcc-c25-rare | `bj08amba2g4f3` | abc:fast | 1.71 | 23 | 0.075x |
| seq-c50-rare | `b10` | tseitin | 3,304 | 43,824 | 0.075x |
| epfl-rare | `sin` | tseitin | 2,257 | 27,843 | 0.081x |
| hwmcc-c50-rare | `bj08amba2g4f3` | abc:fast | 1.04 | 4.95 | 0.209x |
| seq-c50-rare | `s386` | abc:fast | 23,310 | 96,271 | 0.242x |
| seq-c50-random | `b13` | tseitin | 59,004 | 183,848 | 0.321x |
| epfl-rare | `sin` | abc:fast | 4,086 | 12,481 | 0.327x |
| seq-c25-random | `b10` | tseitin | 16,854 | 49,044 | 0.344x |
| hwmcc-c50-rare | `texasparsesysp3` | abc:fast | 49,982 | 141,767 | 0.353x |

## Control — what should NOT have moved

Only GPAP's portfolio changed, so CMSGen's throughput and the formula sizes are a negative control on the pairing. If either moved, the `soft` effect would be confounded with a machine difference and nothing above would be valid.

* CMSGen throughput, soft arm / safe arm: **1.007x** (n=275) — consistent with VM-to-VM noise on the same instance type.
* Deterministic fields (clauses `M`, `|support|`) differing between arms: **0** (identical encodings on every paired cell, as required).

## Reading this

The honest statement is not a single number. Enforcing uniformity costs **0.85x** of throughput averaged over the whole published matrix, but the cost is concentrated: it is ~1.0x wherever the bandit would not have chosen `soft` anyway, and **0.45x** where it would have. Against that, `soft` fails an exact-oracle uniformity test 5 times out of 5, and a sampler is a tool whose output distribution is the product. We keep the guarantee and report the price.

# The price of the uniformity guarantee

GPU **NVIDIA L4**, CPU **Intel(R) Xeon(R) CPU @ 2.20GHz** — both arms, same machine type, same protocol, same budgets, same encoders. The only difference is `--allow-soft`, injected through `run_full_eval.sh`'s single `bench` call so both arms traverse identical code.

`soft` is Barbarik-rejected 5/5 against an exact oracle (median TV 0.63 vs a 0.040 detection floor), so the shipped sampler cannot select it. Ratios below are **safe / soft**: less than 1.0 means the uniformity-safe default is slower.

## Summary

* **Throughput, all paired cells: 0.796x** (n=317) — -20% against a sampler free to choose `soft`.
* **Throughput where the bandit actually chose `soft`: 0.332x** (n=66 of 69 such cells). The all-cells figure is diluted: on the other cells both arms run the same configuration and contribute exactly 1.0x.
* **Capability lost** (only the `soft` arm sampled at all): **6** cells: `b11`/abc:fast (seq-c25-rare, succ 1.6e-03), `b11`/abc:fast (seq-c50-rare, succ 5.3e-04), `b11`/tseitin (seq-c50-rare, succ 1.8e-04), `bj08amba2g4f3`/tseitin (hwmcc-c25-rare, succ 2.1e-04), `bj08amba2g4f3`/tseitin (hwmcc-c50-rare, succ 5.4e-04), `i2c`/tseitin (epfl-rare, succ 6.4e-04)
* **Capability gained** (only the safe arm sampled): **1** cells: `div`/tseitin (epfl-rare, succ 2.1e-04)

  ⚠️ **6 of 7 capability differences sit below success 1e-03**, where the zero-progress stop makes the outcome bimodal — the run halts once every config plus a grace chunk has missed. Treat those as draws, not as evidence about `soft`: `b11`/abc:fast, `b11`/tseitin, `bj08amba2g4f3`/tseitin, `bj08amba2g4f3`/tseitin, `i2c`/tseitin, `div`/tseitin.

* **Effect on the published GPAP/CMSGen headline: 1.30x.** Allowing `soft` would move the reported ratios to 39.1x → **74.7x** (ITC'99 T=25 rare), 14.0x → **64.9x** (ITC'99 T=50 rare), 183.3x → **199.1x** (ITC'99 T=25 random), 147.0x → **253.8x** (ITC'99 T=50 random), 615.4x → **632.4x** (ISCAS-85 T=1 rare), 636.1x → **611.8x** (ISCAS-85 T=1 random), 195.6x → **202.7x** (HWMCC'17 T=25 rare), 421.5x → **395.8x** (HWMCC'17 T=50 rare), 263.8x → **286.6x** (HWMCC'17 T=25 random), 520.6x → **537.5x** (HWMCC'17 T=50 random), 9.4x → **11.1x** (EPFL T=1 rare), 8.2x → **11.8x** (EPFL T=1 random), 98.9x → **101.8x** (Sketch -- --). That is the size of the claim we are choosing not to make.

## Per family

| family | depth | targets | cells | safe/soft (all) | safe/soft (soft-won) | soft-won cells | cap. lost | headline safe | headline soft |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ITC'99 + ISCAS-89 sequential | T=25 | rare | 38 | 0.601x | 0.278x | 14 | 1 | 39.1x | 74.7x |
| ITC'99 + ISCAS-89 sequential | T=50 | rare | 38 | 0.454x | 0.097x | 12 | 2 | 14.0x | 64.9x |
| ITC'99 + ISCAS-89 sequential | T=25 | random | 38 | 0.952x | 0.767x | 9 | 0 | 183.3x | 199.1x |
| ITC'99 + ISCAS-89 sequential | T=50 | random | 38 | 0.876x | 0.248x | 5 | 0 | 147.0x | 253.8x |
| ISCAS-85 combinational | T=1 | rare | 20 | 0.969x | 0.970x | 1 | 0 | 615.4x | 632.4x |
| ISCAS-85 combinational | T=1 | random | 20 | 1.016x | 0.996x | 1 | 0 | 636.1x | 611.8x |
| HWMCC'17 property | T=25 | rare | 14 | 0.733x | 0.511x | 6 | 1 | 195.6x | 202.7x |
| HWMCC'17 property | T=50 | rare | 14 | 0.599x | 0.109x | 3 | 1 | 421.5x | 395.8x |
| HWMCC'17 property | T=25 | random | 14 | 0.959x | — | 0 | 0 | 263.8x | 286.6x |
| HWMCC'17 property | T=50 | random | 14 | 0.974x | — | 0 | 0 | 520.6x | 537.5x |
| EPFL combinational | T=1 | rare | 40 | 0.849x | 0.624x | 10 | 1 | 9.4x | 11.1x |
| EPFL combinational | T=1 | random | 40 | 0.810x | 0.321x | 7 | 0 | 8.2x | 11.8x |
| Sketch / sk_ projected CNF | -- | -- | 29 | 0.971x | 1.011x | 1 | 0 | 98.9x | 101.8x |

## Where the price is paid — by instance difficulty

Bucketed by the **safe** arm's success rate, which is the thing you could know before choosing a portfolio. The cost is not spread evenly; it is a function of how hard the instance is.

| safe-arm success rate | cells | safe/soft | `soft` won the bandit |
|---|---:|---:|---:|
| < 1e-4 | 13 | **0.119x** | 5/13 |
| 1e-4 .. 1e-2 | 22 | **0.313x** | 12/22 |
| 1e-2 .. 0.5 | 86 | **0.822x** | 24/86 |
| > 0.5 | 196 | **0.988x** | 25/196 |

This is the shape that matters for the parked design question: a bandit that fell back to `soft` **only** where every uniformity-safe config yields almost nothing would recover most of the loss while leaving the easy majority — where the safe portfolio is already within a few percent — untouched and uniform.

## Where it actually costs (bandit chose `soft`, worst 15 cells)

| stage | instance | encoder | safe samples/s | soft samples/s | safe/soft |
|---|---|---|---:|---:|---:|
| seq-c50-rare | `b13` | tseitin | 0.40 | 12,953 | 0.000x |
| seq-c25-rare | `b12` | abc:fast | 21 | 11,328 | 0.002x |
| seq-c50-rare | `b12` | abc:fast | 4.28 | 2,187 | 0.002x |
| hwmcc-c50-rare | `pdtswvibs8x8p0` | tseitin | 1.55 | 151 | 0.010x |
| epfl-random | `arbiter` | tseitin | 250 | 20,268 | 0.012x |
| seq-c25-rare | `b11` | tseitin | 0.32 | 9.72 | 0.033x |
| seq-c50-random | `b10` | abc:fast | 2,122 | 54,449 | 0.039x |
| seq-c50-random | `b10` | tseitin | 1,750 | 26,582 | 0.066x |
| hwmcc-c25-rare | `bj08amba2g4f3` | abc:fast | 1.43 | 19 | 0.075x |
| seq-c50-rare | `b10` | tseitin | 1,795 | 22,748 | 0.079x |
| epfl-random | `arbiter` | abc:fast | 305 | 3,189 | 0.096x |
| epfl-rare | `arbiter` | tseitin | 901 | 7,817 | 0.115x |
| seq-c25-rare | `s832` | abc:fast | 183 | 1,453 | 0.126x |
| seq-c50-rare | `b12` | tseitin | 253 | 1,697 | 0.149x |
| hwmcc-c50-rare | `bj08amba2g4f3` | abc:fast | 1.10 | 6.05 | 0.182x |

## Control — what should NOT have moved

Only GPAP's portfolio changed, so CMSGen's throughput and the formula sizes are a negative control on the pairing. If either moved, the `soft` effect would be confounded with a machine difference and nothing above would be valid.

* CMSGen throughput, soft arm / safe arm: **1.004x** (n=274) — consistent with VM-to-VM noise on the same instance type.
* Deterministic fields (clauses `M`, `|support|`) differing between arms: **0** (identical encodings on every paired cell, as required).

## Reading this

The honest statement is not a single number. Enforcing uniformity costs **0.80x** of throughput averaged over the whole published matrix, but the cost is concentrated: it is ~1.0x wherever the bandit would not have chosen `soft` anyway, and **0.33x** where it would have. Against that, `soft` fails an exact-oracle uniformity test 5 times out of 5, and a sampler is a tool whose output distribution is the product. We keep the guarantee and report the price.

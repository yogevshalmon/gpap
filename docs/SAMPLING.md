# Sampling solutions with GPAP

> **Reader:** anyone who wants random solutions of a CNF formula or a circuit — every option of
> `gpap sample`, its default, and how to read what a run reports.

```bash
gpap sample formula.cnf -n 50000 --out samples.txt --stats run.json
gpap sample b10.bench --cycles 25 -n 50000 --out stimuli.txt    # a sequential circuit
```

From Python, `gpap.sample(path, n)` (a CNF) and `gpap.sample_circuit(path, cycles, n)` (a
circuit) run the same pipeline and return the samples as a numpy array plus the same statistics
(see [PYTHON_API.md](PYTHON_API.md)).

## What a sample is

Each sample comes from one randomized trajectory on the GPU: pick free variables, give them
random values, propagate, repeat. A trajectory either reaches a complete assignment, which then
satisfies every clause by construction, or runs into a conflict and is discarded. Thousands of
trajectories run side by side in one kernel launch, so the rate is the number of trajectories per
second times the fraction that succeed.

A sample is **projected** onto the formula's support:
- for a CNF, the variables named in its `c ind` lines, or all variables when there are none;
- for a circuit, the primary inputs at every cycle, one column per `wire@cycle`.

## Defaults

| option | default | what it does |
|---|---|---|
| `--portfolio` | `circuit` | which sampling configurations compete (below) |
| `--no-simplify` | simplification on | unit propagation before sampling; it never changes the solution set |
| `--no-els` | equivalent-literal substitution on | merges literals that imply each other, removing variables and clauses |
| `--mine` | `auto` | mines entailed clauses first (below) |
| `--verify-sound` | off | re-checks every emitted sample against every clause |
| `-n` | 50,000 | how many samples to deliver (a target, see below) |
| `--time-limit` | 60 s | wall cap on the sampling loop; `0` means no cap |
| `--give-up-chunks` | off | stop early when nothing has been found |
| `--seed` | 5 | the random seed |
| `--k-cap` | `auto` | ceiling on the batch of trajectories per launch; `auto` is the same constant on every GPU |
| `--device` | the current CUDA device | which GPU to run on |
| `--n-cov` | 0 (off) | after the run, collect this many full models to count distinct ones and verify them on the CPU; a self-check, kept out of every rate |

**Portfolios.** A configuration is a variable order and a decision batch (how many variables a
trajectory decides before it propagates). A portfolio of several configurations lets a bandit
pick the fastest one for the formula during a short warmup.

| `--portfolio` | configurations | use it for |
|---|---|---|
| `circuit` | random order with batches of 8 and 1 | the default, for every input |
| `default` | `circuit` plus a structured order with batch 1 | plain CNF from other sources |
| `b8`, `b1`, `s1` | exactly one configuration, no bandit | measurements: the bandit can pick differently on another machine |

**Mining.** For formulas of 500 clauses or more, `--mine auto` first proves clauses the formula
already implies, within a 5-second budget shared by its two stages and the rebuild between them. The
mined clauses are kept only if a short trial shows they speed up sampling. `--mine on` mines
smaller formulas too, and `--mine off` skips it. The run reports what it found and whether it kept
it.

## How many, and for how long

**`-n` is a target, not a promise.** The GPU works in batches of trajectories, so a run normally
stops at the end of the batch in which it reaches `-n`, and delivers more than you asked for. It
can also stop short:

| `stop` in the stats | the run ended because |
|---|---|
| `target` | it reached `-n` |
| `time` | `--time-limit` ran out |
| `chunks` | a chunk limit ran out (`max_chunks` in the Python API) |
| `zero_yield` | `--give-up-chunks` fired: nothing found after every configuration had its turn plus N more batches |
| `unsat` | unit propagation, when the formula was loaded, proved it has no solution |

Quote the delivered count (`n_models`), not the request.

**A time budget is spent, not approached.** With `--give-up-chunks` off (the default), a run that
has found nothing keeps searching until `--time-limit`: on tight circuits the first solution can
take a large part of a minute. Turn it on when a quick "this yields nothing" is worth more than
the search. With `--time-limit 0` and no give-up, the run continues until it reaches `-n`.

## Output

- `--out FILE` writes one row per sample, one column per support variable.
- `--out-format` is `bits` (0/1 text, the default), `npy` (a uint8 numpy array) or `dimacs`
  (signed literals, the format CMSGen writes).
- The file holds `min(-n, --max-samples, delivered)` rows; `--max-samples` (default 100,000) can
  only lower that count, so raise `-n` for a bigger file.
- Rows are kept by the sampling run itself. If the winning configuration kept fewer than the
  file needs, a top-up with that configuration runs within what is left of `--time-limit`; a
  spent budget buys none. Writing never changes which samples a seed finds.
- The time spent producing the file is outside the steady-state rate and inside the end-to-end
  one (`phases.output`).
- All rows come from one configuration, the one with the best measured rate, never from a
  mixture.
- `--stats FILE` writes the statistics described below as JSON.

## Circuits

`.bench` (ISCAS-85/89, ITC'99) and AIGER `.aig`/`.aag` files are read directly. Latch reset
values 0 and 1 are honoured and `x` is rejected. A sequential circuit is unrolled `--cycles`
times (default 25) from its reset state; a combinational one is a single cycle. The CNF records
which wire and cycle each support variable is (`c wire` comments, and `wire_map` in the stats).

**The target.** The outputs of the last cycle are pinned to a pattern known to be reachable:

- `--target random` (the default) pins the outputs that one random input sequence produces.
  Patterns like this are easy to hit by chance.
- `--target rare` pins the rarest output pattern seen while simulating 262,144 random sequences:
  still reachable, but rare enough that random simulation almost never hits it.
- `--pin-outputs K` pins only K outputs; `0` pins none and samples the unconstrained inputs.

**The encoder.** `--encoder tseitin` (the default) encodes the unrolled circuit natively.
`--encoder abc` first optimizes it with [berkeley-abc](https://github.com/berkeley-abc/abc),
which usually gives a much smaller CNF with the same solution space on the inputs; build ABC
with `scripts/build_abc.sh`, or point `--abc PATH` at a binary (otherwise `$GPAP_ABC`, then
`PATH`, then the one the script built). `--abc-opt full|fast|auto` sets how hard ABC works. The default
stays `tseitin` so that a result never depends on whether ABC is installed. Encoding time is
reported as `encode_s` and is not part of any rate.

**Two circuit-only checks:**
- `--sim-check` replays every emitted sample through the original circuit and confirms the
  pinned outputs. This checks the whole chain — unrolling, encoding and sampling — independently
  of the CNF. A clean run reports `sim_check_bad: 0`.
- `--sim-baseline` also measures plain rejection sampling (random inputs, keep the ones that hit
  the target) and reports how much more often GPAP's trajectories succeed.

## Checking soundness

Every sample satisfies the formula by construction. If you want proof, `--verify-sound`
re-checks every emitted sample on the GPU. The stats report:

| field | meaning |
|---|---|
| `sound_check` | `ok`, `UNSOUND`, or `off` |
| `unsound_rows` | samples that failed the check |
| `sound_rows_checked` | samples checked |

`off` covers both "not requested" and "requested, but the run produced no sample to check";
`sound_rows_checked` tells them apart. The check's time is reported separately
(`phases_diag.verify_sound`) and never counted in a rate.

## Reading the stats

| field | meaning |
|---|---|
| `n_models` | samples the run delivered |
| `samples_per_s` | the steady-state rate of the winning configuration: sampling only |
| `sps_e2e_total` | the end-to-end rate: everything from reading the file to the last sample |
| `success_rate` | the fraction of trajectories that ended in a solution |
| `phases` | where the time went; the phases add up to `wall_total_s` |
| `phases_diag` | time spent on optional self-checks, kept out of every rate |
| `mined` | clauses found by mining, whether they were kept, and the measured gain |
| `stop`, `chunk_capped` | why the run ended, and whether a limit cut it short |
| `engine` | the kernel, the batch size and the packed layout the engine chose |

Use the end-to-end rate to answer "how long until I have N samples". Use the steady-state rate
to compare sampling engines, together with the other tool's equivalent number; see
[BENCHMARKING.md](BENCHMARKING.md) for how to compare fairly.

## Uniformity

The library does not claim that its samples are uniformly distributed. Testing that needs either
the full solution set or a statistical test such as Barbarik, and both live in the evaluation
layer (`eval/uniformity/`). What we measured, and how, is in
[findings/UNIFORMITY_FINDINGS.md](findings/UNIFORMITY_FINDINGS.md) §4b.

`--allow-soft` adds a configuration that succeeds more often on hard circuits, but it skews the
distribution and fails the uniformity test. Use it only when you need many solutions fast and
their distribution does not matter.

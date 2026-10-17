# TACAS 2027 — *SAT Sampling at Scale via GPU-based Boolean Constraint Propagation*

> **Reader:** readers and reviewers of the TACAS 2027 paper — what its evaluation measures, the rules its tables enforce, and how the paper's numbers are held to the data.

Regular tool paper. Artifact evaluation is **mandatory** for this track.

**The paper lives outside this repository and stands on its own.** No `.tex` section is kept here
and no generated table is committed. This file is the other half: what the evaluation measures,
which rules the generator enforces, and what holds the paper and the data to each other. The
paper's own argument — why any of it matters — is the paper's job.

| | |
|---|---|
| the rows, tables, figures | shipped with the artifact as `artifact-data/paper/`; `eval/artifact/reproduce.sh tables` regenerates them |
| the protocol, in full | [`PROTOCOL.md`](PROTOCOL.md) |
| the evidence behind every settled number | [`docs/findings/`](../../findings/) |
| the artifact's tiers and costs | [`ARTIFACT.md`](ARTIFACT.md) |
| what the PDF prints | [`published_numbers.json`](published_numbers.json) |

The protocol is **frozen** (v1.0, 2026-09-06): a review after the freeze fixes bugs and prose,
never a metric, configuration, screen, budget or cell set. Proposals go in
[`RIVAL_FAIRNESS_FINDINGS.md`](../../findings/RIVAL_FAIRNESS_FINDINGS.md) §17.

## What is measured

The benchmarks are **the rivals' own published sets**, whole — one population per row, never a
subset we chose:

| set | instances | depth | its gradient sampler |
|---|---|---|---|
| ISCAS-85 | 11 combinational | T=1 | DEMOTIC |
| Sequential (ITC'99 + ISCAS-89) | 30 | T=25, and T=50 as an appendix | Recurrent-CircuitSAT |

Each circuit is pinned to **two targets**: the *orig* target fixes the same few output bits the
set's own gradient sampler fixes in its released experiments, and **our rare target** fixes every
last-cycle output to the rarest pattern observed over 262,144 random input sequences. The rare
target is what makes the comparison mean anything — the orig targets turn out to be ones random
simulation already hits often, so no sampler is really needed there.

That is **142 cells** = (circuit, target, depth). One of them, `b19` at T=25, is an encoder
refusal: `abc` itself crashes on it (`aigDup.c:602`). It is reported as a refusal, never a zero.

**Three tools on one constraint, one row.** Every cell runs the rival on the ABC-simplified
netlist, GPAP on the `abc:fast` CNF of that same circuit and pins, and CMSGen on that very CNF
file. So Table 1 and Table 2 are the same cells from the same files, screened once — they cannot
disagree about which instances count.

Budgets: 1,200 s for a gradient sampler's whole run (model build included — twenty times ours,
since building and initialising their model over a deep unroll can take minutes), and 60 s of
sampling for GPAP and CMSGen, each asked for 50,000 samples. GPAP runs three seeds and the median
is published.

### The two clocks, and who gets which

*Steady state* times the sampling run alone; *end to end* times GPAP's whole run from parsing to
the last sample. Against CMSGen we report both — and CMSGen gets a single column, because it
starts sampling as soon as it has read the formula, so its two rates are equal to the printed
precision. Against the gradient samplers we report **steady state only**: their published rates
exclude model build and initialisation, so GPAP's figure excludes its preparation the same way.
That is the one comparison the rivals themselves define.

**Table 2 therefore has no end-to-end column, by rule.** An end-to-end ratio across an in-process
sampler and a subprocess gradient tool had four different definitions in three review rounds
(0.8×, 7.6×, 20.9×/132.0× and 11.8×/68.8× on the same rows), and on DEMOTIC's cells both tools
finish the real work in under a second, so it measured process start-up. The end-to-end comparison
is Table 1's, where both walls are the same kind of thing.

What counts also differs: against CMSGen, every sample, for both tools. Against the gradient
samplers, **unique** samples only, for both tools, because that is the rate their papers report.
Duplicates are removed after the timed run. No tool is timed while writing its samples out —
re-derived from the rivals' own source by `eval/rivals/check_emission_symmetry.py`.

The **first run of every process is discarded** before anything is measured: a cold CUDA context
costs the session's first instance about 3.3×, and whichever cell happened to run first would
otherwise carry it.

## The screens — a cell that measures the *instance* is not evidence about a *sampler*

Each of these, left implicit, produces a number a referee would reject. They live in exactly one
place, `eval/artifact/paper_tables.py`, and every consumer — the tables, the figures, the
publishability gate, the slide tables — *imports* them rather than restating them, so a table and
a figure cannot disagree about which cells were screened.

| screen | what it catches | what it costs us |
|---|---|---|
| **vacuous pin** | the target holds for every input, so pinning it is a tautology and every sampler scores ~100% — read per row from `p_accept`, not by name, because vacuity depends on the depth | at T=25 `b02`, `b05`, `b07`, `s526` leave the sequential geomeans; at T=50 only `b05`, `s526` and `b07` under its own pins do |
| **free support** | propagation alone fixes the support, so sampling needs no search — detected per row from our own counters, not by name | `c432` at a rare pin was once an entire geomean at 89,786× |
| **tiny space** | fewer than 1,000 solutions: both tools enumerate rather than sample, and the ratio is a race between two start-up costs | drops a 45,520× cell from Table 2 while *returning* 1,348× → 1,620× to Table 1 |
| **proved UNSAT at this depth** | neither tool can produce a solution | — |
| **encoder refusal** | a reach limit of ours, never a zero | `b19` at T=25 |

Three of those are one principle wearing three hats. Screening happens **before** the zero ledger,
so a degenerate cell cannot also be booked as a win.

### The zero ledger is symmetric, and the columns add up

CMSGen-zero → `ratio = inf`; GPAP-zero → `0.0`; both-zero → `None`. The last two are falsy, and
branching on `ratio` alone once showed 38 of the rival's zeros as wins while hiding all of ours.

```
Table 1:   inst  = n + CMSGen=0 + GPAP=0 + excluded
Table 2:   cells = n + rival=0  + GPAP=0 + rival failed + excluded
```

`n` is what the geomean stands on. `excluded` is **one** column holding every cell without a ratio
for any other reason, itemised by instance name under each table. `wins` is counted over exactly
the geomean's cells plus the two zero columns. Both identities are pinned by
`tests/experiments/test_paper_tables.py`, on the emitted file as well as the dicts behind it.

Further rules:

- **Absent is not zero.** A row whose CMSGen arm never ran, or whose subprocess crashed, is
  screened per row and named. `verify_artifact.py` refuses a stage that mixes armed and armless
  rows.
- **A tool that scored anything is scored.** There is no thin-denominator floor. Cells resting on
  fewer than `THIN_DENOMINATOR = 30` events on a side are *named* in the caveats with their count,
  and a geomean standing on fewer than three cells is flagged — quote the win count, not the ratio.
- **No best-of-encoders selection.** One row per (set, encoder). Picking the encoder with the
  highest ratio would let the rival's number choose which of our runs gets published.
- **A GPAP/CMSGen ratio without its CPU is meaningless.** CMSGen is single-core, and this
  campaign's Xeon is 0.41× the reference laptop's CPU. Every row records its CPU.

### Why the tables can print absolute rates

Tables 1 and 2 print each side's **absolute rate geomean** where the generator also computes their
ratio. These are one claim, not two: a geomean of quotients *is* the quotient of the geomeans, so
`ours / theirs` reproduces the ratio exactly, over exactly the same cells. `rate_gm_pair()` is the
single computation, and the identity is pinned by a test. A row prints one `n` and two rate
columns; if a row ever carried no end-to-end rate the two columns would cover different cells, and
the generator says so rather than leaving it to the reader.

### Table 3 answers the obvious objection

Table 1's speedups span an order of magnitude, which invites *"you picked your benchmarks."*
Table 3 answers with GPAP's near-constant implication rate against CMSGen's own `--statsevery`
counters, and the identity

```
samples/s ratio = (implications/s ratio) / (implications-per-sample ratio)
```

closing on every row. That identity is **algebraic**, not a validation: defining work-per-sample
as the quotient of the two rates makes it true by construction. It can only fail if an aggregation
mixes row sets — which is exactly the per-column-filtering bug it is there to catch, and all it
should be claimed to catch.

### Rival configuration is a measurement, not a citation

Knobs come from the rival or from a stated rule, never from us, and nothing is swept. Each rival
runs at its **shipped script's** setting, which is also the better of its own published values on
this card. Its batch is a share of the machine, so it is *granted*: the largest rung of a ladder
walked down from 30,000,000 that it runs to completion, rejected only on feasibility, never on how
fast it was. The grant is part of the record
(`eval/rivals/grants/l4_protocol_20260904.json`), and `--grants` pins a re-run.

The full policy, every value measured, and the considered-and-not-adopted list:
[`RIVAL_FAIRNESS_FINDINGS.md`](../../findings/RIVAL_FAIRNESS_FINDINGS.md) §16–§17.

### Uniformity — what a Barbarik verdict does and does not say

A separate, smaller campaign on its own host, as the text states: a CM_test verdict is a property
of the sampler's distribution, not of the card. Parameters are CMSGen's own certification setting
(ε=0.3, η=1.8, δ=0.1), drawing 1,836 samples from the sampler under test per instance (1,872 in
all — the one reference draw per round is the difference).

The guarantee is **asymmetric**: a Reject is unconditional; an Accept means *not caught*, never
"proven uniform". "No verdict" — the sampler could not fill an order Barbarik placed — bounds
coverage, not uniformity. Eligibility buckets on the **negative control's** verdict: SPUR, placed
under test itself, must Accept. Details and both controls:
[`UNIFORMITY_FINDINGS.md`](../../findings/UNIFORMITY_FINDINGS.md).

## The campaign these numbers come from

**Campaign `l4_protocol_20261012`**, measured 2026-10-12/13 on two Colab **NVIDIA L4**
sessions (host CPU Intel Xeon @ 2.20 GHz, recorded per row), from tag `paper_campaign_v2_run`
(e821180), engine `libgpap 0.3.0+318406dfa9dc3199`: the build id every row carries. It re-ran
protocol v1.0 after three fixes: Recurrent's in-loop host copy is deducted from its rate (§17c),
GPAP's early give-up on zero yield is off, and the vacuous screen reads the row (§17d). Every
rival rung and Recurrent `start_point` is pinned from `eval/rivals/grants/l4_protocol_20260904.json`,
and so are the 28 recorded rival initialization failures (marked `not_remeasured`, named by the
gate). Against 2026-09-04 every measured rival unique count is identical, and re-extracting grants
from the new rows reproduces the grants file exactly (`RIVAL_FAIRNESS_FINDINGS.md` §17e).

The uniformity rows are a separate run, campaign `uniformity_20261013` (RTX 4080, e821180),
on the 68 instances of the earlier campaign: every arm 65/65.

It supersedes `l4_protocol_20260904` (kept, with the data its tables were made from),
which in turn superseded `l4_protocol_20260903`, whose numbers rested on four defects that each
moved a number in our favour (§16, §16b, §16c).

The data the paper's numbers come from ships with the artifact as `artifact-data/paper/`: the stage
and calibration JSONs plus the Barbarik rows under `data/`, with a sha256 manifest. The manifest is what makes
a regenerated table one that can only have come from the frozen rows — `reproduce.sh tables`
verifies it before generating anything.

## Regenerating, and the contract with the paper

```bash
eval/artifact/reproduce.sh tables --out DIR     # every table and figure, from the shipped rows
```

It needs no GPU, no CUDA wheels and not even numpy: a stock `python3` runs it, which is what the
TACAS evaluation VM provides.

What comes out:

| | |
|---|---|
| `RESULTS.md` | every table, the ledger, the exclusions by name, and the caveats |
| `paper_numbers.json` | every number the paper prints, machine-readable |
| `PAPER_NUMBERS.txt` | that file checked against what the PDF prints |
| `VERIFICATION.txt` | the publishability gate |
| `emission_audit.txt` | no tool was timed while writing samples out |
| `plots/` | the figures, drawn through the tables' own cell classifiers |

### Numbers, not shapes

The paper typesets its own tables — it drops columns, renames rows and reorders them to fit a
two-column LNCS page — so nothing can byte-compare them against generated LaTeX, and for a while
nothing checked them at all: Tables 1 and 2 printed absolute rate geomeans the generator did not
even compute.

```
rows ──► paper_tables.py --out-json ──► paper_numbers.json
                                             ↕  check_paper_numbers.py
PDF  ──► (hand transcription)       ──► published_numbers.json
```

`published_numbers.json` records what the PDF prints and names the generated row each value
checks. **Shapes may differ; numbers may not.** A rate is compared at its own printed precision —
`2.31e6` is a claim about three significant figures — and a count exactly, because a count has no
precision to round to.

**Reading a failure.** The published file changes only when the PDF does:

- failed right after editing the paper's numbers → the transcription is wrong, or the number in
  the paper is;
- failed with that file untouched → **the data moved under the paper.** Find out which row moved
  before touching anything. A value re-transcribed to make the check pass is a number the campaign
  never produced.

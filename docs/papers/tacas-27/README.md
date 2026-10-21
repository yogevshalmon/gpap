# TACAS 2027 — *SAT Sampling at Scale via GPU-based Boolean Constraint Propagation*

> **Reader:** readers and reviewers of the TACAS 2027 paper — the details of its evaluation that
> did not fit in the paper, the rules its tables follow, and how its numbers are checked against
> the measurements.

This page accompanies the paper's evaluation. It describes what was compared and how, which
cells each table counts and why, and how the numbers the paper prints are checked against the
data.

| file | what it holds |
|---|---|
| this page | the evaluation, explained |
| [PROTOCOL.md](PROTOCOL.md) | the exact rules of the evaluation, frozen before the paper's campaign ran |
| [ARTIFACT.md](ARTIFACT.md) | the artifact: how to regenerate every table without a GPU, and re-measure on one |
| [published_numbers.json](published_numbers.json) | the numbers the paper prints from its measurements, each checked against the data |

## What is compared

GPAP is compared with three samplers, on exactly the same constraints:

- **CMSGen**, the standard CPU sampler built on a SAT solver;
- **DEMOTIC** and **Recurrent-CircuitSAT**, two GPU samplers that search by gradient descent.

A third GPU sampler, HTS, is cited but not compared: its own paper evaluates no circuit
netlists, so it has no benchmarks of the kind the other two publish.

**The benchmarks come from the gradient samplers' own papers.** DEMOTIC's are its 11 ISCAS-85
circuits. Recurrent-CircuitSAT evaluates circuits of two families, ITC'99 and ISCAS-89; the
benchmark set holds 30 of them, all 21 ITC'99 circuits and 9 ISCAS-89 circuits. They include the
16 its paper reports, and `s35932`, which we added. The tables report each whole set, never a
subset.

| set | circuits | unrolled to | taken from |
|---|---|---|---|
| ISCAS-85 | 11 combinational | T=1 | DEMOTIC's paper |
| ITC'99 and ISCAS-89 | 30 sequential | T=25, and T=50 beside it | Recurrent-CircuitSAT's paper |

**Each circuit gets two targets.** The *original* target fixes the output bits that the sampler
whose paper the set comes from fixes in its own experiments. *Our rare target* fixes every output
of the last cycle to the rarest pattern seen in 262,144 random input sequences. The rare target
is the one that needs a sampler: random simulation already hits most original targets often.

A circuit, a target and a depth make a **cell**; there are 142. One of them, `b19` at T=25 under
Recurrent-CircuitSAT's original target, is refused by GPAP's encoder because ABC crashes on it;
it is reported as a refusal, never as a zero.

**Three tools, one constraint, one row.** In every cell, the gradient sampler runs on the circuit
after ABC's sequential simplification. GPAP encodes that same simplified circuit, with one more
ABC pass after unrolling it (the `abc:fast` encoder). CMSGen runs on GPAP's CNF file itself.
Tables 1 and 2 therefore come from the same cells and files, screened once, and cannot disagree
about which instances count.

**Budgets and runs.**
- GPAP and CMSGen each get 60 s of sampling and are asked for 50,000 samples. GPAP runs three
  seeds and the median is reported.
- A gradient sampler gets 1,200 s for its whole run, model building included, because building
  its model over a deep unrolling can take minutes. Recurrent-CircuitSAT also gets a separate
  1,200 s to find its starting point. A sampler that does not initialize or finish within its
  budget is counted as a failure (the `rival failed` column), never as a zero.
- A gradient sampler runs once, at its own default seed, with the settings of its authors'
  shipped script. These are also the better of the settings its paper reports, on this GPU: the
  paper's settings were measured beside them, and `RESULTS.md` prints that measurement under
  Table 2.
- A gradient sampler runs with the largest batch it completes: the largest rung of a ladder walked
  down from 30,000,000 (30,000,000, 10,000,000, 3,000,000, 1,000,000, and so on), a rung rejected
  only on feasibility, never on how fast the tool ran on it.

## How rates are measured

**Two clocks.** *Steady state* times the sampling alone; *end to end* times GPAP's whole run, from
reading the file to the last sample.

- Against CMSGen the paper reports both. CMSGen gets one column, because it starts sampling as
  soon as it has read the formula, so its two rates agree to the printed precision.
- Against the gradient samplers it reports steady state only. Their published rates exclude
  building and initializing their model, so GPAP's excludes its preparation the same way. Table 2
  has no end-to-end column on purpose: across an in-process sampler and a separate gradient tool,
  an end-to-end ratio mostly measures process start-up, and on DEMOTIC's cells both tools finish
  the real work in under a second.

**What is counted.** Against CMSGen, every sample, for both tools. Against the gradient samplers,
unique samples only, for both tools, because that is the rate their papers report. Duplicates are
removed after the timed run.

**What is not timed.** No tool is timed while writing its samples out;
`eval/rivals/check_emission_symmetry.py` checks this against each tool's source.
Recurrent-CircuitSAT copies its candidates to the host inside its timed loop, so that copy is
timed separately and deducted, as GPAP's delivery of samples is excluded. The first run of every
process is discarded: a cold CUDA context makes it about 3.3 times slower, and whichever cell ran
first would carry that cost.

**The host CPU matters.** CMSGen runs on one CPU core, so a GPAP/CMSGen ratio depends on the host
as well as the GPU. Every row records its CPU.

## Which cells count

Some cells measure the instance rather than the sampler, and would produce a ratio no reader
should trust. The tables screen them out first. The screens live in one place,
`eval/artifact/paper_tables.py`, which the tables, the figures and the artifact's checks all
import, so a table and a figure cannot disagree about which cells count.

| screen | what it catches |
|---|---|
| **vacuous target** | the target holds for every input, so every sampler succeeds trivially; decided from each row's own acceptance rate, since it depends on the depth |
| **free support** | after propagation, every assignment to the sampled variables is a solution, so there is nothing to search |
| **tiny space** | fewer than 1,000 solutions: both tools enumerate rather than sample, and the ratio compares start-up costs |
| **proved UNSAT** | the cell has no solution at this depth, so neither tool can produce one |
| **encoder refusal** | GPAP could not encode the cell; a limit of ours, never counted as a zero |

`RESULTS.md`, which the artifact generates, names every screened cell under its table.

**The columns add up.** Screening happens before anything is counted, so a screened cell cannot
also count as a win. Every cell lands in exactly one column:

```
Table 1:   instances = n + CMSGen=0 + GPAP=0 + excluded
Table 2:   cells     = n + rival=0  + GPAP=0 + rival failed + excluded
```

`n` is the number of cells the geometric mean is taken over. `excluded` holds the screened cells
and every other cell without a ratio, each named under its table. Wins are counted over the `n`
cells and the two zero columns. Both identities are checked by
`tests/experiments/test_paper_tables.py`.

Further rules:

- **Absent is not zero.** A cell where a tool never ran, or crashed, is named and kept out of the
  ratio, not scored as a zero.
- **A tool that scored anything is scored.** Cells resting on fewer than 30 events on one side are
  named in the caveats with their count, and a geometric mean over fewer than three cells is
  flagged: quote its win count, not its ratio.
- **A tie is neither a win nor a loss.** In Table 2, a cell whose ratio would cross 1 within the
  spread of GPAP's three seeds keeps its ratio but counts as a tie.
- **No best-of-encoders selection.** Each row uses one GPAP encoding; reporting the better of two
  would let the result choose which of our runs is published.

**Why the tables can print absolute rates.** The paper's Tables 1 and 2 print each side's
geometric-mean rate, and its text quotes their ratio. These are one claim, not two: over the same
cells, the geometric mean of the ratios equals the ratio of the geometric means. The generator
computes both from one function, and a test checks the identity.

**Table 3 answers "you picked your benchmarks".** Table 1's speedups span an order of magnitude.
Table 3 shows that GPAP's implication rate varies little from row to row, with CMSGen's read from
its own counters, so the spread comes from how much work each sample costs, through the identity

```
samples/s ratio = (implications/s ratio) / (implications-per-sample ratio)
```

The identity holds by construction, so it proves no claim by itself. What it does catch is a
table that mixes the cells of different rows.

## Uniformity

Uniformity is tested in a separate, smaller run on its own machine: a Barbarik verdict is a
property of the sampler's distribution, not of the GPU. Barbarik runs with CMSGen's own
certification parameters (ε=0.3, η=1.8, δ=0.1) and draws 1,836 samples per instance from the
sampler under test.

The guarantee is asymmetric. A Reject assumes nothing about the sampler but holds with
confidence 1−δ, here 90%, so even a uniform sampler is sometimes rejected. An Accept means the
test did not catch any bias, never that the sampler is proven uniform. "No verdict" means the
sampler could not deliver what the test asked for, which says something about its coverage and
nothing about its uniformity. An instance counts only if SPUR, a sampler known to be uniform, is
accepted on it; of the 68 instances run, 65 are eligible.
[UNIFORMITY_FINDINGS.md](../../findings/UNIFORMITY_FINDINGS.md) describes both controls.

## How the paper's numbers are checked

The paper typesets its own tables to fit the page: it drops columns, renames rows and reorders
them. So nothing compares the tables' layout with the generated ones. Their numbers are compared
instead:

```
rows ──► paper_tables.py --out-json ──► paper_numbers.json
                                             ↕  check_paper_numbers.py
PDF  ──► (transcribed by hand)      ──► published_numbers.json
```

[published_numbers.json](published_numbers.json) records the numbers the PDF prints from the
measurements and names the generated row each one checks; the few printed numbers that are not
generated are listed apart, with the reason. A rate is compared at the precision printed
(`2.31e6` claims three significant figures), and a count exactly. The check runs whenever the
tables are regenerated (`eval/artifact/reproduce.sh tables`; see [ARTIFACT.md](ARTIFACT.md)).

## Where the numbers come from

| data | campaign | where it ran | code |
|---|---|---|---|
| Tables 1 to 3, and Figure 3 | `l4_protocol_20261012`, 2026-10-12/13 | two Colab NVIDIA L4 sessions, Intel Xeon @ 2.20 GHz | tag `paper_campaign_v2_run` |
| the uniformity results (Sect. 5.4; Table 4b of `RESULTS.md`) | `uniformity_20261013`, 68 instances | an RTX 4080 Laptop GPU | tag `paper_campaign_v2_run` |

The campaign follows [PROTOCOL.md](PROTOCOL.md), with two bug fixes and one change of metric
made after the freeze:
- Recurrent-CircuitSAT's copy to the host is deducted from its rate. This is the change of
  metric, and the protocol's text was amended for it
  ([RIVAL_FAIRNESS_FINDINGS.md](../../findings/RIVAL_FAIRNESS_FINDINGS.md) §17c).
- The vacuous-target screen reads each row's own acceptance rate (the same file, §17d).
- GPAP no longer gives up early when it has found nothing
  ([UNIFORMITY_FINDINGS.md](../../findings/UNIFORMITY_FINDINGS.md) §4b).

Two shortcuts are recorded in every row they affect. Each rival's batch size, and
Recurrent-CircuitSAT's starting point, are taken from an earlier campaign's record instead of
searched for again. The 28 recorded failures of a rival to initialize were carried over instead
of re-measured; this is the one departure from the protocol's text. [ARTIFACT.md](ARTIFACT.md)
explains both. The campaigns this one replaces, and why, are in
[RIVAL_FAIRNESS_FINDINGS.md](../../findings/RIVAL_FAIRNESS_FINDINGS.md) §16 to §17e.

The rows ship with the artifact, with a hash of every file that is checked before any table is
generated.

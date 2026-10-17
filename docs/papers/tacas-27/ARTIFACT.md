# GPAP — artifact for the TACAS 2027 tool paper

> **Reader:** artifact evaluators of the TACAS 2027 tool paper *SAT Sampling at Scale via
> GPU-based Boolean Constraint Propagation* — what to run, what it should print, and how long
> it takes. No knowledge of the code is assumed.

GPAP samples solutions of SAT formulas and circuits on an NVIDIA GPU. This artifact contains the
tool's source, every per-instance measurement behind the paper's evaluation, and one script,
`eval/artifact/reproduce.sh`, that regenerates the paper's tables from those measurements and
re-measures them on a GPU. The measurements are those of the paper's campaign,
`l4_protocol_20261012`, run on two Colab NVIDIA L4 sessions on 2026-10-12/13
([the paper's evaluation notes](README.md) describe it).

- **Archive:** **TODO** the Zenodo DOI, and the release `v0.3.0` (commit `<sha>`) it was built from.
- **Badges applied for:** Available, Functional and Reusable.
- **Licence:** MIT ([LICENSE](../../../LICENSE)). Benchmarks and tools by others keep their own licences
  ([THIRD_PARTY.md](../../../THIRD_PARTY.md)).

## In short

| step | where it runs | what it needs | time | what it shows |
|---|---|---|---|---|
| `check` | the TACAS VM | nothing | seconds | what this machine can run |
| `tables` | the TACAS VM | nothing: offline, the stock `python3` | seconds | every table regenerated from the recorded measurements, and the paper's numbers checked against them |
| `setup`, then `smoke` | a Linux machine with an NVIDIA GPU | network access, about 10 GB of disk | 10 to 30 min, with the installation | the whole pipeline end to end on 4 cells, compared with the published ones |
| `subset` | the same | the same | about 2.9 h on the reference L4 | 66 of the 142 cells re-measured |
| `full` | the same | the same | about 13 h on the reference L4 | all 142 cells re-measured |

Each step prints what it is about to do, and what it will cost, before it starts.

**GPAP needs an NVIDIA GPU, and the TACAS VM has none.** The tool has no CPU fallback: running
propagation on the GPU is what the paper is about. In the VM, the artifact does everything that
needs no GPU. It regenerates every table from the measurements of the paper's campaign, checks
the 163 numbers the paper prints from them, and re-checks the soundness verdict recorded in
every row. Running the tool, and with it re-measuring the rates and re-checking that every sample
is a solution, needs a machine with an NVIDIA GPU
([Re-measuring on a GPU](#re-measuring-on-a-gpu)).

**TODO** the AE chairs' answer on the special-hardware exemption.

## Early light review (about 5 minutes, in the VM)

**1. Unpack and verify the archive** (the `.tar.gz` file at the DOI above).

```bash
tar xzf gpap-artifact-*.tar.gz
cd gpap-artifact-*/
sha256sum -c --quiet MANIFEST.sha256 && echo "archive intact"
```

From here on, run every command from the root of the unpacked archive.

**2. Ask what this machine can run.**

```bash
eval/artifact/reproduce.sh check
```

In the VM, it reports torch and the GPU tools as missing; that is expected. The line that matters:

```
torch is not installed — only the `tables` step can run here
```

It also prints how long each GPU tier would take on the reference card.

**3. Regenerate the tables.**

```bash
eval/artifact/reproduce.sh tables --out vm-tables
```

Among its output are these lines:

```
141 cells | 0 skipped | 1 errors | 0 unsound
163 published values, all reproduced by the generator.
The paper's tables and the campaign agree.
27 ok | 8 warnings | 0 failures
Publishable, PROVIDED every warning above is stated in the text.
```

If you see them, the artifact works. What they mean:
- *1 errors*: of the 142 cells, one is a refusal, not a measurement. GPAP's optional encoder,
  ABC, crashes on `b19` ([Known deviations](#known-deviations)); the paper reports it as a
  failure.
- *8 warnings*: the rival failures the paper reports, and the failures carried over from an
  earlier measurement instead of re-measured
  ([Two shortcuts](#two-shortcuts-both-recorded-in-every-row)). Each warning names what the
  paper's text has to say about it.

Two more things in this output are expected in the VM:
- *figures skipped*: drawing Figure 3 needs `matplotlib`, which the VM lacks. The published
  figure is in `artifact-data/paper/plots/sps_scatter_cmsgen.pdf`. With network access,
  `pip install matplotlib` and a second `tables` run draws it again. With matplotlib 3.10.8 and
  numpy 2.4.4, on Python 3.11 or later, the file is byte-identical to the paper's; another
  version draws the same points with different pixels.
- *Section 1 did NOT run* in the emission audit: that section reads the rival samplers' source
  code, which only `setup` fetches. The full audit, from a machine that had it, ships as
  `artifact-data/paper/emission_audit.txt`.

**4. If you have a machine with an NVIDIA GPU**, the `setup` and `smoke` steps of
[Re-measuring on a GPU](#re-measuring-on-a-gpu) run the tool itself, in 10 to 30 minutes.

## Full review

### The paper's claims, from the recorded measurements (in the VM)

`tables` writes these files:

| file | what it is |
|---|---|
| `RESULTS.md` | every table, with the list of excluded cells and the caveats behind each table |
| `paper_numbers.json` | the numbers the paper prints, computed from the rows |
| `PAPER_NUMBERS.txt` | those numbers compared with what the submitted PDF prints |
| `VERIFICATION.txt` | the publishability gate: soundness, and every caveat the text must state |
| `emission_audit.txt` | checks that no tool was timed while writing its samples out |
| `plots/` | the paper's Figure 3 and the other scatter plots, when `matplotlib` is installed |

The same files, as generated when the archive was built, are in `artifact-data/paper/`.
Comparing the two shows what the VM reproduced:

```bash
diff vm-tables/RESULTS.md artifact-data/paper/RESULTS.md
```

At most the date in the header differs.

Each claim of the paper, and where to check it:

| the paper says | where | check it in | made by | from |
|---|---|---|---|---|
| GPAP reaches 83.4–1,571× CMSGen's throughput | Table 1 | `RESULTS.md`, Table 1 | `paper_tables.py --rivals` | `artifact-data/paper/data/rivals_*.json` |
| 2.2–18.7× the GPU samplers at their own targets; far more at ours | Table 2 | `RESULTS.md`, Table 2 | the same | the same rows |
| the speedup is the cost of rejection, not engine speed | Table 3 | `RESULTS.md`, Table 3 | the same | the same rows as Table 1 |
| Barbarik accepts GPAP on all 65 eligible instances, as it does CMSGen | Sect. Sample Uniformity | `RESULTS.md`, Table 4b | `paper_tables.py --barbarik` | `artifact-data/paper/data/barbarik/*.jsonl` |
| 93 of 110 instances delivered all 50,000 samples, CMSGen 44 | the evaluation's text | `RESULTS.md`, "Numbers the paper's text quotes" | `paper_tables.py --out-json` | the same rows as Table 1 |
| every emitted sample is a solution | the evaluation's text | `VERIFICATION.txt` | `verify_artifact.py`, over the check each row recorded | each row's soundness check and circuit replay |
| the numbers above are what the PDF prints | — | `PAPER_NUMBERS.txt` | `check_paper_numbers.py` | `paper_numbers.json` and `docs/papers/tacas-27/published_numbers.json` |

The scripts are in `eval/artifact/`. The paper typesets its own tables, so their layout differs
from `RESULTS.md`; their numbers may not. `docs/papers/tacas-27/published_numbers.json` records
the 163 numbers the PDF prints from the data, each with the generated value it must equal at the
precision printed, and `tables` fails if any differs. It also lists the few printed numbers that
are not generated, and why.

The rules deciding which cells a table counts are in one file, `eval/artifact/paper_tables.py`,
which the tables, the figures and the gate all import.
[The paper's evaluation notes](README.md) explain each rule, and [PROTOCOL.md](PROTOCOL.md)
states the protocol in full.

### Re-measuring on a GPU

**What you need:**
- Linux on x86-64, with an NVIDIA GPU of compute capability 8.0 or newer (Ampere or later) and
  its driver. GPAP has run on an RTX 4080, an L4 and an A100.
- A driver for CUDA 12.8 or later: `nvidia-smi` shows `CUDA Version: 12.8` or higher. With an
  older driver that supports CUDA 12.6, install the same torch built for 12.6 instead (below).
- cmake, a C++17 compiler, git, and Python 3.10 or later with `venv`. No CUDA toolkit: the
  kernels are compiled by Triton, and the library links only against the driver. (torch's
  wheels bring CUDA runtime libraries as Python packages, one of them named `cuda-toolkit`;
  GPAP's build uses none of them.)
- Network access, to install the Python packages and fetch the tools GPAP is compared with.
- About 10 GB of disk, most of it PyTorch. The reference host had 12 vCPUs (Intel Xeon @
  2.20 GHz) and about 55 GB of free memory; `smoke` also ran on a laptop with 47 GB.
- 24 GB of GPU memory runs every cell at the batch size the paper recorded. A smaller card
  works, with a difference the run reports ([Hardware](#hardware)).

**Steps.** From the unpacked archive:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r eval/artifact/requirements-artifact.txt && pip install -e .
eval/artifact/reproduce.sh setup     # builds the engine, ABC and CMSGen; fetches the GPU rivals
eval/artifact/reproduce.sh smoke     # 4 cells, a few minutes
eval/artifact/reproduce.sh subset    # 66 cells, about 2.9 h on the reference L4
eval/artifact/reproduce.sh full      # all 142 cells, about 13 h on the reference L4
```

On our RTX 4080 machine the installation took 3 minutes, `setup` 2 and `smoke` 3; on a slower
machine or network, allow 30 minutes for the first four lines.

For a driver that supports only CUDA 12.6, replace the `pip install` line with:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
grep -v '^torch==' eval/artifact/requirements-artifact.txt > /tmp/req.txt
pip install -r /tmp/req.txt && pip install -e .
```

`requirements-artifact.txt` pins the three packages that can move a number: torch, the Triton
that compiles the kernels, and the SAT solver that decides the `unsat` screen. If you would
rather not activate the environment, pass `PYTHON=/path/to/python` to each step. In the same
environment, `pytest tests -m "not gpu and not slow"` runs the tool's own test suite in seconds.

**What a GPU tier prints.** Each tier writes to `.experiments/artifact/<gpu>-<tier>/`
(`--out DIR` chooses another), regenerates the tables from its own fresh measurements, and
compares them with the published ones, in `COMPARISON.txt` and `comparison.json`. Its last line
is `ARTIFACT_TIER_COMPLETE`, or `ARTIFACT_TIER_FAILED` followed by the steps that failed.
`smoke` also prints:

```
PASS — 4 cells compared: same formulas, sound samples, same verdicts.
```

`smoke` runs one cell of each of the four T=25 stages, so its two lines
`NOTE: rivals_..._t50: not in the fresh run` are expected.

A tier that is interrupted can be resumed: running the same step again skips finished cells.
`eval/artifact/reproduce.sh compare --out DIR --tier subset` repeats only the comparison over
an existing directory.

**What is compared.** A sampling rate is a property of the card: the same engine runs 1.82× as
fast on an A100 and 0.84× on an L4 as on an RTX 4080
([CROSS_GPU_FINDINGS.md](../../findings/CROSS_GPU_FINDINGS.md)). Asking for our rates back would
test the hardware. So the comparison checks what must hold on any machine, and reports the
rest:

| checked | how | a difference means |
|---|---|---|
| **the same formula in each cell** | `n_vars`, `n_clauses`, `support`, the encoder and `pin_check_ok` match exactly | the two runs did not measure the same thing, usually because of a different `abc` build |
| **soundness** | no emitted model falsifies a clause, and every kept sample replays through the original circuit | a real bug, against the paper's central claim |
| **the same verdicts** | which tool won each cell, and how each cell was classified | the conclusion did not survive the change of hardware |
| *rates* | printed as this card's speed relative to ours | nothing: it describes the machine |

The tier fails if a formula differs, if a sample is unsound, if fewer cells ran than the tier
sets out to run, or if any step fails. A changed verdict does not fail it; the `PASS` line then
says how many verdicts differ, and the report names them.

**What each tier costs.** These are estimates for the reference L4 session, built from the
recorded time of every cell plus a measured overhead the rows do not carry (start-up, the ABC
pass and its equivalence check, the soundness replay):

| tier | cells | on the reference L4 | what it covers |
|---|---|---|---|
| `tables` | — | seconds, no GPU | every published table and figure |
| `smoke` | 4 | about 4 min | one cell of each T=25 stage, run exactly as the real tiers run it |
| `subset` | 66 | about 2.9 h (2 h 50 min measured) | all 22 ISCAS-85 cells, and 22 of the 30 sequential instances at T=25 |
| `full` | 142 | about 12.9 h | every cell, at T=25 and T=50 |
| `full --remeasure-failures` | 142 | about 23.2 h | the same, re-measuring every recorded rival failure |

For another machine, `eval/artifact/estimate_wall.py artifact-data/paper/data` prints the same
table: `--scale` multiplies the GPU's part (below 1 for a card faster than the L4) and
`--cpu-scale` the overhead, which depends on the host CPU (1.9 for the reference host, 1 for the
laptop the overhead was measured on). `check` runs it with both at 1.

The `subset` tier's cells are chosen by cost, not by us: a sequential instance is kept when its
two T=25 tasks together take ten minutes or less on the reference card. `eval/artifact/subset.json`
holds the rule, the selection and the per-instance times. The comparison uses the published
numbers for the same cells, not the paper's averages over all 142.

### Reusing GPAP

GPAP is a library with a command-line tool on top; the paper's evaluation is one use of it. After
`setup`, on a GPU machine, in the activated environment:

```bash
gpap sample benchmarks/circuits/bench/seq/b10.bench --cycles 25 -n 10000 --out samples.txt --sim-check
gpap sample benchmarks/sampling/27.sk_3_32.cnf -n 10000 --out samples.txt
```

The first unrolls the circuit `b10` for 25 cycles, writes 10,000 input sequences that reach a
fixed output pattern, and replays each one through the circuit to confirm it. The second samples
a CNF formula.

| to | read |
|---|---|
| install and use GPAP | [the project README](../../../README.md) |
| every option of `gpap sample` | [docs/SAMPLING.md](../../SAMPLING.md) |
| call it from Python | [docs/PYTHON_API.md](../../PYTHON_API.md) |
| call it from C or C++ | [cpp/README.md](../../../cpp/README.md) |
| understand the engine | [docs/ENGINE.md](../../ENGINE.md) |
| run your own comparison fairly | [docs/BENCHMARKING.md](../../BENCHMARKING.md), [docs/EVAL_RUNBOOK.md](../../EVAL_RUNBOOK.md) |
| change it | [CONTRIBUTING.md](../../../CONTRIBUTING.md) |

## What is in the archive

| path | what it is |
|---|---|
| `README-ARTIFACT.md` | points to this guide, `docs/papers/tacas-27/ARTIFACT.md` |
| `docs/papers/tacas-27/` | this guide, the paper's evaluation notes and protocol, and the numbers the PDF prints |
| `MANIFEST.md`, `MANIFEST.sha256` | how the archive was built, and a hash of every file |
| `eval/artifact/reproduce.sh` | the one entry point: `check`, `tables`, `setup`, `smoke`, `subset`, `full`, `compare` |
| `artifact-data/paper/data/` | the per-instance measurements of the paper's campaign, with their own hashes |
| `artifact-data/paper/` | the tables, gate, audit and figures generated from them when the archive was built |
| `benchmarks/` | every benchmark the evaluation uses, including six ITC'99 circuits the repository fetches |
| `gpap/`, `cpp/` | the tool: the Python package and the C++ engine |
| `eval/` | the evaluation: campaign drivers, the harnesses for the compared tools, the table generator |
| `docs/` | the documentation; [docs/README.md](../../README.md) is its map |
| `tests/` | the test suite; it needs the GPU steps' environment |

The GPU samplers GPAP is compared with, DEMOTIC and Recurrent-CircuitSAT, state no licence, so
the archive does not include them; nor HTS, which `setup` also fetches but the paper's tables do
not use. `setup` fetches each at a pinned commit, and every result row records the commit that
ran. ABC and CMSGen are built from pinned sources.

`setup` changes the compared tools in three ways, none of which changes what they compute:
- CMSGen (`scripts/cmsgen_gpap_eval.patch`) gets `--nowrite`, which generates samples without
  writing them out, so its rate counts generation alone, as GPAP's does; and two options that
  report its progress and stop it at the time budget. All three are off unless the harness
  passes them.
- Recurrent-CircuitSAT (`eval/rivals/recurrent_gpap_eval.patch`) times its copy of every
  candidate to the host on its own, so the harness can deduct it from its rate.
- All three GPU samplers get a one-line fix that Python 3.13 needs: they `exec` generated code,
  which PEP 667 breaks. It is applied on every Python version, and their output is
  bit-identical with and without it on Python 3.12.

Barbarik and SPUR, which test uniformity, are not part of `setup`.
[THIRD_PARTY.md](../../../THIRD_PARTY.md) says where each component comes from, and its licence.

## Details

### Hardware

**24 GB of GPU memory runs every cell at the batch size the paper recorded.** Each rival's batch
size is the largest that ran on the reference L4; for nearly all of them, the next step up ran
out of memory. On a card with less memory, a cell may not run at its recorded batch. It then
steps down to the next smaller batch, which the row records (`grant_source`) and the comparison
counts and warns about, because a rival given a smaller batch than the paper gave it makes our
ratio look better. On our 12 GB card, DEMOTIC stepped down on `c432`. The run still completes.

The other direction is handled by `--vram-cap`, on by default. GPAP's own batch is limited by
memory on 63 of the campaign's 141 rows, so a larger card would give GPAP more parallelism while
the rival stays at its recorded batch, again in our favour. The default `--vram-cap l4` holds
GPAP's memory budget to the 17,280 MiB the L4 run had. It is a ceiling, so it changes nothing on
a smaller card. `--vram-cap none` measures the card as it is.

### Two shortcuts, both recorded in every row

**Each rival's batch size is taken from the record, not searched for again.** The protocol gives
each rival the largest batch it completes with, found by walking down a ladder from 30,000,000;
that search costs many runs per cell. `eval/rivals/grants/l4_protocol_20260904.json` records
the batch each cell was given, and Recurrent's settled `start_point`, so a reproduction pays for
neither. The file is named after the earlier campaign that measured it; the paper's campaign
reproduced it exactly. `--grants none` repeats the whole search.

**Recorded rival failures are carried over, as in the paper's own campaign.** 28 cells are
recorded as "the rival did not initialize within its 1,200 s budget", measured on the L4 in
September 2026. Re-measuring them costs that budget each, 10 of the 22 hours of a full
re-measurement, to produce rows that say "no measurement". The campaign the paper reports
carried them over, and so does this artifact: such a row is marked `not_remeasured`,
`VERIFICATION.txt` names it, and the comparison counts it as evidence for neither side. This can
only be unfair to the rival, since a faster card might initialize where the L4 could not.
`reproduce.sh full --remeasure-failures` re-measures them.

### Known deviations

- `b19` at T=25 under Recurrent's own targets fails in GPAP's encoder: ABC itself crashes on it
  (`aigDup.c:602`). It is one of the 142 cells, and is reported as a failure, never as zero.
- The uniformity rows come from a separate run on a different card, an RTX 4080, as the paper
  states: a Barbarik verdict is a property of the sampler, not of the GPU. `tables` regenerates
  the verdict counts from the shipped rows; running Barbarik again needs Barbarik and SPUR,
  installed by hand as `eval/uniformity/barbarik_gpap.py` describes.
- Rented cloud machines such as Colab reclaim the VM after about an hour. Every tier resumes
  where it stopped; [eval/colab/README.md](../../../eval/colab/README.md) describes how we ran the tiers
  on Colab.

### What has been run end to end

- `check` and `tables`, from an archive built like this one, in an offline Ubuntu 22.04
  container with only the stock `python3` 3.10 (no GPU, no numpy, no network): the archive
  verified, all 163 published values reproduced, the gate passed (2026-10-15). The same in
  Ubuntu 24.04 with Python 3.12 (2026-10-13).
- `setup`, then `smoke`, from the archive, in a fresh virtual environment built from
  `requirements-artifact.txt`, on an RTX 4080 Laptop GPU (12 GB): every tool built from its
  pinned source; 4 of 4 cells with the same formula, sound samples and the same verdicts.
  DEMOTIC stepped down on `c432` on this smaller card, as described above (2026-10-13).
- `subset`, from the archive, on a rented L4 ([ARTIFACT_FINDINGS.md](../../findings/ARTIFACT_FINDINGS.md)
  §5b): all 66 cells reach the same verdict against CMSGen, and all 58 cells whose rival side was
  measured rather than carried over reach the same verdict against the rival. Every count in
  Tables 1–3 equals the published one for the same cells, and every ratio is within 5.5%
  (2026-10-14).

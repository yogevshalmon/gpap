# GPAP — artifact for a tool paper submitted to TACAS 2027

> **Reader:** artifact evaluators of the tool paper submitted to TACAS 2027, *SAT Sampling at
> Scale via GPU-based Boolean Constraint Propagation* — what to run, what it should print, and
> how long it takes. No knowledge of the code is assumed.

The artifact is GPAP's source, every measurement behind the paper's evaluation (the campaign
`l4_protocol_20261012`, two Colab NVIDIA L4 sessions, 2026-10-12/13; [README.md](README.md)), and
a Docker image with all of it installed. One script, `eval/artifact/reproduce.sh`, regenerates
the paper's tables from the measurements and re-measures them on a GPU.

- **Archive:** [10.5281/zenodo.23241912](https://doi.org/10.5281/zenodo.23241912), built from the
  release `v0.3.1` of [the repository](https://github.com/yogevshalmon/gpap).
- **Badges applied for:** Available, Functional and Reusable.
- **Licence:** MIT ([LICENSE](../../../LICENSE)). Benchmarks and tools by others keep their own licences
  ([THIRD_PARTY.md](../../../THIRD_PARTY.md)).

GPAP runs only on NVIDIA GPUs, so the artifact is a Docker image; the AE chairs granted the
special-hardware exemption. Without a GPU, `check` and `tables` still run, in Docker or in the
TACAS VM ([Without Docker](#without-docker)).

## What you need

- Linux on x86-64, or Windows with WSL2.
- An NVIDIA GPU of compute capability 8.0 or newer (Ampere or later), with a driver for CUDA 12.8
  or later.
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  This must list the GPU:

  ```bash
  docker run --rm --gpus all ubuntu:24.04 nvidia-smi
  ```
- About 15 GB of disk.
- For the GPU steps, network access to github.com: the two GPU samplers the paper compares with
  state no licence, so the artifact does not include them, and each GPU step clones them first.

## Smoke test (about 10 minutes)

```bash
sha256sum -c gpap-image-*.tar.gz.sha256
docker load -i gpap-image-*.tar.gz
mkdir -p out
docker run --rm --gpus all gpap-artifact:tacas-27 check
docker run --rm -v "$PWD/out:/out" gpap-artifact:tacas-27 tables --out /out/tables
docker run --rm --gpus all -v "$PWD/out:/out" gpap-artifact:tacas-27 smoke --out /out/smoke
```

`tables` and `smoke` each write the one file to read, `SUMMARY.txt`, and everything else into
`details/` beside it. The results belong to the owner of `out/`, which is why it is created
first.

- `check` lists the GPU and what each step costs on the reference card.
- `tables` regenerates every table from the recorded measurements and checks the 163 numbers
  the paper prints against them. `out/tables/SUMMARY.txt` must read:

  <!-- check_image.sh: the lines out/tables/SUMMARY.txt must hold -->
  ```
  141 cells | 0 skipped | 1 errors | 0 unsound
  163 published values, all reproduced by the generator.
  The paper's tables and the campaign agree.
  27 ok | 8 warnings | 0 failures
  Publishable, PROVIDED every warning above is stated in the text.
  ```

  The one error is the cell where GPAP's encoder fails because ABC crashes on `b19`, a failure in
  the paper too. The 8 warnings, two for each of Recurrent's four stages, are about its 28
  recorded failures: one asks that the paper report them, which its Fail column does; the other
  says they were carried over from an earlier run on the same setup instead of re-measured
  ([Two shortcuts](#two-shortcuts-both-recorded-in-every-row)).
- `smoke` runs the whole pipeline on 4 cells, compiling the engine for this GPU first.
  `out/smoke/SUMMARY.txt` must hold:

  <!-- check_image.sh: the lines out/smoke/SUMMARY.txt must hold -->
  ```
  PASS — 4 cells compared: same formulas, sound samples, same verdicts.
  ARTIFACT_TIER_COMPLETE
  ```

## Full review

### The paper's claims, from the recorded measurements

`out/tables/details/` holds the tables (`RESULTS.md`), the paper's printed numbers checked one by
one (`PAPER_NUMBERS.txt`), the publishability gate (`VERIFICATION.txt`) and the figures
(`plots/`). The archive's own copies, generated when it was built, are in `artifact-data/paper/`,
and only the date differs:

```bash
docker run --rm -v "$PWD/out:/out" gpap-artifact:tacas-27 diff /out/tables/details/RESULTS.md artifact-data/paper/RESULTS.md
```

Figure 3, `plots/sps_scatter_cmsgen.pdf`, comes out byte-identical to the paper's.

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

`subset` re-measures 66 of the 142 cells, in about 2.5 hours, and `full` all of them, in about 12
hours. Start one detached and follow it:

```bash
docker run -d --name gpap-subset --gpus all -v "$PWD/out:/out" gpap-artifact:tacas-27 subset --out /out/subset
docker logs -f gpap-subset
```

`out/subset/SUMMARY.txt` holds the verdict, as for `smoke`, once the tier ends. `docker stop
gpap-subset` stops it, and a stopped tier resumes: remove the container (`docker rm
gpap-subset`) and run the same command again.

**What is compared.** A sampling rate is a property of the card: the same engine runs 1.82× as
fast on an A100 and 0.84× on an L4 as on an RTX 4080
([CROSS_GPU_FINDINGS.md](../../findings/CROSS_GPU_FINDINGS.md)). Asking for our rates back would
test the hardware. So the comparison checks what must hold on any machine, and reports the
rest:

| checked | how | a difference means |
|---|---|---|
| **the same formula in each cell** | `n_vars`, `n_clauses`, `support`, the encoder and `pin_check_ok` match exactly | the two runs did not measure the same thing, usually because of a different `abc` build |
| **soundness** | no emitted model falsifies a clause, and every kept sample replays through the original circuit | a real bug, against the paper's central claim |
| **the same verdicts** | which tool won each cell, and how each cell was classified | the conclusion did not survive the change of hardware, unless the GPU sampler stepped down to a smaller batch on that cell ([Hardware](#hardware)) |
| *rates* | this card's speed relative to ours, per stage, in `details/COMPARISON.txt` | nothing: it describes the machine |

The tier fails if a formula differs, if a sample is unsound, if fewer cells ran than the tier
sets out to run, or if any step fails. A changed verdict does not fail it; the `PASS` line then
says how many verdicts differ, and `SUMMARY.txt` names them. `details/COMPARISON.txt` is the
comparison cell by cell.

**What each tier costs.** These are estimates, built from the recorded time of every cell plus a
measured overhead the rows do not carry (start-up, the ABC pass and its equivalence check, the
soundness replay). The real time depends on the GPU and the host CPU:

| tier | cells | time | what it covers |
|---|---|---|---|
| `tables` | — | seconds, no GPU | every published table and figure |
| `smoke` | 4 | about 4 min | one cell of each T=25 stage, run exactly as the real tiers run it |
| `subset` | 66 | about 2.5 h | all 22 ISCAS-85 cells, and 22 of the 30 sequential instances at T=25 |
| `full` | 142 | about 12 h | every cell, at T=25 and T=50 |
| `full --remeasure-failures` | 142 | about 23 h | the same, re-measuring every recorded rival failure |

Each GPU step also spends about a minute compiling the engine and cloning the compared samplers.
For another machine, `eval/artifact/estimate_wall.py artifact-data/paper/data` prints the same
table: `--scale` multiplies the GPU's part (below 1 for a card faster than the L4) and
`--cpu-scale` the overhead, which depends on the host CPU (1.9 for the reference host, 1 for the
laptop the overhead was measured on). `check` runs it with both at 1.

The `subset` tier's cells are chosen by cost, not by us: a sequential instance is kept when its
two T=25 tasks together take ten minutes or less on the reference card. `eval/artifact/subset.json`
holds the rule, the selection and the per-instance times. The comparison uses the published
numbers for the same cells, not the paper's averages over all 142.

### Reusing GPAP

The image runs GPAP itself, too:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/out:/out" gpap-artifact:tacas-27 gpap sample benchmarks/circuits/bench/seq/b10.bench --cycles 25 -n 10000 --out /out/b10.txt --sim-check
docker run --rm -it --gpus all gpap-artifact:tacas-27 bash
```

The first writes 10,000 input sequences of the circuit `b10`, unrolled for 25 cycles, that reach
a fixed output pattern, and replays each through the circuit. The second opens a shell with
everything installed. The tool alone, without the paper, is the image `gpap:<version>` in the same
file ([docs/DOCKER.md](../../DOCKER.md)).

| to | read |
|---|---|
| install and use GPAP | [the project README](../../../README.md) |
| run it in Docker | [docs/DOCKER.md](../../DOCKER.md) |
| every option of `gpap sample` | [docs/SAMPLING.md](../../SAMPLING.md) |
| call it from Python | [docs/PYTHON_API.md](../../PYTHON_API.md) |
| call it from C or C++ | [cpp/README.md](../../../cpp/README.md) |
| understand the engine | [docs/ENGINE.md](../../ENGINE.md) |
| run your own comparison fairly | [docs/BENCHMARKING.md](../../BENCHMARKING.md), [docs/EVAL_RUNBOOK.md](../../EVAL_RUNBOOK.md) |
| change it | [CONTRIBUTING.md](../../../CONTRIBUTING.md) |

## Without Docker

In the TACAS VM, or on any machine without a GPU, `tables` runs from the source archive with the
stock `python3` and no network:

```bash
tar xzf gpap-artifact-*.tar.gz && cd gpap-artifact-*/
eval/artifact/reproduce.sh tables --out vm-tables
```

The figures need `matplotlib`, which the VM lacks; the published ones are in
`artifact-data/paper/plots/`.

On a machine with a GPU, cmake, a C++17 compiler, git and Python 3.10 or later, the GPU steps run
in a virtual environment:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r eval/artifact/requirements-artifact.txt && pip install -e .
eval/artifact/reproduce.sh setup     # builds the engine, ABC and CMSGen; fetches the GPU rivals
eval/artifact/reproduce.sh smoke     # 4 cells, a few minutes
```

For a driver that supports only CUDA 12.6, install
`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126` first, then the
requirements without their `torch==` line.

`requirements-artifact.txt` pins the three packages that can move a number: torch, the Triton
that compiles the kernels, and the SAT solver that decides the `unsat` screen. The image installs
the same three. In the same environment, `pytest tests -m "not gpu and not slow"` runs the
tool's own test suite in seconds.

## What you receive

| file at the DOI | what it is |
|---|---|
| `gpap-image-<date>-<commit>.tar.gz` | the Docker images: `gpap-artifact:tacas-27`, and the tool alone, `gpap:<version>` |
| `gpap-artifact-<date>-<commit>.tar.gz` | the source archive: the same files as `/opt/gpap` in the image |

The source archive holds:

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
| `eval/` | the evaluation: campaign drivers, the harnesses for the compared tools, the table generator, and the artifact image's Dockerfile |
| `Dockerfile` | the tool image |
| `docs/` | the documentation; [docs/README.md](../../README.md) is its map |
| `tests/` | the test suite |

The GPU samplers GPAP is compared with, DEMOTIC and Recurrent-CircuitSAT, state no licence, so
neither the image nor the archive includes them; nor HTS, which `setup` also fetches but the
paper's tables do not use. `setup` fetches each at a pinned commit, and every result row records
the commit that ran.

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

**The paper's measurements ran on an NVIDIA L4 with 24 GB of GPU memory, which runs every cell
at the batch size the paper recorded.** Each rival's batch size is the largest that ran on the
L4; for nearly all of them, the next step up ran out of memory. A GPU with less memory runs every
step, but its results can differ. Where a rival's recorded batch does not fit, the harness steps
it down to the next smaller batch that does, and the row records it (`grant_source`). A smaller
batch slows the rival, so on that cell our ratio looks better and a verdict against the rival
can change in GPAP's favour. The comparison counts these cells, warns about them and names
each one, and a verdict that changed there is evidence for neither side. On a 12 GB card, most
of Recurrent's cells step down. The run still completes.

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
recorded as "the rival did not initialize within its 1,200 s budget". They were measured in
September 2026, in the campaign before the paper's, on the same setup: an NVIDIA L4, with
Recurrent at the same commit, configuration and budget. The only change to Recurrent since, the
timer on its copy to the host, does not change what it computes
([What you receive](#what-you-receive)). These rows are the paper's Fail column. Re-measuring
them costs that budget each, 10 of the 22 hours of a full re-measurement, to produce rows that
say "no measurement". The campaign the paper reports carried them over, and so does this
artifact: such a row is marked `not_remeasured`, `VERIFICATION.txt` names it, and the comparison
counts it as evidence for neither side. This can only be unfair to the rival, since a faster card
might initialize where the L4 could not. `full --remeasure-failures` re-measures them.

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

- The Docker images, built by `make_image.sh` from an archive built like this one, on an RTX 4080
  Laptop GPU under WSL2 ([ARTIFACT_FINDINGS.md](../../findings/ARTIFACT_FINDINGS.md) §10):
  `check` and `tables` with no GPU and no network, all 163 values reproduced and `RESULTS.md` and
  every figure identical to the shipped ones; `smoke` in 3 minutes, `setup` included, and again
  with the network cut after `setup`; the tool's whole test suite on the GPU (2026-10-19).
  `subset`, in an image built the same way, on the same card (2026-10-21).
- The same images on native Ubuntu 20.04, Docker 26.1 with the NVIDIA Container Toolkit, on a
  GeForce MX150, a GPU too old for GPAP ([ARTIFACT_FINDINGS.md](../../findings/ARTIFACT_FINDINGS.md)
  §11): the smoke test above as written,
  where `check` and `tables` ran and `smoke` stopped at once, saying why (2026-10-19).
- `check` and `tables`, from an archive built like this one, in an offline Ubuntu 22.04
  container with only the stock `python3` 3.10 (no GPU, no numpy, no network): the archive
  verified, all 163 published values reproduced, the gate passed (2026-10-15). The same in
  Ubuntu 24.04 with Python 3.12 (2026-10-13), and in Ubuntu 26.04, the 2027 TACAS VM's base,
  with Python 3.14 (2026-10-21).
- `setup`, then `smoke`, from the archive, in a fresh virtual environment built from
  `requirements-artifact.txt`, on an RTX 4080 Laptop GPU (12 GB): every tool built from its
  pinned source; 4 of 4 cells with the same formula, sound samples and the same verdicts.
  DEMOTIC stepped down on `c432` on this smaller card, as described above (2026-10-13).
- `subset`, from the archive, on a rented L4 ([ARTIFACT_FINDINGS.md](../../findings/ARTIFACT_FINDINGS.md)
  §5b): all 66 cells reach the same verdict against CMSGen, and all 58 cells whose rival side was
  measured rather than carried over reach the same verdict against the rival. Every count in
  Tables 1–3 equals the published one for the same cells, and every ratio is within 5.5%
  (2026-10-14).

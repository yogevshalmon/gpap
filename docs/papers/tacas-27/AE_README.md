# GPAP — artifact for a tool paper submitted to TACAS 2027

> **Reader:** TACAS 2027 artifact evaluators: what the artifact needs, what to run, and how to
> read what it prints.

Paper: *SAT Sampling at Scale via GPU-based Boolean Constraint Propagation*, a regular tool paper.

- **Artifact:** [doi.org/10.5281/zenodo.23241912](https://doi.org/10.5281/zenodo.23241912).
  Download the Docker image `gpap-image-*.tar.gz` and its `.sha256` into one folder, and run
  every command below from it. The third file, the source archive `gpap-artifact-*.tar.gz`, is
  optional ([Where else to look](#where-else-to-look)).
- **Badges applied for:** Available, Functional and Reusable.
- **Licence:** MIT. Benchmarks and tools by others keep their own (`THIRD_PARTY.md`).

## Requirements: an NVIDIA GPU

GPAP runs only on NVIDIA GPUs; the AE chairs granted the special-hardware exemption.

- Linux on x86-64, or Windows with WSL2.
- An NVIDIA GPU of compute capability 8.0 or newer, with a driver for CUDA 12.8 or later. The
  paper's measurements ran on an NVIDIA L4 with 24 GB; a GPU with less memory runs every step
  ([Reading the results](#reading-the-results) says what changes).
- Docker with the NVIDIA Container Toolkit, and about 15 GB of disk.
- Network access to github.com for the GPU steps: the two GPU samplers the paper compares with
  state no licence, so they are cloned, not shipped.

## Smoke test (about 10 minutes, loading the image included)

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
sha256sum -c gpap-image-*.tar.gz.sha256
docker load -i gpap-image-*.tar.gz
mkdir -p out
docker run --rm --gpus all gpap-artifact:tacas-27 check
docker run --rm --gpus all -v "$PWD/out:/out" gpap-artifact:tacas-27 smoke --out /out/smoke
```

`check` lists the GPU and what each step costs. `smoke` runs the whole pipeline on 4 cells, and
`out/smoke/SUMMARY.txt` must hold:

<!-- check_image.sh: the lines out/smoke/SUMMARY.txt must hold -->
```
PASS — 4 cells compared: same formulas, sound samples, same verdicts.
ARTIFACT_TIER_COMPLETE
```

## Full review

### The paper's tables (seconds, no GPU)

`tables` regenerates the paper's tables and figures from the per-cell measurements the artifact
ships, and checks every number the PDF prints from them. Nothing is measured again.

```bash
docker run --rm -v "$PWD/out:/out" gpap-artifact:tacas-27 tables --out /out/tables
```

`out/tables/SUMMARY.txt` must read:

<!-- check_image.sh: the lines out/tables/SUMMARY.txt must hold -->
```
142 cells: 141 measured, 1 GPAP could not encode (b19), 0 unsound
163 published values, all reproduced by the generator.
checks on the measurements: 27 ok | 8 warnings | 0 failures
```

The files to look at are in `out/tables/details/`:

- `RESULTS.md`: the paper's Tables 1–3, its uniformity counts and the numbers its text quotes,
  laid out as the paper prints them, plus the ratios the text quotes. Under each table, every
  cell not counted, by name and reason. Read it beside the PDF.
- `PAPER_NUMBERS.txt`: the check of every number the PDF prints from the measurements against
  the regenerated one: the count, and any number that differs.
- `VERIFICATION.txt`: the checks on the measurements: every emitted sample is a solution, every
  CMSGen count is exact, every failure of a compared tool is reported.
- `plots/`: the figures; the paper's Figure 3 is `sps_scatter_cmsgen.pdf`.

### Re-measuring on a GPU

`subset` re-measures 66 of the paper's 142 cells (none at T=50) in about 2.5 hours, and `full`
all 142 in about 12 hours. Start one detached and follow it:

```bash
docker run -d --name gpap-subset --gpus all -v "$PWD/out:/out" gpap-artifact:tacas-27 subset --out /out/subset
docker logs -f gpap-subset
```

or, for every cell:

```bash
docker run -d --name gpap-full --gpus all -v "$PWD/out:/out" gpap-artifact:tacas-27 full --out /out/full
docker logs -f gpap-full
```

A rate depends on the GPU, so a tier does not ask for the paper's rates back. It compares each
re-measured cell with the paper's and checks what must hold on any machine: the same formula,
every sample sound, and the same verdict (which tool won). When it ends, `out/subset/SUMMARY.txt`
(or `out/full/`) has a line that begins `PASS — 66 cells compared: same formulas, sound samples`
(142 for `full`), then `ARTIFACT_TIER_COMPLETE`; `details/COMPARISON.txt` beside it compares
every cell. A stopped tier resumes: `docker rm gpap-subset` (or `gpap-full`), then the same
command again.

### Reading the results

- **`b19`.** ABC crashes when GPAP encodes `b19` at T=25 under Recurrent-CircuitSAT's own target.
  It is one of the 142 cells, a failure in the paper too, and never counted as a zero.
- **The 8 warnings in `VERIFICATION.txt`** are about Recurrent-CircuitSAT's 28 recorded failures
  to initialize within its 1,200 s budget. They were measured in the campaign before the
  paper's, on the same setup (an L4, and the tool at the same commit, settings and budget), and
  carried over, by the paper's campaign and by every tier here. 25 of them are the paper's Fail
  column; the other 3, all `b19`, are screened out first. `full --remeasure-failures` measures
  them again, in about 23 hours.
- **A GPU with less than 24 GB.** Where a GPU sampler's recorded batch does not fit, it steps down
  to a smaller one, which slows it. A verdict that changed on such a cell favours GPAP and is
  evidence for neither side; the tier's summary names every such cell.
- **Uniformity** comes from a separate Barbarik run, on an RTX 4080 Laptop GPU. `tables`
  regenerates its counts from the shipped rows.

### Reusing GPAP

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/out:/out" gpap-artifact:tacas-27 gpap sample benchmarks/circuits/bench/seq/b10.bench --cycles 25 -n 10000 --out /out/b10.txt --sim-check
```

This writes 10,000 input sequences of the circuit `b10`, unrolled for 25 cycles, that reach a
fixed output pattern, and replays each through the circuit. `docs/README.md` maps the guides for
running GPAP on your own circuits and formulas, from the command line, Python or C++.

## Platforms tested

All on x86-64 (amd64); the image is built for it alone.

- An NVIDIA L4 (24 GB) on Colab, from the source archive: `subset`.
- An RTX 4080 Laptop GPU (12 GB) on Windows with WSL2, in the Docker image: `check`, `tables`,
  `setup`, `smoke` and `subset`.
- Native Ubuntu 20.04 with a GPU too old for GPAP, in the Docker image: the image loads, `check`
  and `tables` run, and the GPU steps stop with a message that says why.
- No GPU, from the source archive: `tables` in offline Ubuntu 22.04, 24.04 and 26.04 with the
  stock `python3`.

## Where else to look

Everything above runs from the image. The source archive (`tar xzf gpap-artifact-*.tar.gz`) holds
the same files as `/opt/gpap` in the image, among them:

- `docs/papers/tacas-27/ARTIFACT.md`: each claim of the paper with the script and data behind
  it, what a re-measurement compares and what each step costs, running `tables` in the TACAS VM
  without Docker, and the evidence behind [Reading the results](#reading-the-results).
- `docs/README.md`: GPAP's documentation.

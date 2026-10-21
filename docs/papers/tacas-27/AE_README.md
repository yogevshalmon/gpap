# GPAP — artifact for a tool paper submitted to TACAS 2027

> **Reader:** TACAS 2027 artifact evaluators: where the artifact is, what it needs, and the smoke
> test. The full guide is ARTIFACT.md, inside the artifact.

Paper: *SAT Sampling at Scale via GPU-based Boolean Constraint Propagation*, a regular tool paper.

- **Artifact:** [doi.org/10.5281/zenodo.23241912](https://doi.org/10.5281/zenodo.23241912): a
  Docker image (`gpap-image-*.tar.gz`) and the source archive (`gpap-artifact-*.tar.gz`). The full
  guide is `docs/papers/tacas-27/ARTIFACT.md` in the source archive.
- **Badges applied for:** Available, Functional and Reusable.
- **Licence:** MIT. Benchmarks and tools by others keep their own (`THIRD_PARTY.md`).

## Requirements: an NVIDIA GPU

GPAP runs only on NVIDIA GPUs; the AE chairs granted the special-hardware exemption.

- Linux on x86-64, or Windows with WSL2.
- An NVIDIA GPU of compute capability 8.0 or newer, with a driver for CUDA 12.8 or later.
- 24 GB of GPU memory to run the GPU samplers at the paper's batch sizes: the paper's
  measurements ran on an NVIDIA L4 (24 GB). A GPU with less memory runs every step, but its
  results can differ: where a GPU sampler's recorded batch does not fit, it steps down to a
  smaller one, and the comparison names each such cell (ARTIFACT.md, "Hardware").
- Docker with the NVIDIA Container Toolkit, and about 15 GB of disk.
- Network access to github.com for the GPU steps: the two GPU samplers the paper compares with
  state no licence, so they are cloned, not shipped.

## Smoke test (about 10 minutes)

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
sha256sum -c gpap-image-*.tar.gz.sha256
docker load -i gpap-image-*.tar.gz
mkdir -p out
docker run --rm --gpus all gpap-artifact:tacas-27 check
docker run --rm -v "$PWD/out:/out" gpap-artifact:tacas-27 tables --out /out/tables
docker run --rm --gpus all -v "$PWD/out:/out" gpap-artifact:tacas-27 smoke --out /out/smoke
```

`tables` and `smoke` each write one file to read, `SUMMARY.txt`. `out/tables/SUMMARY.txt`
includes

```
163 published values, all reproduced by the generator.
```

and `out/smoke/SUMMARY.txt` includes

```
PASS — 4 cells compared: same formulas, sound samples, same verdicts.
ARTIFACT_TIER_COMPLETE
```

## Full review

`subset` re-measures 66 of the paper's 142 cells, in about 2.5 hours, and `full` all of them, in
about 12 hours:

```bash
docker run -d --name gpap-subset --gpus all -v "$PWD/out:/out" gpap-artifact:tacas-27 subset --out /out/subset
docker logs -f gpap-subset
```

ARTIFACT.md maps each claim of the paper to the file that checks it.

## Platforms tested

All on x86-64 (amd64); the image is built for it alone.

- An NVIDIA L4 (24 GB) on Colab, from the source archive: `subset`.
- An RTX 4080 Laptop GPU (12 GB) on Windows with WSL2, in the Docker image: `check`, `tables`,
  `setup`, `smoke` and `subset`.
- Native Ubuntu 20.04 with a GPU too old for GPAP, in the Docker image: the image loads, `check`
  and `tables` run, and the GPU steps stop with a message that says why.
- No GPU, from the source archive: `tables` in offline Ubuntu 22.04, 24.04 and 26.04 with the
  stock `python3`.

# GPAP — artifact for the TACAS 2027 tool paper

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
- Docker with the NVIDIA Container Toolkit, and about 15 GB of disk.
- Network access to github.com for the GPU steps: the two GPU samplers the paper compares with
  state no licence, so they are cloned, not shipped.

## Smoke test (about 10 minutes)

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
sha256sum -c gpap-image-*.tar.gz.sha256
docker load -i gpap-image-*.tar.gz
docker run --rm --gpus all gpap-artifact:tacas-27 check
docker run --rm -v "$PWD/out:/out" gpap-artifact:tacas-27 tables --out /out/tables
docker run --rm --gpus all -v "$PWD/out:/out" gpap-artifact:tacas-27 smoke --out /out/smoke
```

Each step writes one file to read, `SUMMARY.txt`. `out/tables/SUMMARY.txt` includes

```
163 published values, all reproduced by the generator.
```

and `out/smoke/SUMMARY.txt` ends with

```
PASS — 4 cells compared: same formulas, sound samples, same verdicts.
ARTIFACT_TIER_COMPLETE
```

## Full review

`subset` re-measures 66 of the paper's 142 cells, in about 3 hours on an NVIDIA L4, and `full`
all of them, in about 13 hours:

```bash
docker run -d --name gpap-subset --gpus all -v "$PWD/out:/out" gpap-artifact:tacas-27 subset --out /out/subset
docker logs -f gpap-subset
```

ARTIFACT.md maps each claim of the paper to the file that checks it.

## Platforms tested

- The Docker image, every step: x86-64, WSL2 with Ubuntu 24.04, Docker 29.1.3 with the NVIDIA
  Container Toolkit 1.20.1, an RTX 4080 Laptop GPU (12 GB) with driver 596.58 (2026-10-19).
- The source archive without Docker: `tables` in offline Ubuntu 22.04 and 24.04 with the stock
  `python3`, and `subset` on an NVIDIA L4 (Colab), all 66 cells with the paper's verdicts
  (2026-10-13 to 10-15).

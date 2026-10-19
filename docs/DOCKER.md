# Running GPAP in Docker

> **Reader:** anyone who wants to run GPAP without installing its toolchain.

## What you need

- Linux on x86-64, or Windows with WSL2.
- An NVIDIA GPU of compute capability 8.0 or newer (Ampere or later), with a driver for CUDA 12.8
  or later.
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  This checks both:

  ```bash
  docker run --rm --gpus all ubuntu:24.04 nvidia-smi
  ```

## Build and run

```bash
docker build -t gpap .
docker run --rm --gpus all -v gpap-cache:/cache -v "$PWD:/data" -w /data gpap sample formula.cnf -n 50000 --out samples.txt
```

The arguments after the image name are `gpap`'s ([SAMPLING.md](SAMPLING.md)); `bash`, `python`
or `pytest` in their place run as themselves. `-v "$PWD:/data" -w /data` gives the command the
current directory; add `--user "$(id -u):$(id -g)"` to own what it writes.

The first run on a GPU compiles the engine for that GPU, in under a minute, because its kernels
are compiled for one GPU architecture. The volume at `/cache` keeps it for the next run. To check
the whole installation on your card:

```bash
docker run --rm --gpus all -v gpap-cache:/cache gpap pytest -q
```

## When something goes wrong

| message | what to do |
|---|---|
| `no NVIDIA GPU in this container` | add `--gpus all`; if the `nvidia-smi` check fails too, install the NVIDIA Container Toolkit and restart Docker |
| `a GPU is visible but CUDA cannot use it` | the driver is older than CUDA 12.8: build with `--build-arg TORCH=2.11.0+cu126 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126` |
| `GPAP needs compute capability 8.0 or newer` | the GPU is too old for GPAP |
| `the compile failed` | its log is under `/cache`; `GPAP_SKIP_BUILD=1` starts the container without compiling |

With Podman, request the GPU with `--device nvidia.com/gpu=all` instead of `--gpus all`.

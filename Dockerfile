# GPAP, the tool, in a container: the gpap package and command, torch and Triton, ABC, and what
# compiles the engine. The engine itself, libgpap, is NOT in the image: its kernels are compiled
# ahead of time for one GPU architecture and the engine refuses any other (cpp/src/runtime.cpp),
# so the entry point compiles it for the visible GPU on first use, in under a minute, into /cache.
# A volume there keeps it.
#
#   docker build -t gpap .
#   docker run --rm --gpus all -v gpap-cache:/cache -v "$PWD:/data" -w /data gpap sample f.cnf -n 1000 --out s.txt
#
# docs/DOCKER.md is the guide; eval/artifact/Dockerfile builds an artifact image on this one.
# Plain Dockerfile syntax, so the legacy builder builds it too.
ARG UBUNTU=ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac
FROM ${UBUNTU}

# The pins that can move a number. They equal eval/artifact/requirements-artifact.txt's, and a
# test keeps them equal, so the artifact layer installs no second torch. A driver too old for
# CUDA 12.8 takes the cu126 build of the same torch:
#   --build-arg TORCH=2.11.0+cu126 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH=2.11.0+cu128
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TRITON=3.6.0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev g++ make cmake git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python3 -m venv /opt/venv \
 && pip install --extra-index-url "$TORCH_INDEX" "torch==$TORCH" "triton==$TRITON"

# GPAP_HOME holds the built tools (ABC here, CMSGen in the artifact layer) and the fetched
# sources. GPAP_CACHE holds the compiled engine, one directory per build key and architecture.
ENV GPAP_HOME=/opt/gpap-home \
    GPAP_CACHE=/cache \
    TRITON_CACHE_DIR=/cache/triton \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ABC first, from the only files its build reads, so an edit elsewhere does not rebuild it.
COPY gpap/__init__.py gpap/paths.py /opt/gpap/gpap/
COPY scripts/build_abc.sh /opt/gpap/scripts/
RUN /opt/gpap/scripts/build_abc.sh

COPY . /opt/gpap
WORKDIR /opt/gpap
RUN pip install -e ".[solver]" pytest

# cpp/CMakeLists.txt links against `libcuda.so`. On native Linux the NVIDIA Container Toolkit
# mounts the driver as libcuda.so.1 alone, so this link supplies the name; it dangles until a
# container starts with --gpus. WSL2's driver directory carries both names, and CMake searches it
# first.
RUN ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/lib/libcuda.so

# Writable by any user, so `docker run --user "$(id -u):$(id -g)"` works: the compile writes the
# generated kernels into the tree and the library into the cache, and the artifact's `setup`
# builds into build/ and clones into GPAP_HOME.
RUN mkdir -p /cache "$GPAP_HOME/src" build cpp/src/generated \
 && chmod 1777 /cache "$GPAP_HOME" "$GPAP_HOME/src" build cpp/src/generated

# The build key names the compiled engine in the cache: the engine's sources and the Triton
# version, hashed before anything is generated, so a newer image never loads a library an older
# one compiled into the same volume. The package lists say what this image holds.
RUN mkdir -p /opt/gpap-image \
 && { find cpp scripts/build_core.sh -type f -print0 | sort -z | xargs -0 sha256sum; \
      python -c "import triton; print(triton.__version__)"; } \
    | sha256sum | cut -c1-16 > /opt/gpap-image/build-key \
 && pip freeze > /opt/gpap-image/pip-freeze.txt \
 && dpkg-query -W > /opt/gpap-image/dpkg.txt

# Last, so a new revision does not invalidate the layers above.
ARG GPAP_VERSION=dev
ARG GPAP_REVISION=unknown
LABEL org.opencontainers.image.title="gpap" \
      org.opencontainers.image.description="GPU-parallel assignment propagation, and the SAT and circuit sampler built on it" \
      org.opencontainers.image.source="https://github.com/yogevshalmon/gpap" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="$GPAP_VERSION" \
      org.opencontainers.image.revision="$GPAP_REVISION"

ENTRYPOINT ["/opt/gpap/scripts/docker_entrypoint.sh"]
CMD ["gpap", "--help"]

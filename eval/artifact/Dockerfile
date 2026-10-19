# One paper's artifact image: the tool image (the Dockerfile at the repository root) plus the
# evaluation's tools and the paper's own files. eval/artifact/make_image.sh builds it from an
# unpacked artifact archive, with the paper's files alone as its context, then gates and saves it:
#
#   eval/artifact/make_image.sh <unpacked archive>
#
# Inside, /opt/gpap is the archive file for file (MANIFEST.sha256 verifies), and the entry point
# runs reproduce.sh's steps:  docker run --rm --gpus all -v "$PWD/out:/out" IMAGE smoke
# The samplers GPAP is compared with state no licence, so no image carries them: `setup` clones
# them at their pinned commits, the only use of the network at run time.
ARG BASE=gpap:dev
FROM ${BASE}

# The archive's files the tool image leaves out (.dockerignore): the paper's rows, and the
# archive's own pointer and manifest. A test keeps this list equal to that one.
COPY artifact-data /opt/gpap/artifact-data
COPY README-ARTIFACT.md MANIFEST.md MANIFEST.sha256 /opt/gpap/

# The reproduction lock. Then the line eval/rivals/rivals_setup.sh installs for the compared
# samplers, so `setup` finds it satisfied and needs no network for it (a test keeps the two
# equal). Then the numpy and matplotlib with which Figure 3 is byte-identical to the paper's.
RUN pip install -r eval/artifact/requirements-artifact.txt \
 && pip install pycryptosat "python-sat>=0.1.8.dev0,<1.10" pandas tqdm \
 && pip install numpy==2.4.4 matplotlib==3.10.8 \
 && scripts/build_cmsgen.sh \
 && pip freeze > /opt/gpap-image/pip-freeze.txt

# reproduce.sh writes under the experiments root (eval/paths.py); here, the directory to mount.
ENV GPAP_EXPERIMENTS=/out
RUN mkdir -p /out && chmod 1777 /out

ENTRYPOINT ["/opt/gpap/eval/artifact/docker_entrypoint.sh"]
CMD ["check"]

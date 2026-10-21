# The artifact image: the tool image (the repository's Dockerfile) plus the evaluation's tools
# and the paper's rows, built from an unpacked archive by eval/artifact/make_image.sh. /opt/gpap
# is that archive (MANIFEST.sha256 verifies), and the entry point runs reproduce.sh's steps:
#
#   docker run --rm --gpus all -v "$PWD/out:/out" IMAGE smoke
#
# The compared samplers state no licence, so no image carries them: `setup` clones them at pinned
# commits, the only network use at run time.
ARG BASE=gpap:dev
FROM ${BASE}

# What the tool image leaves out (.dockerignore); a test keeps the two lists equal.
COPY artifact-data /opt/gpap/artifact-data
COPY README-ARTIFACT.md MANIFEST.md MANIFEST.sha256 /opt/gpap/

# The reproduction lock; the packages rivals_setup.sh installs, so `setup` needs no network for
# them (a test keeps the two equal); the numpy and matplotlib that redraw Figure 3 byte for byte.
RUN pip install -r eval/artifact/requirements-artifact.txt \
 && pip install pycryptosat "python-sat>=0.1.8.dev0,<1.10" pandas tqdm \
 && pip install numpy==2.4.4 matplotlib==3.10.8 \
 && scripts/build_cmsgen.sh \
 && pip freeze > /opt/gpap-image/pip-freeze.txt

# reproduce.sh writes under the experiments root (eval/paths.py): the mounted /out.
ENV GPAP_EXPERIMENTS=/out
RUN mkdir -p /out && chmod 1777 /out

ENTRYPOINT ["/opt/gpap/eval/artifact/docker_entrypoint.sh"]
CMD ["check"]

# GPAP — guide for coding agents

> **Reader:** coding agents working in this repository. It is short on purpose: the rules live in
> [CONTRIBUTING.md](CONTRIBUTING.md), which the `@CONTRIBUTING.md` line below imports (Claude
> Code's import syntax), and they bind people and agents alike.

GPAP propagates Boolean constraints on the GPU for thousands of partial assignments at once. The
engine is one C++ library, `libgpap` (Triton kernels compiled ahead of time, a C ABI, and Python
and C++ faces over it); its flagship application is a SAT and circuit solution sampler. Users start
at [README.md](README.md); the documentation map is [docs/README.md](docs/README.md).

@CONTRIBUTING.md

## Before you change anything

- Read the comments where you are about to edit. They state the rule the code enforces and the
  refactor that would break it; keep them true.
- Find the owner of a fact before you write it down (CONTRIBUTING.md, "Where things go"). Never
  copy a number from `docs/findings/` into code, a guide or this file: link to it.
- The findings are dated evidence. Read a file's status block before quoting anything from it.

## Before you finish

- Run the checks in CONTRIBUTING.md, "Definition of done", and report their output, failures
  included.
- After a docs edit, run `pytest tests/repo/test_docs_hygiene.py`.
- Do not commit, push, or start a long GPU job unless you were asked to.

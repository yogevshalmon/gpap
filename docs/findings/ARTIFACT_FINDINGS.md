# Artifact-evaluation findings

> **Reader:** contributors and researchers — the evidence behind the artifact's reproduction tiers. A dated research log, not a user guide.
>
> **Status:** settled. Condensed 2026-10-21 to what still holds; the full earlier text is
> `git show 823eefd:docs/findings/ARTIFACT_FINDINGS.md`.
> **Measured:** 2026-09-09/10 against the campaign `l4_protocol_20260904`, and 2026-10-13/14
> against the paper's rows, its re-run `l4_protocol_20261012` (§5b): NVIDIA L4 (24 GB, Colab) for
> the reference rows, RTX 4080 Laptop (12 GB) for the smoke tier. The Docker images 2026-10-19 on
> the same laptop under WSL2 and on native Ubuntu 20.04 (§10, §11).
> **Rule lives in:** `GPAP_VRAM_BUDGET_MIB`, clamped once inside `gpap_mem_available`
> (`cpp/src/calibrate.cpp`); `eval/artifact/reproduce.sh --vram-cap`;
> `eval/artifact/compare_campaign.py`; `eval/artifact/subset.json`; the images: `Dockerfile`,
> `scripts/docker_entrypoint.sh`, `eval/artifact/Dockerfile`, `eval/artifact/check_image.sh`.
> **Gated by:** `tests/engine/test_k_cap.py`, `tests/experiments/test_rival_budget.py`,
> `tests/repo/test_docker.py`, `tests/experiments/test_artifact_image.py`,
> `tests/experiments/test_artifact_steps.py`.
> **Record only:** §8, the review post-mortem on gates that could not fail.

The evidence behind `docs/papers/tacas-27/ARTIFACT.md` and `eval/artifact/reproduce.sh`. Nothing
here changes the protocol; the artifact only chooses which cells a reviewer runs, under two
documented shortcuts.

## 1. 24 GB is a hard requirement, and the grants file proves it

Counted over `eval/rivals/grants/l4_protocol_20260904.json` (unchanged by the re-run):

| tool | cells | pinned | recorded failures | at the card's memory ceiling | granted the ladder top |
|---|---|---|---|---|---|
| DEMOTIC | 22 | 22 | 0 | 0 | 2 |
| Recurrent-CircuitSAT | 120 | 92 | 28 | **92** | 0 |

"At the memory ceiling" means the entry records an OOM one rung above its grant: the L4's 24 GB
stopped it. Every pinned Recurrent cell is there, so a smaller card walks down on almost all of
them and hands the rival less parallelism than the paper did, which inflates our ratio.
`compare_campaign.py` counts those cells and warns; `grant_source` records it per row.

## 2. The other direction: a bigger card silently favours us

63 of the campaign's 141 measured rows have a batch K below the 262,144 cap: they are VRAM-bound.
On a larger card their K grows while the rival stays pinned. `GPAP_VRAM_BUDGET_MIB`, clamped
inside `gpap_mem_available` (the one function C++ and Python both call), holds the engine to the
published budget, and `reproduce.sh --vram-cap l4` defaults to it:

    (23.45 GB free - 800 MB reserve) x 0.80 = 17,280 MiB

The cap is a ceiling, inert on smaller cards. The rival's budget calls the same function, so one
cap reproduces both sides; it used to be a second implementation, 614 MiB apart (§8).

## 3. What a pinned reproduction costs, and what the rows do not time

A cell costs the rival's one recorded run, our arm at three seeds and the CMSGen arm. Two
findings shape `eval/artifact/estimate_wall.py`:
- **The rows do not time everything.** Start-up, the ABC pass with its equivalence check
  (≈ 0.31 ms per gate, over a minute on `b19`) and the soundness replay are a per-cell constant,
  four times the recorded wall on the cheapest cells. The model is constant + per gate +
  fraction, with `--cpu-scale` for another host.
- **Pinning the 28 recorded rival failures** (`not_remeasured`) halves a full reproduction;
  `--remeasure-failures` is the faithful run. A cell that delivers no models is not free: it
  runs three seeds to the 60 s wall.

`reproduce.sh check` prints the current estimates, read off the shipped rows.

## 4. The reviewer tier's cell set is chosen by the clock

`eval/artifact/subset.json` keeps a sequential instance when both its T=25 tasks together cost
ten minutes or less on the reference card: 22 of 30 instances and all 11 ISCAS-85, 66 of 142
cells for about a quarter of the wall. The rule is about the clock alone, since a hand-picked
subset could select for cells we win, and it does not flatter us: on the 2026-09-04 rows the
subset read lower than the full set on three of the four Recurrent numbers. A subset run is
compared cell by cell with the same cells, never with the paper's geomeans. Only
`estimate_wall.py --write-subset` regenerates it.

## 5. The reviewer tier, run for real (2026-09-09/10)

Superseded by §5b. Two lessons stand: a pinned rival cell agrees by construction (its fresh row
is a copy), so the comparator leaves it out of the rival tally (58/58 measured, not "66/66"); and
the cost model carried over between campaigns within 1% (2.86 h modelled, 2.83 h spent).

## 5b. The tiers again, from the submitted package (2026-10-13/14)

Every tier, run from the package that goes to Zenodo (`gpap-artifact-20261013-d1e274d`) against
the rows it bundles (RIVAL_FAIRNESS_FINDINGS.md §17e):
- **`tables`**, offline, with stock `python3` only (Ubuntu 24.04, and Ubuntu 22.04 with Python
  3.10, the TACAS'23 VM's base): 163/163 published values, the gate 27 ok / 8 warnings /
  0 failures, Figure 3 byte-identical to the paper's.
- **`setup` + `smoke`**, RTX 4080 Laptop: every tool built from its pinned source; 4/4 cells,
  same verdicts.
- **`subset`**, Colab L4, three VMs: PASS, 66/66 against CMSGen and 58/58 against the rival,
  every grant honoured. Every count in Tables 1–3 matched the published rows on the same cells,
  and every ratio was within 0–5.5% (GPAP ran ×0.95–0.98 of the reference, the others
  ×0.99–1.01):

| row | Table 1 (ss) fresh / reference | Table 2 fresh / reference |
|---|---|---|
| demotic native | 1,263.8x / 1,335.2x | 2.15x / 2.25x |
| demotic comb rare | 232.4x / 240.0x | 7.6e5x / 8.0e5x |
| recurrent table, T=25 | 543.6x / 562.0x | 8.13x / 8.39x |
| recurrent rare, T=25 | 42.7x / 43.4x | 147.7x / 149.2x |

## 6. A cross-card reproduction, demonstrated

The smoke tier on the 12 GB 4080 against the L4 rows: the same verdicts on 4/4 cells, with speed
factors GPAP ×1.06–1.09, CMSGen ×1.79–1.92 (the host CPU), and the rival ×1.03–1.05 where its
grant fit but ×0.64 where it walked down, the handicap §1 predicts. The verdicts reproduce; the
rates describe the machine.

## 7. Tier 1 needs no GPU, and now no CUDA wheels either

The table generators read the soundness verdict through `gpap/verdict.py`, a stdlib-only module,
not through the sampler, so `tables` needs neither torch nor numpy. Gated by
`tests/experiments/test_paper_tables.py::test_the_table_generators_import_without_torch_or_numpy`,
in a subprocess with those modules blocked, since an in-process check would import them happily.

## 8. What a review of the artifact layer found (2026-09-10)

No published number moved; the artifact's gates were weaker than their prose.
- **A gate that passes what it did not check is worse than no gate.** `compare_campaign.py`
  passed a field missing on one side and a `sound_check` of `off`, and crashed on an empty stage.
  All three fail now, and `--expect-cells` fails a run that compared fewer cells than its tier.
- **A completion marker must be earned.** `ARTIFACT_TIER_COMPLETE` printed after a driver that
  died at cell 3; every step now records into a failure ledger that reaches the exit code.
- **A pinned cell agrees with itself** (§5).
- **One environment variable, two parsers.** C++ read `GPAP_VRAM_BUDGET_MIB` with `strtoll` and
  Python with `int()`, so `1e3` capped the two sides apart; both share one grammar now
  (`tests/engine/test_k_cap.py`).
- **Two budgets under one cap.** The rival's budget was a second implementation, 3.4% apart; it
  is now `gpap_mem_available` too (§2,
  `test_rival_budget.py::test_both_sides_of_a_head_to_head_get_the_same_budget`).
- **The cap was missing from the results cache key**, so a capped run could serve uncapped rows.
- **A test proves nothing about a branch it did not take.** The no-CUDA branches stayed uncapped
  until CPU-only CI ran them; the test now forces the fallback.

## 9. The third leak had no `/home/` in it (2026-09-13)

Three Barbarik error rows carried an agent's scratch path with the operator's username flattened
into a directory name, and the packager's one `/home/` pattern passed it. The shapes now live in
`eval/artifact/check_leaks.sh` (a Linux home, a macOS home, a flattened home, an agent scratch
root), run over the staged bytes and mirrored over the checkout by
`tests/repo/test_docs_hygiene.py`; `tests/experiments/test_artifact_gate.py` proves each shape is
refused on its own. Literal names come from a denylist outside the repository, since the script
ships inside the tarball it checks. The harness error text now names the instance file but not
its directory; the three rows were edited to that form, recorded in the paper folder's README and
manifest.

Running `pytest` inside the packaged tree found two more: gates that listed files with
`git ls-files` (an archive has no `.git`), and an uncapped rival budget when the core cannot
load.

The double-blind copy was retired on 2026-10-10: the TACAS'27 tool track is not double-blind. The
leak gate stays; its subject is the operator's identity, not the authors'.

## 10. The artifact as a Docker image (2026-10-19)

Two images: the tool (`Dockerfile`) and the artifact layered on it (`eval/artifact/Dockerfile`),
built from the archive by `make_image.sh` and gated by `check_image.sh`. On the RTX 4080 Laptop
under WSL2: the tool image is 8.47 GB, the artifact 8.80 GB sharing its layers (4.57 GB saved and
compressed); the engine compiles for the GPU on first use in 14 s and is reused from a cache
volume; `smoke` with `setup` takes 3 min 6 s.

Every gate passed. With no GPU and no network: the images start, the archive verifies against its
manifest, nothing names the operator, `check` and `tables` print what ARTIFACT.md quotes with
`RESULTS.md` and every figure as shipped, and the host suites pass. With the GPU: the engine
compiles once, a non-root user samples, the full test suite passes, and `smoke` passes, again
with the network cut after `setup`.

What building it taught:
- **No engine compiled without a GPU can ship.** CI's stub `libcuda` has no SONAME, so a library
  linked against it would load the stub at run time. The image compiles for the GPU it finds.
- **In a container the tests run as root**, which writes through permission bits; the one test
  that relies on them skips as root.

## 11. A review of the images, and native Linux (2026-10-19)

A review before the pull request found nine faults on the reviewer's path, each fixed with a gate
first seen to fail on the previous code:
- **PID 1 ignores a signal it has no handler for.** `docker stop` waited out its grace period and
  left the results root-owned. Each step now runs in its own process group, which a trap signals
  (exit 143, gate `stop-ends-a-step`).
- **`producer | grep -q` under `pipefail` fails when grep wins**: grep's early exit hands the
  producer SIGPIPE (141), and a note printed in 0 of 300 runs. It reads `setup.log` now.
- **Docker creates a missing bind source as root**, so the guide creates `out/` first.
- **The GPU steps ask the tool's entry point first**: no traceback without `--gpus`, and `check`
  shows that the engine builds.
- **Moving outputs moves every reader**: the packager reads `details/` first.

All 18 gates passed on the 4080 (tree `7f28258`).

**Native Linux:** Ubuntu 20.04, Docker 26.1.3, the NVIDIA Container Toolkit 1.20.1, and a
GeForce MX150 (sm_61) that GPAP cannot run on. The toolkit also creates `libcuda.so`; a compile
forced to sm_80 links against the mounted driver; the guide's smoke test as written runs `check`
and `tables`, refuses `smoke` in one line, and leaves every file in `out/` to the user; the nine
gates that need no GPU pass.

**Still not shown:** the engine running in the image on a GPU other than Ada.

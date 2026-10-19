# Artifact-evaluation findings

> **Reader:** contributors and researchers — the evidence behind the artifact's reproduction tiers. A dated research log, not a user guide.
>
> **Status:** settled.
> **Measured:** 2026-09-09/10 against the frozen campaign `l4_protocol_20260904`, and again
> 2026-10-13/14 against its re-run `l4_protocol_20261012`, the rows the paper prints (§5b) —
> NVIDIA L4 (24 GB, Colab) for the reference rows, RTX 4080 Laptop (12 GB) for the smoke tier;
> the Docker images 2026-10-19 on the same laptop under WSL2 (§10, §11), and on native Ubuntu
> 20.04 with a GeForce MX150 (§11).
> **Rule lives in:** `GPAP_VRAM_BUDGET_MIB`, clamped once inside `gpap_mem_available`
> (`cpp/src/calibrate.cpp`); `eval/artifact/reproduce.sh --vram-cap`;
> `eval/artifact/compare_campaign.py`; `eval/artifact/subset.json`; the images: `Dockerfile`,
> `scripts/docker_entrypoint.sh`, `eval/artifact/Dockerfile`, `eval/artifact/check_image.sh`.
> **Gated by:** `tests/engine/test_k_cap.py`, `tests/experiments/test_rival_budget.py`,
> `tests/repo/test_docker.py`, `tests/experiments/test_artifact_image.py`,
> `tests/experiments/test_artifact_steps.py`.
> **Record only:** §8, the review post-mortem on gates that could not fail.

The evidence behind `docs/papers/tacas-27/ARTIFACT.md` and `eval/artifact/reproduce.sh`
(protocol v1.0, `docs/papers/tacas-27/PROTOCOL.md`). Nothing here changes the protocol; the artifact layer only chooses
which cells a reviewer runs and under which two documented shortcuts.

## 1. 24 GB is a hard requirement, and the grants file proves it

Counted over `eval/rivals/grants/l4_protocol_20260904.json`:

| tool | cells | pinned | recorded failures | at the card's memory ceiling | granted the ladder top |
|---|---|---|---|---|---|
| DEMOTIC | 22 | 22 | 0 | 0 | 2 |
| Recurrent-CircuitSAT | 120 | 92 | 28 | **92** | 0 |

"At the memory ceiling" means the entry records an OOM one rung above the rung it was granted —
i.e. the reference L4's 24 GB is what stopped it, not the ladder and not the wall. **Every
pinned Recurrent cell is in that state**, so a card with less memory walks down on essentially
all of them, and a walked-down rung hands the rival less parallelism than the paper gave it,
which inflates our ratio. `compare_campaign.py` counts those cells and warns; `grant_source`
records it per row.

DEMOTIC's cells are memory-free by contrast: none sits at a ceiling, and two ran at the ladder
top (which `paper_tables` already names, since a top rung that completes first try means the
ladder bound the grant).

## 2. The other direction: a bigger card silently favours us

63 of the campaign's 141 measured rows carry `engine.K` **below** the 262,144 batch cap — they
are VRAM-bound, not cap-bound. On a larger card those Ks grow while the rival stays pinned to
its recorded batch, so an uncapped reproduction on an A100-80GB compares a grown K against a
fixed grant. Hence `GPAP_VRAM_BUDGET_MIB`, clamped inside `gpap_mem_available` — the one funnel
every consumer already calls, C++ and Python alike — with `reproduce.sh --vram-cap l4`
defaulting to the budget the published run had:

    (23.45 GB free - 800 MB reserve) x 0.80 = 18.12 GB = 17,280 MiB

23.45 GB free is what all 142 rows record, to two decimals. The cap is a ceiling only, so it is
inert on smaller cards.

**And it covers both sides, because there is only one budget now.** `rival_vram_budget` used to
be a second implementation of the same idea — `free x 0.80`, with no driver reserve — which read
17,894 MiB where ours read 17,281 on the very same card. One number could therefore reproduce
our budget or the rival's, never both: capping both at 17,280 handed the rival 614 MiB less than
the campaign had granted it, and a rung that just fit could OOM and walk down, inflating our
ratio in exactly the direction this section is about. It now calls `gpap_mem_available` itself,
so the two sides are the same bytes of the same card by construction
(`tests/experiments/test_rival_budget.py::test_both_sides_of_a_head_to_head_get_the_same_budget`).

Changing it costs nothing in the record: the modelled budget only ever picks a rung through
`run_at_batch`'s truncation fallback — "`batch`, the rung our VRAM model predicts, decides
nothing", the grant being the largest rung that COMPLETES — and **no cell in the campaign, 0 of
120, ever reached that fallback**. In a grants-pinned reproduction it cannot: the pinned rung is
the top of the ladder.

## 3. What a pinned reproduction costs, and what the rows do not time

Per cell, a grants-pinned run pays the rival's one recorded run (`wall_s`), our arm at three
seeds (`n_models / sps_e2e_with_delivery` plus `distinct_s`), and the CMSGen arm
(`cms_wall_s`). Summed over the campaign (`eval/artifact/estimate_wall.py`):

| | recorded failures pinned | re-measured |
|---|---|---|
| full campaign, 142 cells | 10.5 h | 19.8 h |

The 28 recorded failures are 9.3 h of the 19.8 — half a full reproduction spent re-deriving rows
that say "no measurement". That is why the artifact pins them by default and says so in every
row (`not_remeasured`); `--remeasure-failures` is the faithful run.

**The rows do not time everything.** Measured on the four smoke cells (RTX 4080 Laptop, 12 GB,
where two DEMOTIC cells also walked their grant down):

| stage | model | measured | residual |
|---|---|---|---|
| demotic native (c432) | 5.7 s | 26 s | +20 s |
| demotic comb rare (c432) | 5.8 s | 26 s | +20 s |
| recurrent table (b03) | 34.9 s | 49 s | +14 s |
| recurrent rare (b03) | 35.2 s | 47 s | +12 s |

So the unrecorded work is a per-cell **constant**, not a fraction: on the cheapest cells it is
four times the recorded wall. A percentage-only model underestimates exactly the tiers made of
many cheap cells.

Part of that constant scales with the netlist. The ABC simplify plus its equivalence check,
timed on the same box:

| circuit | gates | `prepare_variant` |
|---|---|---|
| b03 | 152 | 0.2 s |
| b13 | 342 | 0.2 s |
| b22 | 29,897 | 7.5 s |
| b17 | 32,192 | 9.5 s |
| b18 | 114,561 | 32.1 s |
| b19 | 231,266 | 71.9 s |

≈ 0.31 ms per gate, linear. This matters because the reviewer tier keeps b17–b19 and b22 —
their rival side is free (a pinned failure) while their preparation is a minute each. The
estimator therefore models the overhead as `per-cell constant + per-gate + fraction`, with
`--cpu-scale` for a slower host (the terms are CPU-bound; the Colab Xeon runs CMSGen at ~0.53x
this box, measured on the smoke cells' `cms_sps`).

Estimates that come out of it, for the reference L4 at `--cpu-scale 1.9`:

| tier | cells | pinned failures | re-measured |
|---|---|---|---|
| smoke | 4 | 4 min | 4 min |
| subset | 66 | 2.9 h | 5.8 h |
| full | 142 | 12.7 h | 23.0 h |

A cell that delivers NO models is not a free cell — it ran its three seeds to the harness's 60 s
per-seed wall and stopped. Pricing those at zero (11 cells of the campaign, 7 of them spending
three full seeds) is what the first version of this table did, and it is why the numbers above
are a little higher than the ones the reviewer tier was originally quoted.

## 4. The reviewer tier's cell set is chosen by the clock

`eval/artifact/subset.json`: a sequential instance is kept when both its T=25 tasks together
cost ten minutes or less on the reference card under the artifact's defaults. That keeps 22 of
30 (dropping b06, b14, b15, b20, b21, s15850.1, s35932, s38584) and all 11 ISCAS-85 under both
tasks: 66 of 142 cells for 23% of the wall. The rule is mechanical and about the clock alone,
which is the point — a hand-picked subset could select for cells we win. The selection is
committed, so an estimate and a run cannot describe different cell sets, and it is regenerated
only by `estimate_wall.py --write-subset`.

**It is a different population, and it does not flatter us.** The published rows restricted to
the subset:

| stage | cells | Table 1 (ss) subset / full | Table 2 subset / full |
|---|---|---|---|
| demotic native | 11/11 | 1,243.7x / 1,243.7x | 2.2x / 2.2x |
| demotic comb rare | 11/11 | 230.4x / 230.4x | ~7.8e5x / ~7.8e5x |
| recurrent table, T=25 | 22/30 | 541.8x / 820.4x | 10.3x / 21.5x |
| recurrent rare, T=25 | 22/30 | 41.1x / 78.0x | 176.8x / 173.2x |

Both DEMOTIC rows are reproduced whole; the two Recurrent rows lose their eight most expensive
instances and read LOWER on three of four numbers. So a reviewer's subset run is compared cell
by cell against these rows, never against the paper's geomeans, which are over all 142 cells.

## 5. The reviewer tier, run for real (2026-09-09/10)

`eval/artifact/reproduce.sh subset` on a rented L4 — the same card the campaign was measured on —
mirrored to a local run folder. 66 cells, 2 h 50 min wall across three VM lifetimes,
`--vram-cap l4` and recorded failures pinned (the artifact's defaults).

**Every cell agreed.** `compare_campaign.py`: PASS on all four stages, **66/66 cells reaching
the same verdict against CMSGen and 58/58 against the rival**, 66/66 rows sound and replayed
through the original netlist, **44 grants honoured at the recorded rung and none walked down**,
8 pinned failures (b17/b18/b19/b22 under each task) marked `not_remeasured`. The gate over the
fresh rows read 18 ok / 2 warnings / 0 failures — the same two warnings the published campaign
carries.

The two denominators are the point. A cell whose rival side was PINNED agrees with the reference
by construction — the fresh row is a verbatim copy — so counting it is counting the copy, and the
comparator now drops those 8 from the rival tally rather than folding them in. Their CMSGen arm
really ran, which is why that side keeps all 66. The first reading of this run said "66/66
against both"; it was 58 measurements and 8 tautologies, disclosed a sentence later but summed
into one number.

Same card, so the rates land where they should: ×0.98–×1.02 on every arm (GPAP, CMSGen and the
rival), against the ×1.85 CPU factor and ×0.64 handicapped rival the 4080 smoke run showed.
Table by table, against the published rows restricted to the same cells:

| stage | Table 1 (ss) fresh / reference | Table 2 fresh / reference |
|---|---|---|
| demotic native | 1,298.8x / 1,243.7x | 2.2x / 2.2x |
| demotic comb rare | 234.3x / 230.4x | 7.8e5x / 7.8e5x |
| recurrent table, T=25 | 550.6x / 541.8x | 10.2x / 10.3x |
| recurrent rare, T=25 | 42.5x / 41.1x | 178.1x / 176.8x |

Within 1-4% on every row, over the same `n` — a whole re-measurement of the pipeline, not a
re-read of the data.

**And the estimate held, twice over.** At the host's measured CPU factor (`--cpu-scale 1.9`,
from the smoke cells' `cms_sps`) the model reads 2.86 h off the REFERENCE rows and 2.85 h off
the FRESH ones — under 1% apart, so the per-cell cost model transfers between campaigns rather
than describing the run it was fitted to. The 2 h 50 min (2.83 h) actually spent is inside 1% of
both, and that number already contains two VM re-bootstraps and two in-flight cells lost to
hourly reclaims, so the pure work is if anything slightly under the estimate.

An earlier reading of this run put the estimate at 2.44 h and called the 2 h 50 a +17% overrun
attributable to the VM. That was the cost model's own gap, not the VM's: zero-model cells were
being priced at zero (§3), and the reviewer tier contains two of them.

## 5b. The tiers again, from the submitted package (2026-10-13/14)

The paper's rows moved to the re-run `l4_protocol_20261012` (RIVAL_FAIRNESS_FINDINGS.md §17e),
so every tier was run again from the package that goes to Zenodo
(`gpap-artifact-20261013-d1e274d`), against the rows it bundles:

* **`tables`**, offline Ubuntu 24.04 container, stock `python3` only: manifest verified, 163/163
  published values, gate 27 ok / 8 warnings / 0 failures, the paper's Figure 3
  (`sps_scatter_cmsgen.pdf`) byte-identical to the paper's.
  Again on 2026-10-15, from a package built at `85e9ffb` (after the `check` fix for a machine
  without torch), in an offline Ubuntu 22.04 container with stock `python3` 3.10, the base of the
  TACAS'23 VM: `check` printed no traceback; `tables` gave the same counts, and its `RESULTS.md`
  and `VERIFICATION.txt` were byte-identical to the package's.
* **`setup` + `smoke`**, RTX 4080 Laptop, fresh virtualenv from `requirements-artifact.txt`:
  every tool built from its pinned source; 4/4 cells, same formulas, sound, same verdicts.
* **`subset`**, Colab L4: the package repacked unmodified (each VM checked all 363 files against
  its `MANIFEST.sha256` before starting), about 2 h 50 min of running across three VMs, two reclaims
  resumed. `compare_campaign.py --expect-cells 66`: PASS, 66/66 against CMSGen and 58/58 against
  the rival (8 pinned failures not scored), every grant honoured. The third VM was reclaimed
  after the last cell and before it wrote its reports; they were generated from the mirrored
  rows with the same code instead of paying for a fourth.

Against the published rows restricted to the same 66 cells, **every count in Tables 1-3 is
identical** (`n`, wins, zeros, failures, exclusions) and every ratio is within 0-5.5%, all of
them slightly lower: GPAP ran ×0.95-0.98 of the reference, CMSGen and both rivals ×0.99-1.01.

| row | Table 1 (ss) fresh / reference | Table 2 fresh / reference |
|---|---|---|
| demotic native | 1,263.8x / 1,335.2x | 2.15x / 2.25x |
| demotic comb rare | 232.4x / 240.0x | 7.6e5x / 8.0e5x |
| recurrent table, T=25 | 543.6x / 562.0x | 8.13x / 8.39x |
| recurrent rare, T=25 | 42.7x / 43.4x | 147.7x / 149.2x |

## 6. A cross-card reproduction, demonstrated

The smoke tier on an RTX 4080 Laptop (12 GB) against the L4 rows:

* 4/4 cells: same formula, sound, replayable, **same verdicts** — PASS.
* Speed factors: GPAP ×1.06–1.09, CMSGen ×1.79–1.92 (the host CPU, not the GPU), the rival
  ×1.03–1.05 where its grant was honoured and **×0.64 where it was not** — the walked-down
  cells, exactly the handicap §1 predicts for a card under 24 GB.

That is the shape of the evidence a reviewer on other hardware can produce: the verdicts
reproduce, the rates are about their machine, and the one place the comparison could quietly
flatter us is counted and named.

## 7. Tier 1 needs no GPU, and now no CUDA wheels either

`paper_tables.py` and `verify_artifact.py` reached the soundness verdict through
`gpap.sampling.pipeline`, which imports the sampler, torch and numpy — so regenerating a table
in the TACAS evaluation VM would have meant an 800 MB install for a pure JSON reader. The
reader moved to `gpap/verdict.py`, a stdlib-only leaf that `pipeline` re-exports (one
definition, every existing import site unchanged). Gated by
`tests/experiments/test_paper_tables.py::test_the_table_generators_import_without_torch_or_numpy`, which
asserts it in a subprocess with those modules blocked — this interpreter has them installed and
would import them happily, so an in-process check would prove nothing.

Regenerated output is byte-identical to the tracked tables apart from the date and commit line
in each header.

## 8. What a review of the artifact layer found (2026-09-10)

Three reviews over the four commits. Nothing moved a published number; what they found was the
artifact layer's own gates being weaker than their prose. The four that mattered:

**A gate that passes what it did not check is worse than no gate.** `compare_campaign.py` could
print PASS over a campaign that had barely run. A deterministic field recorded on one side and
absent on the other was skipped as "not a mismatch"; a `sound_check` of `"off"` — which also
means *requested but never reached* — satisfied the safety tier; and a stage whose fresh file
held zero rows crashed the report instead of failing it. All three now fail, and
`--expect-cells`, which `reproduce.sh` fills in per tier, fails a run that compared fewer cells
than the tier set out to produce. Four deliberately broken fixtures (fields stripped, one cell
per stage, an empty stage, every arm delivering nothing) were used to check the new behaviour;
three now exit non-zero and the fourth reports the flips in its headline instead of claiming
"same verdicts".

**A completion marker must be earned.** `reproduce.sh` does not `set -e`, and printed
`ARTIFACT_TIER_COMPLETE` unconditionally after the tables and the comparison — so a measurement
driver that died at cell 3 produced a "Publishable" verdict, a PASS over whatever had been
written, and a marker the Colab supervisor reads as DONE. Every step now records into a failure
ledger, the marker is conditional, and the ledger reaches the exit code.

**A pinned cell agrees with itself.** A rival failure honoured from the grants file is copied
verbatim into the fresh row, so counting it as agreement counts the copy. The comparator now
drops those cells from the rival tally and reports them separately (§5).

**One environment variable, two parsers.** `GPAP_VRAM_BUDGET_MIB` was read by `strtoll` in C++
(longest numeric PREFIX) and by `int()` in Python (all or nothing). `1e3` therefore capped the
ENGINE to 1 MiB while the rival ran with the whole card, and `1_000` split the other way — the
two-sides-capped-apart failure the cap exists to prevent, reached from the opposite direction. A
huge value overflowed the MiB→bytes shift into a small positive cap that bound while `gpap info`
called it inert. Both sides now share one narrow grammar (blanks, an optional `+`, digits) and
answer *no cap* outside it, pinned by `tests/engine/test_k_cap.py::test_the_cap_grammar_is_one_grammar` and
`::test_the_core_reads_the_cap_grammar_the_same_way`, the second of which asserts against the
CORE and not only against the Python mirror.

**Two budgets under one cap.** The rival's budget was a second implementation — the same 0.80
of the same card, minus the driver reserve we take and it did not — so `--vram-cap l4` capped
the two sides 3.4% apart and could not reproduce both. Folded into `gpap_mem_available` (§2).

**And the cap was not in the results cache key.** `gpap_cache_key` hashed the CNF, the build id,
the sampling layer, the ABC fingerprint and the GPU name — so an uncapped run and a capped one
shared cache entries, and `--vram-cap l4` could serve numbers measured with the cap off while
printing the cap in its own provenance. The cap is part of the key now.

**The branch the developer's box takes is not the branch CI takes (2026-09-10).** Folding the
two budgets into `gpap_mem_available` moved the clamp into the core — and left the two no-CUDA
branches behind it uncapped: the rival's 5 GiB fallback, and our own 8 GB CPU placeholder. Both
are placeholders on a machine that cannot run a comparison, but the asymmetry is the one the cap
exists to remove, and the new-in-the-same-commit test could not see it: on any box with a card
it took the CUDA branch and passed. CPU-only CI found it on the first run. Both branches now
read the ceiling through the same reader, and the test forces the fallback
(`monkeypatch.setattr(torch.cuda, "is_available", ...)`) so it fails on either kind of machine.
The general form is the older lesson in one more disguise: a test proves nothing about a branch
it did not take, and "which branch" can be a property of the hardware rather than of the input.

## 9. The third leak had no `/home/` in it (2026-09-13)

Three rows of the published Barbarik campaign (`data/barbarik/circ_abc_b8.jsonl` and
`circ_abc_pipe.jsonl`, verdict `error`) carried the instance's absolute path inside the error
message: an agent's scratch directory, `/tmp/claude-<uid>/-home-<user>-…/`, with the operator's
username flattened into a directory NAME. The packager's gate, one `/home/` pattern inline, passed
it, and a tarball built on 2026-09-13 shipped it. Same lesson as §8, a different shape: a gate that
knows one shape of a leak is a gate for that shape.

**What changed.** The shapes live in one script, `eval/artifact/check_leaks.sh` — a Linux home, a
macOS home, a home flattened into one directory name, an agent scratch root — run by the packager
over the STAGED bytes and mirrored by `tests/repo/test_docs_hygiene.py` over the checkout.
`tests/experiments/test_artifact_gate.py` proves the script refuses each shape on its own and a denylisted
literal case-insensitively, and that a blank or comment line in the denylist matches nothing (to
`grep -F -f` an empty line is a pattern that matches every line, and the failure mode of that is an
operator setting the override for good). Literal names cannot be in the script, which ships inside
the tarball it checks; they come from a denylist outside the repository,
`GPAP_ARTIFACT_DENYLIST=<file>`. The harness now prints `display_path(input_file)`, and the three
rows were edited to the basename the fixed harness prints; the paper folder's README and
`data/MANIFEST.sha256` record the edit, and no verdict moved.

**Two more things the packaged tree taught**, found by running `pytest` inside it — the reviewer's
first command — for the first time. Four gates in `tests/repo/test_suite_hygiene.py` enumerate files
with `git ls-files`, and an archive has no `.git`: they now fall back to a pruned walk, proven to
see what git sees where git works. And with a card visible and no built library — an unpacked
artifact before `setup` — `rival_vram_budget` took the CUDA branch, the core call raised, and the
last-resort return handed the rival the whole fallback uncapped: the third route past the ceiling
(§8 closed the CPU-only one), now capped and gated by
`tests/experiments/test_rival_budget.py::test_rival_budget_is_capped_when_the_core_cannot_load`.

**The double-blind copy — RETIRED 2026-10-10.** The submission is a TACAS'27 regular tool paper,
which is not double-blind, so no anonymised copy is cut and the authors are named on purpose. The
leak gate STAYS, and its subject is now stated as what it always was in practice: the *operator's*
identity, not the authors' — a username or a hostname in a shipped file names the machine that
built the tarball, which is a defect in a signed artifact just as much as in a blind one. Recorded
because a venue can change: it was a transform over the packager's tarball,
a script kept outside the repository because it spells the author's
name — delete the brand kit and the development files, rewrite the identity lines (each
substitution must hit, or the script refuses as stale), the gate with the denylist, the host suite
inside the result, the manifest rewritten, repacked as `<name>-anon`. `GPAP_ARTIFACT_TREE=<tree-ish>`
lets the packager dry-run uncommitted work (`git write-tree` on a temporary index); the name and
the manifest then carry that object's id, so a dry run cannot pass as a release.

## 10. The artifact as a Docker image (2026-10-19)

The AE chairs granted the special-hardware exemption and asked for a Docker image. There are two:
the tool (`Dockerfile`) and the artifact, a layer over it (`eval/artifact/Dockerfile`), both built
from the artifact archive by `eval/artifact/make_image.sh` and gated by
`eval/artifact/check_image.sh`. Measured on an RTX 4080 Laptop GPU (12 GB), driver 596.58
(CUDA 13.2), under WSL2 with Ubuntu 24.04, Docker 29.1.3 (the legacy builder: no buildx) and the
NVIDIA Container Toolkit 1.20.1, from a dry-run archive of the branch (tree `79ea6a1`):

| | |
|---|---|
| the tool image | 8.47 GB |
| the artifact image | 8.80 GB loaded, sharing the tool's layers; both saved with `docker save \| gzip`: 4.57 GB |
| the engine's first-use compile, sm_89 | 14 s, then reused from the cache volume |
| `smoke`, `setup` included | 3 min 6 s: PASS, 4 cells, `ARTIFACT_TIER_COMPLETE` |
| all 15 gates | 9 min |

Every gate passed. With no GPU and no network: the images start; the archive verifies against its
`MANIFEST.sha256` inside the image; no operator identity in `/opt/gpap`, the image history or its
metadata; `check` says only `tables` can run; `tables` prints the five lines ARTIFACT.md quotes;
`RESULTS.md` differs from the shipped one only in its date, and every figure is byte-identical; the
host suites pass. With the GPU: the engine compiles on first use and is reused; a non-root user
samples a circuit (`--sim-check`) and a CNF; the full test suite passes (623); `smoke` passes, and
passes again with the network cut after `setup`; with no network at all it fails within 20 s,
saying why. The suites' one failure was the citation of this section before it existed.

**What building it taught.**
- **No engine compiled without a GPU can ship.** CI links the library against a stub `libcuda`
  that has no SONAME, and the linker then records the stub's absolute path, so the library would
  load the stub at run time (a two-file link test). The image compiles for the GPU it finds.
- **In a container, tests run as root**, which writes through permission bits:
  `test_tmp_dir_falls_back_when_the_root_is_unusable` could not pass there, and now skips as root.
- **`make_artifact.sh` given a relative output directory dropped the uniformity rows.** Their
  paths were resolved before it changed into the staging tree, so the generator found none and
  the packager refused, 4 of 163 numbers unproduced. The refusal was right; the directory is now
  made absolute first.
- **The leak gate's denylist is the operator's**, a username and a hostname kept outside the
  repository. The retired double-blind list names the authors, who ship on purpose.

**Not yet shown:** native Linux, where the toolkit mounts the driver as `libcuda.so.1` alone (the
image carries the `libcuda.so` link the compile needs), and a GPU other than Ada. *(Native Linux,
the same day: §11. The toolkit there also mounts `libcuda.so`.)*

## 11. A review of the images, and native Linux (2026-10-19)

A review of the branch before its pull request found nine faults on the reviewer's path and in
the packager. Each was reproduced, fixed, and given a test or a gate that failed on the previous
code first. What they teach:

- **PID 1 ignores a signal it has no handler for.** The entry point ran each step as a child
  with no trap, so `docker stop` waited out its grace period and killed it (exit 137), Ctrl-C did
  nothing, and the closing chown never ran: 19 of 19 files stayed root-owned. Each step now runs
  in its own process group (`setsid`), which a trap signals: `docker stop` returns at once, exit
  143 (the gate `stop-ends-a-step`).
- **`producer | grep -q` under `pipefail` fails when grep wins.** grep exits at its first match,
  the producer's next write gets SIGPIPE, and the pipeline returns 141: the "needs network
  access" note printed in 0 of 300 runs when the missing sampler was listed before a present one.
  It now reads `setup.log`'s `clone failed` line, so it also no longer blames the network for
  every other setup failure.
- **Docker creates a missing bind source as root**, so giving the results to the owner of `/out`
  changed nothing in a fresh directory, the documented flow: `rm -rf out` needed sudo. The guide
  now creates `out/` first, and only files the container created are given back.
- **The steps bypassed the tool's entry point.** Without `--gpus`, a tier printed a Triton
  traceback and then blamed the network; with a GPU, `check` said "engine core: NOT BUILT". The
  GPU steps now ask the tool's entry point first (`usable_cc`), and `check` gets its first-use
  compile.
- **Moving outputs moves every reader.** `tables` now writes into `details/`, but
  `make_artifact.sh` still copied the audit and the figures from the top of the paper folder, so
  a regeneration would have shipped the previous audit; it reads `details/` first. §10's
  relative-path fix resolved the path after changing into the checkout, so the archive was
  written into the repository; it now resolves where the path was typed.
- A tier removes its `SUMMARY.txt` when it starts, so a re-run that stops cannot show the last
  run's PASS; a bare `compare` prints the whole report and leaves the tier's summary alone; the
  comparator names a changed verdict in its closing block, which is what a tier's summary shows.

All 18 gates passed on the 4080 under WSL2 (tree `7f28258`): the 15 of §10 and three new ones
(`tier-without-gpu-says-why`, `check-with-gpu`, `stop-ends-a-step`), each first seen to fail on
the previous image.

**Native Linux:** Ubuntu 20.04 (kernel 5.15), Docker 26.1.3, the NVIDIA Container Toolkit 1.20.1
and driver 570.133.07, on a GeForce MX150 (sm_61, 2 GB), which GPAP cannot run on. It shows the
plumbing:
- the toolkit mounts `libcuda.so.1` and also creates `libcuda.so` when the container starts, so
  the image's own link (§10) resolves and is redundant here;
- a compile forced to sm_80 links against the mounted driver (`NEEDED libcuda.so.1`);
- the entry point says the card is sm_61 and GPAP needs 8.0 or newer;
- the guide's smoke test as written, from the saved image (loaded in 1 min 46 s): `check` runs,
  `tables` writes its five lines in 18 s, `smoke` stops in 4 s with that one line, and every
  file in `out/` belongs to the user; the nine gates that need no GPU pass.

**Still not shown:** the engine running in the image on a GPU other than Ada.

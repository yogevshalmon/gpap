# TACAS 2027 — *SAT Sampling at Scale via GPU-based Boolean Constraint Propagation*

Regular tool paper. Artifact evaluation is **mandatory** for this track.

The paper itself lives outside this repository (its own LaTeX tree) and stands on its own: no
`.tex` section is kept here, and no generated table is committed. What belongs here is the
*why* — which campaign the published numbers come from, how to re-derive them, and what holds
the paper and the data to each other.

> This directory is a work in progress: the evaluation how-to and the artifact checklist are
> still being written. `published_numbers.json` and this file are what exists so far.

## `published_numbers.json` — the contract with the paper

Every number the submitted PDF prints, transcribed by hand from its `.tex`.

The paper typesets its own tables: it drops columns, renames rows and reorders them to fit a
two-column LNCS page. So no byte comparison against generated LaTeX can work, and for a while
nothing checked the tables at all — Tables 1 and 2 printed absolute rate geomeans that the
generator did not even compute, thirty numbers whose only provenance was a step nobody could
repeat.

The division of labour that replaces it:

| | |
|---|---|
| the paper | chooses its own columns, labels and layout |
| `paper_tables.py --out-json` | derives every published number from the campaign's rows |
| `published_numbers.json` | says what the PDF prints, and which generated row each value checks |
| `check_paper_numbers.py` | compares the two and fails on a disagreement |

**Shapes may differ. Numbers may not.**

```bash
# over the published campaign
bash $GPAP_RESULTS/paper/regenerate.sh

# or as the artifact's no-GPU tier runs it (stock python3 is enough)
eval/artifact/reproduce.sh tables --out /some/dir
```

Both run the check. A value is compared **at its own printed precision**: the PDF's `2.31e6` is
a claim about three significant figures, so any generated value that rounds to `2.31e6` at three
significant figures agrees. A published *count* (`n`, `wins`, `eligible`) must match exactly,
because a count has no precision to round to.

### Reading a failure

The published file changes only when the PDF does. So:

- **failure right after editing the paper's numbers** — the transcription is wrong, or the
  number in the paper is;
- **failure with this file untouched** — the data moved under the paper. Find out which row
  moved before touching anything here. A value re-transcribed to make the check pass is a number
  the campaign never produced.

## Where everything else is

| | |
|---|---|
| the campaign's rows, tables, figures | `$GPAP_RESULTS/paper/` (outside the repo, by rule) |
| the protocol and its rationale | [`docs/EVAL_RUNBOOK.md`](../../EVAL_RUNBOOK.md) §0 · [`RIVAL_FAIRNESS_FINDINGS.md`](../../findings/RIVAL_FAIRNESS_FINDINGS.md) §17 |
| the artifact guide and its tiers | [`ARTIFACT.md`](../../../ARTIFACT.md) |
| how to measure on this hardware without fooling yourself | [`docs/BENCHMARKING.md`](../../BENCHMARKING.md) |
| the evidence behind every settled number | [`docs/findings/`](../../findings/) |

The protocol is **frozen** (v1.0, 2026-09-06): a review after the freeze fixes bugs and prose,
never a metric, config, screen, budget or cell set.

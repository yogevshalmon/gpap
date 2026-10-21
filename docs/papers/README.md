# Papers

> **Reader:** readers of a paper that uses GPAP — where its evaluation, its protocol and its
> artifact are.

| paper | submitted to | status | folder |
|---|---|---|---|
| *SAT Sampling at Scale via GPU-based Boolean Constraint Propagation* | TACAS 2027, tool paper | under review | [tacas-27/](tacas-27/README.md) |

A paper's status lives in this table only; its folder says where it was submitted, which stays
true whatever the outcome.

Each paper has one folder here, named for the venue and year it was submitted to (`tacas-27`),
and every document about the paper lives in it. The name does not change with the outcome: the
artifact and its tooling refer to it. A folder holds:

| file | what it is |
|---|---|
| `README.md` | the paper's evaluation, explained: what was compared, the rules the tables follow, where the data comes from |
| `PROTOCOL.md` | the rules the evaluation ran under, frozen before its campaign |
| `ARTIFACT.md` | how to regenerate the paper's tables, and to re-measure them |
| `published_numbers.json` | the numbers the paper prints from its measurements, checked against the data whenever the tables are regenerated |
| `AE_README.md` | optional: the README a venue's artifact submission asks for, an excerpt of `ARTIFACT.md` that a test keeps faithful |

The paper's own text lives outside this repository, and no generated table is committed: each
paper's artifact regenerates them from the campaign data it ships. The tool itself, in `gpap/`
and `cpp/`, never depends on a paper. The code that runs evaluations is in `eval/`, and the
measurements behind the evaluation's rules are in [../findings/](../findings/).

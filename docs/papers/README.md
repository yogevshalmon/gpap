# Papers

> **Reader:** readers of a paper that uses GPAP — where its evaluation, its protocol and its
> artifact are.

| paper | venue | folder |
|---|---|---|
| *SAT Sampling at Scale via GPU-based Boolean Constraint Propagation* | TACAS 2027, tool paper | [tacas-27/](tacas-27/README.md) |

Each paper has one folder here, named for its venue and year (`tacas-27`), and every document
about the paper lives in it. A folder holds:

| file | what it is |
|---|---|
| `README.md` | the paper's evaluation, explained: what was compared, the rules the tables follow, where the data comes from |
| `PROTOCOL.md` | the rules the evaluation ran under, frozen before its campaign |
| `ARTIFACT.md` | how to regenerate the paper's tables, and to re-measure them |
| `published_numbers.json` | the numbers the paper prints from its measurements, checked against the data whenever the tables are regenerated |

The paper's own text lives outside this repository, and no generated table is committed: each
paper's artifact regenerates them from the campaign data it ships. The tool itself, in `gpap/`
and `cpp/`, never depends on a paper. The code that runs evaluations is in `eval/`, and the
measurements behind the evaluation's rules are in [../findings/](../findings/).

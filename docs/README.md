# GPAP documentation

> **Reader:** everyone — which document answers what you want to do.

## Using GPAP

| you want to… | read |
|---|---|
| sample solutions of a CNF formula or a circuit | [SAMPLING.md](SAMPLING.md) |
| call GPAP from Python | [PYTHON_API.md](PYTHON_API.md) |
| call GPAP from C or C++ | [../cpp/README.md](../cpp/README.md) |
| start from working code | [../examples/](../examples/) and [../cpp/examples/](../cpp/examples/) |
| understand how the engine works and what it guarantees | [ENGINE.md](ENGINE.md) |
| measure GPAP, or compare it with another sampler, without fooling yourself | [BENCHMARKING.md](BENCHMARKING.md) |

Rates depend on the GPU and on its power state, so only ratios measured in one session on one
machine are comparable ([BENCHMARKING.md](BENCHMARKING.md) §1).

## Changing GPAP

| you want to… | read |
|---|---|
| build, test, follow the conventions, know where a fact belongs | [../CONTRIBUTING.md](../CONTRIBUTING.md) |
| know why the engine is built the way it is | [findings/INSIGHTS.md](findings/INSIGHTS.md) |

## The TACAS 2027 paper

| you want to… | read |
|---|---|
| reproduce the paper's tables and figures | [../ARTIFACT.md](../ARTIFACT.md) |
| know what the evaluation measures, and the rules behind its tables | [papers/tacas-27/README.md](papers/tacas-27/README.md) |
| read the evaluation protocol, or run a campaign of your own | [EVAL_RUNBOOK.md](EVAL_RUNBOOK.md) |

The paper's text lives outside this repository. No generated table is committed here: the
artifact regenerates every table from the campaign data it ships.

## The research log: `findings/`

Every design decision in the code rests on a measurement, and the measurement lives here: the
date, the GPU, the numbers, and what was tried and lost. [findings/INSIGHTS.md](findings/INSIGHTS.md)
is the index; each `*_FINDINGS.md` file is one topic and opens with a status block saying what it
settles, when and where it was measured, and where the rule it justifies lives in the code. A
number from a finding holds for the date and the GPU it names. Files marked historical record
directions that were dropped; quote no rate from them.

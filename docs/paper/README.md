# Paper experiment section — data, tables, and how to regenerate them

Everything in this directory is **generated from the experiment JSONs**. Nothing is typed by
hand, and that is deliberate: an earlier version of the deck carried a sequential headline of
`47.7×` that silently included four instances whose output pins are tautologies. The real number
is `36.5×`. The only durable fix is to derive every published figure from the data with the
exclusions applied in exactly one place.

`eval/artifact/slide_tables.py` **imports** its aggregation from `eval/artifact/paper_tables.py`
rather than restating it, so the deck cannot drift away from the paper. When the generator
learned to count our own zeros, the deck learned it in the same commit.

## The campaign these tables report

**`$GPAP_RESULTS/l4_protocol_20260904`** (default `../gpap-results/l4_protocol_20260904`, a
sibling of the checkout — the raw data is too heavy to track). THE converged protocol
([../EVAL_RUNBOOK.md](../EVAL_RUNBOOK.md) §0): measured 2026-09-04 to 2026-09-06 on a Colab **NVIDIA L4**
(host CPU Intel Xeon @ 2.20 GHz, recorded per row), engine build
`libgpap 0.3.0+abf8f5e147090437` — the build id every row records and the header of
`RESULTS.md` re-derives from the rows. Three trees ran: the cells kept from 2026-09-04 at
commit `207e388` plus the uncommitted §16/§16b harness fixes (`worktree_vs_HEAD.patch`, since
committed); the cells re-measured on 2026-09-05/06 under the §16d decisions plus the
calibration at commit `86d5f0c` plus `worktree_vs_HEAD_launch2d_20260905.patch`; and the 17
cells of the 2026-09-06 cleanup run (the ladder-bound cells re-walked from a 30,000,000 top,
`b22`-rare on a clean device) at commit `59bdf20` plus `worktree_vs_HEAD_cleanup_20260906.patch`,
the tree that also generated the tables. All in the campaign directory. It supersedes `l4_protocol_20260903`, whose
numbers two adversarial review rounds found to rest on four defects that each moved a number in
our favour (`docs/findings/RIVAL_FAIRNESS_FINDINGS.md` §16, §16b, §16c).

* **The rivals' own benchmark sets, nothing else**: DEMOTIC's 11 ISCAS-85 (T=1) and every
  ITC'99 `b*` / ISCAS-89 `s*` Recurrent-CircuitSAT evaluates (30 circuits, T=25 — the
  T=50 rows are the ADDITIVE appendix run under the identical protocol, printed beside the
  T=25 rows and never in their place — "not run" until they exist).
* **Two tasks per circuit**: the pins the rival's own run script names, and our rare pin
  (all outputs, the rarest pattern of a 262k-sequence simulation, a miter for the rival).
* **Three tools on one constraint per row**: the rival at its own configuration
  (`num_steps=1`, `lr=15` — its shipped script setting; seed 0; the largest batch rung it
  completes, walked down from the ladder's 30,000,000 top; 1,200 s per run and a separate
  1,200 s initialization budget for Recurrent's `start_point` search, which bounds the search
  and is in no published rate; every tool is handed the SAME sequential-ABC-simplified
  netlist), GPAP on the `abc:fast` CNF of the same
  circuit and pins (`circuit` portfolio, 60 s, the median-rate run of seeds 5/6/7,
  `--verify-sound`, kept rows simulated through the original netlist, exact dedup off the
  clock), and CMSGen (patched, `--nowrite --maxtime 60`) on that very CNF file. Tables 1, 2 and
  3 are read off the same 82 rows (81 measured + 1 GPAP encoder error, `b19`).
* **Uniformity = Barbarik CM_test only** (Table 4b), from `l4_abc_20260902/barbarik_abc/`: run
  locally (RTX 4080) on `abc:fast`-encoded circuits under OUR pins — a separate campaign on
  different instances, stated as such in the table itself; a CM_test verdict does not depend
  on the card.
* **Table 2 has ONE clock, the rivals' own** (unique solutions per second of their
  `model_runtime`, against GPAP's steady-state unique/s) and no end-to-end column, by rule:
  that column had four definitions in three review rounds and measured process start-up on
  cells whose real work is under a second. Table 1 keeps both clocks. The headline is the FULL
  set (30 sequential circuits, 11 ISCAS-85); each rival paper's own subset is printed beside it.
* **The grant each rival received is part of the record**: `eval/rivals/grants/l4_protocol_20260904.json`
  (written by `eval/rivals/extract_grants.py` off these rows) holds every cell's batch rung and
  Recurrent's `start_point`, and a re-run with `--grants` / `GRANTS=` pins them — no ladder
  walk, no probe.
* No family matrix, no tseitin arm, no HTS, no soft table, no recheck arm.

**The protocol is FROZEN as v1.0 (2026-09-06; [../EVAL_RUNBOOK.md](../EVAL_RUNBOOK.md) §0, the
record and the considered-not-adopted list in `docs/findings/RIVAL_FAIRNESS_FINDINGS.md` §17).**
A review from here fixes a bug in a number or fixes prose; it changes no metric, configuration,
screen, budget or cell set. Numbers may still move for three reasons only: the pre-freeze
cleanup run (the cells the old 3,000,000 ladder top had bound, re-walked from 30,000,000, and
`b22`-rare's GPAP arm on a clean device), the T=50 appendix rows (additive), and a bug fix.

The campaign's own `README.md` is the protocol record: what changed against the superseded
run, the scope that was deliberately trimmed, the headline geomeans, every loss and limit by
name (`c1355`/`c499` to DEMOTIC on its pins, `c432` a tie at its 10,000,000 grant;
`b11`/`b12`/`b18` to CMSGen under the rare pin;
the circuits on which Recurrent did not initialize inside the 1,200 s budget (`b17`, `b18`,
`b22` under both tasks, `b19` under its own pin) — a budget statement, not a capability one,
and six that had failed at 300 s initialize at 1,200 s; the structure of the 21.5x Recurrent-table
headline (10.6x without its top four cells) and the batch each big instance was granted on the L4; the vacuous, tiny, near-free, proved-UNSAT and thin cells),
and the VM history.

| campaign path | content |
|---|---|
| `rivals_{demotic_native,demotic_comb_rare,recurrent_table,recurrent_rare}.json` | the 82 rows: rival + GPAP arm + CMSGen arm each |
| `rivals_steps_calibration.json`, `rivals_lr_calibration.json` | the rivals' OTHER author-shipped values (argparse default, paper, paired paper), measured once each on this card — evidence, printed by `paper_tables` under Table 2, never run as configuration |
| `RESULTS.md` / `tables.tex` / `plots/` | generated by `paper_tables.py --rivals` / `eval_plots.py` over the rows |
| `VERIFICATION.txt` / `emission_audit.txt` | `verify_artifact.py --rivals --barbarik …` (0 failures) / `check_emission_symmetry.py --results` (PASS) |
| `prebuilt/`, `worktree_vs_HEAD*.patch`, `supervisor.log`, `run.log`, `split_{A,B}/`, `rerun_20260905/` | the trees the VMs ran, the bootstrap, the VM-cycle history, the two-session completion mirrors and their merge, the removed pre-completion rows with reasons |

Two earlier campaigns are kept and superseded: `l4_protocol_20260903` (the first protocol
run — the record of what was measured and of how it was wrong) and `l4_abc_20260902` (family
matrix, GPAP vs CMSGen only, plus a rivals re-run), which predates the protocol and survives
for the Barbarik arm and as the engine A/B baseline.

## Files in this directory

| file | what it is |
|---|---|
| `RESULTS.md` | every table, plus the caveats that must appear in the text |
| `tables.tex` | the same tables as `booktabs` LaTeX, ready to `\input` |

The WAVE-vs-SCALAR kernel A/B table (`KERNEL_AB.md` / `kernel_ab.tex`,
`eval/artifact/kernel_ab_tables.py`) was removed rather than left stale: its data is the
2026-08-23 pre-shuffle-schedule engine, which the 2026-08-27 wave rewrite superseded
(`docs/findings/WAVE_SHUFFLE_FINDINGS.md` — the tie band's crossover moved and must be
re-derived, not guessed). The dispatch-rule evidence lives in
`docs/findings/KERNEL_AB_FINDINGS.md`; regenerate the table from a fresh `eval/kernel_ab.py`
campaign against the current engine before citing a kernel A/B number.

## Regenerate

```bash
R=${GPAP_RESULTS:-../gpap-results}

eval/artifact/paper_tables.py \
    --rivals   $R/l4_protocol_20260904 \
    --barbarik $R/l4_abc_20260902/barbarik_abc/circ_abc_*.jsonl \
    --out-md docs/paper/RESULTS.md --out-tex docs/paper/tables.tex

eval/artifact/verify_artifact.py --rivals $R/l4_protocol_20260904 \
    --barbarik $R/l4_abc_20260902/barbarik_abc/circ_abc_*.jsonl          # the publishability gate
eval/rivals/check_emission_symmetry.py --results $R/l4_protocol_20260904 --cycles 25
eval/eval_plots.py $R/l4_protocol_20260904        # rival figures, incl. the unique/s scatter
# or all four at once: bash $R/l4_protocol_20260904/finish_campaign.sh
```

The all-cells-at-once scatter (`sps_scatter_rivals.png`: every rival cell on one log-log
unique/s plot) answers the question the geomeans hide — how the instances are distributed
around a ratio — and imports its cell classification from `paper_tables.rival_cells`, so the
figure cannot show a cell Table 2 screens. (`sps_scatter_all.png`, the CMSGen family-matrix
scatter, belongs to the supplementary `gpap bench` stages, which this campaign does not run.)

The header of `RESULTS.md` carries two commits and they mean different things: the **measured
engine** is the per-row build id (unioned over every stage — more than one value means mixed
engines), while `tables generated at` is only the HEAD the generator ran under. Re-running the
block above at a later commit moves the second, never the first.

## What the experiments are

| experiment | question it answers |
|---|---|
| GPAP vs CMSGen (Table 1) | throughput against the SOTA uniform-ish CPU sampler, patched for fair timing |
| GPAP vs 3 GPU rivals (Table 2) | throughput against the gradient-descent GPU samplers, on their metric |
| propagation rate (Table 3) | is the speedup the engine or the benchmark? |
| Barbarik CM_test (Table 4b) | the standard conditional-sampling uniformity certificate |

**Absent by design, and the absence is load-bearing**: no reproducibility re-check arm has been
run against this campaign's engine yet, so `RESULTS.md` carries no Table 6 — an un-run check is
neither a pass nor a failure, and the generator only prints the table when the arm exists.

## The nine rules the generator enforces

These exist because each one, left implicit, produces a number that a referee would reject.
Rules 1, 5 and 6 are three faces of the same principle: **a cell that measures the instance
rather than the sampler does not belong in a geomean.** The incident record behind each rule is in
`docs/findings/` (RIVAL_FAIRNESS, L4_EVAL, UNIFORMITY).

1. **Vacuous pins are excluded from sequential geomeans.** `b02`, `b05`, `b07`, `s526` have
   constant last-cycle outputs at T=25, so pinning them is a tautology and every sampler scores
   ~100%. Including them inflates the *rare* column — the one we tell people to quote.
2. **A ratio against a sampler that produced nothing is not a ratio.** Where CMSGen emitted zero
   samples inside the budget the ratio is ∞; those cells are counted as outright wins and
   excluded from every geomean, which makes the published figures *conservative*.
3. **The zero ledger is symmetric.** A CMSGen-zero row carries `ratio = inf`, a GPAP-zero row
   `0.0`, a both-zero row `None` — the last two are falsy, and branching on `ratio` alone once
   showed 38 of their zeros as wins while hiding all of ours. Table 1 carries `CMSGen=0`,
   `GPAP=0` and `both=0`, and they sum with the geomean's instances to the instance count.
4. **Absent is not zero.** A row whose CMSGen arm never ran (a campaign resumed on a VM that
   lost the binary) or whose subprocess crashed (`cms_error`) is neither a win nor a zero: it is
   screened per row, named in the caveats, and `verify_artifact.py` FAILS a stage that mixes
   armed and armless rows. The producer refuses too — `gpap bench --resume` will not append rows
   with a different arm than the file holds.
5. **Zero-search cells are excluded.** EPFL `ctrl`, `dec`, `square`, `cavlc` emit at hundreds of
   M samples/s with *no decisions and no implications*: after propagation the support is a free
   hypercube, so no search happens. Detected per row (`paper_tables.zero_search`), not by a name
   list, and continuously rather than at exactly zero — `c432` under a rare pin delivers a model
   per 1.09 implications and was an entire published geomean at 89,786×. (The EPFL figures
   sometimes quoted here, 257.0× vs 108.9×, are from the supplementary family sweep, not from the
   rival campaign these tables report.)
6. **A solution space smaller than the sample request is enumerated, not sampled.** ISCAS-85
   `c17` has 5 inputs = 32 points; both tools saturate it and the ratio becomes a race between
   startup costs. DEMOTIC's own Table II says the same thing by carving it out — "a minimum of
   1000 distinct solutions (**except c17**)" — and that 1000 is the threshold
   `paper_tables.tiny_space` uses (`2**support < 1000`, read off the rival's own `num_inputs` on
   a Table 2 row). It is applied to BOTH tables, and it does not flatter us: it removes a
   45,520× cell from Table 2 (DEMOTIC-native would otherwise read ~6.5× instead of 2.7×) while
   *returning* 1,348× → 1,620× to Table 1's combinational row, where the tiny space made us look
   worse. A screen that only ever helped would be the one to distrust.
7. **Every screen applies to every table.** Vacuous pins were screened in Table 1 from the start
   but not in Table 2 — invisible while no vacuous instance was in the rival set, and live the
   moment that set became each rival's own published one (Recurrent's Table II includes `s526`).
   Both tables now share `zero_search`, `tiny_space` and `VACUOUS_PINS`, and both ledgers sum.
8. **A geomean over fewer than three instances is flagged, not quoted.** On HWMCC, CMSGen scores
   zero on 2–3 of 7 instances depending on depth and target; the ledger columns carry what the
   thinned geomean cannot.
9. **A GPAP/CMSGen ratio without its CPU is meaningless.** CMSGen is single-core, so the ratio
   is as much a property of the host as of the sampler: this campaign's Xeon is 0.41× the
   reference laptop's CPU while GPAP itself moves 0.96× between the cards. `gpap bench` records
   `cpu` in every JSON for this reason — quote it with every ratio.

### The propagation table answers the obvious objection

Sample-rate ratios in Table 1 span two orders of magnitude, which invites "you picked your
benchmarks". The counters answer it: GPAP holds a near-constant implication rate across every
family, against CMSGen's own `--statsevery` counters — an engine advantage of two to three orders
of magnitude that is essentially independent of the benchmark — and the identity

⚠️ **Quote the numbers from the generated tables, never from this file** — and note that this
paragraph proved the point on itself: it once carried 2.07×10⁹ / 611× from `l4_abc_20260902`
beside tables reporting a different campaign, and the replacement figures went stale within the
hour, on the next regeneration. Prose is where a stale number survives, so this paragraph now
carries none. `RESULTS.md` is generated; this file is
prose, and prose is where a stale number survives.

```
samples/s ratio = (implications/s ratio) / (implications-per-sample ratio)
```

closes on every family row. ⚠️ It is an algebraic identity over one row set, not a validation:
`work := (a/s)/(b/t)` makes `impl/work ≡ s/t` by construction, so it can only fail if the
aggregation mixes row sets — which is exactly the per-column-filtering bug it is there to catch,
and is all it should be claimed to catch. `verify_artifact.py --rivals` checks precisely that:
Table 3's row set against Table 1's, on the generator's own printed numbers, plus a physical
band on the counters themselves (an identity cancels a corrupt counter; a range does not). Until
2026-09-05 its identity loop read only the bench schema and Table 3 shipped ungated. What varies
between families is how much propagation a conflicted trajectory wastes (rejection cost), not the
propagator.

### Table 4b — what a Barbarik verdict does and does not say

Parameters are the CMSGen paper's own certification setting (ε=0.3, η=1.8, δ=0.1 ⇒ 1,872
samples per instance). Of 92 generated instances, 68 pass the feasibility screen and **65 are
eligible** — eligibility is bucketed on the **negative control's verdict** (SPUR, an exactly
uniform sampler, must Accept), never on the screen's own estimate. Results: `spur` 65/65 Accept,
`cmsgen` 65/65, `gpap:b8` 63 Accept / 0 Reject / 2 no-verdict, `pipe:circuit` 60 / 0 / 5 —
**zero GPAP Rejects, parity with CMSGen up to coverage**. Three readings matter:

* **The guarantee is asymmetric**: a Reject is unconditional; an Accept means *not caught*
  under Barbarik's non-adversality assumption, never "proven uniform".
* **"No verdict" bounds coverage, not uniformity**: Barbarik's kernel enlarges the sampling set
  while pinning the support — adversarial for a rejection sampler by construction — so an order
  GPAP cannot fill is neither verdict.
* The full control story (including the positive control that must be Rejected) is
  `docs/findings/UNIFORMITY_FINDINGS.md`; `eval/uniformity/barbarik_table.py` is the one
  aggregator and refuses to pool parameter settings.

## Comparability notes

* GPAP is the **2026-09-01 core-sampler engine** (RNG moved into libgpap): same-seed model
  *counts* are not comparable across that boundary; *rates* are. Cross-checking these tables
  against the 20260823/20260827 full-matrix campaigns compares two engines, not two protocols.
* `eval/artifact/make_artifact.sh` (the tar-with-MANIFEST packager) still speaks the
  full-matrix campaign layout; it refuses stale or mismatched arms, so it cannot silently
  package this campaign. Until it learns this layout, the campaign directory itself — with
  `VERIFICATION.txt` and its own `README.md` inside — is the organized artifact of record.

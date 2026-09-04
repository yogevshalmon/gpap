#!/usr/bin/env python3
"""Verify the experiment artifact is publishable: sound, complete, and self-describing.

Run this before quoting a number anywhere. It is deliberately a CHECKER, not a summariser — it
answers "is there anything here I would be embarrassed to defend?" and exits non-zero if so.

    eval/artifact/verify_artifact.py <results-dir> \\
        --rivals .experiments/l4_rivals --uniformity .experiments/uniformity/*.json

Checks, and why each one exists:

  SOUNDNESS   no scored cell may REPORT itself unsound. The on-GPU re-check is opt-in
              (`gpap bench --verify-sound`), so most cells simply carry no verdict — that is
              counted, not required. An unsound row means the
              sampler emitted something that is not a model, which invalidates the whole run --
              this is the one check that can never be waived.
  FAIRNESS    CMSGen must be the PATCHED binary in every stage. Stock CMSGen is timed WITH sample
              emission (~88% of its wall on 27.sk_3_32), which silently inflates every ratio.
  PROVENANCE  each stage must record GPU *and* CPU. CMSGen is single-core, so a ratio without its
              CPU is not interpretable (measured: the same suite reads 48.5x and 113x on two
              machines purely because one host is 0.41x the other).
  AGGREGATION the vacuous-pin instances must not sit inside a sequential geomean, and a geomean
              standing on fewer than 3 instances must be flagged rather than quoted.
  COVERAGE    stages short of their family's instance count are listed, so a partial sweep can
              never be mistaken for a complete one.
  ORACLE      the uniformity campaign must include BOTH controls; without an exact-uniform
              negative and a known-TV positive, "no skew detected" is uncalibrated and unusable.
  RECHECK     (--recheck DIR) the reproducibility claim is re-derived from the raw rows: the
              deterministic fields (M, support, packed split, propagate variant) must match the
              original sweep exactly on every shared cell.
"""
import argparse
import glob as globmod
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.artifact.paper_tables import (CMS_FAMILIES, cms_arm_ran,          # noqa: E402
                                        sound_failed, sound_unchecked)

VACUOUS_PINS = {"b02", "b05", "b07", "s526"}


def load(p):
    try:
        return json.load(open(p))
    except Exception:                                   # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmsgen_dir", nargs="?", default=None,
                    help="`gpap bench` stage directory — the supplementary family matrix. "
                         "Optional: the converged protocol reads Table 1 off --rivals")
    ap.add_argument("--rivals", default=None,
                    help="the campaign directory (rivals_*.json) — every row carries the "
                         "rival, the GPAP arm and the CMSGen arm on one CNF")
    ap.add_argument("--uniformity", nargs="*", default=[])
    ap.add_argument("--recheck", default=None,
                    help="reproducibility re-measurement directory (same stage layout); the "
                         "deterministic fields must match the original sweep on every shared "
                         "cell, or the artifact fails")
    args = ap.parse_args()
    if not (args.cmsgen_dir or args.rivals):
        ap.error("nothing to verify: pass --rivals (the campaign) and/or a `gpap bench` directory")

    fail, warn, ok = [], [], []

    # ---------------------------------------------------------------- CMSGen sweep
    stages = (sorted(globmod.glob(os.path.join(args.cmsgen_dir, "*.json")))
              if args.cmsgen_dir else [])
    tot_cells = tot_unsound = tot_unchecked = 0
    fam_inst = {}
    for p in stages:
        d = load(p)
        if not d:
            fail.append(f"unreadable JSON: {p}")
            continue
        stem = os.path.basename(p)[:-5]
        if not isinstance(d, dict) or "results" not in d:
            warn.append(f"{p}: not a `gpap bench` stage (no `results`) — skipped, not checked")
            continue
        rows = [r for r in d.get("results", []) if not r.get("skipped") and not r.get("error")]
        tot_cells += len(rows)
        # MIXED CMSGen provenance: some rows carry the comparison arm and some never ran it —
        # the signature of a campaign resumed on a VM where the cmsgen binary failed to restore.
        # The armless rows carry no `cms_*` keys, and any consumer that read the missing count
        # as a number would score them as CMSGen zeros (phantom wins). paper_tables screens
        # them per row; here the stage FAILS, because the fix is to re-run the cells, not to
        # publish around them. A FULLY armless stage is fine — that is a GPAP-only sweep.
        armed = [r for r in rows if cms_arm_ran(r)]
        if armed and len(armed) != len(rows):
            fail.append(f"{stem}: MIXED CMSGen provenance — {len(rows) - len(armed)} of "
                        f"{len(rows)} rows never ran the CMSGen arm (a resume without the "
                        f"binary?). Re-run those cells or delete them; absent is not zero")
        # A crashed CMSGen subprocess (binary missing/unrunnable on that VM) is recorded as
        # `cms_error` with a count of 0. That zero is an environment failure, not a measurement,
        # and scoring it publishes a broken binary as a stage of outright wins.
        cerr = [r for r in rows if r.get("cms_error")]
        if cerr:
            fail.append(f"{stem}: the CMSGen subprocess ERRORED on {len(cerr)} row(s) "
                        f"({', '.join(sorted(r.get('name', '?') for r in cerr)[:6])}"
                        f"{', …' if len(cerr) > 6 else ''}) — a crashed baseline is not a "
                        f"zero; re-run before publishing")
        # A row that says UNSOUND is a hard failure. A row that ran NO check is not: the
        # re-check is opt-in (`gpap bench --verify-sound`), it is a self-test rather than part
        # of producing the numbers this file gates, and demanding it would make the artifact
        # depend on a flag nobody needs for a throughput claim. Counted, never required.
        uns = [r for r in rows if sound_failed(r)]
        tot_unsound += len(uns)
        for r in uns:
            fail.append(f"UNSOUND cell {stem}/{r.get('name')}/{r.get('enc_key')}")
        tot_unchecked += sum(1 for r in rows if sound_unchecked(r))
        if d.get("cmsgen_patched") is not True and any(r.get("cms_sps") for r in rows):
            fail.append(f"{stem}: CMSGen present but NOT patched — ratios inflated by emission")
        modes = {r.get("cms_mode") for r in rows if r.get("cms_mode")}
        if modes - {"nowrite"}:
            fail.append(f"{stem}: CMSGen timed in mode(s) {modes}, expected only 'nowrite'")
        if not (d.get("gpu") or {}).get("name"):
            fail.append(f"{stem}: no GPU recorded")
        if not (d.get("cpu") or {}).get("model"):
            warn.append(f"{stem}: no CPU recorded — a GPAP/CMSGen ratio is not interpretable "
                        f"without it (this stage predates the cpu provenance fix)")
        # ---- ONE ENGINE PER STAGE -----------------------------------------------------------
        # `gpap_build_id` fingerprints the loaded library's own bytes, and compilation is
        # deterministic, so two ids in one stage means two different engines produced rows that
        # a table then geomeans together. It is what `--resume` does when a stage is topped up
        # after a rebuild (or on a VM that built from a different tree): the 2026-09-02 seq
        # stages carry 19 rows on one build and 10 on another — exactly the instances that were
        # added incrementally. Not fatal, because the added rows may be perfectly good; but the
        # campaign cannot claim one engine, and this is the only place that would notice.
        cores = {}
        for r in rows:
            c = (r.get("engine") or {}).get("core")
            if c:
                cores[c] = cores.get(c, 0) + 1
        if len(cores) > 1:
            warn.append(f"{stem}: rows span {len(cores)} ENGINE BUILDS "
                        + ", ".join(f"{c} ({n} rows)" for c, n in sorted(cores.items()))
                        + " — a resumed stage was topped up under a different libgpap; the "
                          "table geomeans them together. Re-run the stage, or say so.")
        inst = {r["name"].split(".")[0] for r in rows}
        fam_inst.setdefault(stem.split("-")[0], {})[stem] = len(inst)
        # ---- CMSGen's sample count must be EXACT where it is published -----------------------
        # A killed CMSGen reports its last --statsevery multiple, so `cms_n_samples: 0` can mean
        # "fewer than 1000 in the budget" rather than "none". On the 2026-08-23 campaign that
        # was 174 of 357 rows, 81 of them recorded as zeros and counted as our wins; one of them
        # (b14/abc:fast) really delivers 941 samples, i.e. an ~10-25x LOSS published as an
        # infinite win. The patch's --maxtime makes CMSGen stop itself and print an exact final
        # count (`cms_count_exact`), so a floored count is now a defect, not a fact of life.
        floored = [r for r in rows
                   if r.get("cms_count_exact") is False or
                   (r.get("cms_count_exact") is None and r.get("cms_timed_out"))]
        if floored:
            zeros = [r for r in floored if not (r.get("cms_n_samples") or 0)]
            msg = (f"{stem}: {len(floored)} row(s) carry a FLOORED CMSGen count (killed between "
                   f"--statsevery lines), {len(zeros)} of them recorded as an exact 0")
            (fail if zeros else warn).append(
                msg + " — re-run with a --maxtime-capable cmsgen (scripts/build_cmsgen.sh)")
        # ---- aggregation traps. One cell per (instance, encoder): there is no best-of ---------
        best = {(r["name"].split(".")[0], r.get("enc_key") or r.get("encoder")): r
                for r in rows}
        # The same population as the paper's n_gm: finite ratios, minus vacuous pins AND
        # zero-search cells (re-derived below). Counting the zero-search cells here made this
        # warning disagree with the table about how many instances the geomean stands on.
        finite = [r["ratio"] for (k, _e), r in best.items()
                  if r.get("ratio") and math.isfinite(r["ratio"])
                  and k not in VACUOUS_PINS
                  and not ((r.get("n_models") or 0) > 0
                           and not (r.get("decisions_per_s") or 0)
                           and not (r.get("implications_per_s") or 0))]
        if 0 < len(finite) < 3:
            warn.append(f"{stem}: geomean stands on only {len(finite)} instance(s) "
                        f"(a small family, or CMSGen scored zero / the screens removed the "
                        f"rest) — flag it, do not quote it")

        # SYMMETRY of the zero ledger. Reporting the instances CMSGen could not sample while
        # dropping the ones WE could not is the single easiest way to publish a flattering table
        # by accident; it is exactly what the first version of paper_tables.py did, because a
        # GPAP-zero row carries ratio == 0.0 and fell through every branch.
        # Only rows whose comparison RAN enter the ledger: an armless row (GPAP-only sweep)
        # is not "CMSGen scored 0", and reading it into b0 claimed "defeated both samplers"
        # about a tool that never ran.
        g0 = [f"{k}/{e or '-'}" for (k, e), r in best.items()
              if cms_arm_ran(r)
              and (r.get("cms_n_samples") or 0) > 0 and not (r.get("n_models") or 0)]
        b0 = [f"{k}/{e or '-'}" for (k, e), r in best.items()
              if cms_arm_ran(r)
              and not (r.get("cms_n_samples") or 0) and not (r.get("n_models") or 0)]
        if g0:
            warn.append(f"{stem}: GPAP emitted NOTHING on {len(g0)} instance(s) that CMSGen "
                        f"sampled ({', '.join(sorted(g0))}) — these are losses and must appear "
                        f"in the table, not only CMSGen's zeros")
        if b0:
            warn.append(f"{stem}: {len(b0)} instance(s) defeated both samplers "
                        f"({', '.join(sorted(b0))}) — report as a limit of the comparison")

        # ZERO-SEARCH cells: samples emitted with no decisions and no implications means the
        # support is free after propagation, so the ratio measures the instance. Worse, the
        # cell is about the instance, not the sampler.
        free = [f"{k}/{e or '-'}" for (k, e), r in best.items()
                if (r.get("n_models") or 0) > 0 and not (r.get("decisions_per_s") or 0)
                and not (r.get("implications_per_s") or 0)]
        if free:
            warn.append(f"{stem}: {len(free)} instance(s) sample with ZERO search "
                        f"({', '.join(sorted(free))}) — no decisions, no implications, the "
                        f"support is a free hypercube. Exclude as for vacuous pins")
    if tot_unsound == 0 and tot_cells:
        checked = tot_cells - tot_unchecked
        ok.append(f"soundness: 0 unsound of {tot_cells} scored cells"
                  + (f" ({checked} carry an on-GPU re-check; the rest did not run one, which is "
                     f"the default)" if checked else " (no cell ran the opt-in re-check)"))

    # ---------------------------------------------------------------- the propagation identity
    # Table 3 claims  samples/s ratio == (impl/s ratio) / (impl-per-sample ratio)  on every
    # family. It is algebra, so it cannot fail unless the aggregation is wrong — which is exactly
    # what happened once (columns filtered independently let EPFL `bar`, which propagates hard
    # and emits nothing, into the numerator only, breaking the family by 3x). Re-derive it here
    # from the raw rows rather than trusting the generator that produced the claim.
    ident_bad, ident_ok = [], 0
    for p in stages:
        d = load(p)
        if not d:
            continue
        if not isinstance(d, dict) or "results" not in d:
            continue
        stem = os.path.basename(p)[:-5]
        best = {(r["name"].split(".")[0], r.get("enc_key") or r.get("encoder")): r
                for r in d.get("results", []) if not r.get("skipped") and not r.get("error")}
        quad = []
        for (k, _e), r in best.items():
            if k in VACUOUS_PINS:
                continue
            a, b = r.get("implications_per_s"), r.get("cms_implications_per_s")
            s, t = r.get("gpap_sps"), r.get("cms_sps")
            if all(x and x > 0 for x in (a, b, s, t)):
                quad.append((a, b, s, t))
        if len(quad) < 3:
            continue

        def _gm(xs):
            return math.exp(sum(math.log(x) for x in xs) / len(xs))
        impl = _gm([a / b for a, b, _, _ in quad])
        work = _gm([(a / s) / (b / t) for a, b, s, t in quad])
        samp = _gm([s / t for _, _, s, t in quad])
        if not (0.99 <= (impl / work) / samp <= 1.01):
            ident_bad.append(f"{stem} ({impl / work:,.1f}x vs {samp:,.1f}x)")
        else:
            ident_ok += 1
    if ident_bad:
        fail.append("propagation identity BROKEN on: " + ", ".join(ident_bad)
                    + " — the impl/s and work-per-sample columns are not over the same rows")
    elif ident_ok:
        ok.append(f"propagation identity holds on {ident_ok}/{ident_ok} families "
                  f"(impl/s ratio / work-per-sample ratio == samples/s ratio, within 1%)")

    # coverage: a stage below its family's max instance count is partial
    for fam, d in sorted(fam_inst.items()):
        mx = max(d.values())
        for stem, n in sorted(d.items()):
            if n < mx:
                warn.append(f"coverage: {stem} has {n}/{mx} instances (incomplete stage)")

    # coverage: a stage that CRASHED before writing its JSON leaves no other trace — the colab
    # drivers print their completion markers unconditionally, so a lost stage would otherwise
    # be invisible (every check above only sees files that exist). Warn, not fail: sub-campaign
    # directories legitimately hold a subset, but the absence must be a stated choice.
    present_stems = {os.path.basename(p)[:-5] for p in stages}
    known = [stem for stem, _, _, _ in CMS_FAMILIES]
    if present_stems & set(known):
        missing = [stem for stem in known if stem not in present_stems]
        if missing:
            warn.append(f"coverage: standard stage(s) with no JSON at all: "
                        f"{', '.join(missing)} — Table 1 will silently have no row for them; "
                        f"confirm those families were deliberately not run")

    # ---------------------------------------------------------------- reproducibility recheck
    # Table 6 claims the deterministic fields (M, support, packed split, propagate variant)
    # match exactly between the sweep and its fresh-VM re-measurement. RE-DERIVE that from the
    # raw rows — through the same cell reader the comparison tool uses (eval/recheck_compare) —
    # rather than trusting the generator: an accepted-but-ignored flag here was exactly the
    # "VERIFIED per row, enforced by nobody" shape this file exists to prevent.
    if args.recheck:
        from eval.recheck_compare import cells as recheck_cells
        det_bad, n_shared = [], 0
        for p in stages:
            stem = os.path.basename(p)[:-5]
            ca = recheck_cells(p)
            cb = recheck_cells(os.path.join(args.recheck, f"{stem}.json"))
            for k in set(ca) & set(cb):
                ra, rb = ca[k], cb[k]
                n_shared += 1
                ea, eb = ra.get("engine") or {}, rb.get("engine") or {}
                if (ra.get("M") != rb.get("M") or ra.get("support") != rb.get("support")
                        or ea.get("split") != eb.get("split")
                        or ea.get("mode") != eb.get("mode")):
                    det_bad.append(f"{stem}/{k[0]}/{k[1] or '-'}")
        if det_bad:
            fail.append(f"recheck: {len(det_bad)} cell(s) disagree on a DETERMINISTIC field "
                        f"(M / support / packed split / propagate variant): "
                        f"{', '.join(sorted(det_bad)[:8])}"
                        f"{', …' if len(det_bad) > 8 else ''} — the re-measurement did not "
                        f"measure the same thing; no rate comparison over it is meaningful")
        elif n_shared:
            ok.append(f"recheck: deterministic fields match on all {n_shared} shared cells")
        else:
            warn.append("recheck: no shared cells with the original sweep — the "
                        "reproducibility table would stand on nothing")

    # ---------------------------------------------------------------- rivals
    if args.rivals:
        for p in sorted(globmod.glob(os.path.join(args.rivals, "rivals_*.json"))):
            d = load(p)
            rows = (d.get("rows") if isinstance(d, dict) else d) or []
            stem = os.path.basename(p)[:-5]
            # AN ARM THAT MEASURED NOTHING IS NOT A PASSING ARM. A well-formed JSON holding
            # `"rows": []` is what a stage writes when every instance was skipped — e.g. the
            # miter arm pointed at ISCAS-85 names without --bench-dir, which resolves to
            # bench/seq/ and finds none of them (2026-09-02). The campaign ran on, and the empty
            # file was indistinguishable from a stage that had not started.
            # `rivals_steps_calibration.json` is EVIDENCE, not an arm: it has no `rows` key at
            # all (its schema is {generated, tools}). Distinguish by the KEY, not by emptiness —
            # an arm that measured nothing writes `"rows": []`, which is the case below.
            if isinstance(d, dict) and "rows" not in d:
                continue
            if not rows:
                fail.append(f"{stem}: 0 rows — the arm measured NOTHING. An empty result file "
                            f"is a failed stage, not an absent one; check the harness's own "
                            f"exit status and its 'no such bench file' lines.")
                continue
            seen = {}
            for r in rows:
                seen[(r["name"], r.get("variant", "-"))] = r
            if len(seen) != len(rows):
                warn.append(f"{stem}: {len(rows)} rows collapse to {len(seen)} cells "
                            f"(duplicate rows from a resume — de-duplicate before publishing)")
            key = next((k for k in ("demotic", "rcs", "hts")
                        if any(k in r for r in rows)), None)
            errs = sum(1 for r in seen.values() if (r.get(key) or {}).get("error"))
            # COUNT IT THE WAY THE TABLE DOES. A raw error tally and the generator's `error`
            # bucket are different numbers whenever a failed cell is screened first — `b19` is
            # vacuous AND Recurrent fails to initialize on it — and the 2026-09-03 gate warned
            # "10/29 failed" beside a paper that said "9 of 29". The gate's verdict is
            # "publishable PROVIDED every warning is stated", so a warning the paper cannot
            # state verbatim is a defect in the gate.
            from eval.artifact.paper_tables import rival_cells as _cells
            tabled = sum(1 for _n, _v, b, _o, _r in _cells(rows, key) if b == "error") if key else 0
            if errs:
                extra = ("" if tabled == errs else
                         f"; {tabled} of them reach Table 2 as failures, the rest are screened "
                         f"first (a vacuous pin or a proved-UNSAT cell is not a rival failure) "
                         f"— quote {tabled}")
                warn.append(f"{stem}: {errs}/{len(seen)} rival runs failed "
                            f"(reported, not folded in as zeros){extra}")
            # MULTIPLE CIRCUIT VARIANTS ARE NOW EXPECTED (2026-09-04): the table scores the one
            # BETTER FOR THE RIVAL per instance, so a stage holding `raw` and `abc` is the fair
            # comparison, not a double count. What must still be checked is that every instance
            # HAS the same variants — a half-run axis silently scores some instances raw-only.
            variants = {v for _n, v in seen}
            per_inst = {}
            for _n, _v in seen:
                per_inst.setdefault(_n, set()).add(_v)
            ragged = sorted(n for n, vs in per_inst.items() if vs != variants)
            if len(variants) > 1 and ragged:
                warn.append(f"{stem}: circuit variants {sorted(variants)} are not run on every "
                            f"instance ({len(ragged)} short: {', '.join(ragged[:6])}) — the "
                            f"rival gets its better variant only where both exist")
            # A row whose constraint-alignment check FAILED compared two different tasks; the
            # tables exclude it (paper_tables.rival_family), and it must not go unnoticed here.
            # HOW the rival's batch was reached is part of whether it was fairly granted.
            # `probe_truncated`: the ladder walk hit its wall budget and fell back to the VRAM
            # model's rung — the estimator that under-granted DEMOTIC 1.5-3.9x — so those cells
            # may hold a smaller grant than an equally capable cell whose rungs failed fast.
            # `budget_bound`: EVERY rung timed out, so the cell is bounded by our wall rather
            # than by the rival's capability, and is re-runnable at a longer timeout.
            trunc = [n for (n, _v), r in seen.items() if (r.get(key) or {}).get("probe_truncated")]
            bound = [n for (n, _v), r in seen.items() if (r.get(key) or {}).get("budget_bound")]
            if trunc:
                warn.append(f"{stem}: {len(trunc)} cells fell back to the MODELLED batch rung "
                            f"after the probe budget ({', '.join(sorted(trunc)[:6])}) — they may "
                            f"be under-granted relative to cells whose rungs failed fast")
            if bound:
                warn.append(f"{stem}: {len(bound)} cells timed out at EVERY batch rung "
                            f"({', '.join(sorted(bound)[:6])}) — bounded by our wall budget, not "
                            f"by the rival's capability; state it or re-run them longer")
            # A SCREEN THAT NEVER RAN IS NOT A SCREEN THAT FOUND NOTHING. The proved-UNSAT
            # column reads `gpap.unsat_at_T`, which only exists on rows written by a harness
            # that computed it; on an older campaign every cell reads False and the table prints
            # a confident `unsat 0` for a question nobody asked. Same tri-state lesson as
            # `sound_check` — "not run" is neither verdict.
            zero_models = [r for r in seen.values() if not ((r.get("gpap") or {}).get("n_models"))]
            asked = [r for r in zero_models if "unsat_at_T" in (r.get("gpap") or {})]
            if zero_models and not asked:
                warn.append(f"{stem}: {len(zero_models)} cells where GPAP emitted nothing, and "
                            f"NONE carries an unsat_at_T verdict — the proved-UNSAT screen did "
                            f"not run on this campaign, so its column is 'not asked', not 0")
            # The unique/s figure is an exact dedup of the rows the gather returned; if that was
            # a PREFIX of what the run delivered it is a fraction over a prefix, which is the
            # estimate this metric was changed to stop publishing silently.
            # Re-derive rather than trust the stored string: rows written before 2026-09-04
            # carry a `partial-gather` status that compared the WINNER's kept rows against ALL
            # configs' model count, which a healthy adaptive run trips by construction. The
            # condition that matters is whether the gather cap actually bound.
            partial = [n for (n, _v), r in seen.items()
                       if ((r.get("gpap") or {}).get("unique_est_status") == "partial-gather"
                           and ((r.get("gpap") or {}).get("unique_rows") or 0) >= 4_000_000)]
            if partial:
                warn.append(f"{stem}: {len(partial)} cells' distinct fraction is over a PREFIX of "
                            f"the delivered rows ({', '.join(sorted(partial)[:6])}) — raise "
                            f"rivals_recurrent.UNIQUE_ROWS or state it")
            mis = [r["name"] for r in seen.values() if r.get("pin_check_ok") is False]
            if mis:
                fail.append(f"{stem}: {len(mis)} row(s) FAILED the constraint-alignment check "
                            f"({', '.join(sorted(mis))}) — the two sides were not given the "
                            f"same task; re-run before publishing")
            # ---- the CMSGen arm on these rows (the converged protocol: Table 1 is read off
            # them). The SAME gates the `gpap bench` stages get above, because the same numbers
            # are now published from here: mixed provenance FAILS (absent is not zero), a crashed
            # subprocess FAILS, anything but --nowrite FAILS, an unpatched binary FAILS, and a
            # floored count FAILS where it is a zero. `paper_tables.rival_cms_family` screens per
            # row; this refuses the file.
            meas = [r for r in seen.values() if not (r.get("gpap") or {}).get("error")]
            armed = [r for r in meas if cms_arm_ran(r)]
            if armed and len(armed) != len(meas):
                fail.append(f"{stem}: MIXED CMSGen provenance — {len(meas) - len(armed)} of "
                            f"{len(meas)} measured rows never ran the CMSGen arm (a resume "
                            f"without the binary?). Re-run those cells; absent is not zero")
            if armed:
                cerr = [r.get("name", "?") for r in armed if r.get("cms_error")]
                if cerr:
                    fail.append(f"{stem}: the CMSGen subprocess ERRORED on {len(cerr)} row(s) "
                                f"({', '.join(sorted(cerr)[:6])}{', …' if len(cerr) > 6 else ''})"
                                f" — a crashed baseline is not a zero; re-run before publishing")
                modes = {r.get("cms_mode") for r in armed if r.get("cms_mode")}
                if modes - {"nowrite"}:
                    fail.append(f"{stem}: CMSGen timed in mode(s) {modes}, expected only "
                                f"'nowrite'")
                if any(r.get("cms_patched") is False for r in armed):
                    fail.append(f"{stem}: CMSGen present but NOT patched — ratios inflated by "
                                f"emission")
                floored = [r for r in armed
                           if r.get("cms_count_exact") is False or
                           (r.get("cms_count_exact") is None and r.get("cms_timed_out"))]
                if floored:
                    zeros = [r for r in floored if not (r.get("cms_n_samples") or 0)]
                    msg = (f"{stem}: {len(floored)} row(s) carry a FLOORED CMSGen count "
                           f"(killed between --statsevery lines), {len(zeros)} of them recorded "
                           f"as an exact 0")
                    (fail if zeros else warn).append(
                        msg + " — re-run with a --maxtime-capable cmsgen (scripts/build_cmsgen.sh)")
                if not all((r.get("cpu") or {}).get("model") for r in armed):
                    warn.append(f"{stem}: CMSGen arm without CPU provenance on some rows — a "
                                f"GPAP/CMSGen ratio is not interpretable without it")
                ok.append(f"{stem}: CMSGen arm on {len(armed)}/{len(meas)} measured rows, "
                          f"--nowrite, counts exact")
            # ---- the GPAP arm: a soundness verdict and one engine per stage, as for a bench row
            uns = [r.get("name", "?") for r in meas if sound_failed(r.get("gpap") or {})]
            for n in uns:
                fail.append(f"UNSOUND cell {stem}/{n}")
            # ...and the ENCODER-INDEPENDENT check: the kept output rows simulated through the
            # original circuit must reproduce the pinned outputs (`sim_check`, run by the arm on
            # every row with a wire map). BAD is a hard failure; "off" is counted, not required.
            simbad = [r.get("name", "?") for r in meas
                      if (r.get("gpap") or {}).get("sim_check") == "BAD"]
            for n in simbad:
                fail.append(f"{stem}/{n}: kept samples FAIL the circuit-level simulate check "
                            f"(encoder-independent) — the unroll/encoder chain broke")
            nsim = sum(1 for r in meas if (r.get("gpap") or {}).get("sim_check") == "ok")
            if nsim:
                ok.append(f"{stem}: {nsim}/{len(meas)} rows' kept samples reproduce the pinned "
                          f"outputs through the ORIGINAL circuit (encoder-independent)")
            cores = {}
            for r in meas:
                c = ((r.get("gpap") or {}).get("engine") or {}).get("core")
                if c:
                    cores[c] = cores.get(c, 0) + 1
            if len(cores) > 1:
                warn.append(f"{stem}: rows span {len(cores)} ENGINE BUILDS "
                            + ", ".join(f"{c} ({n} rows)" for c, n in sorted(cores.items()))
                            + " — a resumed stage was topped up under a different libgpap; the "
                              "table geomeans them together. Re-run the stage, or say so.")
        ok.append("rivals: error rows counted, never treated as zero")

    # ---------------------------------------------------------------- uniformity oracle
    upaths = []
    for g in args.uniformity:
        upaths += sorted(globmod.glob(g))
    if upaths:
        names = set()
        unsound = 0
        for p in upaths:
            d = load(p)
            for r in (d or {}).get("rows", []):
                for nm, s in (r.get("samplers") or {}).items():
                    names.add(nm)
                    unsound += s.get("unsound") or 0
        neg = [n for n in names if n.startswith("ctl:uniform")]
        pos = [n for n in names if n.startswith("ctl:tv")]
        if not neg:
            fail.append("uniformity: no exact-uniform NEGATIVE control — "
                        "a 'no skew' result is uncalibrated")
        if not pos:
            fail.append("uniformity: no known-TV POSITIVE control — power is unmeasured")
        if unsound:
            fail.append(f"uniformity: {unsound} unsound emissions")
        else:
            ok.append(f"uniformity: {len(names)} samplers, both controls present, 0 unsound")
        if not any("soft" in n for n in names):
            warn.append("uniformity: no `soft` arm — the reason it was removed is unverified here")

    # ---------------------------------------------------------------- report
    print("=" * 78)
    for m in ok:
        print(f"  OK    {m}")
    for m in warn:
        print(f"  WARN  {m}")
    for m in fail:
        print(f"  FAIL  {m}")
    print("=" * 78)
    print(f"{len(ok)} ok | {len(warn)} warnings | {len(fail)} failures")
    if fail:
        print("\nNOT publishable until the failures above are resolved.")
    elif warn:
        print("\nPublishable, PROVIDED every warning above is stated in the text.")
    else:
        print("\nPublishable.")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

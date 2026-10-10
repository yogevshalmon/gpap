#!/usr/bin/env python3
"""Verify the experiment artifact is publishable: sound, complete, and self-describing.

A CHECKER, not a summariser — "is there anything here I would be embarrassed to defend?" — that
exits non-zero if so. Run it before quoting a number anywhere.

    eval/artifact/verify_artifact.py <results-dir> --rivals DIR --uniformity FILES

  SOUNDNESS   no scored cell may REPORT itself unsound (the opt-in re-check's "off" is counted,
              not required). This check can never be waived.
  FAIRNESS    CMSGen must be the PATCHED binary: stock CMSGen is timed WITH sample emission.
  PROVENANCE  each stage records GPU and CPU; CMSGen is single-core, so a ratio without its CPU
              is not interpretable.
  AGGREGATION vacuous pins stay out of a geomean, and a geomean on fewer than three instances is
              flagged rather than quoted.
  COVERAGE    stages short of their family's instance count are listed.
  ORACLE      the uniformity campaign must include BOTH controls, or "no skew detected" is
              uncalibrated. RECHECK (--recheck DIR): the deterministic fields must match the
              original sweep exactly on every shared cell.
"""
import argparse
import glob as globmod
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gpap.verdict import sound_verdict                        # noqa: E402
from eval.artifact.paper_tables import (CMS_FAMILIES, cms_arm_ran,          # noqa: E402
                                        recheck_cells, sound_failed, sound_unchecked)

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
    ap.add_argument("--barbarik", nargs="*", default=[],
                    help="the CM_test campaign's jsonl files (Table 4b) — one parameter "
                         "setting, a negative control that Accepts, every arm over the "
                         "eligible set")
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
        # MIXED CMSGen provenance: reading a missing count as a number mints phantom zeros.
        armed = [r for r in rows if cms_arm_ran(r)]
        if armed and len(armed) != len(rows):
            fail.append(f"{stem}: MIXED CMSGen provenance — {len(rows) - len(armed)} of "
                        f"{len(rows)} rows never ran the CMSGen arm (a resume without the "
                        f"binary?). Re-run those cells or delete them; absent is not zero")
        # A crashed CMSGen subprocess is `cms_error` with a count of 0: scoring that zero
        # publishes a broken binary as a stage of outright wins.
        cerr = [r for r in rows if r.get("cms_error")]
        if cerr:
            fail.append(f"{stem}: the CMSGen subprocess ERRORED on {len(cerr)} row(s) "
                        f"({', '.join(sorted(r.get('name', '?') for r in cerr)[:6])}"
                        f"{', …' if len(cerr) > 6 else ''}) — a crashed baseline is not a "
                        f"zero; re-run before publishing")
        # A row that says UNSOUND fails; a row that ran NO check does not, the re-check being an
        # opt-in self-test. Counted, never required.
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
        # ---- ONE ENGINE PER STAGE: two `gpap_build_id`s in one stage means two engines produced
        # rows a table then geomeans together. Not fatal, but the campaign cannot claim one.
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
        # ---- CMSGen's count must be EXACT where published: a killed CMSGen reports its last
        # --statsevery multiple, so a 0 can mean "fewer than 1000". `--maxtime` prints an exact one.
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
        # The same population as the paper's n_gm (finite ratios minus vacuous and zero-search
        # cells), or this warning disagrees with the table.
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

        # SYMMETRY of the zero ledger: reporting the instances CMSGen could not sample while
        # dropping the ones WE could not flatters a table by accident.
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

        # ZERO-SEARCH cells: no decisions and no implications means the ratio measures the instance.
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
    # Table 3 claims  samples/s ratio == (impl/s ratio) / (impl-per-sample ratio). It is algebra,
    # so it can only fail when the aggregation is wrong — filtering columns independently once
    # let an instance that propagates hard and emits nothing into the numerator only.
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

    # coverage: a stage that CRASHED before writing its JSON leaves no other trace, since every
    # check above only sees files that exist. Warn, not fail — a subset can be a stated choice.
    present_stems = {os.path.basename(p)[:-5] for p in stages}
    known = [stem for stem, _, _, _ in CMS_FAMILIES]
    if present_stems & set(known):
        missing = [stem for stem in known if stem not in present_stems]
        if missing:
            warn.append(f"coverage: standard stage(s) with no JSON at all: "
                        f"{', '.join(missing)} — Table 1 will silently have no row for them; "
                        f"confirm those families were deliberately not run")

    # Table 6's deterministic-field claim, RE-DERIVED from the raw rows through the generator's
    # own cell reader, so this gate and the table it gates read the same cells.
    if args.recheck:
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
            # AN ARM THAT MEASURED NOTHING IS NOT A PASSING ARM: `"rows": []` is what a stage
            # writes when every instance was skipped, while a calibration file has no `rows` key
            # at all — distinguish by the KEY, not by emptiness.
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
            # COUNT IT THE WAY THE TABLE DOES: a raw error tally and the generator's `error`
            # bucket differ when a failed cell is screened first, and a warning the paper cannot
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
            # MULTIPLE CIRCUIT VARIANTS ARE EXPECTED (the table scores the one BETTER FOR THE
            # RIVAL per instance); what must be checked is that every instance has the same ones.
            variants = {v for _n, v in seen}
            per_inst = {}
            for _n, _v in seen:
                per_inst.setdefault(_n, set()).add(_v)
            ragged = sorted(n for n, vs in per_inst.items() if vs != variants)
            if len(variants) > 1 and ragged:
                warn.append(f"{stem}: circuit variants {sorted(variants)} are not run on every "
                            f"instance ({len(ragged)} short: {', '.join(ragged[:6])}) — the "
                            f"rival gets its better variant only where both exist")
            # A row whose constraint-alignment check FAILED compared two different tasks. HOW the
            # batch was reached matters too: `probe_truncated` = the walk hit its wall budget and
            # fell back to the under-granting VRAM model, `budget_bound` = every rung timed out.
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
            # A SCREEN THAT NEVER RAN IS NOT A SCREEN THAT FOUND NOTHING: the proved-UNSAT column
            # reads `gpap.unsat_at_T`, absent on older rows (the `sound_check` tri-state).
            zero_models = [r for r in seen.values() if not ((r.get("gpap") or {}).get("n_models"))]
            asked = [r for r in zero_models if "unsat_at_T" in (r.get("gpap") or {})]
            if zero_models and not asked:
                warn.append(f"{stem}: {len(zero_models)} cells where GPAP emitted nothing, and "
                            f"NONE carries an unsat_at_T verdict — the proved-UNSAT screen did "
                            f"not run on this campaign, so its column is 'not asked', not 0")
            # The unique/s figure is an exact dedup of the rows the gather returned; a PREFIX
            # makes it a fraction over a prefix. Re-derive rather than trust the stored string.
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
            # ---- the CMSGen arm on these rows: the SAME gates the `gpap bench` stages get above.
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
            # AN UN-RUN CHECK IS NOT A PASSING ONE: gating only on "UNSOUND" lets a run without
            # --verify-sound produce a byte-identical "Publishable" verdict.
            emitted = [r for r in meas if ((r.get("gpap") or {}).get("n_models") or 0) > 0]
            nok = sum(1 for r in emitted if sound_verdict(r.get("gpap") or {}) == "ok")
            missing = [r.get("name", "?") for r in emitted
                       if sound_verdict(r.get("gpap") or {}) != "ok"]
            if missing:
                fail.append(f"{stem}: {len(missing)} row(s) emitted models with NO soundness "
                            f"verdict ({', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}) "
                            f"— the arm ran without --verify-sound; re-run before publishing")
            elif emitted:
                ok.append(f"{stem}: device soundness re-check ok on {nok}/{len(emitted)} rows "
                          f"that emitted a model ({len(meas) - len(emitted)} emitted nothing: "
                          f"counted as 'off', not as passing)")
            # ...and the ENCODER-INDEPENDENT check: kept rows simulated through the original
            # circuit must reproduce the pinned outputs. BAD fails, "off" is counted.
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

        # ---- Table 3 under the converged protocol. The identity is ALGEBRA (work := (a/s)/(b/t)
        # makes impl/work == s/t by construction), so all it can catch is a ROW-SET mismatch
        # against Table 1 — and while this block only understood the `gpap bench` schema, Table 3
        # shipped under --rivals with no gate at all. Check the row sets AGAINST THE GENERATOR'S
        # OWN OUTPUT, and add a plausibility band on the counters.
        from eval.artifact.paper_tables import (RIVAL_CMS_SETS, _RIVAL_KEY,  # noqa: E402
                                                rival_cms_family, rival_prop_rates)
        for stem, _label, _tgt in RIVAL_CMS_SETS:
            key = _RIVAL_KEY[stem]
            t1 = rival_cms_family(args.rivals, stem, key)
            t3 = rival_prop_rates(args.rivals, stem, key)
            if not t1 or not t3:
                continue
            if t3["n"] != t1["n_gm"]:
                fail.append(f"rivals_{stem}: Table 3 is over {t3['n']} cells and Table 1 over "
                            f"{t1['n_gm']} — the propagation columns and the sample-rate column "
                            f"are not the same rows (a screen applied to one table and not the "
                            f"other)")
            elif t1["gm"] and t3["work"] and not (0.99 <= (t3["ratio"] / t3["work"]) / t1["gm"] <= 1.01):
                fail.append(f"rivals_{stem}: propagation identity BROKEN "
                            f"({t3['ratio'] / t3['work']:,.1f}x vs Table 1's {t1['gm']:,.1f}x)")
            else:
                ok.append(f"rivals_{stem}: Table 3 is over the same {t3['n']} cells as Table 1 "
                          f"and the identity closes on the printed numbers")
        # counters outside a physical band: GPAP's packed-word kernel holds ~1e9 impl/s on every
        # family, CMSGen's own counters 1e6-1e7. An identity cancels a wrong counter, a range not.
        odd = []
        for p in sorted(globmod.glob(os.path.join(args.rivals, "rivals_*.json"))):
            d = load(p)
            if not isinstance(d, (list, dict)) or (isinstance(d, dict) and "rows" not in d):
                continue
            for r in ((d.get("rows") if isinstance(d, dict) else d) or []):
                g = r.get("gpap") or {}
                a, b = g.get("implications_per_s"), r.get("cms_implications_per_s")
                if a and not (1e6 <= a <= 5e10):
                    odd.append(f"{os.path.basename(p)[:-5]}/{r.get('name')} GPAP {a:.3g}")
                if b and not (1e3 <= b <= 1e9):
                    odd.append(f"{os.path.basename(p)[:-5]}/{r.get('name')} CMSGen {b:.3g}")
        if odd:
            fail.append("implication counters outside the physical band on "
                        + ", ".join(odd[:6]) + (", …" if len(odd) > 6 else "")
                        + " — a corrupt counter cancels out of the identity; it does not "
                          "cancel out of Table 3")
        else:
            ok.append("rivals: every implication counter inside its physical band")

    # ---------------------------------------------------------------- Barbarik (Table 4b)
    bpaths = []
    for g in args.barbarik:
        bpaths += sorted(globmod.glob(g))
    if bpaths:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "uniformity"))
        from barbarik_table import CONTROL, arm_summary, buckets, check_one_setting  # noqa: E402
        from barbarik_table import load as bload  # noqa: E402
        brows = bload(bpaths)
        if not brows:
            fail.append("barbarik: no rows in " + ", ".join(bpaths))
        else:
            try:
                eps, eta, delta = check_one_setting(brows)
                eligible, rejected, errored = buckets(brows)
            except (ValueError, AssertionError) as e:
                fail.append(f"barbarik: {e}")
            else:
                summ = arm_summary(brows, eligible)
                if CONTROL not in summ or summ[CONTROL]["accept"] != summ[CONTROL]["n"]:
                    fail.append("barbarik: the negative control does not Accept every eligible "
                                "instance — the eligibility rule is broken")
                for arm, a in summ.items():
                    if a["n"] != len(eligible):
                        warn.append(f"barbarik: `{arm}` has verdicts on {a['n']}/{len(eligible)} "
                                    f"eligible instances — the arms are not over one set")
                    if a["reject"]:
                        warn.append(f"barbarik: `{arm}` REJECTED on {a['reject']} instance(s) "
                                    f"({', '.join(a['bad'][:5])}) — a Reject is unconditional; "
                                    f"state it")
                ok.append(f"barbarik: one setting (eps={eps} eta={eta} delta={delta}), "
                          f"{len(eligible)} eligible / {len(rejected)} control-rejected / "
                          f"{len(errored)} control-errored, {len(summ)} arms")

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

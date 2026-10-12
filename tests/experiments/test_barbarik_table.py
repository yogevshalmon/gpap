"""The uniformity table's own rules (`eval/uniformity/barbarik_table.py`).

This file holds the arithmetic and the eligibility rule the headline uniformity claim rests on
("accepted on 60 of 65 eligible instances"), and it had no test: `paper_tables.barbarik()`
imports `buckets` and `arm_summary` precisely so the paper cannot re-derive them differently, so
a silent change here moves a published count with nothing to stop it.
"""
import json

import pytest

from eval.uniformity.barbarik_table import (SETTINGS, arm_summary, buckets, check_one_setting,
                                            load, order_budget)

CONTROL = "spur(control)"


def row(sampler, instance, verdict, **kw):
    r = {"sampler": sampler, "instance": instance, "verdict": verdict, "seed": 1,
         "epsilon": 0.3, "eta": 1.8, "delta": 0.1, "order_seconds": 30.0, "k_cap": 4096}
    r.update(kw)
    return r


def test_settings_arithmetic_is_self_consistent():
    """`rounds`, `per_round` and the two totals are four statements about one unroll, quoted in
    four places (the paper, Table 4b, this file's header, the screen's budget). Derive the
    totals from the parts so the first one to move cannot stay plausible."""
    s = SETTINGS[(0.3, 1.8, 0.1)]
    parts = [int(x) for x in s["per_round"].split("+")]
    assert sum(parts) * s["rounds"] == s["samples"], \
        f"{s['rounds']} rounds x {s['per_round']} != {s['samples']} samples"
    # the gap between the two totals is exactly the one reference draw per round
    assert s["samples"] - s["under_test"] == s["rounds"]
    assert sum(parts[1:]) * s["rounds"] == s["under_test"]


def test_only_control_accepted_instances_are_eligible_and_the_buckets_partition():
    """An instance where the trusted reference is itself Rejected is an instance property, not a
    sampler property; scoring an arm there would publish the one as the other."""
    rows = [row(CONTROL, "ok", "Accept"), row(CONTROL, "bad_ind", "Reject"),
            row(CONTROL, "crashed", "error"),
            row("gpap:b8", "ok", "Accept"), row("gpap:b8", "bad_ind", "Reject")]
    eligible, rejected, errored = buckets(rows)
    assert eligible == {"ok"} and rejected == {"bad_ind"} and errored == {"crashed"}
    # the arm's Reject on a control-failed instance must NOT reach the summary
    s = arm_summary(rows, eligible)["gpap:b8"]
    assert (s["n"], s["accept"], s["reject"]) == (1, 1, 0), s


def test_a_table_without_the_control_is_refused():
    """The control gates everything else, so its absence is a failure, never an empty screen."""
    with pytest.raises(ValueError, match="negative control"):
        buckets([row("gpap:b8", "ok", "Accept")])


def test_an_unfilled_order_is_counted_as_no_verdict_never_as_a_reject():
    """Barbarik's kernel is adversarial for a rejection sampler, so an order the sampler could
    not fill is "no verdict". Counting it as a Reject would report our own budget as a
    uniformity failure (docs/findings/UNIFORMITY_FINDINGS.md §4b)."""
    rows = [row(CONTROL, "hard", "Accept"),
            row("gpap:b8", "hard", "error",
                error="RuntimeError: GPAP produced 0/50 samples on hard.cnf "
                      "[1 of 24 attempts; the 30s order budget was spent]")]
    s = arm_summary(rows, buckets(rows)[0])["gpap:b8"]
    assert (s["error"], s["reject"], s["n"]) == (1, 0, 1), s


def test_two_parameter_settings_are_refused_in_one_table():
    with pytest.raises(ValueError, match="mix Barbarik parameter settings"):
        check_one_setting([row(CONTROL, "a", "Accept"),
                           row(CONTROL, "b", "Accept", epsilon=0.1, eta=0.9, delta=0.05)])


def test_two_per_order_budgets_are_refused_in_one_table():
    """The budget decides how often an order goes unfilled, and unfilled orders are what this
    table reports — so two budgets pooled is a budget comparison wearing a uniformity label,
    the same defect as running two arms on two budgets."""
    mixed = [row(CONTROL, "a", "Accept"), row("gpap:b8", "a", "Accept"),
             row("cmsgen", "a", "Accept", order_seconds=300.0)]
    with pytest.raises(ValueError, match="mix PER-ORDER BUDGETS"):
        order_budget(mixed)
    with pytest.raises(ValueError, match="mix PER-ORDER BUDGETS"):
        check_one_setting(mixed)              # the gate paper_tables actually calls


def test_rows_from_before_the_budget_existed_are_not_a_third_budget():
    """Every published row predates `--order-seconds`. A None must not read as its own value,
    or the whole recorded campaign becomes unsummarisable."""
    legacy = [row(CONTROL, "a", "Accept", order_seconds=None),
              row("gpap:b8", "a", "Accept", order_seconds=None)]
    assert order_budget(legacy) is None
    # ...and SPUR's None beside a budgeted arm is consistent, not a mix: the budget never
    # reaches the reference, so the harness records it as the None it is.
    assert order_budget([row(CONTROL, "a", "Accept", order_seconds=None),
                         row("gpap:b8", "a", "Accept")]) == 30.0


def test_load_reads_jsonl_and_skips_blank_lines(tmp_path):
    """The harness appends one row per instance as it goes, so a killed sweep leaves a file that
    still has to parse."""
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(row(CONTROL, "a", "Accept")) + "\n\n"
                 + json.dumps(row("gpap:b8", "a", "Accept")) + "\n")
    assert len(load([str(p)])) == 2

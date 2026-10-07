"""Host-side tests for `gpap bench` (gpap/benchmark.py): the packed-envelope pre-check, encoder
specs, resume keys, CMSGen stats and rate accounting (including killed runs), --help rendering
and the bench table layout.
"""
import json
import os
import subprocess
import sys
import textwrap

from itertools import product
import pytest

from gpap.benchmark import (
    _cms_arm_mismatch,
    _dimacs_header,
    _parse_cms_stats,
    _row_key,
    cmsgen_run,
    cmsgen_supports_nowrite,
    enc_key,
    parse_encoder_specs,
)
from gpap.engine.bcp_engine import clauses_envelope, packed_envelope


# --------------------------------------------------------------------- packed envelope

def test_packed_envelope_picks_the_narrow_split_when_it_fits():
    assert packed_envelope(100, 3) == (16, "")
    assert packed_envelope(32767, 8)[0] == 16          # 2N = 65534, the last narrow formula


def test_packed_envelope_falls_back_to_wide_then_rejects():
    # 2N past 65535 -> the 8|24 split, which needs clauses of at most 255 literals
    assert packed_envelope(40_000, 200)[0] == 8
    shift, why = packed_envelope(40_000, 256)
    assert shift is None and "max clause len=256" in why and "len<=255" in why
    # and past 24 bits of literal index nothing represents it
    shift, why = packed_envelope(9_000_000, 3)
    assert shift is None and "2N=18000000" in why


def test_packed_envelope_guards_the_count_field_of_the_narrow_split():
    """A clause longer than the sat_count field would carry into the sum field. Only reachable
    with repeated literals."""
    assert packed_envelope(10, 65_535)[0] == 16
    assert packed_envelope(10, 65_536)[0] is None


def test_clauses_envelope_reads_the_max_length_off_the_clause_list():
    assert clauses_envelope(5, [[1, 2], [3]]) == (16, "")
    assert clauses_envelope(5, [])[0] == 16


# --------------------------------------------------------------------- encoder specs

def test_parse_encoder_specs_defaults_to_the_single_encoder_pair():
    assert parse_encoder_specs(None, "tseitin", "auto") == [("tseitin", None)]
    assert parse_encoder_specs(None, "abc", "fast") == [("abc", "fast")]


def test_parse_encoder_specs_ab_list_and_keys():
    specs = parse_encoder_specs(["tseitin", "abc:fast"], "tseitin", "auto")
    assert specs == [("tseitin", None), ("abc", "fast")]
    assert [enc_key(e, o) for e, o in specs] == ["tseitin", "abc:fast"]
    # a bare 'abc' inherits --abc-opt
    assert parse_encoder_specs(["abc"], "tseitin", "full") == [("abc", "full")]


def test_parse_encoder_specs_rejects_nonsense():
    with pytest.raises(SystemExit):
        parse_encoder_specs(["tsetin"], "tseitin", "auto")


def test_row_key_separates_encoder_target_and_depth():
    """The resume key distinguishes every cell, or a resumed sweep skips rows it never ran."""
    base = {"name": "b03.bench", "enc_key": "tseitin", "cycles": 25, "target": "rare"}
    assert _row_key(base) != _row_key({**base, "enc_key": "abc:fast"})
    assert _row_key(base) != _row_key({**base, "cycles": 50})
    assert _row_key(base) != _row_key({**base, "target": "random"})


def test_resume_refuses_to_mix_armed_and_armless_rows():
    """A results file never mixes rows with and without the CMSGen arm, in either direction:
    consumers read a missing CMSGen count as zero, which scores as a GPAP win."""
    armed = [{"name": "a.cnf", "cms_n_samples": 5, "n_models": 9}]
    armless = [{"name": "a.cnf", "n_models": 9}]
    assert _cms_arm_mismatch(armed, None)                      # armed file, armless run
    assert _cms_arm_mismatch(armless, "/bin/cmsgen")           # armless file, armed run
    assert _cms_arm_mismatch(armed, "/bin/cmsgen") is None     # same arm: fine
    assert _cms_arm_mismatch(armless, None) is None
    # skipped/error rows carry no arm by construction and must not trip the guard
    assert _cms_arm_mismatch([{"name": "x", "skipped": "too big"}], "/bin/cmsgen") is None
    # a crashed CMSGen run still wrote the arm's keys, so it counts as armed
    assert _cms_arm_mismatch([{"name": "a.cnf", "cms_n_samples": 0, "cms_error": "boom",
                               "n_models": 1}], "/bin/cmsgen") is None


# --------------------------------------------------------------------- CMSGen accounting

def test_parse_cms_stats_takes_the_LAST_flushed_line():
    txt = ("c gpap-stats samples=1000 props=10 decisions=5 conflicts=1 cpu_s=0.1 wall_s=0.2 "
           "final=0\n"
           "c noise\n"
           "c gpap-stats samples=2000 props=40 decisions=9 conflicts=3 cpu_s=0.3 wall_s=0.4 "
           "final=1\n")
    s = _parse_cms_stats(txt)
    assert s == {"samples": 2000, "props": 40, "decisions": 9, "conflicts": 3,
                 "cpu_s": 0.3, "wall_s": 0.4, "final": True}


def test_parse_cms_stats_reads_a_pre_maxtime_line_as_not_final():
    """A line without `final=` (a binary older than --maxtime) still parses, as not exact:
    that binary's counts really are floored."""
    s = _parse_cms_stats("c gpap-stats samples=7000 props=1 decisions=1 conflicts=0 "
                         "cpu_s=0.1 wall_s=0.2\n")
    assert s["samples"] == 7000 and s["final"] is False


def test_parse_cms_stats_none_on_stock_output():
    assert _parse_cms_stats("c Finished generating all 10 samples\n") is None
    assert _parse_cms_stats("") is None


def test_dimacs_header(tmp_path):
    p = tmp_path / "x.cnf"
    p.write_text("c a comment\nc ind 1 2 0\np cnf 7 3\n1 2 0\n-1 3 0\n2 -3 0\n")
    assert _dimacs_header(str(p)) == (7, 3)
    q = tmp_path / "noheader.cnf"
    q.write_text("1 2 0\n")
    assert _dimacs_header(str(q)) == (None, None)


# A stand-in for the patched cmsgen: it advertises the flags in --help and prints the same
# progress lines, so detection and accounting are testable without a C++ build.
_FAKE_CMSGEN = textwrap.dedent("""
    import sys, time
    args = sys.argv[1:]
    MAXTIME = __MAXTIME__
    if "--help" in args:
        print("usage: cmsgen [--samples VAR] [--samplefile VAR] [--nowrite] "
              "[--statsevery VAR]" + (" [--maxtime VAR]" if MAXTIME else ""))
        sys.exit(0)
    n = int(args[args.index("--samples") + 1])
    out = args[args.index("--samplefile") + 1] if "--samplefile" in args else None
    budget = float(args[args.index("--maxtime") + 1]) if "--maxtime" in args else 0.0
    RATE = 20000.0                      # fake samples per second, so a budget is meaningful
    stop = int(budget * RATE) if budget else n
    n = min(n, stop)
    def line(i, final):
        print(f"c gpap-stats samples={i} props={i*10} decisions={i*3} conflicts={i//100} "
              f"cpu_s={i/1e5:.4f} wall_s={i/1e5:.4f} final={1 if final else 0}", flush=True)
    if out:
        with open(out, "w") as f:
            for i in range(n):
                f.write("1 -2 3 0\\n")
    for i in range(1000, n + 1, 1000):
        line(i, False)
    if MAXTIME:                          # the post-loop line: only this one is exact
        line(n, True)
    print("c Finished generating all %d samples" % n)
""")


def _fake_bin(tmp_path, name, maxtime):
    p = tmp_path / f"{name}.py"
    # str.replace, not str.format: the fake's body is full of f-string braces
    p.write_text(_FAKE_CMSGEN.replace("__MAXTIME__", repr(bool(maxtime))))
    sh = tmp_path / name
    sh.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{p}" "$@"\n')
    sh.chmod(0o755)
    return str(sh)


@pytest.fixture()
def fake_cmsgen(tmp_path):
    """The current patched binary: --nowrite, --statsevery and --maxtime."""
    return _fake_bin(tmp_path, "cmsgen", True)


@pytest.fixture()
def fake_cmsgen_no_maxtime(tmp_path):
    """An older patched binary without --maxtime; when killed, its counts are floored to the
    --statsevery cadence."""
    return _fake_bin(tmp_path, "cmsgen_old", False)


def test_cmsgen_supports_nowrite_detects_the_patch(fake_cmsgen):
    assert cmsgen_supports_nowrite(fake_cmsgen) is True
    assert cmsgen_supports_nowrite("/nonexistent/cmsgen") is False


def test_cmsgen_run_nowrite_counts_from_the_stats_line(fake_cmsgen, tmp_path):
    cnf = tmp_path / "f.cnf"
    cnf.write_text("p cnf 3 1\n1 2 3 0\n")
    r = cmsgen_run(str(cnf), fake_cmsgen, 5000, str(tmp_path), write=False)
    assert r["mode"] == "nowrite" and r["patched"] is True
    assert r["n_samples"] == 5000 and r["models"] == []
    assert r["samples_per_s"] > 0
    # implications = propagations - decisions (as in the GPAP column), over CMSGen's sampling
    # wall, the same denominator as samples_per_s, so the propagation identity stays exact.
    assert r["implications_per_s"] == pytest.approx(
        (5000 * 10 - 5000 * 3) / r["sampling_wall_s"], rel=1e-6)
    assert r["count_exact"] is True                    # the run finished; the count is CMSGen's
    # The two rates differ only in their wall: CMSGen's sampling clock (paired with GPAP's steady
    # state) and the whole subprocess (paired with GPAP's wall_total).
    assert r["samples_per_s"] == pytest.approx(5000 / r["sampling_wall_s"], rel=1e-9)
    assert r["samples_per_s_e2e"] == pytest.approx(5000 / r["wall_s"], rel=1e-9)


def test_cmsgen_run_write_mode_parses_models_and_removes_the_file(fake_cmsgen, tmp_path):
    cnf = tmp_path / "f.cnf"
    cnf.write_text("p cnf 3 1\n1 2 3 0\n")
    r = cmsgen_run(str(cnf), fake_cmsgen, 2000, str(tmp_path), keep=5, write=True)
    assert r["mode"] == "write"
    assert r["n_samples"] == 2000          # every sample counted for the rate
    assert len(r["models"]) == 5           # only `keep` retained in memory
    assert r["models"][0] == [1, -2, 3]    # trailing 0 stripped
    assert not [f for f in os.listdir(tmp_path) if f.startswith("cms_")]


def test_cmsgen_run_timeout_keeps_partial_progress(fake_cmsgen, tmp_path, monkeypatch):
    """A killed CMSGen still scores what it produced, and marks the count not exact: it is the
    last multiple of the --statsevery cadence."""
    cnf = tmp_path / "f.cnf"
    cnf.write_text("p cnf 3 1\n1 2 3 0\n")
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        p = real_run(cmd, **{**kw, "timeout": None})
        raise subprocess.TimeoutExpired(cmd, 0.01, output=p.stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)
    r = cmsgen_run(str(cnf), fake_cmsgen, 3000, str(tmp_path), write=False, timeout=0.01)
    assert r["timed_out"] is True
    assert r["n_samples"] == 3000 and r["samples_per_s"] > 0
    assert r["count_exact"] is False


def test_maxtime_makes_the_count_exact_instead_of_a_cadence_multiple(fake_cmsgen, tmp_path):
    """With --maxtime CMSGen stops itself, so its final count is exact and lands wherever the
    budget lands, not on a --statsevery multiple (RIVAL_FAIRNESS_FINDINGS.md §7)."""
    cnf = tmp_path / "f.cnf"
    cnf.write_text("p cnf 3 1\n1 2 3 0\n")
    r = cmsgen_run(str(cnf), fake_cmsgen, 10_000_000, str(tmp_path), write=False, timeout=0.3)
    assert r["timed_out"] is False and r["count_exact"] is True
    assert r["n_samples"] == 6000                       # 0.3 s x the fake's 20k/s
    r2 = cmsgen_run(str(cnf), fake_cmsgen, 10_000_000, str(tmp_path), write=False, timeout=0.317)
    assert r2["n_samples"] == 6340 and r2["n_samples"] % 1000                # not a multiple


def test_maxtime_capability_is_probed_separately_from_nowrite(fake_cmsgen,
                                                              fake_cmsgen_no_maxtime, tmp_path):
    """An older binary with --nowrite but no --maxtime still runs, and its counts are marked
    not exact so verify_artifact can refuse them as zeros."""
    from gpap.benchmark import cmsgen_supports_maxtime
    assert cmsgen_supports_nowrite(fake_cmsgen_no_maxtime) is True
    assert cmsgen_supports_maxtime(fake_cmsgen_no_maxtime) is False
    assert cmsgen_supports_maxtime(fake_cmsgen) is True
    cnf = tmp_path / "f.cnf"
    cnf.write_text("p cnf 3 1\n1 2 3 0\n")
    r = cmsgen_run(str(cnf), fake_cmsgen_no_maxtime, 3000, str(tmp_path), write=False)
    assert r["n_samples"] == 3000 and r["count_exact"] is False


def test_cmsgen_run_missing_binary_is_an_error_row_not_a_crash(tmp_path):
    cnf = tmp_path / "f.cnf"
    cnf.write_text("p cnf 3 1\n1 2 3 0\n")
    r = cmsgen_run(str(cnf), "/nonexistent/cmsgen", 10, str(tmp_path))
    assert r["samples_per_s"] == 0.0 and "error" in r


# --------------------------------------------------------------------- driver plumbing

def test_bench_driver_skips_an_oversized_instance_and_still_writes_json(tmp_path):
    """--max-clauses records an oversized instance as skipped, with the reason, and the sweep
    goes on. Runs the real main() to cover the JSON; the instance never reaches the GPU."""
    cnf = tmp_path / "big.cnf"
    cnf.write_text("p cnf 10 999999\n1 2 0\n")
    out = tmp_path / "res.json"
    from gpap import benchmark
    benchmark.main(["--glob", str(cnf), "--max-clauses", "10", "--output", str(out)])
    payload = json.load(open(out))
    assert payload["schema"] >= 3
    row, = payload["results"]
    assert "max-clauses" in row["skipped"] and row["M"] == 999999


# ---------------------------------------------------------------- cli help renders
# argparse %-formats help strings only when rendering, so a literal `%` in help text crashes
# --help with a TypeError while every real invocation works.

@pytest.mark.parametrize("sub", ["sample", "mine", "bench", "circuit2cnf", "info"])
def test_subcommand_help_renders(sub):
    from gpap.cli import main

    with pytest.raises(SystemExit) as e:
        main([sub, "--help"])
    assert e.value.code == 0


def test_top_level_help_renders():
    from gpap.cli import main

    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


# ---------------------------------------------------------------- quality pass metrics
# Distinct-projection counting and 2-wise coverage. Neither is a uniformity test; that lives in
# eval/uniformity/.

def test_two_wise_vectorized_matches_bruteforce():
    import itertools
    import random

    from gpap.benchmark import two_wise_coverage

    rng = random.Random(9)
    support = [2, 5, 7, 9]                          # 1-based vars, non-contiguous
    models = [[v if rng.random() < 0.5 else -v for v in support] for _ in range(40)]
    got = two_wise_coverage(models, support)
    # brute-force reference: a pair is covered iff all 4 joint polarities appear
    covered = total = 0
    for a, b in itertools.combinations(range(len(support)), 2):
        total += 1
        seen = {(support[a] in set(m), support[b] in set(m)) for m in models}
        covered += len(seen) == 4
    assert got == pytest.approx(100.0 * covered / total)
    assert two_wise_coverage([], support) == 0.0
    assert two_wise_coverage(models, [3]) == 0.0    # S < 2


def test_distinct_projections_counts_the_projection_not_the_model():
    """Two models that agree on the support are one distinct sample."""
    from gpap.benchmark import distinct_projections

    sup = [1, 2]
    models = [[1, 2, 3], [1, 2, -3], [1, -2, 3]]     # first two share the projection (1, 2)
    assert distinct_projections(models, sup) == (2, 3)
    assert distinct_projections([], sup) == (0, 0)


# ---------------------------------------------------------------- table alignment
# Two layout invariants of the bench table, over every combination of optional columns:
#   same total width  every row branch (e.g. the three sim-baseline shapes) matches the header.
#   no glued cells    a value that exactly fills its right-aligned field runs into its left
#                     neighbour at the same total width. Labels and cells hold no spaces, so a
#                     token count against the header detects it.

class _TableArgs:                                   # only the flags the formatters read
    def __init__(self, **kw):
        self.seeds = self.twise = self.sim_baseline = self.cmsgen = None
        self.__dict__.update(kw)


# A row at the wide end of what campaigns produce: a mid-range row can pass while real output
# glues, and a row maxing every field would demand widths we never print. `unique_sps_est` and
# `accepted_per_s` are values seen gluing; the rest are the maxima `_format_header` documents.
_TABLE_ROW = {
    "name": "registerlesSwap-c50", "enc_key": "abc:fast", "nv": 999_999, "M": 1_000_000,
    "support": 65536, "gpap_sps": 62_345_678.0, "gpap_sps_e2e": 51_234_567.0,
    "decisions_per_s": 6.7e7, "implications_per_s": 2.4e9, "succ": 1.0,
    "n_models": 10_000_000, "chunk_capped": True, "wall_timed_s": 123.4,
    "wall_total_s": 234.5, "sps_spread_frac": 0.42, "encode_s": 123.4,
    "unique_sps_est": 12_234_239.0, "twise_gpap": 100.0, "mined": {"n_units": 1},
    "engine": {"mode": "wave", "shift": 16}, "sound_check": "ok",
    "cms_sps": 1_234_567.0, "cms_decisions_per_s": 5.4e6,
    "cms_implications_per_s": 5.4e6, "ratio": 1234.56, "ratio_e2e": 999.99,
    "twise_cms": 100.0,
}


@pytest.mark.parametrize("multi_enc", [False, True])
@pytest.mark.parametrize("has_circuits", [False, True])
@pytest.mark.parametrize("quality", [False, True])
@pytest.mark.parametrize("flags", [
    {}, {"seeds": [1, 2]}, {"twise": True}, {"sim_baseline": True},
    {"cmsgen": "/bin/true"}, {"twise": True, "cmsgen": "/bin/true"},
    {"seeds": [1, 2], "twise": True, "sim_baseline": True, "cmsgen": "/bin/true"},
])
def test_row_width_matches_header_width(multi_enc, has_circuits, quality, flags):
    from gpap.benchmark import _format_header, _format_row

    args = _TableArgs(**flags)
    hdr = _format_header(args, multi_enc, has_circuits, quality)
    # The three sim_baseline shapes are separate format branches. succ varies with them because
    # `fmt_pct` is widest at its small end ("4e-07%" fills the field); 3.6e-9 is below every
    # recorded rate (docs/findings/CLI_DELIVERY_FINDINGS.md §7).
    for sim, succ in product((None,
                              {"hits": 7, "p_accept": 4.9e-4, "accepted_per_s": 162_912_098.0},
                              {"hits": 0, "p_accept_lt": 1e-9}),
                             (1.0, 1e-4, 3.6e-9)):
        row = dict(_TABLE_ROW, sim_baseline=sim, amplification=2047.0, succ=succ)
        line = _format_row(row, args, multi_enc, has_circuits, quality)
        why = f"(sim={None if sim is None else sim['hits']}, succ={succ})\n{hdr}\n{line}"
        assert len(line) == len(hdr), \
            f"row is {len(line) - len(hdr):+d} chars off the header {why}"
        assert len(line.split()) == len(hdr.split()), \
            f"{len(hdr.split()) - len(line.split())} cell(s) ran together {why}"


def test_both_invariants_actually_fire():
    """Each layout check rejects the shape it exists for: a row too long, and a row of the
    right length with a glued cell."""
    from gpap.benchmark import _format_header, _format_row

    args = _TableArgs(sim_baseline=True)
    hdr = _format_header(args, False, False, False)
    # (a) overflow past the field: 15 digits + commas is well past `rej/s`.
    over = dict(_TABLE_ROW, amplification=2047.0,
                sim_baseline={"hits": 1, "p_accept": 0.5, "accepted_per_s": 1e14})
    assert len(_format_row(over, args, False, False, False)) > len(hdr)
    # (b) exactly filling the field: same total width, one token short, which the length
    #     check cannot see.
    glued = dict(_TABLE_ROW, amplification=2047.0,
                 sim_baseline={"hits": 1, "p_accept": 0.5,
                               "accepted_per_s": 162_912_098.0 * 1000})  # 15 chars w/ commas
    line = _format_row(glued, args, False, False, False)
    assert len(line) == len(hdr), "precondition: this row is the RIGHT length"
    assert len(line.split()) < len(hdr.split()), "...but two of its cells ran together"

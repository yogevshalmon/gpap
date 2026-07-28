"""Host-side tests for the evaluation harness (no CUDA needed).

Covers the parts of a long unattended sweep that must not be wrong: the packed-envelope
pre-check that decides whether an instance CAN run at all, the encoder-spec parsing that drives
the tseitin-vs-abc A/B, and the CMSGen stats/rate accounting (including the killed-run case,
where the counters come from the last flushed progress line).
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

from gpap.benchmark import (
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
    """A clause longer than the sat_count field would carry into the SUM field at construction.
    Only reachable with repeated literals, but it is the other half of the overflow story."""
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
    """The resume key must distinguish every cell of the matrix, or a resumed sweep silently
    skips rows it never ran."""
    base = {"name": "b03.bench", "enc_key": "tseitin", "cycles": 25, "target": "rare"}
    assert _row_key(base) != _row_key({**base, "enc_key": "abc:fast"})
    assert _row_key(base) != _row_key({**base, "cycles": 50})
    assert _row_key(base) != _row_key({**base, "target": "random"})


# --------------------------------------------------------------------- CMSGen accounting

def test_parse_cms_stats_takes_the_LAST_flushed_line():
    txt = ("c gpap-stats samples=1000 props=10 decisions=5 conflicts=1 cpu_s=0.1 wall_s=0.2\n"
           "c noise\n"
           "c gpap-stats samples=2000 props=40 decisions=9 conflicts=3 cpu_s=0.3 wall_s=0.4\n")
    s = _parse_cms_stats(txt)
    assert s == {"samples": 2000, "props": 40, "decisions": 9, "conflicts": 3,
                 "cpu_s": 0.3, "wall_s": 0.4}


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


# A stand-in for the patched cmsgen: it advertises the flags in --help and emits the same
# progress lines, so the harness's detection + accounting can be tested without a C++ build.
_FAKE_CMSGEN = textwrap.dedent("""
    import sys, time
    args = sys.argv[1:]
    if "--help" in args:
        print("usage: cmsgen [--samples VAR] [--samplefile VAR] [--nowrite] [--statsevery VAR]")
        sys.exit(0)
    n = int(args[args.index("--samples") + 1])
    out = args[args.index("--samplefile") + 1] if "--samplefile" in args else None
    if out:
        with open(out, "w") as f:
            for i in range(n):
                f.write("1 -2 3 0\\n")
    for i in range(1000, n + 1, 1000):
        print(f"c gpap-stats samples={i} props={i*10} decisions={i*3} conflicts={i//100} "
              f"cpu_s={i/1e5:.4f} wall_s={i/1e5:.4f}", flush=True)
    print("c Finished generating all %d samples" % n)
""")


@pytest.fixture()
def fake_cmsgen(tmp_path):
    p = tmp_path / "fake_cmsgen.py"
    p.write_text(_FAKE_CMSGEN)
    sh = tmp_path / "cmsgen"
    sh.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{p}" "$@"\n')
    sh.chmod(0o755)
    return str(sh)


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
    # implications = propagations - decisions, the same definition as the GPAP column
    assert r["implications_per_s"] == pytest.approx(
        (5000 * 10 - 5000 * 3) / r["wall_s"], rel=1e-6)


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
    """A killed CMSGen must still score what it produced — otherwise a hard instance reads 0."""
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


def test_cmsgen_run_missing_binary_is_an_error_row_not_a_crash(tmp_path):
    cnf = tmp_path / "f.cnf"
    cnf.write_text("p cnf 3 1\n1 2 3 0\n")
    r = cmsgen_run(str(cnf), "/nonexistent/cmsgen", 10, str(tmp_path))
    assert r["samples_per_s"] == 0.0 and "error" in r


# --------------------------------------------------------------------- driver plumbing

def test_bench_driver_skips_an_oversized_instance_and_still_writes_json(tmp_path):
    """--max-clauses is the wall-time guard for a sweep: the row is recorded as skipped (with the
    reason) and the run continues. Exercised through the real main() so the JSON contract is
    covered too; no GPU is touched because the instance never reaches the sampler."""
    cnf = tmp_path / "big.cnf"
    cnf.write_text("p cnf 10 999999\n1 2 0\n")
    out = tmp_path / "res.json"
    from gpap import benchmark
    benchmark.main(["--glob", str(cnf), "--max-clauses", "10", "--output", str(out)])
    payload = json.load(open(out))
    assert payload["schema"] >= 3
    row, = payload["results"]
    assert "max-clauses" in row["skipped"] and row["M"] == 999999


# --------------------------------------------------------------------------- rival fairness

def test_prepare_variant_raw_is_the_untouched_source(tmp_path):
    from eval.rival_common import prepare_variant
    src = "benchmarks/circuits/bench/seq/b03.bench"
    path, info = prepare_variant(src, "raw", str(tmp_path), cycles=5, seed=5)
    assert path == src and info == {"variant": "raw"}


def test_prepare_variant_rejects_unknown(tmp_path):
    from eval.rival_common import prepare_variant
    with pytest.raises(ValueError):
        prepare_variant("x.bench", "aigier", str(tmp_path), cycles=5, seed=5)


def test_prepare_variant_falls_back_to_raw_when_abc_is_missing(tmp_path, monkeypatch):
    """A rival must never be compared on a circuit we could not verify: with no abc binary the
    RAW path comes back, tagged with the reason, rather than a silently different circuit."""
    import gpap.circuits.abc_backend as ab
    from eval.rival_common import prepare_variant, variant_note
    monkeypatch.setattr(ab, "abc_available", lambda *a, **k: False)
    src = "benchmarks/circuits/bench/seq/b03.bench"
    path, info = prepare_variant(src, "abc", str(tmp_path), cycles=5, seed=5)
    assert path == src and "no abc binary" in info["error"]
    assert "-> raw" in variant_note(info)

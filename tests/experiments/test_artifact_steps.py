"""The artifact's steps, run on hand-built rows: what a reader is shown, and where files go.

  * the comparator names a changed verdict in its closing block, which a tier's SUMMARY.txt is;
  * a bare `reproduce.sh compare` shows every missing stage and leaves SUMMARY.txt to the tier;
  * the packager reads the tables where `reproduce.sh tables` writes them;
  * the packager resolves a relative output directory where it was typed, not in the checkout.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

from tests.conftest import REPO_ROOT

REPO = pathlib.Path(REPO_ROOT)
ARTIFACT = REPO / "eval" / "artifact"
RULE = "=" * 78


def _row(gpap_rate, rival_rate, name="c432"):
    return {"name": name, "variant": "abc", "demotic": {"unique_per_s": rival_rate},
            "gpap": {"unique_per_s_est": gpap_rate, "n_models": 5,
                     "sound_check": "ok", "sim_check": "ok"}}


def _stage(d, stem, rows):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"rivals_{stem}.json").write_text(json.dumps({"rows": rows}))


def test_a_changed_verdict_is_named_in_the_closing_block(tmp_path):
    """The PASS line counts a flipped verdict and says it is listed above; a tier's summary is
    the closing block alone, so the cell must be named there, not only in the stage report."""
    _stage(tmp_path / "ref", "demotic_native", [_row(100.0, 10.0)])
    _stage(tmp_path / "new", "demotic_native", [_row(1.0, 10.0)])
    r = subprocess.run([sys.executable, str(ARTIFACT / "compare_campaign.py"),
                        str(tmp_path / "ref"), str(tmp_path / "new")],
                       capture_output=True, text=True)
    closing = r.stdout.split(RULE)[-1]
    assert "1 DIFFER" in closing, r.stdout
    assert "c432" in closing, f"the flipped cell is not named in the closing block:\n{closing}"


def test_a_bare_compare_shows_missing_stages_and_leaves_the_summary_to_the_tier(tmp_path):
    """Over a run missing a stage, a bare `compare` prints the note that says so (only smoke and
    subset drop it, by design), and does not overwrite the SUMMARY.txt the tier wrote: it used
    to replace a tier's ARTIFACT_TIER_FAILED with a bare PASS."""
    _stage(tmp_path / "ref", "demotic_native", [_row(100.0, 10.0)])
    _stage(tmp_path / "ref", "demotic_comb_rare", [_row(100.0, 10.0)])
    _stage(tmp_path / "out" / "details", "demotic_native", [_row(100.0, 10.0)])
    summary = tmp_path / "out" / "SUMMARY.txt"
    summary.write_text("ARTIFACT_TIER_FAILED: 1 step(s) did not succeed\n")
    r = subprocess.run(["bash", str(ARTIFACT / "reproduce.sh"), "compare",
                        "--reference", str(tmp_path / "ref"), "--out", str(tmp_path / "out")],
                       capture_output=True, text=True, env={**os.environ, "PYTHON": sys.executable})
    assert "rivals_demotic_comb_rare: not in the fresh run" in r.stdout, r.stdout + r.stderr
    assert summary.read_text() == "ARTIFACT_TIER_FAILED: 1 step(s) did not succeed\n"


def test_the_packager_reads_the_tables_where_tables_writes_them():
    """`reproduce.sh tables` writes the audit, the gate and the figures into details/, and the
    paper folder's regenerate.sh runs it: a packager reading the top of the folder shipped the
    previous audit after a regeneration, and refused a folder generated only in the new layout."""
    reproduce = (ARTIFACT / "reproduce.sh").read_text()
    packager = (ARTIFACT / "make_artifact.sh").read_text()
    assert 'step_tables "$REFERENCE" "$DETAILS"' in reproduce
    assert '"$SRC_DATA/details/' in packager, "make_artifact.sh does not look in details/"


def test_the_packager_resolves_a_relative_output_where_it_was_typed(tmp_path):
    """make_artifact.sh changes into the checkout first, so a relative OUT used to land inside it:
    the archive was written into the repository and the next run refused a dirty tree. With no
    published rows the packager stops right after creating OUT, which is all this needs."""
    name = f"relative-out-{os.getpid()}"
    try:
        r = subprocess.run(["bash", str(ARTIFACT / "make_artifact.sh"), name], cwd=tmp_path,
                           capture_output=True, text=True,
                           env={**os.environ, "GPAP_PAPER_DIR": str(tmp_path / "no-paper")})
        assert r.returncode != 0 and "no published rows" in r.stderr, r.stdout + r.stderr
        assert (tmp_path / name).is_dir(), r.stdout
        assert not (REPO / name).exists(), "the relative output was resolved in the checkout"
    finally:
        shutil.rmtree(REPO / name, ignore_errors=True)

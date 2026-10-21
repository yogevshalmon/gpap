"""The artifact's steps, run on hand-built rows: what a reader is shown, and where files go.

  * the comparator names a changed verdict in its closing block, which a tier's SUMMARY.txt is;
  * a bare `reproduce.sh compare` shows every missing stage and leaves SUMMARY.txt to the tier;
  * the packager reads the tables where `reproduce.sh tables` writes them;
  * the packager resolves a relative output directory and a relative PYTHON where they were
    typed, not in the checkout;
  * the packager writes the venue's zip from the archive's own README and licence.
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


def test_the_packager_resolves_a_relative_python_where_it_was_typed(tmp_path):
    """make_artifact.sh changes into the checkout, then into its stage, so a relative PYTHON used
    to name no file there and the table gate failed after the tree was staged. With no published
    rows the packager stops right after printing the interpreter it resolved."""
    (tmp_path / "py").symlink_to(sys.executable)
    r = subprocess.run(["bash", str(ARTIFACT / "make_artifact.sh"), str(tmp_path / "out")],
                       cwd=tmp_path, capture_output=True, text=True,
                       env={**os.environ, "PYTHON": "./py",
                            "GPAP_PAPER_DIR": str(tmp_path / "no-paper")})
    assert r.returncode != 0 and "no published rows" in r.stderr, r.stdout + r.stderr
    assert f"python     : {tmp_path}/./py" in r.stdout, r.stdout


def test_the_packager_zips_the_archives_own_readme_and_licence(tmp_path):
    """The venue's form takes one zip holding a README and a LICENSE. The packager's own zip code,
    run here on two files, writes exactly those two names; it reads them from the finished archive
    ($FINAL), so the zip cannot drift from the files the archive ships; and the packager refuses
    to finish if the paper folder has no AE_README.md."""
    import zipfile

    packager = (ARTIFACT / "make_artifact.sh").read_text()
    assert '"$FINAL/docs/papers/tacas-27/AE_README.md" "$FINAL/LICENSE"' in packager
    assert "for f in AE_README.md ARTIFACT.md" in packager
    # The heredoc's body: after the `|| { ... }` line that ends the command, up to its `ZIP`.
    code = packager.split("<<'ZIP'")[1].split("\nZIP\n")[0]
    code = code[code.index("\nimport ") + 1:]
    (tmp_path / "AE_README.md").write_text("the guide\n")
    (tmp_path / "LICENSE").write_text("MIT\n")
    out = tmp_path / "out.zip"
    subprocess.run([sys.executable, "-", str(out), str(tmp_path / "AE_README.md"),
                    str(tmp_path / "LICENSE")], input=code, text=True, check=True)
    with zipfile.ZipFile(out) as z:
        assert z.namelist() == ["README.md", "LICENSE"]
        assert z.read("README.md") == b"the guide\n" and z.read("LICENSE") == b"MIT\n"

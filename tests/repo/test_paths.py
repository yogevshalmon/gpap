"""`gpap/paths.py`: where built tools, fetched benchmarks and scratch files go.

A wrong answer here moves every path in the repo at once. `tmp_dir()`'s fallback (an unusable
root) and its refusal of a planted symlink are exercised directly, since a healthy box never takes
either branch. `eval/paths.py` is tested in tests/experiments/test_eval_paths.py."""
import os


from gpap import paths

REPO = paths.repo_root()


# ------------------------------------------------------------------ the machinery root

def test_gpap_home_is_repo_local_and_the_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("GPAP_HOME", raising=False)
    assert paths.gpap_home() == REPO / paths.DIRNAME
    monkeypatch.setenv("GPAP_HOME", str(tmp_path / "elsewhere"))
    assert paths.gpap_home() == tmp_path / "elsewhere"
    for accessor, leaf in ((paths.bin_dir, "bin"), (paths.src_dir, "src"),
                           (paths.benchmarks_dir, "benchmarks"), (paths.tmp_dir, "tmp")):
        assert accessor() == tmp_path / "elsewhere" / leaf, f"{leaf} did not follow the root"


def test_scratch_lands_in_the_one_root_and_is_named_for_its_contents(monkeypatch, tmp_path):
    monkeypatch.setenv("GPAP_HOME", str(tmp_path / "h"))
    f = paths.scratch_file("b10-c25", ".cnf")
    d = paths.scratch_dir("rivals_demotic")
    try:
        assert os.path.dirname(f) == str(paths.tmp_dir()) == os.path.dirname(d)
        assert os.path.basename(f).startswith("b10-c25-") and f.endswith(".cnf"), \
            "the path is PRINTED, so it has to say what it holds"
        assert os.path.isdir(d)
        second = paths.scratch_file("b10-c25", ".cnf")
        assert second != f, "mkstemp: two runs of one instance must not collide"
        os.unlink(second)
    finally:
        os.unlink(f)
        os.rmdir(d)


def test_tmp_dir_falls_back_when_the_root_is_unusable(monkeypatch, tmp_path):
    """The branch that never runs on a healthy box: a read-only shared tool install. Without it the
    PermissionError surfaced from whatever line happened to open the file."""
    ro = tmp_path / "readonly"
    ro.mkdir()
    (ro / "tmp").mkdir()
    ro.chmod(0o500)
    (ro / "tmp").chmod(0o500)
    monkeypatch.setenv("GPAP_HOME", str(ro))
    try:
        got = paths.tmp_dir()
        assert got != ro / "tmp", "an unwritable scratch dir was handed back anyway"
        assert os.access(got, os.W_OK), f"the fallback {got} is not writable either"
    finally:
        ro.chmod(0o700)
        (ro / "tmp").chmod(0o700)


def test_tmp_dir_refuses_a_symlink_someone_else_could_have_planted(monkeypatch, tmp_path):
    """The per-UID name in the shared temp directory is PREDICTABLE, so a symlink sitting there is
    not ours to write through — it would redirect this run's scratch somewhere readable."""
    monkeypatch.delenv("GPAP_HOME", raising=False)
    monkeypatch.setattr(paths, "repo_root", lambda: None)          # pretend: installed wheel
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    # The target must EXIST: pointed at nothing, `_writable`'s mkdir fails on the dangling link
    # and the refusal would happen for the wrong reason — the test then passed with the
    # symlink guard deleted, which is the failure mode this whole file is about.
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / f"gpap-{os.getuid()}").symlink_to(tmp_path / "elsewhere",
                                                  target_is_directory=True)
    assert paths.tmp_dir() == tmp_path, "followed a planted symlink instead of refusing it"

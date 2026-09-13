"""The two modules that decide where every generated file goes.

Nothing tested these. The hygiene gates check that no OTHER file spells a managed directory, which
is a claim about spelling; this is the claim about behaviour, and it is the one that matters — a
wrong answer here moves every path in the repo at once. Three subjects, each picked because its
docstring makes a promise that used to go unexercised:

* `tmp_dir()`'s fallback chain. Three branches, and the repo's own recorded lesson is a cap test
  that only ever took one of them.
* `results_root()`'s `realpath`, which its docstring calls load-bearing — and is, but only on a
  checkout reached through a symlink, which is exactly the case a normal run never takes.
* `display_path`, which settles what a SHIPPED provenance header says. It has leaked a username
  twice, and a mismatched normalization silently demoted its output to a bare basename.
"""
import os
import subprocess
import sys

import pytest

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


@pytest.mark.parametrize("value", ["~/x", "rel"])
@pytest.mark.parametrize("var", ["GPAP_HOME", "GPAP_RESULTS", "GPAP_EXPERIMENTS"])
def test_every_root_override_is_expanded_and_made_absolute(var, value, monkeypatch):
    """ALL THREE roots, one normalization. They had three: raw, a `~` had a shell `mkdir -p` a
    directory literally NAMED `~`, and a relative root resolved against whatever directory the
    caller happened to stand in. Parametrized over the roots because the bug was that they
    differed — testing one proves nothing about the others."""
    from eval import paths as ev
    ask = {"GPAP_HOME": lambda: str(paths.gpap_home()),
           "GPAP_RESULTS": ev.results_root,
           "GPAP_EXPERIMENTS": ev.experiments_dir}[var]
    monkeypatch.setenv(var, value)
    got = ask()
    assert os.path.isabs(got), f"{var}={value!r} stayed relative: {got}"
    assert "~" not in got, f"{var}={value!r} kept a literal tilde: {got}"
    if value.startswith("~"):
        assert got == os.path.expanduser(value)


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


# ------------------------------------------------------------------ the experiment roots

def test_the_three_roots_stay_distinct_and_the_record_is_outside_the_checkout(monkeypatch):
    from eval import paths as ev
    monkeypatch.delenv("GPAP_RESULTS", raising=False)
    monkeypatch.delenv("GPAP_EXPERIMENTS", raising=False)
    monkeypatch.delenv("GPAP_HOME", raising=False)
    record, runs, machinery = ev.results_root(), ev.experiments_dir(), str(paths.gpap_home())
    assert not record.startswith(str(REPO) + os.sep), \
        "the record must survive deleting the checkout, so it cannot live inside it"
    assert runs.startswith(str(REPO) + os.sep), "a run's output is repo-local and gitignored"
    assert not runs.startswith(machinery + os.sep), \
        "output must not live under the root that is safe to delete"


def test_results_root_and_display_path_survive_a_symlinked_checkout(tmp_path):
    """`realpath` is load-bearing, not decorative: reached through a symlink from another cwd, an
    imported module's `__file__` KEEPS the symlink, so a lexical parent lands in a different tree —
    and `display_path` then failed its prefix match and demoted a `$GPAP_RESULTS/...` provenance
    header to a bare basename. Both are asserted through the symlink, because on a plain checkout
    the lexical and physical answers coincide and this proves nothing."""
    link = tmp_path / "checkout"
    link.symlink_to(REPO, target_is_directory=True)
    env = {k: v for k, v in os.environ.items() if k not in ("GPAP_RESULTS", "GPAP_EXPERIMENTS")}
    probe = ("from eval.paths import results_root, display_path\n"
             "r = results_root()\n"
             "print(r)\n"
             "print(display_path(r + '/paper/data/rivals_demotic_native.json'))\n")
    out = subprocess.run([sys.executable, "-c", probe], cwd=str(tmp_path),
                         env={**env, "PYTHONPATH": str(link)}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    root, shown = out.stdout.split("\n")[:2]
    assert root == os.path.join(os.path.dirname(str(REPO)), "gpap-results"), \
        f"through a symlink the record root moved to {root}"
    assert shown == "$GPAP_RESULTS/paper/data/rivals_demotic_native.json", shown


def test_display_path_ships_a_var_prefix_or_a_bare_basename_and_nothing_else(monkeypatch):
    """Two generators leaked a home directory into a published artifact before this existed, so the
    guarantee is: a path under a known root becomes `$VAR/...`, and anything else is reduced to ONE
    component — never a local directory tree, which is what carries a username.

    `/` is the one input that comes back as itself: there is no basename to fall back to, and it
    discloses nothing, so it is allowed rather than special-cased in the function."""
    from eval.paths import display_path
    # An inherited $GPAP_RESULTS pointing at an ancestor of the checkout would make the repo
    # paths below match the RECORD root first and render `$GPAP_RESULTS/...`.
    monkeypatch.delenv("GPAP_RESULTS", raising=False)
    assert display_path(str(REPO / "eval" / "paths.py")) == "$GPAP_REPO/eval/paths.py"
    for outside in ("/etc/passwd", "/var/tmp/somebody/secret.json",
                    os.path.expanduser("~/private/rows.json")):
        got = display_path(outside)
        assert os.sep not in got, f"{outside} -> {got}: shipped a directory tree"
        assert got == os.path.basename(outside)
    assert display_path("/") == "/"


def test_the_vm_layout_is_the_vms_and_reads_its_override_at_call_time(monkeypatch):
    """Reading the LOCAL `$GPAP_HOME` for a VM path pointed a download at a directory that exists
    only on this machine. `/root/.gpap` — the wrong default it was given next — exists on no
    machine the bootstrap builds."""
    from eval import paths as ev
    monkeypatch.delenv("GPAP_VM_ROOT", raising=False)
    monkeypatch.delenv("GPAP_VM_REPO", raising=False)
    monkeypatch.delenv("GPAP_VM_HOME", raising=False)
    monkeypatch.setenv("GPAP_HOME", "/somewhere/local/only")
    assert ev.vm_gpap_home() == ev.vm_repo() + "/" + paths.DIRNAME
    assert "/somewhere/local/only" not in ev.vm_gpap_home()
    monkeypatch.setenv("GPAP_VM_ROOT", "/mnt/vm")
    assert ev.vm_gpap_home() == "/mnt/vm/gpap/" + paths.DIRNAME, "read at import instead of call"

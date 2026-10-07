"""`eval/paths.py`: the experiments root, the results record outside the checkout, and
`display_path`, which decides what a shipped provenance header says."""
import os
import subprocess
import sys

import pytest

from gpap import paths

REPO = paths.repo_root()


@pytest.mark.parametrize("value", ["~/x", "rel"])
@pytest.mark.parametrize("var", ["GPAP_HOME", "GPAP_RESULTS", "GPAP_EXPERIMENTS"])
def test_every_root_override_is_expanded_and_made_absolute(var, value, monkeypatch):
    """All three roots expand `~` and resolve a relative override to an absolute path, the same
    way. Parametrized over the roots because each has its own code path."""
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
    """Through a symlinked checkout, `__file__` keeps the symlink, so the roots must use
    `realpath`; otherwise `display_path` misses its prefix match. Run through a symlink because on
    a plain checkout the lexical and physical paths coincide."""
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
    """A path under a known root becomes `$VAR/...`; anything else is reduced to its basename, so
    no local directory (which can carry a username) ships. `/` comes back as itself: it has no
    basename and discloses nothing."""
    from eval.paths import display_path
    # An inherited $GPAP_RESULTS at an ancestor of the checkout would match the repo paths below
    # first and render them as `$GPAP_RESULTS/...`.
    monkeypatch.delenv("GPAP_RESULTS", raising=False)
    assert display_path(str(REPO / "eval" / "paths.py")) == "$GPAP_REPO/eval/paths.py"
    for outside in ("/etc/passwd", "/var/tmp/somebody/secret.json",
                    os.path.expanduser("~/private/rows.json")):
        got = display_path(outside)
        assert os.sep not in got, f"{outside} -> {got}: shipped a directory tree"
        assert got == os.path.basename(outside)
    assert display_path("/") == "/"


def test_the_vm_layout_is_the_vms_and_reads_its_override_at_call_time(monkeypatch):
    """The VM's `.gpap` path is derived from the VM repo layout, never from the local
    `$GPAP_HOME`, and `GPAP_VM_ROOT` is read at call time."""
    from eval import paths as ev
    monkeypatch.delenv("GPAP_VM_ROOT", raising=False)
    monkeypatch.delenv("GPAP_VM_REPO", raising=False)
    monkeypatch.delenv("GPAP_VM_HOME", raising=False)
    monkeypatch.setenv("GPAP_HOME", "/somewhere/local/only")
    assert ev.vm_gpap_home() == ev.vm_repo() + "/" + paths.DIRNAME
    assert "/somewhere/local/only" not in ev.vm_gpap_home()
    monkeypatch.setenv("GPAP_VM_ROOT", "/mnt/vm")
    assert ev.vm_gpap_home() == "/mnt/vm/gpap/" + paths.DIRNAME, "read at import instead of call"

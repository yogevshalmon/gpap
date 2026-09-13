"""Where GPAP puts what it GENERATES, and the one place that decides it.

Three roots, told apart by what losing one costs:

    `.gpap/`           DISPOSABLE MACHINERY: built tools, fetched third-party sources, fetched
                       benchmark sets, and ALL scratch. Delete it and you lose only time.
                       `$GPAP_HOME` moves the whole root.
    `.experiments/`    a run's OUTPUT, on its way to the record.     -> eval/paths.py
    `../gpap-results`  the RECORD itself, outside the checkout.      -> eval/paths.py

This module owns the first. The other two are the experiment layer's and live in `eval/paths.py`,
because the library must not know what a campaign is.

**Nothing outside those two modules names a directory.** That is the whole convention, and it is
what makes the rule unspellable-two-ways rather than merely documented: shell scripts ASK
(`python3 -m gpap.paths tmp`) instead of recomputing, so bash and Python cannot end up disagreeing
about symlinks, `..` or `~` — there is one implementation and nothing to keep in step.

Repo-local by default, deliberately. These artifacts run to gigabytes once benchmarks are
fetched, and a home directory is the wrong place to put them silently: on a shared or quota'd
machine — the normal case for artifact evaluation — that is the difference between a build that
works and one that fails halfway with no obvious cause. Keeping them beside the checkout also
means removing the checkout removes everything it created.

Set `GPAP_HOME` to override (a fast scratch filesystem, a shared read-only tool install).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

#: Name of the repo-local directory. Gitignored.
DIRNAME = ".gpap"


def repo_root() -> Path | None:
    """The source checkout this package lives in, or None when installed as a wheel."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() or (parent / ".git").exists():
            return parent
    return None


def gpap_home() -> Path:
    """The disposable-machinery root: `$GPAP_HOME`, else `<repo>/.gpap`, else `./.gpap`."""
    env = os.environ.get("GPAP_HOME")
    if env:
        return Path(env).expanduser()
    root = repo_root()
    return (root or Path.cwd()) / DIRNAME


def bin_dir() -> Path:
    """Where `scripts/build_abc.sh` and `scripts/build_cmsgen.sh` install their binaries."""
    return gpap_home() / "bin"


def src_dir() -> Path:
    """Checkouts of optional third-party sources (abc, cmsgen, the rival samplers, barbarik)."""
    return gpap_home() / "src"


def benchmarks_dir() -> Path:
    """Downloaded benchmark sets (`scripts/fetch_benchmarks.sh`)."""
    return gpap_home() / "benchmarks"


def tool(name: str) -> Path:
    """Path a GPAP-built tool would have, whether or not it exists yet."""
    return bin_dir() / name


def _writable(d: Path, *, follow_symlink: bool = True) -> bool:
    """Can this process create a file in `d`, making it if needed? Permission bits only: a full
    or over-quota filesystem still passes here and fails at the open, which is the one case the
    probe cannot pre-empt.

    `follow_symlink=False` for a path OTHER USERS can predict: a symlink someone else planted
    there would silently redirect this run's scratch into a directory they can read."""
    if not follow_symlink and d.is_symlink():
        return False
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(d, os.W_OK | os.X_OK)


def tmp_dir() -> Path:
    """THE scratch directory, for every layer: files a run writes and then deletes — the CNF a
    circuit is unrolled to, the intermediate AIGs the ABC route hands to `abc`, a rival harness's
    working copies, a screen's throwaway output.

    `$GPAP_HOME/tmp` (so `<repo>/.gpap/tmp` by default) rather than the system temp directory,
    for the reason at the top of this file plus two of its own. `/tmp` is a tmpfs capped at half
    of RAM on several current distributions (Fedora, Arch, recent Ubuntu — not all, and not
    WSL2), and a deep unroll written there competes for host memory with the run that is about to
    sample it; the checkout's disk has room either way. And the path is PRINTED, where
    `.gpap/tmp/b10-c25-kzdub143.cnf` says what it holds and `/tmp/tmpuo7r0zb0.cnf` does not.

    One directory, no per-file judgement — that question used to be re-answered per harness and
    the answers drifted to five. A caller that passes an explicit scratch directory still wins:
    `gpap bench --scratch` and the campaign drivers put converted CNFs beside the rows they
    produced, on purpose, so a row's input stays auditable.

    Falls back to the system temp directory when there is no checkout and no `$GPAP_HOME` — an
    installed wheel, where `./.gpap` would mean creating a directory in whatever directory the
    user happens to be in — and whenever the directory is not WRITABLE: a read-only shared tool
    install, or a `<tmp>/gpap-<uid>` someone else on the box got to first. Writability, not just
    creation: `mkdir(exist_ok=True)` succeeds on a directory nobody can write to, and the
    PermissionError then surfaced from the unrelated line that opened the file. (A full or
    over-quota filesystem still fails at the open — no probe short of writing catches that.) The
    last resort is the system temp directory itself, which is what every version before this one
    used.

    Nothing here is MEANT to survive a run, but a kill leaves files behind, so they do collect —
    inside the checkout rather than somewhere the OS reclaims. `rm -rf .gpap/tmp` is safe against
    anything but a run in flight, and `gpap bench --keep-cnf` fills it on purpose.
    """
    if os.environ.get("GPAP_HOME") or repo_root() is not None:
        d = gpap_home() / "tmp"
        if _writable(d):        # symlinked on purpose is legitimate here: $GPAP_HOME is chosen
            return d
    # Per-UID, and never through a symlink: the system temp directory is shared and this name is
    # predictable, so a directory someone else created there is not ours to write into.
    d = Path(tempfile.gettempdir()) / f"gpap-{os.getuid()}"
    return d if _writable(d, follow_symlink=False) else Path(tempfile.gettempdir())


def scratch_file(stem: str, suffix: str) -> str:
    """Create an empty file in `tmp_dir()` and return its path. Named after what it holds —
    `b10-c25-kzdub143.cnf`, not `tmpuo7r0zb0.cnf` — because the path is printed, and a run that dies
    before its own cleanup leaves something the reader can identify. `mkstemp`, so two runs of
    the same instance never collide."""
    fd, path = tempfile.mkstemp(prefix=f"{stem}-", suffix=suffix, dir=tmp_dir())
    os.close(fd)
    return path


def scratch_dir(stem: str) -> str:
    """Create a directory in `tmp_dir()` and return its path — the several-files form of
    `scratch_file`, for a harness that hands a whole working directory to a subprocess. The
    caller owns removing it; `mkdtemp`, so concurrent runs never share one."""
    return tempfile.mkdtemp(prefix=f"{stem}-", dir=tmp_dir())


#: What `python3 -m gpap.paths <role>` prints, so a shell script asks instead of recomputing.
ROLES = {"repo": lambda: repo_root() or Path.cwd(), "home": gpap_home, "bin": bin_dir,
         "src": src_dir, "benchmarks": benchmarks_dir, "tmp": tmp_dir}


def _main(argv) -> int:
    if len(argv) != 1 or argv[0] not in ROLES:
        import sys
        print(f"usage: python3 -m gpap.paths {{{'|'.join(ROLES)}}}", file=sys.stderr)
        return 2
    print(ROLES[argv[0]]())
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))

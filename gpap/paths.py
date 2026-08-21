"""Where GPAP keeps things that are built or fetched rather than shipped.

Optional third-party tools (abc, cmsgen), their sources, and downloaded benchmark sets do not
live in the repository and are never vendored. They land in ONE place:

    $GPAP_HOME  ->  <repo>/.gpap  ->  ./.gpap

Repo-local by default, deliberately. These artifacts run to gigabytes once benchmarks are
fetched, and a home directory is the wrong place to put them silently: on a shared or quota'd
machine — the normal case for artifact evaluation — that is the difference between a build that
works and one that fails halfway with no obvious cause. Keeping them beside the checkout also
means removing the checkout removes everything it created.

Set `GPAP_HOME` to override (a fast scratch filesystem, a shared read-only tool install).
"""
from __future__ import annotations

import os
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
    """Base directory for built tools, fetched sources and benchmarks."""
    env = os.environ.get("GPAP_HOME")
    if env:
        return Path(env).expanduser()
    root = repo_root()
    return (root or Path.cwd()) / DIRNAME


def bin_dir() -> Path:
    """Where `scripts/build_abc.sh` and `scripts/build_cmsgen.sh` install their binaries."""
    return gpap_home() / "bin"


def src_dir() -> Path:
    """Checkouts of optional third-party sources."""
    return gpap_home() / "src"


def benchmarks_dir() -> Path:
    """Benchmark sets fetched by `scripts/fetch_benchmarks.sh`."""
    return gpap_home() / "benchmarks"


def tool(name: str) -> Path:
    """Path a GPAP-built tool would have, whether or not it exists yet."""
    return bin_dir() / name

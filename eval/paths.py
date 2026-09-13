"""Where the EXPERIMENT layer's directories are, and the one place that decides them.

The companion to `gpap/paths.py`, which owns the disposable-machinery root (`.gpap`: built tools,
fetched sources, ALL scratch). This owns the two roots that are not disposable:

    `.experiments/`    a run's OUTPUT while it is being produced — rows, logs, plots. Repo-local
                       and gitignored, then mirrored to the record. `$GPAP_EXPERIMENTS` moves it.
    `../gpap-results`  THE RECORD: campaign data, the paper folder, manifests. OUTSIDE the
                       checkout on purpose, so deleting the checkout — or deleting `.gpap` —
                       cannot delete evidence. `$GPAP_RESULTS` moves it.

Plus the layout of a RENTED MACHINE, which is a third thing entirely and was the subject of the
worst confusion this module exists to end: a Colab VM's directories are not this machine's, so
reading `$GPAP_HOME` to build a VM path points a download at a directory that exists only here.

Nothing outside this module and `gpap/paths.py` names a directory. Shell asks
(`python3 -m eval.paths results`) rather than recomputing, so there is one implementation of each
rule instead of one per language.
"""
from __future__ import annotations

import os

from gpap.paths import DIRNAME  # the machinery root's NAME has one owner

#: The checkout. One spelling, because a hand-counted depth is what broke the repo root in eleven
#: files the last time `eval/` was reorganized.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _from_env(var: str, default: str) -> str:
    """A root the caller may override, normalized ONE way for all of them: `~` expanded and made
    absolute. Left raw, `GPAP_EXPERIMENTS=~/runs` had a shell `mkdir -p` a directory literally
    named `~`, and a relative value resolved against wherever the caller happened to stand."""
    env = os.environ.get(var)
    return os.path.abspath(os.path.expanduser(env)) if env else default


def experiments_dir() -> str:
    """Where a run writes its rows before they are mirrored to the record."""
    return _from_env("GPAP_EXPERIMENTS", os.path.join(REPO, ".experiments"))


def results_root() -> str:
    """The record: `$GPAP_RESULTS`, else `gpap-results` beside the checkout.

    PHYSICAL, because the fallback's `..` is not inert: left in a string it resolves physically
    when the path is opened (the kernel follows the symlink, then the `..`) but lexically when the
    same string is compared through `abspath`/`relpath`, so for a checkout reached through a
    symlink one spelling named two directories — and `display_path` settles a SHIPPED provenance
    header by comparing exactly that. `realpath` is load-bearing here, not decorative: `__file__`
    is made absolute by the interpreter but never symlink-resolved, so without it `REPO` can be a
    symlink and the parent comes out somewhere else entirely.

    An explicit `$GPAP_RESULTS` is taken as given (made absolute, never resolved): the caller
    named that directory and a symlink they chose is theirs to keep.
    """
    return _from_env("GPAP_RESULTS",
                     os.path.join(os.path.dirname(os.path.realpath(REPO)), "gpap-results"))


def display_path(path) -> str:
    """A path as it belongs in a file we SHIP, not as it happens to sit on this disk.

    Generated headers are provenance, read by people who do not have our home directory — and an
    absolute one also carries a username into an artifact that goes to Zenodo and stays there.
    Two separate generators had already leaked one by the time this was written (a committed slide
    table for a month, and `emission_audit.txt` into the packaged artifact), which is what makes
    it a shared function rather than a local fix.

    Rewrites a path under the record or under the checkout to a `$VAR/`-prefixed form, and falls
    back to the basename for anything else — never the absolute path. Every comparison is
    physical, matching `results_root()`: mixing one physical root with one lexical one silently
    demoted a `$GPAP_RESULTS/...` header to a bare basename on a symlinked checkout.
    """
    roots = [("$GPAP_RESULTS", results_root()), ("$GPAP_REPO", os.path.realpath(REPO))]
    full = os.path.realpath(str(path))
    for label, base in roots:
        if not base:
            continue
        rel = os.path.relpath(full, os.path.realpath(base))
        if not rel.startswith(os.pardir):
            return f"{label}/{rel}" if rel != "." else label
    return os.path.basename(full.rstrip("/")) or full


# ------------------------------------------------------------------ a rented machine's layout
# NOT this machine's. The supervisor runs locally and drives a VM, so every path it hands over must
# come from the VM's convention: the bootstrap untars the checkout at `vm_repo()`, exports
# `GPAP_HOME=<that>/.gpap`, and every driver resolves its tools under it. Reading the LOCAL
# `$GPAP_HOME` here pointed a download at a directory that exists only on this box. Call-time like
# every accessor above, so setting the override after import still takes effect.

def vm_root() -> str:
    """The rented machine's working root (Colab's). Everything else over there derives from it, so
    a differently-laid-out VM is one variable rather than a dozen string literals."""
    return os.environ.get("GPAP_VM_ROOT", "/content")


def vm_repo() -> str:
    """Where the bootstrap untars the checkout."""
    return os.environ.get("GPAP_VM_REPO", vm_root() + "/gpap")


def vm_gpap_home() -> str:
    """The VM's machinery root — the same `<repo>/.gpap` rule, resolved over there. NOT
    `/root/.gpap`, which was the default for a while and exists on no machine the bootstrap
    builds."""
    return os.environ.get("GPAP_VM_HOME", vm_repo() + "/" + DIRNAME)


def vm_prebuilt() -> str:
    """Where the supervisor uploads prebuilt binaries for the bootstrap to install from."""
    return os.environ.get("GPAP_VM_PREBUILT", vm_root() + "/prebuilt")


#: What `python3 -m eval.paths <role>` prints, so a shell script asks instead of recomputing.
#: Only the roles something actually asks for — the VM layout is imported, not shelled out to.
ROLES = {"experiments": experiments_dir, "results": results_root}


def _main(argv) -> int:
    if len(argv) != 1 or argv[0] not in ROLES:
        import sys
        print(f"usage: python3 -m eval.paths {{{'|'.join(ROLES)}}}", file=sys.stderr)
        return 2
    print(ROLES[argv[0]]())
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))

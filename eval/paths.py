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

#: The checkout. One spelling, because a hand-counted depth is what broke the repo root in eleven
#: files the last time `eval/` was reorganized.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def experiments_dir() -> str:
    """Where a run writes its rows before they are mirrored to the record."""
    return os.environ.get("GPAP_EXPERIMENTS") or os.path.join(REPO, ".experiments")


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
    env = os.environ.get("GPAP_RESULTS")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(os.path.dirname(os.path.realpath(REPO)), "gpap-results")


def display_path(path, root=None) -> str:
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
    roots = [("$GPAP_RESULTS", results_root()),
             ("$GPAP_REPO", os.path.realpath(REPO))] if root is None else [(root[0], root[1])]
    full = os.path.realpath(str(path))
    for label, base in roots:
        if not base:
            continue
        rel = os.path.relpath(full, os.path.realpath(base))
        if not rel.startswith(os.pardir):
            return f"{label}/{rel}" if rel != "." else label
    return os.path.basename(full.rstrip("/")) or full


# ------------------------------------------------------------------ a rented machine's layout
# NOT this machine's. The supervisor runs locally and drives a VM, so every path it hands to the
# VM must come from the VM's convention: the bootstrap untars the checkout at VM_REPO, exports
# `GPAP_HOME=<VM_REPO>/.gpap`, and every driver then resolves its tools under it. Reading the
# LOCAL `$GPAP_HOME` here pointed a download at a directory that exists only on this box.
#: The rented machine's working root — Colab's. Everything else over there derives from it,
#: so a differently-laid-out VM is one variable, not a dozen string literals.
VM_ROOT = os.environ.get("GPAP_VM_ROOT", "/content")
#: Where the bootstrap untars the checkout.
VM_REPO = os.environ.get("GPAP_VM_REPO", VM_ROOT + "/gpap")
#: The VM's machinery root — the same `<repo>/.gpap` rule, resolved over there. It is NOT
#: `/root/.gpap`: that was this constant's default for a while and exists on no machine the
#: bootstrap builds, which is why the prebuilt-cache fill never found anything.
VM_GPAP_HOME = os.environ.get("GPAP_VM_HOME", VM_REPO + "/.gpap")
#: Where the supervisor uploads prebuilt binaries for the bootstrap to install from.
VM_PREBUILT = os.environ.get("GPAP_VM_PREBUILT", VM_ROOT + "/prebuilt")


def vm_tool(name: str) -> str:
    """Path of a GPAP-built tool ON THE VM — the same `bin/` rule as `gpap.paths.tool`, over
    there. `name` is the INSTALLED name, which is not always the uploaded one (the supervisor
    pushes `abc_stripped` and the bootstrap installs it as `abc`)."""
    return f"{VM_GPAP_HOME}/bin/{name}"


#: What `python3 -m eval.paths <role>` prints, so a shell script asks instead of recomputing.
ROLES = {"repo": lambda: REPO, "experiments": experiments_dir, "results": results_root,
         "vm-repo": lambda: VM_REPO, "vm-home": lambda: VM_GPAP_HOME,
         "vm-prebuilt": lambda: VM_PREBUILT}


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

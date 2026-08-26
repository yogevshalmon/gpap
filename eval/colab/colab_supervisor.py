#!/usr/bin/env python3
"""Run a long GPAP sweep on a Colab VM that will be taken away before it finishes.

Measured: a rented GPU assignment was reclaimed after **~1 h** with the keep-alive
daemon running and no error logged — not idle pruning, just the assignment's lifetime. A 4-hour
sweep therefore cannot be "launched and collected"; it has to survive the VM dying repeatedly.

The rule this encodes, learned by losing a 1-hour run: **results only exist once they are on the
local disk.** Checkpointing to the VM's disk buys nothing, because that disk dies with the VM.

    ensure session -> bootstrap -> push local partials -> launch -> poll+PULL every cycle
         ^                                                              |
         +------------------ VM reclaimed, work not done ---------------+

So each cycle downloads the result files, and each new VM is seeded with what we already have.
The remote command must be RESUMABLE (`--resume` on the rival harnesses, `--resume` on
`gpap bench`) or every restart would redo finished work.

    eval/colab/colab_supervisor.py --session l4riv --gpu L4 \
        --tarball /tmp/gpap_src.tgz --bootstrap /tmp/bootstrap.sh \
        --remote-cmd 'bash eval/colab/colab_rivals_l4.sh' \
        --remote-dir /content/gpap/.experiments/full_eval/nvidia-l4 \
        --files rivals_demotic_native.json rivals_demotic_miter.json rivals_hts.json \
                rivals_recurrent_rare.json rivals_recurrent_table.json \
        --local-dir .experiments/l4_rivals --done-marker RIVALS_STAGE_COMPLETE

⚠️ **Stop the SESSION before restarting this script.** Killing the supervisor mid-bootstrap and
relaunching it starts a SECOND bootstrap on the same VM; both `rm -rf /content/gpap` and untar
into it, one loses the race and writes `/content/bootstrap.fail`, and the run burns a VM
recovering (observed 2026-08-26 while extending --max-hours). Either wait for the cycle to reach
'launching', or `colab stop -s <session>` first and let the fresh supervisor build a clean VM.

Console quirks this works around (all hit the hard way, see the colab-cli notes):
`colab exec` is broken in 0.6.0, so everything goes through `colab console`; the console exits on
piped EOF, so commands are fire-and-forget and progress is polled separately; tmux line-wraps and
corrupts long one-liners, so commands are short and newline-separated.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

COLAB = os.environ.get("COLAB_BIN") or shutil.which("colab") or ".venv/bin/colab"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, timeout=600):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def console(session, lines, timeout=600):
    """Fire shell lines into the tmux console. Returns whatever came back (escapes stripped).
    NEVER rely on this for long synchronous work — the console exits on EOF."""
    payload = "".join(l + "\n" for l in lines)
    try:
        p = subprocess.run([COLAB, "console", "-s", session], input=payload,
                           capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    import re
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", out).replace("\r", "")


def session_alive(session):
    rc, out = run([COLAB, "status", "-s", session], timeout=120)
    return rc == 0 and "not found" not in out and "lost" not in out.lower()


def ensure_session(session, gpu):
    if session_alive(session):
        return True
    log(f"session '{session}' is gone — creating a new {gpu}")
    run([COLAB, "stop", "-s", session], timeout=180)          # clear stale local state
    rc, out = run([COLAB, "new", "-s", session, "--gpu", gpu], timeout=900)
    if rc != 0 or "READY" not in out:
        log(f"could not create session: {out.strip()[:300]}")
        return False
    return True


def marker(session, path):
    """True iff `path` exists on the VM.

    Two traps stacked here, both of which produced wrong answers:

    1. The console ECHOES the command it runs, so a plain `grep -c BOOT_FAIL ...` matches its own
       command text (and `set -x` prints a script's `|| { echo BOOT_FAIL...; }` branches even on
       success) — any literal-string probe reads as a hit the moment you look.
    2. `ls | wc -l` answers with a bare digit, which dodges (1) but NOT tmux: a redraw can emit a
       lone digit line of its own, which false-positived a bootstrap into "FAILED" twice.

    So the reply token is SPLIT in the command text (`"MRK""OK"`) and joined only by the shell.
    The echoed command therefore contains `MRK""OK`, never the token itself, and any line holding
    the joined token with no stray quotes is genuine output.
    """
    out = console(session, [f'ls {path} >/dev/null 2>&1 && echo "MRK""OK" || echo "MRK""NO"'])
    return any("MRKOK" in ln and '"' not in ln for ln in out.splitlines())


def bootstrap(session, tarball, bootstrap_sh):
    log("bootstrapping VM ...")
    for local, remote in ((tarball, "/content/gpap_src.tgz"),
                          (bootstrap_sh, "/content/bootstrap.sh")):
        rc, out = run([COLAB, "upload", "-s", session, local, remote], timeout=900)
        if rc != 0:
            log(f"upload failed {local}: {out.strip()[:200]}")
            return False
    console(session, ["nohup bash /content/bootstrap.sh > /dev/null 2>&1 &", "echo LAUNCHED"])
    deadline = time.time() + 1800
    while time.time() < deadline:
        time.sleep(30)
        if not session_alive(session):
            log("VM died during bootstrap")
            return False
        if marker(session, "/content/bootstrap.ok"):
            log("bootstrap complete")
            return True
        if marker(session, "/content/bootstrap.fail"):
            log("bootstrap FAILED: " + console(session, ["cat /content/bootstrap.fail"]).strip()
                .replace("\n", " ")[:160])
            return False
    log("bootstrap timed out")
    return False


def push_partials(session, local_dir, remote_dir, files):
    """Seed the fresh VM with everything already collected, so --resume skips it."""
    console(session, [f"mkdir -p {remote_dir}"])
    for f in files:
        p = os.path.join(local_dir, f)
        if os.path.isfile(p):
            rc, _ = run([COLAB, "upload", "-s", session, p, f"{remote_dir}/{f}"], timeout=600)
            log(f"  pushed {f} ({os.path.getsize(p):,} B)" if rc == 0 else f"  push FAILED {f}")


def _intact(path):
    """Is this download complete? A file copied while the VM was writing it reads back truncated,
    and a truncated result silently REPLACING a good local copy is the one failure this script
    exists to prevent. Checked by type: JSON must parse, a tarball must open. Anything else is
    accepted on being non-empty — note that validating EVERY file as JSON (which this did until
    2026-08-25) silently discards every non-JSON artifact, e.g. the GPAP-arm cache tarball that
    makes a rival-side re-run cheap."""
    import json
    import tarfile
    if path.endswith(".json"):
        try:
            json.load(open(path))
            return True
        except Exception:           # noqa: BLE001
            return False
    if path.endswith((".tgz", ".tar.gz")):
        try:
            with tarfile.open(path) as t:
                return bool(t.getmembers())
        except Exception:           # noqa: BLE001
            return False
    return True


def pull_results(session, local_dir, remote_dir, files):
    """THE point of this script: get results onto local disk every cycle."""
    os.makedirs(local_dir, exist_ok=True)
    got = []
    for f in files:
        tmp = os.path.join(local_dir, f + ".part")
        rc, _ = run([COLAB, "download", "-s", session, f"{remote_dir}/{f}", tmp], timeout=600)
        if rc == 0 and os.path.isfile(tmp) and os.path.getsize(tmp) > 0 and _intact(tmp):
            os.replace(tmp, os.path.join(local_dir, f))
            got.append(f)
        elif os.path.exists(tmp):
            os.remove(tmp)
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--gpu", default="L4")
    ap.add_argument("--tarball", required=True, help="repo tarball to unpack on the VM")
    ap.add_argument("--bootstrap", required=True, help="bootstrap script (must echo BOOTSTRAP_DONE)")
    ap.add_argument("--remote-cmd", required=True, help="run from /content/gpap, must be resumable")
    ap.add_argument("--remote-dir", required=True, help="where the remote job writes its JSONs")
    ap.add_argument("--files", nargs="+", required=True, help="result filenames to mirror")
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--cache", nargs="*", default=[], metavar="REMOTE:LOCAL",
                    help="binary to carry between VMs: pushed to REMOTE before bootstrap when the "
                         "LOCAL copy exists, otherwise downloaded into LOCAL after the first VM "
                         "builds it. ONLY for self-contained executables — cmsgen is NOT one "
                         "(it links libcmsgen.so from its build tree, so a cached copy cannot "
                         "run and its bootstrap check correctly rejects it); just let it rebuild.")
    ap.add_argument("--push", nargs="*", default=[], metavar="LOCAL:REMOTE",
                    help="extra files to upload to every fresh VM, e.g. a prebuilt cmsgen "
                         "binary — rebuilding it on each ~1 h VM would cost a fifth of its life")
    ap.add_argument("--done-marker", required=True, help="string the job prints when finished")
    ap.add_argument("--poll", type=float, default=180.0)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--log-name", default="run.log")
    args = ap.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    local_log = os.path.join(args.local_dir, args.log_name)
    end = time.time() + args.max_hours * 3600
    launched_on = None                       # endpoint we last launched the job on
    boot_fails = 0
    had_results = False                      # have we ever pulled anything off this VM?

    while time.time() < end:
        if not ensure_session(args.session, args.gpu):
            log("no session; retrying in 120 s")
            time.sleep(120)
            continue
        rc, st = run([COLAB, "status", "-s", args.session], timeout=120)
        endpoint = st.split("|")[0].strip() if "|" in st else st.strip()

        if endpoint != launched_on:          # fresh VM: build it, seed it, start the job
            # Cached binaries go up BEFORE bootstrap, so the bootstrap script can use them
            # instead of rebuilding. CMSGen is a full CryptoMiniSat build (~10 min); on a VM
            # that only lives ~1 h that is a sixth of the machine, every hour.
            for spec in args.cache:
                remote, _, local = spec.partition(":")
                if os.path.isfile(local):
                    # the upload does NOT create parent dirs — without this the push fails and
                    # the VM silently rebuilds from source (measured: cost a whole CMSGen build)
                    console(args.session, [f"mkdir -p {os.path.dirname(remote)}"])
                    rc, _ = run([COLAB, "upload", "-s", args.session, local, remote], timeout=900)
                    console(args.session, [f"chmod +x {remote}"])
                    log(f"  cache HIT -> pushed {os.path.basename(local)}"
                        if rc == 0 else f"  cache push FAILED {local}")
            if not bootstrap(args.session, args.tarball, args.bootstrap):
                boot_fails += 1
                # Don't churn through VMs on a bug of our own: a bootstrap that fails on several
                # fresh machines in a row is a broken script, not bad luck, and each retry costs
                # a billable assignment.
                if boot_fails >= 3:
                    log("bootstrap failed on 3 consecutive VMs — stopping (fix the script)")
                    run([COLAB, "stop", "-s", args.session], timeout=180)
                    return 1
                run([COLAB, "stop", "-s", args.session], timeout=180)
                continue
            boot_fails = 0
            # Fill the cache from the first VM that built it, so no later VM has to.
            for spec in args.cache:
                remote, _, local = spec.partition(":")
                if not os.path.isfile(local):
                    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
                    src = spec.split(":", 1)[0]
                    built = src.replace("/content/prebuilt/", os.environ.get("GPAP_HOME", "/root/.gpap") + "/bin/")
                    rc, _ = run([COLAB, "download", "-s", args.session, built, local], timeout=900)
                    if rc == 0 and os.path.isfile(local) and os.path.getsize(local) > 0:
                        os.chmod(local, 0o755)
                        log(f"  cache FILL <- {built} ({os.path.getsize(local):,} B)")
                    elif os.path.exists(local):
                        os.remove(local)
            for spec in args.push:
                local, _, remote = spec.partition(":")
                if not os.path.isfile(local):
                    log(f"  push source missing, skipped: {local}")
                    continue
                rc, _ = run([COLAB, "upload", "-s", args.session, local, remote], timeout=900)
                console(args.session, [f"chmod +x {remote}"])
                log(f"  pushed {os.path.basename(local)} -> {remote}"
                    if rc == 0 else f"  push FAILED {local}")
            push_partials(args.session, args.local_dir, args.remote_dir, args.files)
            log(f"launching: {args.remote_cmd}")
            # Write the command to a script and nohup THAT. `nohup <cmd>` execs directly, so an
            # env-var prefix (`FOO=bar bash x.sh`) makes nohup look for a program literally named
            # "FOO=bar" and fail with "No such file or directory" — silently, in a log nobody
            # reads, after the VM has already been built. Measured: 45 minutes of an empty run.
            console(args.session, [
                "cd /content/gpap",
                f"cat > /content/job.sh <<'GPAPJOBEOF'\n{args.remote_cmd}\nGPAPJOBEOF",
                "nohup bash /content/job.sh > /content/run.log 2>&1 &",
                "echo LAUNCHED_JOB"])
            launched_on = endpoint

        time.sleep(args.poll)
        if not session_alive(args.session):
            log("VM reclaimed — will rebuild and resume from mirrored results")
            launched_on = None
            continue

        got = pull_results(args.session, args.local_dir, args.remote_dir, args.files)
        # A VM can be reset WITHOUT the session changing endpoint — `colab status` still says OK
        # while /content has been wiped. Observed: 3 straight cycles pulling nothing from a
        # machine that had been mirroring 12/13 a moment earlier, with no rebuild triggered.
        # Going from "results" to "none" is that signal; force a rebuild instead of polling a
        # dead disk for the rest of the budget.
        if not got and had_results and not marker(args.session, args.remote_dir):
            log("remote results vanished (VM reset, same endpoint) — rebuilding")
            launched_on = None
            continue
        had_results = had_results or bool(got)
        tail = console(args.session, ["tail -c 3000 /content/run.log"])
        with open(local_log, "a") as fh:
            fh.write(f"\n===== {time.strftime('%F %T')} =====\n{tail}\n")
        log(f"mirrored {len(got)}/{len(args.files)}: {', '.join(got) or 'nothing yet'}")
        if args.done_marker in tail:
            log("DONE marker seen — final pull")
            pull_results(args.session, args.local_dir, args.remote_dir, args.files)
            return 0
    log("max-hours reached")
    return 1


if __name__ == "__main__":
    sys.exit(main())

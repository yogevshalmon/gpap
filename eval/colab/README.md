# Running a campaign on a rented Colab GPU

The drivers in this directory push a GPAP checkout to a Colab VM, run a campaign stage there, and
mirror the results back. The platform rules below are operational — they say nothing about GPAP
and nothing about a GPU's speed — which is why they live beside the drivers rather than in
`docs/findings/`. They were learned on `google-colab-cli` 0.6.0 against Colab's L4 and A100
assignments, 2026-08-03 onward, and each one cost real time before it was written down.

## The lifetime is the design constraint

**A Colab GPU VM is reclaimed after about an hour, keep-alive or not.** One session, unambiguous:
created 13:45:29 UTC, keep-alive running normally, terminated 14:49:18 — 1 h 04 m, no error
logged. It recurred every session.

The rule it forces: **results exist only once they are on the local disk.** Per-row checkpointing
to the VM's own disk buys nothing, and the first rivals run was lost exactly that way.

`colab_supervisor.py` encodes the surviving loop:

```
ensure session -> push cached binaries -> bootstrap -> push partials
              -> launch -> poll and PULL every cycle -> reseed a fresh VM
```

Every harness takes `--resume`; a downloaded result replaces the local copy only once it
**parses**; built binaries (CMSGen is ~10 min) are cached locally and pushed to each fresh VM
along with the mirrored partials.

## Driving the CLI

`pip install google-colab-cli`, then ADC with all four scopes (`openid`, `cloud-platform`,
`userinfo.email`, `colaboratory` — it 403s without the last). `colab whoami` verifies.

`colab exec` is **broken** in google-colab-cli 0.6.0: it calls
`jupyter_kernel_client.KernelClient`, renamed upstream to `JupyterKernelClient`. Use
`colab console`, a raw-websocket tmux shell, and always detach:

```bash
printf 'nohup bash /content/job.sh > /content/job.log 2>&1 &\necho LAUNCHED\n' \
    | colab console -s <session>                       # fire and detach
printf 'grep -ac DONE_MARKER /content/job.log\n' | colab console -s <session>   # poll separately
```

The console exits on piped EOF, so a foreground run dies with the pipe. Keep commands short and
newline-separated: tmux line-wraps long one-liners and `&&` arrives as `& &`. Filter output with
`grep -a` and `sed 's/\x1b\[[0-9;]*[a-zA-Z]//g'`. Move results back with `tar` + `colab download`.
`colab stop -s <name>` when done; the keep-alive is a local daemon with a hard 24 h cap.

Bootstrap is ~2 min (`apt-get install build-essential cmake zlib1g-dev`, deps,
`scripts/build_core.sh`); libgpap AOT-builds on sm_80 and sm_89 with no CUDA toolkit. Where two
checkouts must coexist on one VM, select the package by cwd + `PYTHONPATH`, never `pip install -e .`.

## Four traps

* **Never grep a log for a status string over the console.** The console echoes the command it
  runs, and `set -x` prints a script's `|| { echo BOOT_FAIL…; }` branch on success — so both read
  as a failure that did not happen. Use **marker files**, counted with `ls … | wc -l`.
* **Resume must retry error rows, and replace them rather than append.** Treating an error row as
  finished froze 18 HTS rows behind a missing dependency; appending the retry produced 25 rows in
  an 18-instance file.
* **Never wait on a grep against an append-only mirror** — it matches a previous run and fires
  early. And **one poller per session**: concurrent console users corrupt each other's commands.
* **Never edit a running bash script.** The shell reads it by byte offset, so the edit lands
  inside the process.

## Where the measurements went

The campaigns these drivers ran are recorded in `docs/findings/CROSS_GPU_FINDINGS.md` (what a
bigger GPU buys), `docs/findings/L4_EVAL_FINDINGS.md` (superseded) and `docs/paper/README.md`
(the frozen campaign the paper reports).

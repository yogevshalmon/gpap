# Running a campaign on a rented Colab GPU

> **Reader:** people running a campaign on a rented Colab GPU — how the drivers work, and the platform rules that keep a run alive.

The drivers in this directory push a GPAP checkout to a Colab VM, run a campaign stage there, and
mirror the results back. The rules below are about the rented machine, not about GPAP, which is
why they live beside the drivers rather than in `docs/findings/`.

## The lifetime is the design constraint

**A Colab GPU VM is reclaimed after about an hour, whatever it is doing** (every session from
2026-08-03 on, with google-colab-cli 0.6.0; not yet measured on 0.8, which leaves liveness to the
Colab backend). So **results exist only once they are on the local disk**: checkpointing to the
VM's own disk buys nothing.

`colab_supervisor.py` runs the loop that survives it:

```
ensure session -> push cached binaries -> bootstrap -> push partials
              -> launch -> poll and PULL every cycle -> reseed a fresh VM
```

The remote command must resume where it stopped; a downloaded result replaces the local copy only
once it parses; built binaries are cached locally and pushed to each fresh VM with the mirrored
partials. The bootstrap script ends by creating `bootstrap.ok`, or `bootstrap.fail` holding the
reason. `colab_supervisor.py --help` lists the rest.

## The CLI

The supervisor needs google-colab-cli 0.8 or later (`colab ssh`) and refuses an older one;
`COLAB_BIN` picks the `colab` it runs. Authenticate with ADC and the four scopes `openid`,
`cloud-platform`, `userinfo.email` and `colaboratory` (it 403s without the last); check with
`colab --auth=adc whoami`. Stop the session when done: `colab stop -s <name>`.

## Four traps

* **Decide on marker files, not on log text.** `set -x` prints a script's
  `|| { echo BOOT_FAIL…; }` branch even on success, so a grep for the string reads as a failure
  that did not happen.
* **A retried error row replaces the old one, never appends** (appending gave 25 rows for 18
  instances). Retry only with `--retry-errors`, for rows an environment fault produced: a timeout
  at the current budget is a finished measurement.
* **Never wait on a grep against an append-only mirror**: it matches a previous run and fires
  early. And **one poller per session**.
* **Never edit a running bash script**: the shell reads it by byte offset, so the edit lands
  inside the process.

## Where the measurements went

The campaigns these drivers ran are recorded in `docs/findings/CROSS_GPU_FINDINGS.md` (what a
bigger GPU buys), `docs/findings/L4_EVAL_FINDINGS.md` (superseded), `docs/papers/tacas-27/README.md`
(the frozen campaign the paper reports) and `docs/findings/OPENCL_FINDINGS.md` (the OpenCL
backend against CUDA, `colab_opencl_l4.sh`).

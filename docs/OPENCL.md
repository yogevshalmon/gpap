# The OpenCL backend

> **Reader:** users who run the engine through OpenCL, and contributors who change a kernel —
> what the backend covers, how to get a runtime and select it, and how its kernels are kept equal
> to the Triton ones.

GPAP's engine reaches the GPU through one small interface, the GPU socket
(`cpp/src/gpu_api.hpp`), with one plug per GPU library:

| plug | kernels | use it for |
|---|---|---|
| `cuda` (the default) | Triton, compiled ahead of time (`cpp/kernels/bcp.py`, `helpers.py`) | everything: it is the reference and the fast path |
| `opencl` | OpenCL C, compiled on the device at first use (`cpp/kernels/bcp.cl`) | a host that can only load OpenCL source, or a device with no CUDA |

The OpenCL backend exists to run **entailed-clause mining and batched assumption checking**
where CUDA binaries cannot be loaded. It is a second implementation of part of the engine, held
to the first by tests; it is not a replacement. It follows the CUDA backend in what a caller
gets: the same answers, through the same API, with the differences below and their reasons.

**No NVIDIA driver needed.** libgpap does not link `libcuda`. The CUDA plug opens it (`dlopen`)
the first time the `cuda` backend is used, as the OpenCL plug opens the OpenCL library, so
libgpap builds and loads on a machine without the NVIDIA driver, and an OpenCL engine never
opens it. On such a machine only the `cuda` backend is missing. Under `GPAP_BACKEND=opencl` the
Python faces and `gpap mine` / `gpap info` need no CUDA either: the engine takes host arrays,
so their tensors live on the CPU, and a CUDA GPU in the same machine is left alone unless
`--device` (or a `device` argument) names it. `gpap info` then lists that GPU as not the
engine's; the engine's device is on its `GPU backend` line.

## What it covers

| | `cuda` | `opencl` |
|---|---|---|
| batched assumption checking (`AssumptionChecker`, `gpap_check_sets`) | yes | yes |
| probing and mining (`LiteralProbe`, `gpap_probe_*`, `gpap.mining`) | yes | yes |
| selections and readouts (`gpap_select_complete`, `gpap_var_bits`, `gpap_check_sound`, …) | yes | yes |
| the scalar propagate kernel, both packed splits | yes | yes |
| the WAVE propagate kernel | yes | yes |
| CLEAN mode | yes | yes |
| sampling (`gpap sample`, `gpap_sampler_*`, `gpap_engine_trajectory`) | yes | no: refused |
| device pointers (`gpap_engine_apply_dev`, `gpap_engine_trajectory_dev`) | yes | no: refused |

"Refused" means the call fails with `GPAP_ERR_KERNEL` or `GPAP_ERR_ARG` and a message that names
the backend. Nothing falls back silently to a wrong or different answer. Sampling is refused
before any work: by `gpap_sampler_create`, and by `gpap sample` before it preprocesses.

Kernel dispatch follows the one rule (`gpap_dispatch_wave`): WAVE from the clause threshold up.
Below it the rule asks how many warps the device keeps resident, which OpenCL cannot report, so
a small formula runs scalar at any batch. `GPAP_KERNEL` and `GPAP_CLEAN` pin the choice as on
CUDA.

## Getting an OpenCL runtime

libgpap needs no OpenCL at build time. At run time it opens the system's ICD loader
(`libOpenCL.so.1`), or the library `$GPAP_OPENCL_LIB` names.

- **Native Linux, NVIDIA GPU.** The driver ships OpenCL. Install the ICD loader
  (`ocl-icd-libopencl1` on Debian and Ubuntu) and check that `clinfo -l` lists the card.
- **WSL2, NVIDIA GPU.** The driver exposes CUDA but not OpenCL. PoCL's CUDA driver runs OpenCL
  kernels on the card through CUDA: `scripts/setup_opencl.sh cuda` installs it without root.
  Its compiler is not NVIDIA's, so use it for correctness and take rates on native Linux.
  The script also arranges the two things that driver needs and conda does not provide:
  NVIDIA's `libdevice` bitcode, taken from the triton wheel because GPAP installs no CUDA
  toolkit, and a loader directory that holds PoCL alone. A second driver with its own LLVM in
  the same process (Oclgrind is one) makes PoCL's compiler abort on every kernel.
- **Any machine, no GPU.** `scripts/setup_opencl.sh cpu` installs PoCL's CPU driver. It runs
  the kernels on the processor, which is how CI executes them.

After either script: `source <(scripts/setup_opencl.sh env)`. With both installed it prints the
GPU runtime; `env cpu` and `env cuda` choose one.

The Docker image does not run this backend yet ([DOCKER.md](DOCKER.md), What you need).

## Selecting the backend

```bash
GPAP_BACKEND=opencl gpap info          # which device libgpap would run on
GPAP_BACKEND=opencl gpap mine f.cnf    # anything that does not sample
```

`$GPAP_BACKEND` is read when an engine is created and fixed for its life, so one process can
hold engines on both backends; that is how the tests compare them. Any value other than `cuda`
or `opencl` is an error. Under `opencl` the engine's device ordinal indexes the OpenCL devices,
GPUs first, then the rest in the loader's order. `gpap_engine_backend` and
`gpap_engine_device_name` (Python: `Engine.backend`, `Engine.device_name`) say what an engine
runs on.

Two sizing rules differ from CUDA, because OpenCL cannot answer the questions CUDA can:

- **Free memory.** OpenCL has no free-memory query, so the budget starts from the device's
  total memory. On a card that something else uses — another process, or another engine or
  torch in the same one — cap it with `GPAP_VRAM_BUDGET_MIB` or pass an explicit batch.
- **One buffer's size.** A device limits a single allocation (`CL_DEVICE_MAX_MEM_ALLOC_SIZE`),
  so the batch is also bounded by the widest per-row buffer (`engine_row_limit` in
  `cpp/src/engine.cpp`). An explicit batch is not clamped; one that is too large fails at the
  allocation with the device's limit in the message. A raw-ABI caller sizing its own batch
  should take it from `gpap_check_chunk_size`, which applies this limit. On a big formula this
  bound, not the memory, can set the batch, and the smaller batch, not the kernel, is then why
  probing runs slower than on CUDA ([findings/OPENCL_FINDINGS.md](findings/OPENCL_FINDINGS.md)
  §12).

**On a CPU** (PoCL's CPU driver, which `scripts/setup_opencl.sh` installs) the backend gives the
same answers, and two settings matter. Pin `GPAP_KERNEL=scalar`: the automatic choice is made
for GPUs and takes WAVE on all but small formulas, the slower kernel on a CPU. Cap
`GPAP_VRAM_BUDGET_MIB`: the device's memory is the host's RAM, the budget would take most of
it, and a CPU gains little from a large batch
([findings/OPENCL_FINDINGS.md](findings/OPENCL_FINDINGS.md) §13).

The launch shapes are fixed in the plug (`kSpecs`, `kGroup` in `cpp/src/gpu_opencl.cpp`), from
measurements ([findings/OPENCL_FINDINGS.md](findings/OPENCL_FINDINGS.md) §8, §9).
`GPAP_KERNEL` and `GPAP_CLEAN` pin the kernel and the mode as on CUDA.


## How it is built

- **The socket.** `gpap::Gpu` (`cpp/src/gpu_api.hpp`) is every device operation the engine
  performs: allocate, copy, read back, and run a kernel named by its key over a work extent
  (rows, or rows × columns) with its arguments in the order of the Triton signature. The engine
  above it names no GPU library.
- **The plug.** `cpp/src/gpu_opencl.cpp` owns the list of kernels the backend ships, how an
  extent becomes a launch, and the parameter block. It opens the OpenCL library with `dlopen`.
  One context and one compiled program per device are shared by every engine in the process.
- **The kernels.** `cpp/kernels/bcp.cl` is plain OpenCL C 1.2. CMake embeds it in the library,
  so the source a library compiles is always the one it was built with. It ships as source
  because an OpenCL binary belongs to one device and one driver: the driver compiles it once per
  process, and every engine there shares the program. The Triton kernels ship compiled because a
  build targets one GPU architecture, and the CUDA plug refuses any other. Scalars travel in a
  parameter buffer (the layout is at the top of the file), because some hosts pass buffers
  only. Every launch is one-dimensional: an OpenCL built over CUDA allows 65,535 work groups in
  every dimension but the first, and a batch has more rows than that.
- **A row.** The scalar kernel gives a row one work item, so it needs no atomics. The WAVE
  kernel gives a row one work group of 32 work items. Triton's lanes exchange values through
  warp shuffles, which OpenCL C does not have: here the lanes share the trail's tail and the
  conflict flag in local memory and meet at two barriers per wave. The packed word and the
  bitmap word are updated by the same single atomics as in Triton. The WAVE kernels declare
  that group size (`reqd_work_group_size`), so a host that launches them itself cannot get it
  wrong silently: any other launch is refused.
- **A conflicted row under WAVE.** The lanes race for the order of the trail, so a conflicted
  row's counters can differ between two launches. The kernel contract leaves a conflicted row
  unspecified; the conflict flags, and every surviving row, are exact.

## Changing a kernel

`bcp.py` and `helpers.py` are the reference; `bcp.cl` follows. It follows in what a caller
gets and in the logic of each kernel, not line by line: where OpenCL C needs another
mechanism, or a faster one exists, the kernel uses it and the file's header says where and
why. An idea that changes what a kernel does (not only how fast) is built in the Triton
kernel first.

1. A change of **behaviour** in a Triton kernel that `bcp.cl` also implements must be made in
   both, in the same change. A change of **speed** need not be.
2. Keep the translation function by function: each OpenCL function names its Triton
   counterpart, and the Triton docstring is the contract.
3. A new OpenCL kernel is one entry in `kSpecs` (`gpu_opencl.cpp`) and one `gpap_<key>`
   function in `bcp.cl`; `test_the_kernel_list_matches_bcp_cl` keeps the two lists equal.
4. Run the tests on both backends (below).

## Testing

```bash
pytest tests/engine/test_opencl_backend.py     # the backend's own tests
GPAP_BACKEND=opencl pytest                     # the whole suite on the OpenCL backend
scripts/setup_opencl.sh oclgrind pytest tests/engine tests/mining -m "(engine or opencl) and not gpu and not slow"
```

The last line runs every engine and mining test that needs no CUDA on the Oclgrind simulator
(the mode sets `GPAP_BACKEND=opencl`), which reports out-of-bounds accesses and data races; the
script fails if it reports anything. The simulator is slow, so the selection leaves out the
tests that also drive the CUDA backend. It needs the `cuda` mode's install.

`tests/engine/test_opencl_backend.py` compares the OpenCL kernels with the pure-Python
reference (`opencl` marker: any OpenCL device, a CPU included) and with the CUDA backend row
for row (`gpu` and `opencl` together). Under `GPAP_BACKEND=opencl` the rest of the suite runs
on OpenCL too. An `engine` test needs only an engine on the selected backend, so it runs on any
OpenCL device; CI runs those on PoCL's CPU driver, on a runner with no NVIDIA driver. A `gpu`
test needs CUDA tensors, a `cuda_only` one tests what only the CUDA backend has, and a test that
samples is skipped when the backend refuses the trajectory kernel by name. Every other failure
is a real one, and an OpenCL install that cannot compile `bcp.cl` fails the OpenCL tests
instead of skipping them.

What was measured, and on which devices, is in
[findings/OPENCL_FINDINGS.md](findings/OPENCL_FINDINGS.md).

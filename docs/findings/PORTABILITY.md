# Running the GPAP engine on non-NVIDIA GPUs

> **Status:** analysis, not a measurement — estimates except where a line says MEASURED.
> **Measured:** 2026-07-29, after the libgpap port. Hardware-independent by subject: it is about
> other vendors' GPUs, and no GPAP rate appears in it. The gfx cross-compile result predates
> CLEAN mode and `tl.gather` — re-run it before quoting.
> **Rule lives in:** nowhere — nothing here is implemented. `cpp/tools/make_libcuda_stub.py` is
> the executable inventory of the CUDA surface this file sizes.
> **Record only:** the port estimates and the recommendation.

Short answer: AMD and Intel are a small port the current design already anticipates. Apple is
not a port but a reimplementation of the kernel layer, because Triton, which makes the other two
cheap, has no Metal backend.

## 1. What is coupled to CUDA

MEASURED by grepping the core for `cu*` / `CU*` / `cuda.h`: 75 of 1,273 lines (5.9%) touch CUDA
— `formula.cpp` 0 of 288, `engine.cpp` 25, `readout.cpp` 11, `calibrate.cpp` 15, `runtime.cpp`
14, `gpap_internal.hpp` 10 — through 18 distinct driver entry points:

```
cuInit  cuDeviceGet  cuDeviceGetAttribute  cuDevicePrimaryCtxRetain  cuDevicePrimaryCtxRelease
cuCtxSetCurrent  cuMemAlloc  cuMemFree  cuMemGetInfo  cuMemsetD8
cuMemcpyHtoD  cuMemcpyDtoH  cuMemcpyDtoD
cuModuleLoadData  cuModuleGetFunction  cuModuleUnload  cuLaunchKernel  cuGetErrorString
```

Init, query, allocate, copy, load a binary, launch. Every one has a direct counterpart in HIP,
Level Zero, Metal and WebGPU. The formula builder, the K calibrator's arithmetic, selections and
all orchestration are plain C++. `cpp/tools/make_libcuda_stub.py` is the executable inventory of
this surface.

## 2. What is coupled to NVIDIA in the kernels: nothing obvious

MEASURED: `cpp/kernels/bcp.py` and `helpers.py` contain zero occurrences of `shfl`, `ballot`,
`__syncwarp`, `inline_asm` or `tl.extern`. The Triton surface is `tl.load`, `tl.store`,
`tl.constexpr`, `tl.where`, `tl.program_id`, `tl.sum`, `tl.arange`, `tl.atomic_or`,
`tl.atomic_add`, `tl.cumsum`, `tl.minimum/min/max`, `tl.zeros` and, since 2026-08-27,
`tl.gather` (the WAVE lane schedule; a warp shuffle on NVIDIA).

The WAVE kernel's correctness does not rest on warp-synchronous execution. Its cross-lane races
are resolved by the return values of `tl.atomic_add` (old count 1: this lane emptied the clause;
old count 2: this lane made it unit and the sum field is the forced literal) and `tl.atomic_or`
(old bit `l`: duplicate; old bit `l^1`: contradiction). Device-scope atomics returning the
previous value exist in CUDA, HIP, Metal and WGSL. `W = 32` is a tile width, not a hardware warp
requirement. The algorithm is portable; the toolchain is what is not.

## 3. AMD (ROCm / HIP): small, and the kernel half is done

MEASURED 2026-07-29 on the NVIDIA dev box with no AMD hardware: pointing `build_kernels.py` at
HIP compiled all 29 kernel specializations, unmodified, for `gfx90a` (MI200, wavefront 64),
`gfx942` (MI300, 64) and `gfx1100` (RDNA3, 32) — 29 compiled, 0 failed on each, including the
wave trajectory kernel. Triton 3.6 ships the `amd` backend in the wheel we build with
(`binary_ext`: `cubin` → `hsaco`), and the AOT builder already routes through
`triton.compiler.make_backend(target)`. That result predates CLEAN mode (33 specializations
2026-08-21, 27 since duplicates were dropped) and the `tl.gather` schedule (Triton ≥ 3.2); re-run
the cross-compile before quoting it for the current kernels.

Remaining work: `build_kernels.py` parses `arch` as an int (`sm_89`) where AMD's is a string
(`gfx942`) and hard-codes `threads = 32 * num_warps` (must be `warp_size * num_warps`; CDNA 64,
RDNA 32); a `gpu_api.h` shim over the 18 calls (HIP is close to 1:1); and re-tuning, not
rewriting: the wave/scalar crossover (`M >= 2048`), `W` and `compute_k_safe`'s constants were
all measured on NVIDIA. The no-change gate (`tests/engine/test_core.py` against `reference_bcp.py`)
transfers unchanged and is what proves correctness on the new device. Estimate 1–2 weeks. Main
risk: whether Triton's AMD backend lowers the atomic-with-return pattern as efficiently on CDNA —
a benchmark, not a rewrite.

## 4. Intel (Level Zero / SPIR-V): small-to-medium

Same story via `intel-xpu-backend-for-triton` (`binary_ext` = `spv`); Level Zero covers the same
18 operations, more verbosely, and the backend is younger. Estimate 2–4 weeks.

## 5. Apple (M-series): a reimplementation

There is no Triton backend for Metal and no maintained path from Triton/LLVM IR to AIR, which
removes the one assumption the architecture rests on. The C++ core, the two-tier API and the
Python facade carry over; the kernel layer does not. Routes, none cheap: hand-written Metal
Shading Language (device atomics with return, `simd_sum` / `simd_prefix_exclusive_sum`; 2 core
kernels + 8 helpers, 4–8 weeks for a validated scalar path, and then two kernel sources to keep
in sync, exactly what libgpap eliminated); WGSL/WebGPU (the scalar kernel maps cleanly, the wave
kernel needs subgroup extensions or a threadgroup fallback, and native Metal will beat it);
waiting for a Triton Metal backend (community attempts, none production). There is also a model
problem: Apple GPUs are unified-memory, so `compute_k_safe`, the 800 MB WDDM reserve, the
free-VRAM query and the spill-to-host story (docs/BENCHMARKING.md §2) are discrete-GPU
assumptions; that calibrator needs rethinking, not porting.

## 6. Recommendation

- For reach, do AMD first: cheap, anticipated by the design, and the other major HPC vendor.
- Do not promise Apple support. The honest public statement: the engine is CUDA-only today; the
  host layer is 18 driver calls and the kernels use no NVIDIA-specific intrinsics, so a HIP/ROCm
  port is small and an Apple port is a Metal reimplementation of the kernel layer.
- Cheap insurance regardless: factor the 75 CUDA-touching lines behind a `gpu_api.h` (types
  `gpu_device/gpu_ctx/gpu_module/gpu_fn/gpu_ptr` + the 18 calls) so the backend is one file.

# Running the GPAP engine on non-NVIDIA GPUs

*Analysis 2026-07-29, after the libgpap port. Measured facts are marked MEASURED; the rest is
engineering estimate and should be treated as such.*

Short answer: **AMD and Intel are a small port that the current design already anticipates.
Apple is not a port — it is a reimplementation of the kernel layer**, because the thing that
makes the other two cheap (Triton) has no Metal backend.

## 1. What is actually coupled to CUDA

MEASURED, by grepping the core for `cu*` / `CU*` / `cuda.h`:

| file | lines | lines touching CUDA |
|---|---|---|
| `formula.cpp` (DIMACS, CSR, packed init) | 288 | **0** |
| `engine.cpp` | 406 | 25 |
| `readout.cpp` | 254 | 11 |
| `calibrate.cpp` | 68 | 15 |
| `runtime.cpp` | 103 | 14 |
| `gpap_internal.hpp` | 154 | 10 |
| **total** | **1273** | **75 (5.9%)** |

And the *distinct* driver entry points used are **18**:

```
cuInit  cuDeviceGet  cuDeviceGetAttribute  cuDevicePrimaryCtxRetain  cuDevicePrimaryCtxRelease
cuCtxSetCurrent  cuMemAlloc  cuMemFree  cuMemGetInfo  cuMemsetD8
cuMemcpyHtoD  cuMemcpyDtoH  cuMemcpyDtoD
cuModuleLoadData  cuModuleGetFunction  cuModuleUnload  cuLaunchKernel  cuGetErrorString
```

That is the entire host-side surface: init, query, allocate, copy, load a binary, launch. Every
one has a direct counterpart in HIP, Level Zero, Metal and WebGPU. The formula builder, the K
calibrator's arithmetic, selections, slicing and all the orchestration are plain C++.

## 2. What is actually coupled to NVIDIA in the *kernels* — nothing obvious

MEASURED: `gpap/engine/kernels.py` and `cpp/kernels/helpers.py` contain **zero** occurrences of
`shfl`, `ballot`, `__syncwarp`, `inline_asm` or `tl.extern`. The Triton surface used is small
and entirely standard:

```
tl.load  tl.store  tl.constexpr  tl.where  tl.program_id  tl.sum  tl.arange
tl.atomic_or  tl.atomic_add  tl.cumsum  tl.minimum/min/max  tl.zeros
```

This matters more than it looks. **The WAVE kernel's correctness does not rest on
warp-synchronous execution.** Its cross-lane races are resolved by the *return values* of
`tl.atomic_add` (old count 1 ⇒ this lane emptied the clause; old count 2 ⇒ this lane made it
unit and the sum field is the forced literal) and `tl.atomic_or` (old bit *l* ⇒ duplicate; old
bit *l^1* ⇒ contradiction). Device-scope atomics that return the previous value exist in CUDA,
HIP, Metal (`atomic_fetch_add_explicit`, `atomic_fetch_or_explicit`) and WGSL (`atomicAdd`,
`atomicOr`). `W = 32` is a **tile width**, not a hardware warp requirement.

So the algorithm is portable in principle. What is not portable is the *toolchain*.

## 3. AMD (ROCm / HIP) — small, and the kernel half is already DONE

MEASURED, 2026-07-29, on the NVIDIA dev box with **no AMD hardware present**: pointing
`build_kernels.py`'s target at HIP compiles **all 29 kernel specializations, unmodified**, for
three AMD generations —

| target | wavefront | result |
|---|---|---|
| `gfx90a` (MI200) | 64 | **29 compiled, 0 failed** |
| `gfx942` (MI300) | 64 | **29 compiled, 0 failed** |
| `gfx1100` (RDNA3) | 32 | **29 compiled, 0 failed** |

including the full wave trajectory kernel (`trajectory.s16.w1.v1.f0`, 33 KB of hsaco, 128 B
dynamic shared). Triton 3.6 ships the `amd` backend *in the same wheel we already build with*
(`binary_ext`: `cubin` → `hsaco`), and the AOT builder already routes through
`triton.compiler.make_backend(target)`.

So the kernel-compilation half of an AMD port is not work — it already functions. What remains
is the host layer and validation.

Work:
- **Kernels: none** (demonstrated above). Same Triton source, different `target`.
- **`build_kernels.py`**: two lines of assumption — `arch` is parsed as an `int` (`sm_89`) while
  AMD's is a string (`gfx942`), and `threads = 32 * num_warps` hard-codes the NVIDIA warp size
  (must become `warp_size * num_warps`; CDNA is 64, RDNA 32).
- **Host**: a `gpu_api.h` shim over those 18 calls. HIP is close to 1:1
  (`hipModuleLoadData`, `hipModuleLaunchKernel`, `hipMalloc`, `hipMemcpyDtoH`, …), so this is
  mostly mechanical.
- **Re-tuning, not re-writing**: the wave/scalar crossover (30× resident warps), `W`, and
  `compute_k_safe`'s constants were all measured on a 4080 and must be re-measured. The
  no-change gate (`tests/test_core.py` against `reference_bcp.py`) transfers
  unchanged and is what proves correctness on the new device.

Estimate: **1–2 weeks** to a validated build, and essentially none of it is kernel work — it is
the HIP shim, the two builder assumptions, and re-tuning. Main risk: whether Triton's AMD backend
lowers the atomic-with-return pattern as *efficiently* on CDNA. That is a benchmark, not a
rewrite, and it cannot be answered without the hardware.

## 4. Intel (Level Zero / SPIR-V) — small-to-medium

Same story via `intel-xpu-backend-for-triton` (`binary_ext` = `spv`). The host API is Level
Zero, which is more verbose than HIP but covers the same 18 operations. The backend is younger
than AMD's, so budget more validation. Estimate: **2–4 weeks**.

## 5. Apple (M-series) — a reimplementation, not a port

**There is no Triton backend for Metal**, and no maintained path from Triton/LLVM IR to Apple's
AIR. That removes the single assumption the whole architecture rests on — "the kernels are
written once, in Triton". Everything else (the C++ core, the two-tier API, the Python facade)
carries over untouched; the kernel layer does not.

Three routes, none cheap:

1. **Hand-write Metal Shading Language.** MSL has everything the algorithm needs: device
   atomics with return, `simd_sum` / `simd_prefix_exclusive_sum` for the tile reductions,
   threadgroup memory. The work is 2 core kernels (the trajectory kernel is the hard one — the
   wave tile logic, segment scheduling and the fused decide/propagate loop) plus 8 helpers.
   Estimate: **4–8 weeks** for a validated scalar path, more for wave and tuning. The real cost
   is not the first version — it is that GPAP would then have **two kernel sources to keep in
   sync**, which is precisely what the libgpap port was designed to eliminate.
2. **WGSL / WebGPU** as one portable compute layer (runs on Metal, Vulkan, D3D12). WGSL has
   `atomicAdd`/`atomicOr` returning the old value on storage buffers, so the **scalar** kernel —
   one thread per row, global atomics only, no cross-lane cooperation — maps cleanly. The wave
   kernel needs subgroup ops, which are an extension, so it would need a threadgroup-memory
   fallback. Attractive as "runs everywhere", but it is still a second kernel source and native
   Metal will beat it.
3. **Wait for a Triton Metal backend.** Community attempts exist; none is production. Not a plan.

There is also a **model** problem, not just a compiler problem: Apple GPUs are unified-memory.
`compute_k_safe`, the 800 MB WDDM reserve, the free-VRAM query and the whole "this box spills to
host RAM" story (docs/BENCHMARKING.md §2) are discrete-GPU assumptions. On Apple, K would be
bounded by system memory pressure and there is no spill cliff to avoid — that calibrator needs
rethinking, not porting.

## 6. Recommendation

- **If the goal is reach**, do AMD first. It is cheap, the design already accommodates it, and
  it covers the other major HPC vendor — which is what a reviewer of a GPU-SAT paper will ask
  about.
- **Do not promise Apple support.** The honest public statement is: *the engine is CUDA-only
  today; the host layer is 18 driver calls and the kernels use no NVIDIA-specific intrinsics, so
  a HIP/ROCm port is small and an Apple port is a Metal reimplementation of the kernel layer.*
- **Cheap insurance, worth doing regardless of any port**: factor the 75 CUDA-touching lines
  behind a `gpu_api.h` (types `gpu_device/gpu_ctx/gpu_module/gpu_fn/gpu_ptr` + those 18 calls)
  so the backend is one file rather than a grep across five. That is a day of work, makes the
  claim above verifiable, and costs nothing at runtime.

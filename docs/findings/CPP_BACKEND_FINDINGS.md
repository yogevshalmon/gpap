# A C++ core for the GPAP engine — what was measured, and what it cost

Investigated 2026-07-28 (RTX 4080 laptop, WSL2, torch 2.11 + triton 3.6). Question: should GPAP
have a C++ backend so other projects can link against it, and what does that do to performance?
Verdict: linkability is the entire benefit, the port is a performance tie, and it was cheap
because the kernels did not have to be rewritten.

## 1. Python is not on the critical path

Per-chunk CUDA-event profile of the fused sampling loop at production K on b06/b03/b10/b04/
b12-c25: the trajectory kernel is 78.8–85.9% of chunk wall, the `reset()` broadcast 5.9–10.8%,
order build + readout 2.4–4.6%, and a 5.8–5.9% host gap that a bare `torch.matmul` shows too
(WSL2 submit latency). Python's own cost is the Triton launch, 22–41 µs per chunk, ≤ 0.07%. A
C++ host loop cannot recover it. End to end, `gpap sample b12-c25.cnf -n 20000` took 9.28 s, of
which timed sampling was 1.57 s and mining 5.07 s: the e2e number is host setup and the
budget-capped miner, not the loop.

## 2. The kernels do not need rewriting

Triton AOT-compiles the kernel sources (`cpp/kernels/`) unmodified into a cubin plus a plain-C
loader that needs only the CUDA driver API. Proven before any porting: a 36 KB binary linking
`libcuda.so.1` alone (no Python, torch or nvcc) reproduced the engine's model counts on real
instances (b03 2337, b06 19087, b12 5635). The AOT and JIT cubins are the same binary: 55,008
bytes, REG:80, shared 128, `nvdisasm` diff clean over 1008 SASS lines, so the AOT route carries
no behavioural or performance risk by construction. No CUDA toolkit is needed: Triton bundles
`ptxas` and a driver `cuda.h`; build deps are cmake + a C++17 compiler + triton (build time
only); the runtime dependency is `libcuda`.

Editing a kernel source at all, even a comment, changes cubin bytes (Triton embeds line tables);
compare executable sections, not bytes. The AOT-vs-JIT identity gate compares one source through
two paths and is unaffected.

## 3. The port is performance-neutral

Interleaved same-process A/B, C++ core vs the torch/Triton-JIT engine, same formula, K, order and
seed, five instances spanning scalar, wave and wide (K 6,017–92,893): per-chunk ratios
0.99–1.03×, **geomean 1.013×**, identical model sets throughout. Identical SASS in, identical
performance out.

## 4. Retracted: "owning the memory is worth 1.2-1.35×"

Before the port the same cubin ran at 43–44 ms from a standalone C process and 54–61 ms from the
Python/torch process on b03-c25, attributed to torch's caching allocator after ruling out codegen
(bit-identical SASS), the launch path (0.97–1.05×), `expandable_segments`, clocks, footprint, arm
order and Python itself (python + ctypes + `cuMemAlloc`: 44.0 ms). Then the real core, which
allocates exactly that way, landed at 53.9 ms on the identical input. The variable is where the
block lands, not who allocates it: allocating the 2 GB `(K, M)` buffer first gave the fast
placement consistently in a standalone process and intermittently in a Python one; allocating it
after a few small buffers, as the core does, gave 44.3 ms once and 55.7 ms the next time. A ~25%
device-memory placement effect exists, but it is not an allocator-ownership win and it did not
transfer to the port. The C++ core ties the Python engine; do not quote the earlier number.

Tracked follow-up: give the core a single state arena — one allocation covering packed + abits +
trail + flags + counters, sub-allocated at a large alignment — and A/B it across instances and K.
If placement is really the lever, that is a ~1.2× the Python engine could never reach and the
first genuine performance argument for C++.

## 5. Not done: hand-written CUDA kernels

The one thing Triton cannot express is a thread-per-row mapping (32 independent scattered streams
per warp), the natural fix for the memory-level-parallelism starvation the WAVE kernel works
around. The other half of that bet, replacing the O(W²) one-hot segment schedule with shuffles,
was done in Triton itself with `tl.gather` on 2026-08-27
([WAVE_SHUFFLE_FINDINGS.md](WAVE_SHUFFLE_FINDINGS.md)). A rewrite would forfeit the free ROCm
path and force a full re-validation, on a kernel already near the scattered-DRAM ceiling on
saturated instances. Not justified.

## 5b. Two API tiers, and why

The first `gpap.hpp` was a 1:1 RAII wrapper over the C ABI, and the sampling example written
against it produced zero models: the independent support must be decided first, with `n_support`
marking the phase boundary, and nothing in the API said so. That knowledge — support-first order,
VRAM-sized K, the wave/scalar threshold, selection slots, pair masks, 2N literal indexing — is
engine knowledge, so tier 1 (`gpap::Sampler`, `gpap::AssumptionChecker`, `gpap::Prober`) carries
it: ~250 lines of header over tier 2 (`Formula`/`Engine`, RAII + exceptions over the ABI), no
state the engine does not already own, and the sampling example dropped from ~40 lines of setup
to three.

Defaults that are right on easy instances and wrong on hard ones came out of writing it. Stopping
after two empty chunks is wrong at a 0.01% success rate (give up only when nothing at all has
been found after a grace period). `CMAKE_CXX_STANDARD 17` does not travel through an exported
target; `target_compile_features(... PUBLIC cxx_std_17)` does. b12 unrolled 50 cycles returns 0
solutions at B=8 or B=1 without lookahead while the soft polarity finds 3.1% success. Since
2026-09-01 the whole sampling loop is in the core (`cpp/src/sample.cpp`, `gpap_sampler_*`), both
faces marshal over it, and tier 1 races the same portfolio the pipeline does
(`SampleOptions::portfolio`). The soft retry is `retry_with_lookahead`, default off: soft is
Barbarik-rejected on 64/65 circuit instances and Python had made it opt-in, while the earlier
`auto_retry` default silently enabled it. Renamed rather than reversed, so an out-of-tree caller
gets a compile error instead of a silent flip; `Solutions::lookahead_used()` still reports it.

Two ABI rules follow. An ABI break must move the soname and an additive change must not: libgpap
is 0.3.0 / SOVERSION 1 (the 0.2.0 renames moved SOVERSION 0 → 1; 0.3.0 only added
`gpap_sampler_*`). And `{0}` must equal NULL on every options struct: every field's zero is its
default, and a field whose useful default is "on" is named for the opt-out (`same_var`;
`GPAP_MODE_AUTO` is 0).

## 5c. The clean-machine build, verified in an environment that was not clean

CI `build-core` failed twice on the first PR for reasons a local check with torch and a GPU could
not catch: the kernel module imported torch at module scope (a build box with triton but no
torch could not build; now a lazy import), and `compile_one` reached the kernels through
`fn.create_binder()`, whose driver reports itself active only with a CUDA-capable torch, so
`--arch NN` cross-compilation could not work on a GPU-less box (now constructs
`triton.compiler.ASTSource` directly; byte-identical output with the source held constant).
Reproducing an environment means removing what it does not have (a `sys.meta_path` blocker for
torch, a stub `libcuda`), not merely not using it.

## 6. Method

Every A/B interleaved in one process or alternated across processes, arms also run reversed:
this laptop's clocks drift enough that a fixed-order comparison is not evidence. Model counts
compared exactly. The durable artifacts are `cpp/` and `tests/test_core.py`, which re-proves the
core's contract against the pure-Python oracles; the torch engine was deleted 2026-08-08.

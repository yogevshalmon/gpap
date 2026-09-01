# A C++ core for the GPAP engine — what was measured, and what it cost

Investigated 2026-07-28 (RTX 4080 laptop, WSL2, torch 2.11 + triton 3.6). The question was
whether GPAP should have a C++ backend so other projects can link against it, and what that
would do to performance. Short version: **linkability is the entire benefit, and it is much
cheaper than expected because the kernels do not have to be rewritten.** The performance
argument I first believed turned out to be a measurement artifact of a different kind — see §4.

---

## 1. Python is not on the critical path

Per-chunk profile of the fused sampling loop (CUDA events around each stage, production K):

| | b06-c25 | b03-c25 | b10-c25 | b04-c25 | b12-c25 |
|---|---|---|---|---|---|
| chunk wall | 166 ms | 52 ms | 74 ms | 133 ms | 140 ms |
| **fused trajectory kernel** | **85.7%** | **78.8%** | **83.4%** | **85.2%** | **85.9%** |
| `reset()` broadcast copy | 6.1% | 10.8% | 7.4% | 6.3% | 5.9% |
| order build + readout | 2.4% | 4.6% | 3.3% | 2.7% | 2.4% |
| host gap | 5.8% | 5.9% | 5.9% | 5.8% | 5.8% |

The "host gap" is **not** Python: a single `torch.matmul` with no Python in between shows the
same 7-10% wall-vs-event gap on this box (WSL2 submit/completion latency). Python's actual cost
is the Triton launch — **22-41 µs per chunk**, i.e. ≤0.07%. A C++ host loop cannot recover it.

End-to-end is a different story: `gpap sample b12-c25.cnf -n 20000` spends 9.28 s as
parse 0.10 · construct 1.08 · warmup 0.31 · **mine 5.07** · rebuild 0.25 · recheck 0.90 ·
**timed sampling 1.57**. Steady-state throughput is ~100% GPU; the e2e number is dominated by
host-side setup and the (budget-capped) miner.

## 2. The kernels do not need rewriting

Triton AOT-compiles the kernel sources (`cpp/kernels/`, then at `gpap/engine/kernels.py`)
**unmodified** into a cubin plus a plain-C loader
that needs only the CUDA driver API. Proven end to end before any porting work started: a 36 KB
binary, linking `libcuda.so.1` alone, no Python/torch/nvcc, reproducing the engine's results.
On real instances with the same K/order/seed it produced **identical model counts**
(b03 2337, b06 19087, b12 5635).

The AOT and JIT cubins are the **same binary**: 55,008 bytes, REG:80, shared 128, and
`nvdisasm` output diffs clean (1008 SASS lines). So the AOT route carries zero behavioural or
performance risk by construction — which is what made option C tractable at all.

Toolchain consequence, worth stating plainly: **no CUDA toolkit is needed**. Triton bundles
`ptxas` and a driver `cuda.h`; the driver supplies `libcuda`. Build deps are cmake + a C++17
compiler + triton (build time only).

## 3. The port is performance-neutral (as it should be)

Interleaved same-process A/B, C++ core vs the torch/Triton-JIT engine, same formula, same K,
same torch-built decision order (passed by pointer, zero copy), same seed:

| | K | torch engine | C++ core | ratio |
|---|---|---|---|---|
| b06-c25 (scalar) | 92,893 | 99.9 ms | 98.8 ms | 1.01× |
| b03-c25 (wave) | 34,775 | 62.7 ms | 63.1 ms | 0.99× |
| b10-c25 (wave) | 16,384 | 62.3 ms | 61.8 ms | 1.01× |
| b04-c25 (wave) | 8,495 | 99.8 ms | 96.5 ms | 1.03× |
| b12-c25 (wide) | 6,017 | 122.1 ms | 120.1 ms | 1.02× |

**geomean 1.013×**, identical model sets throughout. Identical SASS in, identical performance
out — the expected result, and the bar the migration had to clear.

## 4. RETRACTED: "owning the memory is worth 1.2-1.35×"

Before the port I measured, repeatedly and with the arms interleaved, that the *same* cubin ran
at **43-44 ms** from a standalone C process and **54-61 ms** from the Python/torch process on
b03-c25. I attributed it to torch's caching allocator, having ruled out:

- **codegen** — bit-identical SASS;
- **the launch path** — AOT-cubin-via-ctypes vs the JIT launcher, in one process: 0.97-1.05×;
- **`expandable_segments`** — True and False are both on the slow side (54.5/58.4, 60.3/61.1);
- **clocks/power** — 2 GB of device-side traffic immediately before the timed launch: no change;
- **footprint** — +1 GB / +3 GB of touched padding: no change (45.1 / 44.1 ms);
- **measurement order** — reversing the arms preserves the gap;
- **Python itself** — python + ctypes + `cuMemAlloc`, torch never imported: 44.0 ms.

The apparent smoking gun was that inside torch's own primary context, swapping torch tensors for
`cuMemAlloc` buffers gave 44.8 / 44.4 ms against 53.8 / 56.8 ms.

**Then the real core — which allocates exactly that way — landed at 53.9 ms on the identical
input** (2337 models, so identical work). The distinguishing variable is not *who allocates* but
*where the block lands*: allocating the 2 GB `(K, M)` buffer FIRST gave the fast placement
consistently in a standalone process and intermittently in a Python one (43.1 / 43.8 ms), while
allocating it after a handful of small buffers — what the core does — gave 44.3 ms once and
55.7 ms the next time.

So: **a ~25% device-memory placement effect exists and is worth chasing, but it is not an
allocator-ownership win and it did not transfer to the port.** The honest statement is that the
C++ core ties the Python engine. Anyone quoting the earlier 1.2-1.35× number should stop.

Tracked follow-up (docs/plans/CPP_CORE_PLAN.md §9): give the core a single **state arena** — one
allocation covering packed + abits + trail + flags + counters, sub-allocated with a large
alignment — and A/B it across instances and K. If placement is really the lever, that is a ~1.2×
the Python engine could never have reached, and the first genuine performance argument for C++.

## 5. What was NOT done, and why

**Hand-written CUDA C++ kernels.** The one thing Triton cannot express is a *thread-per-row*
mapping (32 independent scattered streams per warp instead of one), which is the natural fix for
the MLP starvation the WAVE kernel exists to work around; a CUDA wave kernel could also replace
the O(W²) one-hot segment scheduling with shuffles — DONE 2026-08-27 in Triton itself
(`tl.gather`), so that half of the bet is settled without leaving the toolchain; see
docs/findings/WAVE_SHUFFLE_FINDINGS.md. The thread-per-row bet remains, on
a memory-bound kernel that already sits near the scattered-DRAM ceiling on the saturated
instances. Rewriting would also forfeit the free ROCm path Triton gives us and force a full
re-validation. Not justified by current evidence.

## 5b. The C++ API needed a second tier, and finding that out was itself the evidence

The first `gpap.hpp` was a faithful 1:1 RAII wrapper over the C ABI — and writing the sampling
example against it produced **zero models**. The cause was the decision order: the independent
support has to be decided first, with `n_support` marking the phase boundary, or every decision
is spent on a variable propagation would have implied. Nothing in the API said so; the engine
accepted the wrong order happily and returned nothing.

That is the argument for a tier 1 (`gpap::Sampler`, `gpap::AssumptionChecker`): the knowledge
needed to use the engine *correctly* — support-first ordering, VRAM-sized K, the wave/scalar
threshold, selection slots, pair masks, the internal 2N literal indexing — is engine knowledge,
not application knowledge, and every consumer would otherwise have to re-derive it (and could
re-derive it wrong, silently). Tier 1 is ~250 lines of header over tier 2, adds no state the
engine does not already own, and reduced the sampling example from ~40 lines of setup to three.

Two bugs came out of writing it, both in the same family as the original one — defaults that are
correct on easy instances and wrong on hard ones:

- Stopping after two consecutive empty chunks looks reasonable and is wrong: at a 0.01% success
  rate most chunks *are* empty. The Python sampler's rule (give up only when nothing at all has
  been found, after a grace period) is the one that survives b11/b13-class instances.
- `CMAKE_CXX_STANDARD 17` does not travel through an exported CMake target, so a consumer on
  C++11 would fail to compile the header it was handed. `target_compile_features(... PUBLIC
  cxx_std_17)` does.

A third came out of measuring it. With B=8 and no lookahead — a perfectly reasonable default —
**b12 unrolled 50 cycles returns 0 solutions**, and so does B=1; only the in-kernel risk polarity
finds anything there (3.1% success, ~1100 models/s). A C++ caller would have read that 0 as "this
formula is hard" when the same library solves it. The portfolio deliberately stays in Python, so
tier 1 got the bounded version: **one** retry with the polarity flipped before giving up
(`SampleOptions::auto_retry`, on by default, reported through `Solutions::retried()`). Measured:
0 → 11,141 solutions at 1111/s on the default settings.

> ⚠️ **Superseded 2026-09-01.** The portfolio no longer "stays in Python": the whole sampling
> loop moved into the core (`cpp/src/sample.cpp`, `gpap_sampler_*`) and BOTH faces are
> marshalling over it, so tier 1 races the same portfolio the pipeline does
> (`SampleOptions::portfolio` / `gpap::default_portfolio()`). And `auto_retry` is now
> `retry_with_lookahead`, **default OFF**: the flip it performs turns the soft polarity on —
> Barbarik-rejected on 64/65 circuit instances — which Python had made opt-in (`--allow-soft`
> + a printed warning) while this default silently enabled it. The rename-not-reverse is the
> options-struct ground rule (a compile error instead of a silent flip). The b12 rescue is one
> field away, and `Solutions::lookahead_used()` still reports it. This is the same lesson as the
regime-dependence finding for `soft` — a lever that is net-negative in one regime must stay
reachable, not be discarded.

## 5c. "I verified the clean-machine build" — in an environment that was not clean

The CI `build-core` job failed on the first PR, twice, for reasons the local verification could
not have caught, and both are worth keeping:

1. `gpap/engine/kernels.py` imported **torch** at module scope. `build_kernels.py` loads that
   file to AOT-compile the kernels, so a build box with triton but no torch could not build the
   library at all — for a runtime GPU probe (`kernel_available`) that the AOT path never calls.
   Fixed by importing torch lazily inside that function.
2. Under that, a second failure: `compile_one` reached the kernels through
   `fn.create_binder()`, which calls `driver.active.get_current_target()`. Triton's NVIDIA
   driver reports itself ACTIVE only when `import torch and torch.cuda.is_available()` — i.e.
   only on a machine with a real GPU. So `--arch NN` cross-compilation, whose entire purpose is
   building without a GPU, could not work. Fixed by constructing `triton.compiler.ASTSource`
   directly; verified a **pure no-op** for codegen (byte-identical generated file when the
   kernel source is held constant).

The methodology lesson: the local "clean machine" reproduction used the repo virtualenv, which
has torch, and ran on a box with a GPU — so it validated neither claim it was written to test.
Reproducing an environment means removing what it does not have (`sys.meta_path` blocker for
torch, stub driver for libcuda), not merely not using it.

Related gotcha, worth knowing before comparing generated artifacts: **editing kernels.py at all —
even a comment — changes the cubin bytes**, because Triton embeds source line tables. The
AOT-vs-JIT bit-identity gate compares the same source through two paths, which is unaffected.

## 6. Method notes

- Every A/B is interleaved within one process or alternated back-to-back across processes, and
  the arms were also run in reversed order — this laptop's clocks drift enough that a
  fixed-order comparison is not evidence (INSIGHTS: the power-state gotcha).
- Model counts are compared exactly, never approximately: the two engines run the same kernels
  over the same state, so anything other than equality is a bug.
- The measurement scripts and the standalone-C harness live in the session scratchpad, which is
  wiped between sessions; the durable artifacts are `cpp/` and
  `tests/test_core.py`, which re-proves the core's contract against the pure-Python oracles on
  every run. (The torch engine itself was deleted 2026-08-08 — the A/B had served its purpose.)

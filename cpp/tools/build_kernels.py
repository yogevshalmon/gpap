"""Ahead-of-time compile every GPAP kernel specialization into the C core.

The kernels stay written in Triton — `cpp/kernels/bcp.py` is the single source of truth for
the two BCP kernels, `cpp/kernels/helpers.py` for the support kernels. This script
compiles them for a fixed set of `constexpr` specializations and emits

    cpp/src/generated/gpap_kernels.c   (cubin blobs + the lookup table)
    cpp/src/generated/gpap_kernels.h   (the descriptor type + externs)

so the shipped library embeds the binaries and needs nothing but `libcuda.so.1` at runtime.

We call `triton.compile()` directly rather than `python -m triton.tools.compile`: that CLI
bakes the launch grid into its wrapper, ignores dynamic shared memory in the generated launch
call, and mangles symbol names with a signature hash. We only want the cubin plus its launch
metadata (entry name, dynamic shared bytes, threads per block) and we launch it ourselves.

Verified: the cubin produced here is BIT-IDENTICAL (SASS) to what the Triton JIT produces for
the same source and constants, so this changes no sampling behaviour whatsoever.

Usage:
    python cpp/tools/build_kernels.py [--arch 89] [--out cpp/src/generated]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import triton                                                    # noqa: E402
import triton.backends                                           # noqa: E402

# ---------------------------------------------------------------- specialization matrix

# Field splits the engine supports (see gpap.engine.bcp_engine.PACKED_SPLITS).
SHIFTS = (16, 8)
# Wavefront lane width: 32 only, and W must be a power of two (tl.arange enforces it). BOTH
# ways of raising it are measured out: over W/32 REAL warps (num_warps>1) it was 2-5x WORSE —
# cross-warp reduction sync per step, docs/findings/INSIGHTS.md — and since every BCP spec below
# pins num_warps=1, raising W here instead gives W/32 elements per lane (ILP), which measured
# 0.918x at W=64 (docs/findings/WAVE_SHUFFLE_FINDINGS.md).
WAVE_W = 32

_PTR_I32, _PTR_I8, _PTR_I64 = "*i32:16", "*i8:16", "*i64:16"

# pointer tails: occ_off, occ_mid, occ_cls, trail, nprops, nops, ndecs (+ nrounds/order for
# the trajectory kernel) — occ_mid is the binary/long split of each occ segment.
_FIX_PTRS = [_PTR_I32, _PTR_I32, _PTR_I8, _PTR_I8, _PTR_I64] + [_PTR_I32] * 7
_FIX_CLEAN_PTRS = _FIX_PTRS[:8] + [_PTR_I32] + _FIX_PTRS[8:]   # + init_ptr after occ_cls
_TRAJ_PTRS = [_PTR_I32, _PTR_I32, _PTR_I8, _PTR_I8] + [_PTR_I32] * 9


def _require_triton_gather():
    """The WAVE schedule uses tl.gather (triton >= 3.2). Check BEFORE compiling: this script
    overwrites cpp/src/generated/gpap_kernels.c, so failing part-way through the specializations
    leaves the tree in a worse state than not starting."""
    if not hasattr(triton.language, "gather"):
        raise SystemExit(
            f"error: triton {triton.__version__} has no tl.gather; the WAVE kernels need >= 3.2.\n"
            f"       pip install -U 'triton>=3.2'")


def vscan_is_read(wave: int, soft: int) -> bool:
    """Does the trajectory kernel's VSCAN constant reach any code for this (wave, soft)?

    Its only use is guarded `if WAVE and VSCAN and not SOFT` (cpp/kernels/bcp.py), so for a
    scalar or a SOFT trajectory the two VSCAN values compile to THE SAME kernel. We used to
    emit both anyway: 6 of the 33 shipped cubins were byte-identical duplicates of another
    entry, 0.19 MB of a 0.83 MB payload, and every one of them was a second copy of a
    trajectory the engine can reach by two keys. `traj_key` in cpp/src/engine.cpp applies the
    same rule, so the engine asks for the key that exists.

    THE RULE LIVES IN TWO LANGUAGES and must agree; gated by
    tests/test_kernel_table.py, which hashes the generated table and launches
    every reachable key.
    """
    return bool(wave and not soft)


def kernel_specs():
    """(key, module, fn_name, signature, num_warps) for every kernel binary we ship."""
    out = []
    for shift in SHIFTS:
        for wave in (0, 1):
            # _bcp_fixpoint_kernel(... M, NW, B, TRAIL_CAP, SHIFT, WAVE, W)
            out.append((f"fixpoint.s{shift}.w{wave}", "bcp", "_bcp_fixpoint_kernel",
                        _FIX_PTRS + ["i32"] * 4 + [shift, wave, WAVE_W], 1))
            for soft in (0, 1):
                for vscan in ((0, 1) if vscan_is_read(wave, soft) else (0,)):
                    # _bcp_trajectory_kernel(... M, NW, P, NSUP, B, TRAIL_CAP, MAX_ROUNDS,
                    #                        seed, SOFT, SHIFT, WAVE, W, VSCAN)
                    out.append((f"trajectory.s{shift}.w{wave}.v{vscan}.f{soft}", "bcp",
                                "_bcp_trajectory_kernel",
                                _TRAJ_PTRS + ["i32"] * 7 + ["i64"]
                                + [soft, shift, wave, WAVE_W, vscan], 1))
    # ---- support kernels (BLOCK is the trailing constexpr) ----
    out += [
        ("bcast_rows", "helpers", "bcast_rows", [_PTR_I32, _PTR_I32, "i32", 1024], 4),
        ("fill_i8", "helpers", "fill_i8", [_PTR_I8, "i32", "i32", 1024], 4),
        ("complete_mask", "helpers", "complete_mask",
         [_PTR_I32, _PTR_I32, _PTR_I8, _PTR_I8, "i32", 256], 1),
        ("var_bits", "helpers", "var_bits",
         [_PTR_I32, _PTR_I32, _PTR_I32, _PTR_I8, "i32", "i32", 256], 4),
        ("unpack_rows", "helpers", "unpack_rows",
         [_PTR_I32, _PTR_I32, _PTR_I8, "i32", "i32", 1024], 4),
        ("sum_i8_to", "helpers", "sum_i8_to", [_PTR_I8, "i32", _PTR_I32, 1024], 1),
        ("sum_i32_acc", "helpers", "sum_i32_acc", [_PTR_I32, "i32", _PTR_I64, 1024], 1),
    ]
    for shift in SHIFTS:
        # sound_rows(st, ab, rows, M, NW, pairs, n_pairs, bad, SHIFT, BLOCK)
        out.append((f"sound_rows.s{shift}", "helpers", "sound_rows",
                    [_PTR_I32, _PTR_I32, _PTR_I32, "i32", "i32", _PTR_I32, "i32", _PTR_I8,
                     shift, 1024], 4))
    for shift in SHIFTS:
        for wave in (0, 1):
            # _bcp_fixpoint_clean_kernel(... occ_cls, INIT, trail ... M, NW, B, TRAIL_CAP,
            # SHIFT, WAVE, W) — CLEAN mode (docs/plans/RESET_RESTORE_PLAN.md §9). Appended
            # LAST so the 29 pre-existing blobs keep their order in the generated file (the
            # no-regression check diffs them key by key).
            out.append((f"fixpoint_clean.s{shift}.w{wave}", "bcp", "_bcp_fixpoint_clean_kernel",
                        _FIX_CLEAN_PTRS + ["i32"] * 4 + [shift, wave, WAVE_W], 1))
    return out


# ---------------------------------------------------------------- compilation

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compile_one(fn, signature: list, num_warps: int, target, num_stages: int = 3):
    """Compile one specialization. `signature` is positional, using triton's CLI conventions:
    a string like "*i32:16" is a typed argument with a 16-byte divisibility hint, an int is a
    constexpr value. Returns (cubin bytes, entry name, dynamic shared bytes)."""
    hints = {(i,): int(s.split(":")[1]) for i, s in enumerate(signature)
             if isinstance(s, str) and ":" in s}
    constants = {fn.arg_names[i]: s for i, s in enumerate(signature) if not isinstance(s, str)}
    sig = {fn.arg_names[i]: (s.split(":")[0] if isinstance(s, str) else "constexpr")
           for i, s in enumerate(signature)}
    for h in hints.values():
        assert h == 16, f"only 16-byte divisibility hints are used here, got {h}"
    attrs = {k: [["tt.divisibility", 16]] for k in hints}
    # Import ASTSource directly rather than via fn.create_binder(): the binder calls
    # `driver.active.get_current_target()`, and triton's NVIDIA driver reports itself INACTIVE
    # unless torch is installed AND torch.cuda.is_available() — i.e. unless a real GPU is
    # present. Going through the binder would make cross-compilation (`--arch`) impossible on
    # exactly the machines it exists for: CI runners and build containers with no GPU.
    from triton.compiler import ASTSource
    src = ASTSource(fn=fn, constexprs=constants, signature=sig, attrs=attrs)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    cc = triton.compile(src, target=target, options=options.__dict__)
    if getattr(cc.metadata, "global_scratch_size", 0) or \
            getattr(cc.metadata, "profile_scratch_size", 0):
        raise RuntimeError(f"{fn.__name__}: kernel needs scratch buffers; AOT path assumes none")
    return cc.asm[backend.binary_ext], cc.metadata.name, int(cc.metadata.shared)


# ---------------------------------------------------------------- C emission

HEADER = """/* GENERATED by cpp/tools/build_kernels.py -- DO NOT EDIT.
 * Cubins AOT-compiled from cpp/kernels/bcp.py + cpp/kernels/helpers.py.
 */
#ifndef GPAP_KERNELS_GENERATED_H
#define GPAP_KERNELS_GENERATED_H
#include <stdint.h>

typedef struct {
    const char *key;          /* e.g. "trajectory.s16.w1.v1.f0" */
    const char *entry;        /* kernel symbol inside the cubin */
    const unsigned char *cubin;
    int32_t cubin_size;
    int32_t shared;           /* dynamic shared memory bytes required at launch */
    int32_t threads;          /* block size = 32 * num_warps */
    int32_t arch;             /* sm_XX this cubin was built for */
} gpap_kernel_desc;

#ifdef __cplusplus
extern "C" {
#endif
extern const gpap_kernel_desc gpap_kernel_table[];
extern const int32_t gpap_kernel_count;
#ifdef __cplusplus
}
#endif
#endif /* GPAP_KERNELS_GENERATED_H */
"""


def c_blob(name: str, data: bytes, chunk: int = 4096) -> str:
    """Emit the cubin as concatenated octal-escaped string literals: 4 source chars per byte
    (vs ~5-6 for an int array) and much faster for the C compiler to parse."""
    parts = []
    for i in range(0, len(data), chunk):
        body = "".join(f"\\{b:03o}" for b in data[i:i + chunk])
        parts.append(f'    "{body}"')
    return (f"static const unsigned char {name}[{len(data)}] = \n"
            + "\n".join(parts) + ";\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", type=int, default=None,
                    help="sm arch to build for (e.g. 89). Default: the local GPU's.")
    ap.add_argument("--out", type=Path, default=REPO / "cpp" / "src" / "generated")
    args = ap.parse_args()
    _require_triton_gather()

    if args.arch is None:
        target = triton.runtime.driver.active.get_current_target()
        arch = int(target.arch)
    else:
        arch = args.arch
        target = triton.backends.compiler.GPUTarget("cuda", arch, 32)

    mods = {"bcp": _load_module(REPO / "cpp" / "kernels" / "bcp.py", "gpap_kernels_src"),
            "helpers": _load_module(REPO / "cpp" / "kernels" / "helpers.py", "gpap_helpers_src")}

    args.out.mkdir(parents=True, exist_ok=True)
    blobs, rows, total = [], [], 0
    for i, (key, mod, fn_name, sig, nw) in enumerate(kernel_specs()):
        fn = getattr(mods[mod], fn_name)
        cubin, entry, shared = compile_one(fn, sig, nw, target)
        blobs.append(c_blob(f"k{i}_cubin", cubin))
        rows.append(f'    {{"{key}", "{entry}", k{i}_cubin, {len(cubin)}, {shared}, '
                    f'{32 * nw}, {arch}}},')
        total += len(cubin)
        print(f"  {key:<34} {len(cubin):>7} B  shared={shared:<4} threads={32 * nw}", flush=True)

    (args.out / "gpap_kernels.h").write_text(HEADER)
    with open(args.out / "gpap_kernels.c", "w") as f:
        f.write('/* GENERATED by cpp/tools/build_kernels.py -- DO NOT EDIT. */\n')
        f.write('#include "gpap_kernels.h"\n\n')
        f.writelines(blobs)
        f.write("\nconst gpap_kernel_desc gpap_kernel_table[] = {\n")
        f.write("\n".join(rows))
        f.write("\n};\n")
        f.write(f"const int32_t gpap_kernel_count = {len(rows)};\n")
    print(f"\n{len(rows)} kernels, {total / 1024:.0f} KiB of cubin -> {args.out}/gpap_kernels.c "
          f"(sm_{arch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

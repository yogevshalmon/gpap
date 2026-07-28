"""Shared pieces of the GPU-rival head-to-heads (eval/rivals_{demotic,recurrent,hts}.py).

The one policy that lives here: **circuit variants**. GPAP's headline circuit numbers are measured
on two encodings of the same instance — our native Tseitin unroll, and an ABC-optimized one. The
circuit-level rivals (DEMOTIC, Recurrent-CircuitSAT) consume `.bench` and unroll it themselves, so
if only GPAP sees the ABC-optimized circuit the encoder's win is silently credited to the sampler.
`prepare_variant` gives BOTH sides the same circuit:

    raw  — the benchmark file as committed
    abc  — `strash; zero; dc2; dc2; fraig` on the SEQUENTIAL AIG, written back as a
           primitive-gate .bench (gpap.circuits.abc_backend.simplify_bench_abc)

and gates the swap on a bit-exact simulation check: both circuits are fed identical stimuli and
must accept exactly the same input sequences for the pinned target.

Honest limit, worth stating in any write-up: the rivals can only be given the SEQUENTIAL
optimization. GPAP's `abc` encoder also runs a pass AFTER unrolling, where the cross-frame
equivalence collapse lives (b12-class 12-14x) — that pass has no sequential equivalent, because
once frames are expanded the circuit can no longer be handed to a tool that unrolls it itself.
Measured on the ITC'99/ISCAS-89 benches, the sequential-only pass is nearly free (b03 129 -> 127
ANDs, s386 166 -> 113), so in practice the rivals see essentially the same circuit either way —
which is itself the point: their gap is not an encoding artifact.
"""
import os

VARIANTS = ("raw", "abc")


def prepare_variant(src: str, variant: str, tmpdir: str, *, cycles: int, seed: int,
                    target_mode: str = "random", pin_outputs=None, target=None) -> tuple:
    """(path, info) for one circuit variant of `src`. `info` carries the ABC stats and the
    equivalence-check result; `info["error"]` is set (and the RAW path returned) when ABC is
    unavailable or the check fails — a rival comparison must never silently run on a circuit
    that is not the instance."""
    if variant == "raw":
        return src, {"variant": "raw"}
    if variant != "abc":
        raise ValueError(f"unknown circuit variant {variant!r} (raw | abc)")
    from gpap.circuits.abc_backend import abc_available, bench_equiv_check, simplify_bench_abc
    info = {"variant": "abc"}
    if not abc_available():
        return src, {**info, "error": "no abc binary (scripts/build_abc.sh)"}
    dst = os.path.join(tmpdir, f"{os.path.splitext(os.path.basename(src))[0]}_abc.bench")
    try:
        info.update(simplify_bench_abc(src, dst))
        chk = bench_equiv_check(src, dst, cycles, seed, pin_outputs=pin_outputs,
                                target_mode=target_mode, target=target)
    except Exception as e:  # noqa: BLE001 — fall back to raw rather than compare the wrong circuit
        return src, {**info, "error": f"{type(e).__name__}: {e}"}
    info["equiv_ok"] = chk["ok"]
    info["equiv_hits"] = [chk["hits_src"], chk["hits_dst"], chk["n"]]
    if not chk["ok"]:
        return src, {**info, "error": "ABC copy is NOT equivalent on the pinned target"}
    return dst, info


def variant_note(info: dict) -> str:
    """One-line human summary for the table row."""
    if info.get("variant") == "raw":
        return "raw"
    if info.get("error"):
        return f"abc FAILED ({info['error'][:40]}) -> raw"
    return f"abc {info.get('ands_before')}->{info.get('ands_after')} ANDs"

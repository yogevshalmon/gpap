#!/usr/bin/env bash
# Fetch/build everything the GPAP evaluation compares against, and report what is ready.
#
#   eval/rivals/rivals_setup.sh            # fetch + build everything missing, then print the status table
#   eval/rivals/rivals_setup.sh check      # only print the status table (no network, no builds)
#
# Installs into $GPAP_HOME (default <repo>/.gpap):
#   bin/abc              berkeley-abc  — the `--encoder abc` circuit preprocessor  (scripts/build_abc.sh)
#   bin/cmsgen           CMSGen + the GPAP eval patch (--nowrite/--statsevery)     (scripts/build_cmsgen.sh)
#   src/Demotic                      ASP-DAC'25 arXiv 2502.08086  -> eval/rivals/rivals_demotic.py
#   src/Recurrent-CircuitSAT-Sampler MLCAD'25  arXiv 2502.21226   -> eval/rivals/rivals_recurrent.py
#   src/High-Throughput-SAT-Sampler  arXiv 2502.08673 (HTS)       -> eval/rivals/rivals_hts.py
#
# The three rivals are PyTorch programs run as subprocesses on the same GPU; nothing is vendored.
set -uo pipefail

# This file lives in eval/rivals/, so the repo root is TWO levels up ("..", from before the
# eval/ reorganization, silently pointed GPAP_HOME at eval/.gpap — every tool read MISSING).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPAP_HOME="${GPAP_HOME:-$REPO_ROOT/.gpap}"
SRC="$GPAP_HOME/src"
MODE="${1:-setup}"
PY_BIN="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY_BIN" ] || PY_BIN="$(command -v python3)"

declare -A RIVALS=(
  [Demotic]="https://github.com/arashardakani/Demotic"
  [Recurrent-CircuitSAT-Sampler]="https://github.com/arashardakani/Recurrent-CircuitSAT-Sampler"
  [High-Throughput-SAT-Sampler]="https://github.com/arashardakani/High-Throughput-SAT-Sampler"
)
# PINNED at the commits the recorded head-to-heads were measured with, like abc and cmsgen —
# an unpinned rival can move (or rewrite) HEAD and silently change the comparison. Every
# harness also records the checkout's actual HEAD into its results JSON (rival_common.py).
# Override (empty = branch tip): GPAP_DEMOTIC_COMMIT / GPAP_RECURRENT_COMMIT / GPAP_HTS_COMMIT.
declare -A PINS=(
  [Demotic]="${GPAP_DEMOTIC_COMMIT-f4e8ede73cb1b68d4806c96fb384ddddf53c2b2b}"
  [Recurrent-CircuitSAT-Sampler]="${GPAP_RECURRENT_COMMIT-b9b048aecb5954fb66333821dc6c1adf94d3432f}"
  [High-Throughput-SAT-Sampler]="${GPAP_HTS_COMMIT-5cf5888074ca47484d3eef6a7d58810c7bdf6a91}"
)

if [ "$MODE" != "check" ]; then
    mkdir -p "$SRC"
    for name in "${!RIVALS[@]}"; do
        pin="${PINS[$name]}"
        if [ -d "$SRC/$name/src" ]; then
            head="$(git -C "$SRC/$name" rev-parse HEAD 2>/dev/null || echo '?')"
            if [ -n "$pin" ] && [ "$head" != "$pin" ]; then
                echo "[warn] $name is at ${head:0:12}, pinned ${pin:0:12} —" \
                     "rm -rf '$SRC/$name' and re-run to fetch the pin"
            else
                echo "[ok]   $name already present @ ${head:0:12}"
            fi
        elif git clone --depth 1 "${RIVALS[$name]}" "$SRC/$name"; then
            # a depth-1 clone lands on the branch tip; move to the pin (GitHub serves SHA fetches)
            if [ -n "$pin" ] && [ "$(git -C "$SRC/$name" rev-parse HEAD)" != "$pin" ]; then
                git -C "$SRC/$name" fetch -q --depth 1 origin "$pin" \
                    && git -C "$SRC/$name" checkout -q FETCH_HEAD \
                    || echo "[warn] $name: pinned commit ${pin:0:12} not fetchable — left at" \
                            "branch tip $(git -C "$SRC/$name" rev-parse --short HEAD)"
            fi
            echo "[get]  $name @ $(git -C "$SRC/$name" rev-parse --short=12 HEAD)"
        else
            echo "[warn] clone failed: $name (it will be reported as missing)"
        fi
    done

    # --- Python >= 3.13 compatibility shim, function-preserving --------------------------------
    # All three rivals load their GENERATED circuit model with `exec(src)` followed by
    # `locals()[name]`. PEP 667 (Python 3.13) makes exec unable to fill a function's locals(),
    # so on a 3.13 runtime EVERY run of every rival dies with `KeyError: '<module>'` (observed
    # on a fresh Colab L4, 2026-08-22, python 3.13.15 — the same checkouts run fine on 3.12).
    # Exec-ing into an explicit namespace (module globals still visible) is bit-identical
    # semantics on <= 3.12 and works on 3.13. Idempotent: touches only files still carrying
    # the old pattern; like scripts/cmsgen_gpap_eval.patch, it changes no rival behaviour.
    SRC_DIR="$SRC" "$PY_BIN" - <<'PYFIX'
import os, re
files = [os.path.join(os.environ["SRC_DIR"], p) for p in (
    "Demotic/src/pytorch/model/circuit_comb.py",
    "Demotic/src/pytorch/model/circuit_seq.py",
    "Recurrent-CircuitSAT-Sampler/src/pytorch/model/circuit_seq.py",
    "High-Throughput-SAT-Sampler/src/pytorch/model/circuit.py")]
for p in files:
    if not os.path.isfile(p):
        continue
    s = open(p).read()
    if "        exec(self.pytorch_model)\n" not in s:
        continue
    s = s.replace(
        "        exec(self.pytorch_model)\n",
        "        _gpap_exec_ns = {}   # PEP 667 (py3.13): exec cannot fill locals() any more\n"
        "        exec(self.pytorch_model, globals(), _gpap_exec_ns)\n")
    s = re.sub(r"class_object = locals\(\)\[", "class_object = _gpap_exec_ns[", s)
    open(p, "w").write(s)
    print(f"[fix]  py3.13 exec-namespace shim: {p}")
PYFIX

    # Rival python deps: none of the three ships a requirements file, and HTS's baseline_sat
    # imports pycryptosat UNCONDITIONALLY — on a fresh box every HTS run dies on the import.
    # pandas/tqdm usually come with the Colab image; install whatever is missing.
    "$PY_BIN" -m pip install -q pycryptosat pandas tqdm \
        || echo "[warn] rival python deps (pycryptosat/pandas/tqdm) failed to install"

    # THE ENGINE. Since the engine became a C++ library there is no Python fallback: without
    # libgpap every GPAP row of every stage errors out. Build it FIRST and fail loudly, because
    # a missing core turns a multi-hour sweep into a few hundred error rows.
    if ! bash "$REPO_ROOT/scripts/build_core.sh"; then
        echo "[FATAL] libgpap did not build — the sweep would produce nothing but error rows." >&2
        echo "        Needs cmake + a C++17 compiler + the NVIDIA driver + python/triton." >&2
        echo "        On Colab: apt-get install -y build-essential cmake" >&2
        exit 1
    fi
    bash "$REPO_ROOT/scripts/build_abc.sh"    || echo "[warn] abc build failed (tseitin-only)"
    bash "$REPO_ROOT/scripts/build_cmsgen.sh" || echo "[warn] cmsgen build failed (GPAP-only runs)"
fi

echo
echo "=== evaluation environment ==="
printf '%-34s %s\n' "component" "status"
printf '%-34s %s\n' "----------------------------------" "------------------------------------"

py="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$py" ] || py="$(command -v python3)"
"$py" - <<'PY'
import importlib, shutil, torch
ok = torch.cuda.is_available()
name = torch.cuda.get_device_properties(0).name if ok else "-"
print(f'{"torch / CUDA":<34} {torch.__version__} cuda={torch.version.cuda} avail={ok} {name}')
# THE engine is libgpap now, not the triton JIT — probe the library, or a green table would
# promise a run that cannot happen (kernel_available() stays True with no core built).
try:
    from gpap.engine import core_available
    from gpap.engine import _core
    if core_available():
        print(f'{"libgpap (THE engine)":<34} {_core.version()}  {_core.load()._name}')
    else:
        print(f'{"libgpap (THE engine)":<34} MISSING -- run scripts/build_core.sh')
except Exception as e:
    print(f'{"libgpap (THE engine)":<34} UNAVAILABLE: {e}')
try:
    import triton
    print(f'{"triton (kernel compiler)":<34} {triton.__version__}  [build time only]')
except Exception as e:
    print(f'{"triton (kernel compiler)":<34} MISSING ({e}) -- cannot build the core')
for mod in ("pysat", "numpy", "matplotlib"):
    try:
        m = importlib.import_module(mod)
        print(f'{mod:<34} {getattr(m, "__version__", "ok")}')
    except Exception as e:
        print(f'{mod:<34} MISSING ({e})')
PY

for b in abc cmsgen; do
    p="$GPAP_HOME/bin/$b"
    if [ -x "$p" ]; then
        extra=""
        if [ "$b" = cmsgen ]; then
            "$p" --help 2>&1 | grep -q -- --nowrite && extra="  [eval patch: --nowrite]" \
                                                   || extra="  [STOCK: rates include writing samples]"
        fi
        printf '%-34s %s%s\n' "$b" "$p" "$extra"
    else
        printf '%-34s %s\n' "$b" "MISSING (scripts/build_${b}.sh)"
    fi
done

for name in "${!RIVALS[@]}"; do
    if [ -d "$SRC/$name/src" ]; then
        head="$(git -C "$SRC/$name" rev-parse HEAD 2>/dev/null || echo '?')"
        state="@ ${head:0:12}"
        if [ -n "${PINS[$name]}" ]; then
            [ "$head" = "${PINS[$name]}" ] && state="$state (pinned)" \
                                           || state="$state (DRIFTED from ${PINS[$name]:0:12})"
        fi
        printf '%-34s %s  %s\n' "$name" "$SRC/$name" "$state"
    else
        printf '%-34s %s\n' "$name" "MISSING (${RIVALS[$name]})"
    fi
done
echo
echo "harnesses: eval/rivals/rivals_demotic.py  eval/rivals/rivals_recurrent.py  eval/rivals/rivals_hts.py"

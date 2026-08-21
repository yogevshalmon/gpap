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

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPAP_HOME="${GPAP_HOME:-$REPO/.gpap}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$GPAP_HOME/src"
MODE="${1:-setup}"

declare -A RIVALS=(
  [Demotic]="https://github.com/arashardakani/Demotic"
  [Recurrent-CircuitSAT-Sampler]="https://github.com/arashardakani/Recurrent-CircuitSAT-Sampler"
  [High-Throughput-SAT-Sampler]="https://github.com/arashardakani/High-Throughput-SAT-Sampler"
)

if [ "$MODE" != "check" ]; then
    mkdir -p "$SRC"
    for name in "${!RIVALS[@]}"; do
        if [ -d "$SRC/$name/src" ]; then
            echo "[ok]   $name already present"
        else
            echo "[get]  cloning $name ..."
            git clone --depth 1 "${RIVALS[$name]}" "$SRC/$name" \
                || echo "[warn] clone failed: $name (it will be reported as missing)"
        fi
    done
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
        printf '%-34s %s\n' "$name" "$SRC/$name"
    else
        printf '%-34s %s\n' "$name" "MISSING (${RIVALS[$name]})"
    fi
done
echo
echo "harnesses: eval/rivals/rivals_demotic.py  eval/rivals/rivals_recurrent.py  eval/rivals/rivals_hts.py"

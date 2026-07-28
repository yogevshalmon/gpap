#!/usr/bin/env bash
# Fetch/build everything the GPAP evaluation compares against, and report what is ready.
#
#   eval/rivals_setup.sh            # fetch + build everything missing, then print the status table
#   eval/rivals_setup.sh check      # only print the status table (no network, no builds)
#
# Installs into $GPAP_HOME (default ~/.cache/gpap):
#   bin/abc              berkeley-abc  — the `--encoder abc` circuit preprocessor  (scripts/build_abc.sh)
#   bin/cmsgen           CMSGen + the GPAP eval patch (--nowrite/--statsevery)     (scripts/build_cmsgen.sh)
#   src/Demotic                      ASP-DAC'25 arXiv 2502.08086  -> eval/rivals_demotic.py
#   src/Recurrent-CircuitSAT-Sampler MLCAD'25  arXiv 2502.21226   -> eval/rivals_recurrent.py
#   src/High-Throughput-SAT-Sampler  arXiv 2502.08673 (HTS)       -> eval/rivals_hts.py
#
# The three rivals are PyTorch programs run as subprocesses on the same GPU; nothing is vendored.
set -uo pipefail

GPAP_HOME="${GPAP_HOME:-$HOME/.cache/gpap}"
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
try:
    import triton
    from gpap.engine.kernels import kernel_available
    print(f'{"triton kernels (THE engine)":<34} triton {triton.__version__}, '
          f'kernel_available={kernel_available()}')
except Exception as e:
    print(f'{"triton kernels (THE engine)":<34} UNAVAILABLE: {e}')
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
echo "harnesses: eval/rivals_demotic.py  eval/rivals_recurrent.py  eval/rivals_hts.py"

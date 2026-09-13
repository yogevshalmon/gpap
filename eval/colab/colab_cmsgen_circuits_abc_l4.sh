#!/usr/bin/env bash
# GPAP vs CMSGen — the CIRCUIT families only (seq / comb / epfl / hwmcc; no sk), ABC ENCODING
# ONLY (`abc:fast`). The same stages, budgets and portfolio as colab_cmsgen_l4.sh, so rows are
# protocol-comparable to the full-matrix campaigns; the two deliberate restrictions are the
# family list and ENCODERS. Resumable under eval/colab/colab_supervisor.py exactly like the
# full driver: every stage JSON is written after each row and --resume skips finished cells.
#
# The CMSGen preflight is a hard fail on purpose: a stage resumed without the binary would
# append armless rows to an armed file, which `gpap bench --resume` refuses, so a soft warning
# here would burn a VM producing nothing.
#
#   bash eval/colab/colab_cmsgen_circuits_abc_l4.sh
#   TARGETS=rare CYCLES=25 bash eval/colab/colab_cmsgen_circuits_abc_l4.sh   # headline slice
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # this file lives in eval/colab/
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PYTHON="${PYTHON:-python3}"

export ENCODERS="${ENCODERS:-abc:fast}"
export TARGETS="${TARGETS:-rare random}"
export CYCLES="${CYCLES:-25 50}"
export N_TIME="${N_TIME:-50000}"
export TIME_LIMIT="${TIME_LIMIT:-60}"
# Its own OUT: this matrix is a subset of the full one, and sharing a directory with a
# full-matrix run would let --resume interleave rows measured under different ENCODERS.
EXPERIMENTS="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m eval.paths experiments)"
: "${EXPERIMENTS:?cannot resolve — python3 must be able to import the repo (eval/paths.py)}"
export OUT="${OUT:-$EXPERIMENTS/full_eval/nvidia-l4-abc}"
# Ask, never recompute: gpap/paths.py is the one definition (see its docstring).
GPAP_HOME="$(PYTHONPATH="$REPO_ROOT" "${PYTHON:-python3}" -m gpap.paths home)"
: "${GPAP_HOME:?cannot resolve — python3 must be able to import the repo (gpap/paths.py, eval/paths.py)}"
export CMSGEN="${CMSGEN:-$GPAP_HOME/bin/cmsgen}"

if [ ! -x "$CMSGEN" ]; then
    echo "[FATAL] no cmsgen at $CMSGEN — this sweep IS the CMSGen comparison." >&2
    exit 1
fi
if ! "$CMSGEN" --help 2>&1 | grep -q -- --nowrite; then
    echo "[FATAL] cmsgen lacks scripts/cmsgen_gpap_eval.patch: it would be timed WITH sample" >&2
    echo "        emission and every ratio would be inflated." >&2
    exit 1
fi
if ! "$CMSGEN" --help 2>&1 | grep -q -- --maxtime; then
    echo "[FATAL] cmsgen predates --maxtime — a killed run reports a FLOORED count." >&2
    echo "        Rebuild: scripts/build_cmsgen.sh" >&2
    exit 1
fi

# seq first: the flagship family, and the run may be cut short. STAGES narrows the sweep for an
# incremental run — seeded with a previous campaign's JSONs, `--resume` skips finished cells, so
# re-running only the families that gained instances costs just the new cells. The seq stage
# globs benchmarks/circuits/bench/seq/*.bench and b17-b22 are fetched rather than committed, so
# pull them first or a fresh VM sweeps a smaller set.
bash "$REPO_ROOT/scripts/fetch_benchmarks.sh" itc99 2>&1 | sed 's/^/[bench] /' || \
    echo "[warn] could not fetch ITC'99 b17-b22 — the seq stage will run without them"
echo "[bench] sequential family: $(ls "$REPO_ROOT"/benchmarks/circuits/bench/seq/*.bench 2>/dev/null | wc -l) circuits"

STAGES="${STAGES:-seq comb epfl hwmcc}"
for stage in $STAGES; do
    echo; echo "########## STAGE $stage  $(date -Is)"
    bash eval/run_full_eval.sh "$stage" || echo "[warn] stage $stage returned non-zero"
    echo "STAGE_${stage}_DONE"
done

echo "CMSGEN_CIRCUITS_ABC_COMPLETE"

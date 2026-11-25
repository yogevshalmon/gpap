"""The memory limit and the ledger (cpp/src/memory.cpp, docs/ENGINE.md §7).

What GPAP holds on a device is counted buffer by buffer; a limit caps that count across every
engine on the device; each call's batch is sized to fit under it, everything the call allocates
afterwards included; and a batch that fits no row is GPAP_ERR_MEMORY, never one silent row.
"""
import gc
import os
import random

import numpy as np
import pytest

from gpap.engine import _core
from tests.support import random_cnf


@pytest.fixture(autouse=True)
def _no_limit(monkeypatch):
    """Every test starts and ends with no limit and the default reserve: both are process state,
    and a shell's GPAP_MEM_LIMIT, GPAP_MEM_RESERVE or GPAP_VRAM_BUDGET_MIB would size every batch
    here."""
    monkeypatch.delenv("GPAP_MEM_LIMIT", raising=False)
    monkeypatch.delenv("GPAP_MEM_RESERVE", raising=False)
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    yield
    # The CLI's --mem-limit and gpap.set_mem_limit write GPAP_MEM_LIMIT into the environment, for
    # child runs (--mem-reserve likewise); monkeypatch restores only what it set, so a limit
    # would outlive the test.
    os.environ.pop("GPAP_MEM_LIMIT", None)
    os.environ.pop("GPAP_MEM_RESERVE", None)
    if _core.available():
        _core.set_mem_limit(None, 0)
        _core.set_mem_reserve(None, 0)


def _formula(nv=600, nc=1800, seed=3):
    rng = random.Random(seed)
    return _core.Formula.from_clauses(nv, random_cnf(rng, nv, nc, kmin=2, kmax=4))


def _flat(sets):
    lits = np.array([x for s in sets for x in s], dtype=np.int32)
    offsets = np.cumsum([0] + [len(s) for s in sets]).astype(np.int64)
    return lits, offsets


def _sets(nv, n, width, seed=5):
    rng = random.Random(seed)
    return [[v if rng.random() < 0.5 else -v for v in rng.sample(range(1, nv + 1), width)]
            for _ in range(n)]


def _settled_held():
    """GPAP's count on device 0 once earlier tests' engines are gone: an engine in a reference
    cycle frees its buffers whenever the collector runs, which would move a baseline taken
    before it."""
    gc.collect()
    return _core.mem_held(0)


def _row_bytes(e):
    return _core.row_bytes(e.num_long, e.num_vars, _core.trail_cap(e.num_vars))


# ---------------------------------------------------------------------------------- the grammar

@pytest.mark.core
@pytest.mark.parametrize("spec,want", [
    ("8G", 8 << 30), ("8g", 8 << 30), ("8GiB", 8 << 30), ("8GB", 8 << 30), ("8 G", 8 << 30),
    ("512M", 512 << 20), ("1.5G", 3 << 29), ("2T", 2 << 40), ("64K", 64 << 10),
    ("100000B", 100000), (" 1G ", 1 << 30), ("none", 0), ("NONE", 0), (4096, 4096),
])
def test_every_documented_spelling_parses_to_its_bytes(spec, want):
    """The grammar of the header comment, unit for unit: binary units, case ignored, a blank
    allowed between number and unit, an int taken as bytes."""
    assert _core.parse_mem_limit(spec) == want


@pytest.mark.core
@pytest.mark.parametrize("spec,why", [
    ("8", "needs its unit"), ("0G", "more than zero"), ("-1G", "not a memory size"),
    ("1e3G", "unit"), ("0x10G", "unit"), ("101%", "at most 100%"), ("0%", "more than 0%"),
    ("8Q", "unit"), ("8G free", "after the unit"), ("", "not a memory size"),
    ("1.G", "digits after"), ("75% of it", "after the '%'"),
])
def test_a_malformed_spec_is_refused_with_its_reason(spec, why):
    """No bare number, zero, exponent, hex, overshare or trailing text slips through as some
    other limit: each is a ValueError that says what is wrong."""
    with pytest.raises(ValueError, match="memory limit") as err:
        _core.parse_mem_limit(spec)
    assert why in str(err.value)


@pytest.mark.core
def test_a_share_is_of_the_device_total():
    """"75%" is bytes only against a total; without one the spec is valid and means 0."""
    assert _core.parse_mem_limit("75%", 1000) == 750
    assert _core.parse_mem_limit("12.5%", 8000) == 1000
    assert _core.parse_mem_limit("75%") == 0


@pytest.mark.core
def test_the_api_overrides_the_variable_and_hands_the_device_back(monkeypatch):
    """GPAP_MEM_LIMIT applies, read on every call; gpap_set_mem_limit beats it on its device;
    "none" from the API means no limit despite it; NULL follows it again."""
    monkeypatch.setenv("GPAP_MEM_LIMIT", "8G")
    assert (_core.mem_limit(0), _core.mem_limit_source(0)) == (8 << 30, "GPAP_MEM_LIMIT")
    _core.set_mem_limit("2G", 0)
    assert (_core.mem_limit(0), _core.mem_limit_source(0)) == (2 << 30, "set_mem_limit")
    _core.set_mem_limit("none", 0)
    assert (_core.mem_limit(0), _core.mem_limit_source(0)) == (0, "set_mem_limit")
    _core.set_mem_limit(None, 0)
    assert _core.mem_limit(0) == 8 << 30
    monkeypatch.setenv("GPAP_MEM_LIMIT", "3G")                  # read on every call
    assert _core.mem_limit(0) == 3 << 30
    monkeypatch.delenv("GPAP_MEM_LIMIT")
    assert (_core.mem_limit(0), _core.mem_limit_source(0)) == (0, None)


@pytest.mark.core
def test_python_sets_the_process_default_or_one_device(monkeypatch):
    """gpap.set_mem_limit with no device writes GPAP_MEM_LIMIT, which child runs inherit, and
    refuses a bad spec before anything changes; with a device it is that device's API limit."""
    import gpap
    gpap.set_mem_limit("6G")
    assert os.environ["GPAP_MEM_LIMIT"] == "6G"
    with pytest.raises(ValueError):
        gpap.set_mem_limit("6")
    assert os.environ["GPAP_MEM_LIMIT"] == "6G"
    gpap.set_mem_limit(None)
    assert "GPAP_MEM_LIMIT" not in os.environ
    gpap.set_mem_limit(1 << 30, device=0)
    assert _core.mem_limit_source(0) == "set_mem_limit" and _core.mem_limit(0) == 1 << 30


# ---------------------------------------------------------------------------------- the reserve

@pytest.mark.core
def test_the_reserve_follows_the_api_then_the_variable_then_the_default(monkeypatch):
    """The reserve is resolved as the limit is: gpap_set_mem_reserve on its device, else
    GPAP_MEM_RESERVE (read on every call), else the library's default of 1 GiB; "none" from
    the API is no reserve despite the variable, and NULL follows the variable again. Its grammar
    is the limit's, so a bare number is refused with the reason."""
    assert _core.mem_reserve(0) == 1 << 30
    monkeypatch.setenv("GPAP_MEM_RESERVE", "2G")
    assert _core.mem_reserve(0) == 2 << 30
    _core.set_mem_reserve("512M", 0)
    assert _core.mem_reserve(0) == 512 << 20
    _core.set_mem_reserve("none", 0)
    assert _core.mem_reserve(0) == 0
    _core.set_mem_reserve(None, 0)
    monkeypatch.setenv("GPAP_MEM_RESERVE", "3G")
    assert _core.mem_reserve(0) == 3 << 30
    with pytest.raises(ValueError, match="unit"):
        _core.set_mem_reserve("512", 0)
    monkeypatch.setenv("GPAP_MEM_RESERVE", "512")
    with pytest.raises(ValueError, match="GPAP_MEM_RESERVE"):
        _core.mem_reserve(0)


@pytest.mark.core
def test_python_sets_the_process_reserve_or_one_device(monkeypatch):
    """gpap.set_mem_reserve with no device writes GPAP_MEM_RESERVE, which child runs inherit,
    and refuses a bad spec before anything changes; with a device it is that device's API
    reserve. gpap.engine.memory() reports it."""
    import gpap
    from gpap.engine import memory
    gpap.set_mem_reserve("2G")
    assert os.environ["GPAP_MEM_RESERVE"] == "2G" and memory()["reserve"] == 2 << 30
    with pytest.raises(ValueError):
        gpap.set_mem_reserve("2")
    assert os.environ["GPAP_MEM_RESERVE"] == "2G"
    gpap.set_mem_reserve(None)
    assert "GPAP_MEM_RESERVE" not in os.environ and memory()["reserve"] == 1 << 30
    gpap.set_mem_reserve(256 << 20, device=0)
    assert _core.mem_reserve(0) == 256 << 20


@pytest.mark.gpu
def test_the_default_budget_is_free_memory_less_the_reserve():
    """Without a limit a new engine's rows are sized against free memory less the reserve, all
    of it on a device that measures free memory: 256 MiB more reserve is 256 MiB less budget,
    and with no reserve the budget is the free memory itself, not 80% of it. On OpenCL, which
    cannot measure free memory, the budget is a share (80%) of the total less what GPAP holds
    there, and the reserve comes out before the share. Every face defaults to the engine's own
    budget (fraction 1)."""
    import torch

    from gpap.engine.calibrate import gpu_mem_available
    opencl = os.environ.get("GPAP_BACKEND", "cuda") == "opencl"
    share, slack = (0.8, 1 << 10) if opencl else (1.0, 8 << 20)  # free memory moves a little
    _core.set_mem_reserve("none", 0)
    whole = _core.mem_available(0)
    if not opencl:
        assert abs(whole - torch.cuda.mem_get_info(0)[0]) <= slack
    _core.set_mem_reserve("256M", 0)
    assert abs((whole - _core.mem_available(0)) - share * (256 << 20)) <= slack
    _core.set_mem_reserve(None, 0)
    assert abs((whole - _core.mem_available(0)) - share * (1 << 30)) <= slack
    assert abs(gpu_mem_available(torch.device("cuda", 0)) - _core.mem_available(0, 1.0)) <= slack


@pytest.mark.gpu
def test_the_cli_sets_the_reserve_for_the_run(capsys):
    """--mem-reserve sets GPAP_MEM_RESERVE for the run, and `gpap info` reports it."""
    from gpap.cli import main as cli_main
    cli_main(["info", "--mem-reserve", "2G"])
    assert os.environ["GPAP_MEM_RESERVE"] == "2G"
    assert "2.00 GiB left free" in capsys.readouterr().out


_CPP_MEMORY_SRC = r"""
#include <gpap/gpap.hpp>
#include <cstdio>
int main() {
    std::printf("%lld %lld %lld\n", (long long)gpap::parse_mem_limit("8G"),
                (long long)gpap::parse_mem_limit("75%", 1000),
                (long long)gpap::parse_mem_limit("none"));
    try {
        gpap::parse_mem_limit("8");
        std::printf("accepted\n");
    } catch (const gpap::Error &e) {
        std::printf("refused: %s\n", e.what());
    }
    gpap::set_mem_limit("64M");
    gpap::set_mem_reserve("256M");
    gpap::reset_peak();
    const gpap::Memory m = gpap::memory();
    std::printf("%lld %lld %lld %lld\n", (long long)m.limit, (long long)m.held, (long long)m.peak,
                (long long)m.reserve);
    std::printf("%d\n", gpap::mem_available() == gpap::mem_available(0, 1.0) ? 1 : 0);
    return 0;
}
"""


@pytest.mark.core
def test_the_cpp_faces_parse_set_and_report_a_limit(tmp_path):
    """The C++ wrapper has the C ABI's whole memory family: parse_mem_limit gives a spec's bytes
    without setting it (binary units, a share of a given total, "none" as 0) and throws, with the
    reason, for a bare number; set_mem_limit, set_mem_reserve, reset_peak and memory set and read
    a device; mem_available defaults to the engine's own budget (fraction 1). No engine exists,
    so GPAP holds nothing and its peak is nothing. Built against cpp/include and the library
    Python loads; no GPU needed."""
    import shlex
    import shutil
    import subprocess
    from pathlib import Path
    cxx = shlex.split(os.environ["CXX"]) if os.environ.get("CXX") else (
        [c] if (c := shutil.which("c++") or shutil.which("g++")) else None)
    if not cxx:
        pytest.skip("no C++ compiler")
    lib = next((c for c in _core._candidates() if c.exists()), None)
    if lib is None:
        pytest.skip("libgpap not built (scripts/build_core.sh)")
    src, exe = tmp_path / "memory.cpp", tmp_path / "memory"
    src.write_text(_CPP_MEMORY_SRC)
    r = subprocess.run([*cxx, "-std=c++17", "-O1",
                        f"-I{Path(__file__).parents[2] / 'cpp/include'}", str(src), "-o",
                        str(exe), str(lib), f"-Wl,-rpath,{lib.parent}"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    env = {k: v for k, v in os.environ.items() if k not in ("GPAP_MEM_LIMIT", "GPAP_MEM_RESERVE")}
    out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60, env=env,
                         check=True).stdout.splitlines()
    assert out[0] == f"{8 << 30} 750 0"
    assert out[1].startswith("refused: ") and "unit" in out[1]
    assert out[2] == f"{64 << 20} 0 0 {256 << 20}"
    assert out[3] == "1"


# ---------------------------------------------------------------------------------- the ledger

@pytest.mark.engine
def test_the_ledger_counts_the_row_state_exactly():
    """Growing the batch from the flush's one row to K adds (K - 1) rows of gpap_row_bytes to
    the ROWS count, byte for byte, and nothing to the FORMULA count: the ledger and the sizing
    model are the same allocation list, and the two kinds are kept apart."""
    base = _settled_held()
    e = _core.Engine(_formula())
    formula, rows = e.mem_held_formula, e.mem_held_rows
    assert formula > 0 and rows > 0 and e.mem_held == formula + rows
    assert _core.mem_held(0) - base == e.mem_held
    e.reset(1000)
    assert e.mem_held_rows - rows == 999 * _row_bytes(e)
    assert e.mem_held_formula == formula
    assert _core.mem_held(0) - base == e.mem_held


@pytest.mark.engine
def test_the_device_count_sums_engines_and_returns_when_they_go():
    """Two engines on one device add up, and freeing both gives every byte back."""
    base = _settled_held()
    a = _core.Engine(_formula(seed=1))
    b = _core.Engine(_formula(seed=2))
    b.reset(64)
    assert _core.mem_held(0) - base == a.mem_held + b.mem_held
    del a, b
    assert _settled_held() == base


@pytest.mark.engine
def test_trim_keeps_the_formula_and_changes_no_result():
    """A trim gives back every row byte and keeps every formula byte (ROWS 0, FORMULA unchanged:
    a row buffer trim does not release shows here), the next call takes the rows again, and it
    answers exactly as before the trim."""
    f = _formula()
    e = _core.Engine(f)
    formula = e.mem_held_formula
    lits, off = _flat(_sets(f.info.num_vars, 300, 2))
    before = e.check_sets(lits, off, chunk=64, assignments=True, sat_count=True)
    assert e.mem_held_rows > 0
    e.trim()
    assert (e.mem_held_rows, e.mem_held_formula, e.K) == (0, formula, 0)
    after = e.check_sets(lits, off, chunk=64, assignments=True, sat_count=True)
    for x, y in zip(before, after):
        assert np.array_equal(x, y)
    assert e.mem_held_rows > 0 and e.mem_held_formula == formula


# ---------------------------------------------------------------------------------- the limit

@pytest.mark.engine
def test_a_limit_below_the_formula_refuses_the_engine():
    """The formula's own buffers are counted too: a limit they do not fit in stops the engine
    at creation, and the message names the limit."""
    _core.set_mem_limit("1K", 0)
    with pytest.raises(MemoryError, match="memory limit"):
        _core.Engine(_formula())


@pytest.mark.engine
def test_the_batch_fits_under_the_limit_and_says_so():
    """Under a limit of the engine plus n rows plus room for what a call takes once, the batch
    uses that room (at least n rows, at most what the whole of it holds), bound "limit", and
    fits when it is allocated; under a limit that leaves less than one row, sizing fails and
    says how much a row needs."""
    e = _core.Engine(_formula())
    held, row = _settled_held(), _row_bytes(e) + 8 * 2
    room = 1 << 20
    _core.set_mem_limit(held + 400 * row + room, 0)
    k = e.check_chunk_size(10 ** 6, 2)
    assert 400 <= k <= 400 + room // row and e.batch_bound == "limit"
    e.reset(k)
    assert _core.mem_held(0) <= _core.mem_limit(0)
    _core.set_mem_limit(held + 16, 0)
    with pytest.raises(MemoryError, match="not even one row fits"):
        e.check_chunk_size(10 ** 6, 2)


@pytest.mark.engine
def test_under_a_limit_a_smaller_batch_holds_only_its_own_rows():
    """Row buffers only grow, so a batch smaller than the engine's last keeps the bigger one's
    buffers. The sizing credits the engine's row state as reused by the next batch, which a
    smaller batch does not do, so under an exact bound a reset fits the buffers to its own rows:
    the row state shrinks by exactly the rows it dropped, and the room the sizing counted for the
    call's later buffers is really there. Without a limit the buffers are kept for the next
    bigger batch, as before."""
    f = _formula()
    e = _core.Engine(f)
    row = _row_bytes(e)
    e.reset(2000)
    big = e.mem_held_rows
    e.reset(100)
    assert e.mem_held_rows == big, "without a limit a smaller batch keeps the buffers"
    _core.set_mem_limit(_settled_held() + (64 << 20), 0)
    e.reset(2000)
    e.reset(100)
    assert big - e.mem_held_rows == 1900 * row


@pytest.mark.engine
def test_under_a_limit_a_chunk_loops_short_last_chunk_keeps_the_first_chunks_rows():
    """The refit is per call, not per chunk: a chunk loop's short last chunk keeps the rows of
    its first, which the call's sizing counted, rather than give them back for the next call to
    take again. Under a limit, a check of 150 sets in chunks of 100 ends holding what a check of
    one chunk of 100 holds, not a last chunk of 50."""
    f = _formula()
    lits, off = _flat(_sets(f.info.num_vars, 150, 3))
    _core.set_mem_limit(_settled_held() + (64 << 20), 0)
    one = _core.Engine(f)
    one.check_sets(lits[:off[100]], off[:101], chunk=100, assignments=True)
    loop = _core.Engine(f)
    loop.check_sets(lits, off, chunk=100, assignments=True)
    assert loop.K == 50, "precondition: the last chunk was short"
    assert loop.mem_held_rows == one.mem_held_rows


@pytest.mark.engine
def test_a_lowered_limit_applies_at_the_next_sizing():
    """A limit set after an engine sized itself bounds that engine's next batch; nothing is
    freed to meet it, and it holds what it held."""
    e = _core.Engine(_formula())
    k0 = e.check_chunk_size(10 ** 6, 2)
    e.reset(min(k0, 2000))
    held = e.mem_held
    _core.set_mem_limit(_settled_held() + (4 << 20), 0)
    assert e.mem_held == held
    assert e.check_chunk_size(10 ** 6, 2) < k0 and e.batch_bound == "limit"


@pytest.mark.engine
def test_a_second_engine_sizes_beside_the_first():
    """The limit is per device: what one engine holds leaves the other less, and trimming the
    first gives its rows to the second."""
    a = _core.Engine(_formula(seed=1))
    b = _core.Engine(_formula(seed=2))
    _core.set_mem_limit(_settled_held() + 600 * (_row_bytes(b) + 16) + (2 << 20), 0)
    a.reset(a.check_chunk_size(10 ** 6, 2))
    alone = b.check_chunk_size(10 ** 6, 2)
    a.trim()
    assert b.check_chunk_size(10 ** 6, 2) > alone


@pytest.mark.engine
def test_no_row_fits_is_an_error_without_a_limit_too(monkeypatch):
    """The default rule refuses as the limit does: a formula whose one row is past the budget
    (GPAP_VRAM_BUDGET_MIB=1, a row over 1 MiB) fails at sizing instead of running one row."""
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", "1")
    e = _core.Engine(_core.Formula.from_clauses(400_000, [[1, 2, 3], [-1, 4]]))
    assert _row_bytes(e) > 1 << 20
    with pytest.raises(MemoryError, match="not even one row fits"):
        e.check_chunk_size(10, 1)
    lits, off = _flat([[1], [2]])
    with pytest.raises(MemoryError):
        e.check_sets(lits, off, chunk=0)


@pytest.mark.engine
@pytest.mark.parametrize("call", ["assignments", "flags", "probe1", "probe2"])
def test_without_a_limit_a_call_stays_inside_the_default_budget(monkeypatch, call):
    """The default rule counts what a call allocates after it is sized, as a limit does: under
    a default budget of 2 MiB (GPAP_VRAM_BUDGET_MIB caps it), a fresh engine's call takes no more
    than that beside what GPAP held before: checking with assignments and sat counts, flags-only
    checking, probe sweeps of width 1 and 2. Before, the default rule left a call's readout
    slices (up to 256 MiB) and selection lists to its 20% margin, and the check with assignments
    took 15% more than its budget."""
    monkeypatch.setenv("GPAP_VRAM_BUDGET_MIB", "2")
    f = _formula(nv=400, nc=1200, seed=9)
    e = _core.Engine(f)
    base = _settled_held()
    _core.mem_reset_peak(0)
    lits, off = _flat(_sets(f.info.num_vars, 3000, 3))
    if call == "assignments":
        e.check_sets(lits, off, chunk=0, assignments=True, sat_count=True)
    elif call == "flags":
        e.check_sets(lits, off, chunk=0)
    else:
        e.probe_combinations(None, int(call[-1]), max_tuples=20_000).conflicts()
    assert e.batch_bound == "memory"
    assert _core.mem_peak(0) - base <= 2 << 20


@pytest.mark.engine
@pytest.mark.parametrize("rows", [1, 3, 40, 1000])
def test_every_entry_point_stays_under_a_tight_limit(rows):
    """Sizing covers what each call allocates afterwards: under a limit of the device plus a few
    rows, checking with assignments and sat counts, flags-only checking and probe sweeps of
    width 1 and 2, all in chunks (the queue's launch buffers do not fit), run, and GPAP's peak
    never passes the limit. A buffer the batch rule misses (engine.cpp, exact_terms) fails one
    of these."""
    f = _formula(nv=400, nc=1200, seed=9)
    e = _core.Engine(f)
    nv = f.info.num_vars
    _core.set_mem_limit(_settled_held() + rows * (_row_bytes(e) + 64) + 40 * nv + (1 << 16), 0)
    _core.mem_reset_peak(0)
    limit = _core.mem_limit(0)
    lits, off = _flat(_sets(nv, 3000, 3))
    e.check_sets(lits, off, chunk=0, assignments=True, sat_count=True)
    e.check_sets(lits, off, chunk=0)
    for k in (1, 2):
        sweep = e.probe_combinations(None, k, max_tuples=20_000)
        sweep.conflicts()
    assert _core.mem_peak(0) <= limit


# ---------------------------------------------------------------------------------- the pipeline

@pytest.mark.gpu
def test_sampling_mines_under_a_limit_as_without_one(tmp_path):
    """A limited run still mines (the sampler gives its rows back first, pipeline.py), finds the
    clauses an unlimited run finds, sizes its batch from the limit and stays under it."""
    import json

    from gpap.cli import main as cli_main
    from tests.conftest import REPO_ROOT
    cnf = os.path.join(REPO_ROOT, "benchmarks", "sampling", "71.sk_3_65.cnf")
    runs = {}
    for limit in (None, "32M"):
        out = tmp_path / f"{limit}.json"
        argv = ["sample", cnf, "-n", "300", "--portfolio", "b8", "--mine", "on",
                "--stats", str(out)]
        cli_main(argv + (["--mem-limit", limit] if limit else []))
        runs[limit] = json.loads(out.read_text())
    free, capped = runs[None], runs["32M"]
    # 32M holds fewer rows than the 1024 the warmup runs at, so the sampler's warmup rows fill the
    # limit when mining starts: without the trim mining fits no row and is skipped.
    assert capped["mined"] is not None, "mining was skipped under the limit"
    assert capped["mined"]["n_units"] == free["mined"]["n_units"] > 0
    assert capped["engine"]["batch_bound"] == "limit"
    assert capped["engine"]["mem_peak"] <= capped["engine"]["mem_limit"] == 32 << 20


def _queue_call_bytes(n_sets, width):
    """queue_call_bytes (cpp/src/gpap_internal.hpp), restated: the launch buffers of one queued
    call, a launch being at most 2^20 jobs and 2^24 literals (one set, however wide)."""
    w = max(width, 1)
    jobs = max(1, min(n_sets, 1 << 20))
    return min(jobs * w, max(1 << 24, w)) * 4 + jobs * 5 + 16


@pytest.mark.gpu
def test_sampling_gives_the_samplers_rows_back_before_it_mines(monkeypatch):
    """Mining sizes from the formula and the card, not from what the sampler's warmup rows
    leave: a sampling run trims the sampler before every mining call, with no limit too (GitHub
    issue 76), and the run still delivers. The spies call through to the real methods."""
    import gpap.sampling.pipeline as pipeline
    from gpap.cli import main as cli_main
    from gpap.sampling.sampler import GPUSampler
    from tests.conftest import REPO_ROOT
    events = []
    trim, mine = GPUSampler.trim, pipeline.mine_entailed

    def spy_trim(self):
        events.append("trim")
        return trim(self)

    def spy_mine(*a, **kw):
        events.append("mine")
        return mine(*a, **kw)
    monkeypatch.setattr(GPUSampler, "trim", spy_trim)
    monkeypatch.setattr(pipeline, "mine_entailed", spy_mine)
    cnf = os.path.join(REPO_ROOT, "benchmarks", "sampling", "71.sk_3_65.cnf")
    cli_main(["sample", cnf, "-n", "300", "--portfolio", "b8", "--mine", "on"])
    assert "mine" in events, "precondition: the run mined"
    assert all(i > 0 and events[i - 1] == "trim" for i, ev in enumerate(events) if ev == "mine"), \
        events


@pytest.mark.engine
def test_the_queue_runs_under_a_limit_that_holds_only_what_its_calls_take():
    """The queue takes off the budget what a call's launches can allocate, for its own sets and
    its widest set, not a fixed worst case of about 70 MB (GitHub issue 72). Under a limit of
    the device plus a few rows plus those buffers, flags-only checks and probe sweeps both run on
    the queue (each adds launches to its count), and GPAP's peak never passes the limit: the
    buffers are sized for, not just tolerated."""
    f = _formula(nv=400, nc=1200, seed=9)
    e = _core.Engine(f)
    nv = f.info.num_vars
    calls = max(_queue_call_bytes(3000, 3), _queue_call_bytes(20_000, 2))
    _core.set_mem_limit(_settled_held() + calls + 16 * (_row_bytes(e) + 24 + 64) + 40 * nv
                        + (1 << 20), 0)
    _core.mem_reset_peak(0)
    limit = _core.mem_limit(0)
    lits, off = _flat(_sets(nv, 3000, 3))
    before = e.queue_stats()[0]
    e.check_sets(lits, off, chunk=0)
    after_check = e.queue_stats()[0]
    assert after_check > before, "the flags-only check did not run on the queue"
    e.probe_combinations(None, 2, max_tuples=20_000).conflicts()
    assert e.queue_stats()[0] > after_check, "the probe sweep did not run on the queue"
    assert _core.mem_peak(0) <= limit


@pytest.mark.engine
def test_a_reset_refused_part_way_leaves_no_rows_and_a_usable_engine():
    """A grow frees each buffer before it allocates the next, so a limit can refuse it half
    done. The engine then holds no rows (K reads 0), a readout touches nothing freed, and the
    next reset that fits takes rows again and answers as a fresh engine does."""
    f = _formula()
    e = _core.Engine(f)
    e.reset(1000)
    row = _row_bytes(e)
    # Room for the first buffer of a 5000-row batch (the packed words), not for the rest.
    _core.set_mem_limit(_settled_held() - e.mem_held_rows + 5000 * 4 * e.num_long + (1 << 16), 0)
    with pytest.raises(MemoryError):
        e.reset(5000)
    assert e.K == 0
    e.flags()                                    # zero rows: nothing to read, nothing freed
    _core.set_mem_limit(None, 0)
    lits, off = _flat(_sets(f.info.num_vars, 200, 2))
    fresh = _core.Engine(f).check_sets(lits, off, chunk=50, assignments=True)
    again = e.check_sets(lits, off, chunk=50, assignments=True)
    for x, y in zip(fresh, again):
        assert np.array_equal(x, y)
    assert row > 0


def _on_wsl2():
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


@pytest.mark.gpu
def test_a_batch_past_the_device_fails_as_memory_and_the_engine_recovers():
    """A pinned batch twice the device's size fails as GPAP_ERR_MEMORY (MemoryError) on every
    plug, and the engine keeps no rows; the same engine then runs a small batch and answers as a
    fresh one does, and freeing it gives every byte back. A runtime that backs a buffer only when
    a command first uses it (NVIDIA's OpenCL) used to create the buffers and fail at the first
    launch with a driver error, leaving the rows with no memory behind them (GitHub issue 79).
    Not on WSL2, which pages device memory to the host instead of failing."""
    if _on_wsl2():
        pytest.skip("WSL2 pages device memory to the host instead of failing")
    base = _settled_held()
    _core.set_mem_limit("100%", 0)               # the device's total, as the limit resolves it
    total = _core.mem_limit(0)
    _core.set_mem_limit(None, 0)
    f = _formula(nv=5000, nc=20000, seed=9)
    e = _core.Engine(f)
    with pytest.raises(MemoryError):
        e.reset(min(2 * total // _row_bytes(e) + 1, 2**31 - 1))
    assert e.K == 0
    e.reset(256)
    lits, off = _flat(_sets(f.info.num_vars, 600, 3))
    fresh = _core.Engine(f).check_sets(lits, off, chunk=256, assignments=True)
    again = e.check_sets(lits, off, chunk=256, assignments=True)
    for x, y in zip(fresh, again):
        assert np.array_equal(x, y)
    del e, fresh, again
    assert _settled_held() == base


@pytest.mark.gpu
def test_on_an_unmeasured_device_a_second_engine_sizes_from_what_the_first_leaves():
    """OpenCL cannot measure free memory, so the default rule sizes against a share of the
    device's total; what GPAP's other engines hold there comes out of it first, as it does out
    of CUDA's measured free memory. Beside engines that hold most of the device, one more sizes
    to what is left and runs, answering as the others do; before, every engine planned on the
    whole device and the last could not allocate. Not on WSL2, which pages device memory to the
    host instead of failing."""
    if os.environ.get("GPAP_BACKEND", "cuda") != "opencl":
        pytest.skip("CUDA measures free memory; this is the rule for a device that cannot")
    if _on_wsl2():
        pytest.skip("WSL2 pages device memory to the host instead of failing")
    _core.set_mem_limit("100%", 0)               # the device's total, as the limit resolves it
    total = _core.mem_limit(0)
    _core.set_mem_limit(None, 0)
    f = _formula(nv=4000, nc=60000, seed=11)     # rows of ~180 KB
    # A device that caps one buffer (NVIDIA's OpenCL: a quarter of the card) sets a lone
    # engine's batch by that cap, not by the share; engines at that batch fill the device first.
    held, k0 = [], None
    while _core.mem_held(0) < 0.6 * total and len(held) < 8:
        e = _core.Engine(f)
        k = e.check_chunk_size(1 << 40, 3)
        k0 = k if k0 is None else k0
        e.reset(k)
        held.append(e)
    last = _core.Engine(f)
    k = last.check_chunk_size(1 << 40, 3)
    assert 0 < k < k0, (k0, k, len(held))
    last.reset(k)
    lits, off = _flat(_sets(f.info.num_vars, 300, 3))
    for x, y in zip(held[0].check_sets(lits, off, chunk=100, assignments=True),
                    last.check_sets(lits, off, chunk=100, assignments=True)):
        assert np.array_equal(x, y)


@pytest.mark.engine
def test_trim_then_recache_is_refused_cleanly():
    """recache_flush caches row 0; a trimmed engine has none, so it refuses before touching a
    buffer, and the engine works on."""
    f = _formula()
    e = _core.Engine(f)
    e.trim()
    with pytest.raises(_core.GpapError, match="holds no rows"):
        e.recache_flush()
    lits, off = _flat(_sets(f.info.num_vars, 50, 2))
    e.check_sets(lits, off, chunk=16)


@pytest.mark.engine
def test_a_malformed_variable_fails_engines_and_is_reported_as_such(monkeypatch):
    """GPAP_MEM_LIMIT=8 is not "no limit": engine creation fails, and the limit reads as an
    error, never as none."""
    monkeypatch.setenv("GPAP_MEM_LIMIT", "8")
    with pytest.raises(_core.GpapError, match="needs its unit"):
        _core.Engine(_formula())
    with pytest.raises(ValueError, match="needs its unit"):
        _core.mem_limit(0)


@pytest.mark.engine
def test_a_share_is_of_the_real_device_and_never_rounds_to_no_limit():
    """"50%" resolves against the device's total; a share under one byte is refused, not
    taken for "none"."""
    e = _core.Engine(_formula())
    _core.set_mem_limit("50%", 0)
    assert 0 < _core.mem_limit(0)
    _core.set_mem_limit("0.0000000000001%", 0)
    with pytest.raises(ValueError, match="less than one byte"):
        _core.mem_limit(0)
    assert e.mem_held > 0


@pytest.mark.engine
def test_a_pinned_batch_reports_the_pin():
    """A caller's chunk is not a sizing: the bound says "pin", not the last sizing's bound."""
    f = _formula()
    e = _core.Engine(f)
    lits, off = _flat(_sets(f.info.num_vars, 100, 2))
    e.check_sets(lits, off, chunk=0, assignments=True)
    assert e.batch_bound in ("cap", "memory", "work")
    e.check_sets(lits, off, chunk=10, assignments=True)
    assert e.batch_bound == "pin"


def _binary_rich_formula():
    """A formula whose width-2 probe proves binary clauses: (a|b|c) and (a|b|~c) entail (a|b),
    so mining strengthens the formula and rebuilds its probe engine before width 3."""
    rng = random.Random(7)
    nv, clauses, near = 4000, [], list(range(1, 201))
    for _ in range(150):
        a, b = rng.sample(near, 2)
        c = rng.randint(201, nv)
        clauses += [[a, b, c], [a, b, -c]]
    for _ in range(12000):
        vs = rng.sample(range(201, nv + 1), 3)
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    return nv, clauses, near + [-v for v in near]


@pytest.mark.gpu
def test_mining_rebuilds_its_probe_under_a_limit():
    """Mining rebuilds its probe engine between widths while the old one is alive; under a
    limit the old one's rows fill it, so it gives them back first (mining.py). The limited run
    mines what the unlimited one does, width 3 included, and stays under the limit."""
    import torch

    from gpap.mining import mine_entailed
    nv, clauses, pool = _binary_rich_formula()
    runs = {}
    for limit in (None, "16M"):
        _core.set_mem_limit(limit, 0)
        _settled_held()
        _core.mem_reset_peak(0)
        res = mine_entailed(nv, clauses, torch.device("cuda"), pool=pool, widths=(1, 2, 3),
                            max_tuples={1: 0, 2: 1_000_000, 3: 40_000}, seconds=60)
        runs[limit] = (dict(res.by_width), dict(res.skipped), _core.mem_peak(0))
    (free_by, _, _), (cap_by, cap_skipped, cap_peak) = runs[None], runs["16M"]
    assert free_by.get(2, 0) > 0, "precondition: width 2 found binaries, so mining rebuilt"
    assert cap_by == free_by and -1 not in cap_skipped, cap_skipped
    assert cap_peak <= 16 << 20


@pytest.mark.gpu
def test_mining_hands_back_every_byte_it_took():
    """Mining frees its probe engines before it returns, also when the device refuses the first
    one: GPAP then holds what it held before, with no collector run in between. A probe engine
    kept alive past the call (a reference cycle, a traceback) would hold its rows, and under a
    limit the next component would size against them."""
    import torch

    from gpap.mining import mine_entailed
    nv, clauses, pool = _binary_rich_formula()
    before = _settled_held()
    res = mine_entailed(nv, clauses, torch.device("cuda"), pool=pool, widths=(1, 2, 3),
                        max_tuples={1: 0, 2: 1_000_000, 3: 40_000}, seconds=60)
    assert res.by_width.get(2, 0) > 0, "precondition: width 2 found binaries, so mining rebuilt"
    assert _core.mem_held(0) == before
    _core.set_mem_limit(before + (64 << 10), 0)          # below the formula's own buffers
    with pytest.raises(MemoryError):
        mine_entailed(nv, clauses, torch.device("cuda"), pool=pool, widths=(1, 2),
                      max_tuples={1: 0, 2: 1000}, seconds=10)
    assert _core.mem_held(0) == before


@pytest.mark.gpu
def test_the_sampler_with_soundness_and_support_stays_under_a_tight_limit(tmp_path):
    """The sampler's own buffers after sizing (pair masks, the support columns, the soundness
    flags, the readout slices) fit under a limit the sampler fills: a run with the soundness
    check and a declared support finishes, sound, under it."""
    import json

    from gpap.cli import main as cli_main
    from tests.conftest import REPO_ROOT
    cnf = os.path.join(REPO_ROOT, "benchmarks", "sampling", "71.sk_3_65.cnf")
    out = tmp_path / "stats.json"
    cli_main(["sample", cnf, "-n", "300", "--portfolio", "b8", "--mine", "off", "--verify-sound",
              "--mem-limit", "4M", "--stats", str(out)])
    r = json.loads(out.read_text())
    assert r["engine"]["batch_bound"] == "limit"
    assert r["engine"]["mem_peak"] <= r["engine"]["mem_limit"] == 4 << 20
    assert r["unsound_rows"] == 0 and r["sound_rows_checked"] > 0

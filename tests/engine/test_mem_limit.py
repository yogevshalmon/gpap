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
    """Every test starts and ends with no limit: a limit is process state, and a shell's
    GPAP_MEM_LIMIT or GPAP_VRAM_BUDGET_MIB would size every batch here."""
    monkeypatch.delenv("GPAP_MEM_LIMIT", raising=False)
    monkeypatch.delenv("GPAP_VRAM_BUDGET_MIB", raising=False)
    yield
    # The CLI's --mem-limit and gpap.set_mem_limit write GPAP_MEM_LIMIT into the environment, for
    # child runs; monkeypatch restores only what it set, so a limit would outlive the test.
    os.environ.pop("GPAP_MEM_LIMIT", None)
    if _core.available():
        _core.set_mem_limit(None, 0)


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
    ("8", "needs its unit"), ("0G", "more than zero"), ("-1G", "not a memory limit"),
    ("1e3G", "unit"), ("0x10G", "unit"), ("101%", "at most 100%"), ("0%", "more than 0%"),
    ("8Q", "unit"), ("8G free", "after the unit"), ("", "not a memory limit"),
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


@pytest.mark.engine
def test_the_queue_runs_under_a_limit_that_holds_its_buffers():
    """Under a limit of the device plus the queue's launch buffers plus a few rows, flags-only
    checks and probe sweeps run on the queue (its launch count says so), and GPAP's peak never
    passes the limit: the queue's buffers are sized for, not just tolerated."""
    f = _formula(nv=400, nc=1200, seed=9)
    e = _core.Engine(f)
    nv = f.info.num_vars
    queue_buffers = 70 << 20                     # kQueueBufferBytes, cpp/src/gpap_internal.hpp
    _core.set_mem_limit(_settled_held() + queue_buffers + 16 * (_row_bytes(e) + 64) + 40 * nv
                        + (1 << 20), 0)
    _core.mem_reset_peak(0)
    limit = _core.mem_limit(0)
    lits, off = _flat(_sets(nv, 3000, 3))
    e.check_sets(lits, off, chunk=0)
    e.probe_combinations(None, 2, max_tuples=20_000).conflicts()
    assert e.queue_stats()[0] > 0, "the queue did not engage"
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

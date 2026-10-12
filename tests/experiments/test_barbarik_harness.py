"""The Barbarik certificate harness (`eval/uniformity/barbarik_gpap.py`): it covers every shipped
portfolio config, and its collector and shell shim behave."""
import pytest

from tests.conftest import REPO_ROOT


def test_every_default_portfolio_config_is_barbarik_certifiable():
    """Every portfolio config has an arm in the Barbarik harness. The pipeline delivers only the
    winning config's rows, so the shipped tool is certified only if every member is."""
    import ast
    import pathlib

    from gpap.sampling.pipeline import CIRCUIT_PORTFOLIO, DEFAULT_PORTFOLIO

    src = pathlib.Path(REPO_ROOT) / "eval/uniformity/barbarik_gpap.py"
    tree = ast.parse(src.read_text())
    configs = next(ast.literal_eval(node.value)
                   for node in ast.walk(tree)
                   if isinstance(node, ast.Assign)
                   and any(getattr(t, "id", None) == "CONFIGS" for t in node.targets))
    covered = {(bool(rv), b, str(la or "")) for rv, b, la in configs.values()}
    for cfg in (*CIRCUIT_PORTFOLIO, *DEFAULT_PORTFOLIO):
        key = (bool(cfg[0]), cfg[1], str(cfg[2] or ""))
        assert key in covered, (
            f"portfolio config {cfg.describe()} has no arm in barbarik_gpap.py CONFIGS "
            f"({sorted(configs)}) — the bandit could select a sampler nothing has certified")


def test_barbarik_harness_silences_upstream_rm_noise_and_nothing_else(tmp_path, capfd, monkeypatch):
    """barbarik deletes its kernel CNF and then shells out `rm *.nnf <that file>`. The shim runs
    that `rm` quietly, even when the file is already gone, and passes every other os.system call
    (the SPUR launch among them) through unchanged."""
    import os

    from eval.uniformity.barbarik_gpap import _QuietOS

    shim = _QuietOS(os)
    monkeypatch.chdir(tmp_path)
    nnf, kernel = tmp_path / "x.nnf", tmp_path / "k_t.cnf"
    nnf.write_text(""); kernel.write_text("p cnf 1 1\n1 0\n")
    assert shim.system(f"rm *.nnf {kernel}") == 0
    assert not nnf.exists() and not kernel.exists()
    assert shim.system(f"rm *.nnf {kernel}") == 0           # the upstream double-delete
    out, err = capfd.readouterr()
    assert out == "" and err == "", f"housekeeping must be silent, got {err!r}"
    # passthrough: exit status and output of anything else are untouched
    assert shim.system("exit 3") >> 8 == 3
    assert shim.system("echo passthrough") == 0
    assert "passthrough" in capfd.readouterr()[0]
    assert shim.path.join("a", "b") == os.path.join("a", "b")   # attribute forwarding


def test_barbarik_fill_gives_up_on_dry_attempts_not_dry_calls(monkeypatch):
    """`_FILL_GIVE_UP` counts consecutive attempts that produced nothing, not dry calls, so a
    low-yield sampler fills its order in both the `batch` and `iid` draws. A fully dry one is
    still abandoned at once."""
    import sys
    import types

    import numpy as np

    import eval.uniformity.barbarik_gpap as bg

    cols = [1, 2, 3]

    class Stub:
        """A low-yield sampler that sometimes returns nothing. It gives up after 3 consecutive
        empty chunks, which is harsher than the real `sample_bits`, so the collector's rule is
        tested against the worse case."""
        p_dry = 0.5

        def __init__(self):
            self.rng = np.random.default_rng(11)

        @classmethod
        def from_clauses(cls, *a, **k):
            return cls()

        # The signature MIRRORS the real one, `max_seconds` included: the harness now hands
        # every order an explicit wall budget instead of relying on the sampler's own
        # give-up rule, which is off by default. A stub that omits a parameter the caller
        # passes fails as a TypeError, which is the right outcome -- it is how this test
        # noticed the budget had been added.
        def sample_bits(self, n, config=None, max_chunks=None, max_seconds=None):
            rows, empty = [], 0
            while len(rows) < n and empty < 3:
                if self.rng.random() < Stub.p_dry:
                    empty += 1
                    continue
                empty = 0
                rows.extend([np.ones(len(cols), dtype=np.uint8)] * min(3, n - len(rows)))
            return (np.array(rows) if rows else None), cols

    class CNFStub:
        def __init__(self, from_file=None):
            self.nv, self.clauses = 4, [[1, 2]]

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=lambda s: s))
    monkeypatch.setitem(sys.modules, "pysat", types.ModuleType("pysat"))
    pf = types.ModuleType("pysat.formula")
    pf.CNF = CNFStub
    monkeypatch.setitem(sys.modules, "pysat.formula", pf)
    sm = types.ModuleType("gpap.sampling.sampler")
    sm.GPUSampler = Stub
    monkeypatch.setitem(sys.modules, "gpap.sampling.sampler", sm)

    def fills(draw, p_dry, trials=12):
        Stub.p_dry = p_dry
        ok = 0
        for t in range(trials):
            try:
                got = bg._gpap_solutions("f.cnf", 50, cols, t, "cpu", (True, 8, False), 4096,
                                         draw=draw)
                ok += len(got) == 50
            except RuntimeError:
                pass
        return ok / trials

    for p_dry in (0.4, 0.6):                    # low yield, not dead
        assert fills("batch", p_dry) == 1.0, "the batch arm must fill an order it can fill"
        assert fills("iid", p_dry) == 1.0, (
            f"the iid arm gave up at p_dry={p_dry} while the batch arm filled — the collector's "
            f"stopping rule, not the sampler's yield")
    # a fully dry instance is abandoned immediately, in both draws
    Stub.p_dry = 1.0
    for draw in ("batch", "iid"):
        with pytest.raises(RuntimeError, match="produced 0/50"):
            bg._gpap_solutions("f.cnf", 50, cols, 1, "cpu", (True, 8, False), 4096, draw=draw)


@pytest.mark.parametrize("draw", ["batch", "iid"])
def test_one_order_is_bounded_by_the_order_budget_not_by_the_attempt_count(monkeypatch, draw):
    """`--order-seconds` bounds ONE of barbarik's orders, so `72 x it` bounds a certificate.

    It has to be a deadline shared by every call the collector makes, not an argument handed to
    each: per-call it would bound an order at 24x (batch) to 1,200x (iid) the stated number. The
    stub is a trickling instance -- one row, then dry, the shape a Barbarik kernel produces from
    a rejection sampler -- because that is the case that walks the whole attempt budget.
    """
    import math
    import sys
    import types

    import numpy as np

    import eval.uniformity.barbarik_gpap as bg

    cols, budget, per_call = [1, 2, 3], 30.0, 4.0
    clock, seen = [1000.0], []

    class Trickle:
        """Yields one row per call and burns `per_call` seconds doing it: never dry, so the
        collector's `_FILL_GIVE_UP` never fires and only the deadline can stop it."""

        @classmethod
        def from_clauses(cls, *a, **k):
            return cls()

        def sample_bits(self, n, config=None, max_chunks=None, max_seconds=None):
            seen.append(max_seconds)
            clock[0] += per_call
            return np.ones((1, len(cols)), dtype=np.uint8), cols

    class CNFStub:
        def __init__(self, from_file=None):
            self.nv, self.clauses = 4, [[1, 2]]

    monkeypatch.setattr(bg, "time", types.SimpleNamespace(monotonic=lambda: clock[0],
                                                          perf_counter=lambda: clock[0]))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=lambda s: s))
    monkeypatch.setitem(sys.modules, "pysat", types.ModuleType("pysat"))
    pf = types.ModuleType("pysat.formula")
    pf.CNF = CNFStub
    monkeypatch.setitem(sys.modules, "pysat.formula", pf)
    sm = types.ModuleType("gpap.sampling.sampler")
    sm.GPUSampler = Trickle
    monkeypatch.setitem(sys.modules, "gpap.sampling.sampler", sm)

    with pytest.raises(RuntimeError, match="order budget was spent"):
        bg._gpap_solutions("f.cnf", 50, cols, 1, "cpu", (True, 8, False), 4096, draw=draw,
                           order_seconds=budget)
    # The order spent its budget ONCE, not a multiple of it: the last call is the one that
    # starts with time left and runs past the deadline. `_FILL_ATTEMPTS` alone would have
    # allowed 24 calls in the batch arm (96 s) and up to 1,200 in the iid arm.
    assert len(seen) == math.ceil(budget / per_call), \
        f"{len(seen)} calls x {per_call}s against a {budget}s order budget"
    # ...and no single call was ever handed more than the order had left.
    assert seen == [budget - i * per_call for i in range(len(seen))], seen


@pytest.mark.parametrize("marker", [True, False], ids=["budget-stop", "real-crash"])
def test_a_cmsgen_maxtime_exit_is_a_budget_stop_not_a_harness_failure(tmp_path, marker):
    """`--maxtime` makes CMSGen exit through its NORMAL path with the DIMACS "unknown" code (15,
    measured), having written the samples it managed. The arm must read them.

    Measured: a `--maxtime` stop on `c6288-c1-random` came back as `RuntimeError: cmsgen failed`
    -- a harness error where the truth is a short count on a spent budget. It is the difference
    between reading the samples CMSGen wrote and discarding them with the kill.
    `gpap.benchmark.cmsgen_run` ignores the return code for this reason; here it is still
    checked, so the patch's own marker separates a budget stop from a crash.
    """
    import stat

    from eval.uniformity.barbarik_gpap import _cmsgen_solutions

    said = "echo 'c gpap-maxtime reached after 30.0010s'" if marker else "echo 'c oh no'"
    fake = tmp_path / "cmsgen"
    # --help must advertise --maxtime or the arm never passes a deadline (capability-probed).
    fake.write_text(
        "#!/bin/sh\n"
        'case "$1" in --help) echo "--nowrite --statsevery --maxtime"; exit 0;; esac\n'
        'for a in "$@"; do case "$prev" in --samplefile) out="$a";; esac; prev="$a"; done\n'
        'printf "1 -2 3 0\\n" > "$out"\n'
        f"{said}\n"
        "exit 15\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    cnf = tmp_path / "toy.cnf"
    cnf.write_text("p cnf 3 1\n1 0\n")

    # one sample written, one asked for: the budget-stop path must SUCCEED on what it read
    if marker:
        assert _cmsgen_solutions(str(cnf), 1, [1, 2, 3], 0, str(fake),
                                 order_seconds=30.0) == [[1, -2, 3]]
        # ...and two asked for is a short count, reported with the budget, never as a failure
        with pytest.raises(RuntimeError) as err:
            _cmsgen_solutions(str(cnf), 2, [1, 2, 3], 0, str(fake), order_seconds=30.0)
        assert "produced 1/2" in str(err.value) and "30s budget" in str(err.value), err.value
        assert "failed" not in str(err.value), str(err.value)
    else:
        # the same exit code WITHOUT the patch's marker is still a crash
        with pytest.raises(RuntimeError, match="cmsgen failed"):
            _cmsgen_solutions(str(cnf), 1, [1, 2, 3], 0, str(fake), order_seconds=30.0)

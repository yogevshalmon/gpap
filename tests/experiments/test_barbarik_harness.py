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
        def sample_bits(self, n, config=None, max_chunks=400, max_seconds=None):
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

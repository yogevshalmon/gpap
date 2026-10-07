"""Which propagate kernel runs is the engine's choice, not the application's.

Scalar and wave reach the same fixpoints (tests/engine/test_propagation.py), so the choice can
change only speed, and every consumer gets it by default. `GPAP_KERNEL=scalar|wave` or an
explicit mode pins it.
"""
import pytest

from gpap.engine.bcp_engine import BCPEngine
from gpap.engine.formula import CnfFormula

from tests.support import DEV

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _no_kernel_override(monkeypatch):
    """A `GPAP_KERNEL` in the caller's shell would pin every engine below."""
    monkeypatch.delenv("GPAP_KERNEL", raising=False)


def _engine(mode=None, nv=6):
    f = CnfFormula(DEV)
    f.init_formula(nv, [[1, 2], [-1, 3], [-2, -3], [4, 5], [-4, 6], [-5, -6]])
    return BCPEngine(f, mode=mode)


def test_auto_is_the_default():
    eng = _engine()
    eng.reset(64)                                  # tiny batch -> wave's regime
    assert eng.mode == "wave"


def test_auto_dispatches_on_the_clause_count():
    """The deciding variable is M, not the batch.

    A batch-only rule (`batch < 30 * resident_warps`) only ever worked as a proxy for M, and
    once K grew, mid-size formulas sat at the batch cap and were all classified scalar. A
    mid-size formula at a huge batch must come back "wave"."""
    from gpap.engine.calibrate import dispatch_mode, resident_warps
    assert resident_warps(DEV) > 0
    huge = 262_144                                   # the batch cap
    assert dispatch_mode(50_000, huge, DEV) == "wave", "big M at a huge batch must be wave"
    assert dispatch_mode(6, huge, DEV) == "scalar", "a 6-clause row cannot fill a warp"
    assert dispatch_mode(6, 64, DEV) == "wave", "...unless the batch cannot fill the device"


def test_auto_re_dispatches_per_reset():
    """Auto is applied on every reset, so changing the batch can still change the kernel on a
    formula small enough for the batch term to bind."""
    from gpap.engine.calibrate import resident_warps
    big = resident_warps(DEV) + 1                   # past what scalar can hold in flight
    eng = _engine()                                 # M = 6
    eng.reset(64)
    assert eng.mode == "wave"
    eng.reset(big)
    assert eng.mode == "scalar", "auto must re-dispatch when the batch changes"


@pytest.mark.parametrize("pinned", ["scalar", "wave"])
def test_explicit_mode_is_never_overridden(pinned):
    """A measurement pins the kernel, and reset() must not quietly change it back."""
    eng = _engine(mode=pinned)
    for k in (64, 200_000):
        eng.reset(k)
        assert eng.mode == pinned


def test_set_mode_disables_auto():
    eng = _engine()
    eng.reset(64)
    eng.set_mode("scalar")
    eng.reset(64)                                  # would be wave under auto
    assert eng.mode == "scalar"


def test_env_var_pins(monkeypatch):
    monkeypatch.setenv("GPAP_KERNEL", "scalar")
    eng = _engine()
    eng.reset(64)
    assert eng.mode == "scalar"


def test_auto_dispatch_reports_whether_a_kernel_is_pinned(monkeypatch):
    """The flag a caller consults before pinning a mode of its own (the sampler does): it
    must see BOTH pin channels — the env override and an explicit set_mode."""
    eng = _engine()
    assert eng.auto_dispatch
    eng.set_mode("wave")
    assert not eng.auto_dispatch
    eng.set_mode("auto")
    assert eng.auto_dispatch
    monkeypatch.setenv("GPAP_KERNEL", "scalar")
    assert not _engine().auto_dispatch

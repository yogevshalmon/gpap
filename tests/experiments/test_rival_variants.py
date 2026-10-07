"""`prepare_variant` (eval/rivals/rival_common.py): the circuit variant a rival is run on."""
import os

import pytest

from tests.conftest import SEQ_DIR


def test_prepare_variant_raw_is_the_untouched_source(tmp_path):
    from eval.rivals.rival_common import prepare_variant
    src = os.path.join(SEQ_DIR, "b03.bench")
    path, info = prepare_variant(src, "raw", str(tmp_path), cycles=5, seed=5)
    assert path == src and info == {"variant": "raw"}


def test_prepare_variant_rejects_unknown(tmp_path):
    from eval.rivals.rival_common import prepare_variant
    with pytest.raises(ValueError):
        prepare_variant("x.bench", "aigier", str(tmp_path), cycles=5, seed=5)


def test_prepare_variant_falls_back_to_raw_when_abc_is_missing(tmp_path, monkeypatch):
    """With no abc binary the raw path comes back, tagged with the reason, so a rival is never run
    on an unverified circuit."""
    import gpap.circuits.abc_backend as ab
    from eval.rivals.rival_common import prepare_variant, variant_note
    monkeypatch.setattr(ab, "abc_available", lambda *a, **k: False)
    src = os.path.join(SEQ_DIR, "b03.bench")
    path, info = prepare_variant(src, "abc", str(tmp_path), cycles=5, seed=5)
    assert path == src and "no abc binary" in info["error"]
    assert "-> raw" in variant_note(info)

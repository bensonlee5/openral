"""Hardware-free tests for the SmolVLA TensorRT runtime module."""

from __future__ import annotations

import pytest
from openral_core.exceptions import ROSConfigError
from openral_rskill.smolvla_trt import (
    _device_index,
    _slug,
    attach_trt_sample_actions,
    maybe_attach_trt_from_env,
)


def test_slug_is_filesystem_safe() -> None:
    assert _slug("sapanostic/so_101_smolvla_pen_placement") == (
        "sapanostic--so_101_smolvla_pen_placement"
    )
    assert "/" not in _slug("a/b:c d")


def test_device_index_parses_torch_device_strings() -> None:
    assert _device_index("cuda:0") == 0
    assert _device_index("cuda:1") == 1
    assert _device_index("cuda") == 0
    assert _device_index("cpu") == 0


def test_fp16_precision_is_rejected_before_any_heavy_work() -> None:
    """fp16 is numerically unusable (measured ~60% action error) — hard error."""
    with pytest.raises(ROSConfigError, match="fp16"):
        attach_trt_sample_actions(object(), "any/repo", precision="fp16")


def test_env_gate_off_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the env knob unset, no attach is attempted (returns False)."""
    monkeypatch.delenv("OPENRAL_SMOLVLA_TRT", raising=False)
    # object() has no policy surface — proves nothing heavy runs when off.
    assert maybe_attach_trt_from_env(object(), "any/repo", device="cuda:0") is False


def test_env_gate_on_reaches_attach(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the knob on, the shared seam routes into attach (fp16 rejected here)."""
    monkeypatch.setenv("OPENRAL_SMOLVLA_TRT", "1")
    monkeypatch.setenv("OPENRAL_SMOLVLA_TRT_PRECISION", "fp16")
    with pytest.raises(ROSConfigError, match="fp16"):
        maybe_attach_trt_from_env(object(), "any/repo", device="cuda:0")

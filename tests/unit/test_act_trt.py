"""Hardware-free tests for the ACT ONNX/TensorRT runtime module."""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.exceptions import ROSConfigError
from openral_rskill.act_trt import (
    ensure_act_onnx,
    maybe_attach_act_trt_from_env,
)


def test_env_gate_off_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the env knob unset, no attach is attempted (returns False)."""
    monkeypatch.delenv("OPENRAL_ACT_TRT", raising=False)
    # object() has no policy surface — proves nothing heavy runs when off.
    assert maybe_attach_act_trt_from_env(object(), "any/repo", onnx_uri=None) is False


def test_missing_onnx_uri_is_hard_error() -> None:
    """No ONNX to resolve → loud config error, never a silent local export."""
    with pytest.raises(ROSConfigError, match="export_act_onnx"):
        ensure_act_onnx("any/repo", onnx_uri=None)


def test_malformed_hf_uri_is_rejected() -> None:
    with pytest.raises(ROSConfigError, match="malformed"):
        ensure_act_onnx("any/repo", onnx_uri="hf://nofilehere")


def test_non_hf_non_file_uri_is_rejected() -> None:
    with pytest.raises(ROSConfigError, match="unresolvable"):
        ensure_act_onnx("any/repo", onnx_uri="s3://bucket/model.onnx")


def test_local_file_uri_resolves_directly(tmp_path: Path) -> None:
    """A local path that exists is used as-is (dev / test loop)."""
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"\x00")  # content irrelevant to path resolution
    assert ensure_act_onnx("any/repo", onnx_uri=str(onnx)) == onnx


def test_env_gate_on_reaches_attach(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the knob on, the seam routes into attach (unresolvable onnx raises)."""
    monkeypatch.setenv("OPENRAL_ACT_TRT", "1")
    with pytest.raises(ROSConfigError, match="export_act_onnx"):
        maybe_attach_act_trt_from_env(object(), "any/repo", onnx_uri=None)

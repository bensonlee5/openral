"""Hardware-free tests for the SmolVLA TensorRT runtime module."""

from __future__ import annotations

import pytest
from openral_core.exceptions import ROSConfigError
from openral_rskill.smolvla_trt import _slug, attach_trt_sample_actions


def test_slug_is_filesystem_safe() -> None:
    assert _slug("sapanostic/so_101_smolvla_pen_placement") == (
        "sapanostic--so_101_smolvla_pen_placement"
    )
    assert "/" not in _slug("a/b:c d")


def test_fp16_precision_is_rejected_before_any_heavy_work() -> None:
    """fp16 is numerically unusable (measured ~60% action error) — hard error."""
    with pytest.raises(ROSConfigError, match="fp16"):
        attach_trt_sample_actions(object(), "any/repo", precision="fp16")

"""Units heuristic: infer degrees vs radians from normalizer stats.

``_detect_joint_units_are_degrees`` walks the loaded preprocessor's
normalizer stats. Two stats layouts exist across lerobot processor
versions (flat ``"observation.state.q99"`` keys vs nested
``"observation.state" → {"q99"/"max"/...}``) and two stat families
(quantile q99 — pi05-style — vs MEAN_STD mean/std/min/max — the SmolVLA
SO-101 checkpoints). The original q99-only lookup silently defaulted a
degrees-trained MEAN_STD checkpoint to radians, which feeds the policy
radian state against degree normalizer stats on `openral deploy run`.

Real numpy stats arrays throughout (values lifted from the
sapanostic/so_101_smolvla_pen_placement normalizer: state spans ±100).
"""

from __future__ import annotations

import numpy as np
from openral_rskill_ros.rskill_runner_node import _detect_joint_units_are_degrees


class _StatsStep:
    """Carrier for a normalizer step's ``stats`` dict (duck-typed pipeline step)."""

    def __init__(self, stats: dict[str, object]) -> None:
        self.stats = stats


class _Pipeline:
    def __init__(self, *steps: object) -> None:
        self.steps = list(steps)


class _Adapter:
    def __init__(self, stats: dict[str, object]) -> None:
        self._preprocessor = _Pipeline(_StatsStep(stats))


def test_nested_mean_std_degrees_detected() -> None:
    """MEAN_STD SmolVLA SO-101 stats (±100 span, no q99) → degrees."""
    stats = {
        "observation.state": {
            "count": 60277.0,
            "mean": np.array([-13.6, -30.0, 36.4, 45.5, -28.0, 8.1]),
            "std": np.array([23.6, 46.5, 40.9, 36.7, 26.9, 8.1]),
            "min": np.array([-86.2, -99.7, -96.7, -91.3, -88.0, 0.6]),
            "max": np.array([83.2, 71.2, 100.0, 98.9, 64.6, 56.6]),
        }
    }
    assert _detect_joint_units_are_degrees(_Adapter(stats)) is True


def test_nested_radian_stats_stay_radians() -> None:
    """A radians-trained checkpoint (all stats below π) → radians."""
    stats = {
        "observation.state": {
            "count": 1000.0,
            "mean": np.array([0.1, -0.4, 0.9, 1.2, -0.3, 0.5]),
            "std": np.array([0.4, 0.7, 0.6, 0.5, 0.4, 0.2]),
            "min": np.array([-1.5, -1.7, -1.6, -1.5, -1.5, 0.0]),
            "max": np.array([1.5, 1.2, 2.4, 2.4, 1.5, 1.0]),
        }
    }
    assert _detect_joint_units_are_degrees(_Adapter(stats)) is False


def test_flat_q99_degrees_detected() -> None:
    """The original flat-layout quantile path still detects degrees."""
    stats = {"observation.state.q99": np.array([80.0, 95.0, 99.0, 90.0, 60.0, 55.0])}
    assert _detect_joint_units_are_degrees(_Adapter(stats)) is True


def test_no_state_stats_defaults_to_radians() -> None:
    """No usable stats anywhere → the safe radians default."""
    assert _detect_joint_units_are_degrees(_Adapter({"action": {"count": 1.0}})) is False

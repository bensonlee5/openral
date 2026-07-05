"""Real-mode camera leg: open every deploy-bound sensor and publish to ROS.

``openral deploy run`` composes the same launch graph as ``deploy sim``,
but where the sim HAL publishes camera frames itself (``SimSensorBridge``
renders MuJoCo cameras onto ``/openral/cameras/<name>/image``), real
hardware has no camera publisher. The physical ``/dev/video*`` devices are
described by :attr:`~openral_core.SensorSpec.deploy_binding` — on the robot
manifest for robot-mounted cameras (wrist / head) and on
:attr:`~openral_core.DeployScene.sensors` for workcell-mounted ones
(overhead / front).

This module is that leg. :func:`open_deploy_sensor_readers` builds one
reader per bound spec via the runner's ``SENSOR_BACKEND_REGISTRY`` and
guarantees every camera ends up on the WorldState subscription topic
``<topic_prefix>/<name>/image``:

* ``gstreamer`` backend — the reader's built-in ROS tee publishes
  directly from the pipeline (``pipeline._build_ros_tee_branch``).
* ``opencv_thread`` (and any backend without a native tee) — the open
  reader is wrapped in a polling
  :class:`~openral_sensors.ros_publisher.SensorRosPublisher`.

Publishers use the CLAUDE.md §2 sensor-stream QoS (BEST_EFFORT); the
WorldState image subscription requests BEST_EFFORT so both match.

The caller owns teardown: :meth:`SensorLeg.close` stops publishers and
closes readers idempotently. ``runtime_node`` wires this in when its
``deploy_config`` parameter is set (real deploys only — sim keeps the
HAL bridge as its single camera source).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterable

    from openral_core import SensorSpec

__all__ = ["SensorLeg", "merge_deploy_sensors", "open_deploy_sensor_readers"]

log = structlog.get_logger(__name__)

#: WorldState's camera subscription prefix (`<prefix>/<name>/image`).
DEFAULT_TOPIC_PREFIX = "/openral/cameras"

#: Publish cadence when the binding's backend_params carry no fps.
#: Matches the WorldStateAggregator staleness-gate expectation (10 Hz cameras).
_DEFAULT_PUBLISH_RATE_HZ = 10.0


class _Closeable(Protocol):
    """Structural type for anything with a no-arg close/stop."""

    def close(self) -> None: ...  # pragma: no cover — Protocol


@dataclass
class SensorLeg:
    """Open readers + started publishers for one deploy session.

    Attributes:
        readers: Open :class:`SensorReader` instances, one per deploy-bound
            :class:`SensorSpec` (gstreamer readers publish via their
            internal ROS tee).
        publishers: Started :class:`SensorRosPublisher` pumps for the
            readers without a native ROS tee. Parallel list, NOT
            index-aligned with ``readers``.
    """

    readers: list[object] = field(default_factory=list)
    publishers: list[object] = field(default_factory=list)

    def close(self) -> None:
        """Stop publishers first (they poll the readers), then close readers.

        Idempotent and exception-safe: a failing stop/close never blocks
        the remaining teardown — deploy shutdown must always reach the
        HAL/lifecycle teardown behind it.
        """
        for publisher in self.publishers:
            try:
                publisher.stop()  # type: ignore[attr-defined]  # reason: duck-typed pump
            except Exception as exc:  # reason: teardown must not raise
                log.warning("sensor_leg.publisher_stop_failed", error=str(exc))
        self.publishers.clear()
        for reader in self.readers:
            try:
                reader.close()  # type: ignore[attr-defined]  # reason: SensorReader protocol
            except Exception as exc:  # reason: teardown must not raise
                log.warning("sensor_leg.reader_close_failed", error=str(exc))
        self.readers.clear()


def merge_deploy_sensors(
    manifest_sensors: Iterable[SensorSpec],
    scene_sensors: Iterable[SensorSpec],
) -> list[SensorSpec]:
    """Robot-manifest sensors ∪ ``DeployScene.sensors``, scene wins on name collision.

    A scene entry named like a manifest sensor is that sensor's deploy-time
    binding (ADR-0078 amendment) — keeping both would double-open the device
    and publish the same topic twice.
    """
    scene = list(scene_sensors)
    scene_names = {s.name for s in scene}
    return [s for s in manifest_sensors if s.name not in scene_names] + scene


def _publish_rate_hz(spec: SensorSpec) -> float:
    """The ROS publish cadence for ``spec`` — binding fps, else spec rate, else 10 Hz."""
    assert spec.deploy_binding is not None  # reason: caller filters on binding
    fps = spec.deploy_binding.backend_params.get("fps")
    if isinstance(fps, (int, float)) and fps > 0:
        return float(fps)
    if spec.rate_hz > 0:
        return float(spec.rate_hz)
    return _DEFAULT_PUBLISH_RATE_HZ


def open_deploy_sensor_readers(
    sensors: Iterable[SensorSpec],
    *,
    topic_prefix: str = DEFAULT_TOPIC_PREFIX,
) -> SensorLeg:
    """Open every deploy-bound sensor in ``sensors`` and publish each onto ROS.

    Args:
        sensors: Robot-manifest sensors plus :attr:`DeployScene.sensors`
            (the caller concatenates). Specs without a
            :attr:`~openral_core.SensorSpec.deploy_binding` are skipped —
            committed reference manifests leave the binding unset.
        topic_prefix: WorldState's ``camera_topic_prefix``. The final
            topic is ``<topic_prefix>/<spec.name>/image``.

    Returns:
        A :class:`SensorLeg` holding the open readers + started
        publishers. Call :meth:`SensorLeg.close` on shutdown.

    Raises:
        ROSConfigError: A binding names an unknown backend, or a backend's
            optional dependency (PyGObject / opencv-python) is missing.

    Example:
        >>> from openral_core import DeployScene, RobotDescription
        >>> desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")  # doctest: +SKIP
        >>> scene = DeployScene.from_yaml("scenes/deploy/so101_bench.yaml")  # doctest: +SKIP
        >>> leg = open_deploy_sensor_readers([*desc.sensors, *scene.sensors])  # doctest: +SKIP
        >>> try:  # doctest: +SKIP
        ...     ...  # spin the graph
        ... finally:
        ...     leg.close()
    """
    # Deferred imports — openral_runner pulls torch-adjacent modules; keep
    # this module importable for AST/shape tests on minimal hosts.
    from openral_core import SensorReaderBackend, SensorReaderConfig
    from openral_runner.factory import SENSOR_BACKEND_REGISTRY

    leg = SensorLeg()
    try:
        for spec in sensors:
            binding = spec.deploy_binding
            if binding is None:
                continue
            topic = f"{topic_prefix}/{spec.name}/image"
            if binding.backend == SensorReaderBackend.GSTREAMER:
                # Native in-pipeline tee: force it on so the frames reach ROS.
                cfg = SensorReaderConfig(
                    sensor_id=spec.name,
                    backend=binding.backend,
                    backend_params=binding.backend_params,
                    max_age_ms=binding.max_age_ms,
                    publish_to_ros=True,
                    publish_topic=topic,
                    publish_rate_hz=_publish_rate_hz(spec),
                )
                reader = SENSOR_BACKEND_REGISTRY[cfg.backend.value](cfg)
                reader.open()
                leg.readers.append(reader)
            else:
                # No native tee (opencv_thread): open the reader bare and
                # attach the polling ROS publisher pump.
                from openral_sensors.ros_publisher import SensorRosPublisher

                cfg = SensorReaderConfig(
                    sensor_id=spec.name,
                    backend=binding.backend,
                    backend_params=binding.backend_params,
                    max_age_ms=binding.max_age_ms,
                )
                reader = SENSOR_BACKEND_REGISTRY[cfg.backend.value](cfg)
                reader.open()
                leg.readers.append(reader)
                publisher = SensorRosPublisher(
                    reader=reader,
                    topic=topic,
                    rate_hz=_publish_rate_hz(spec),
                )
                publisher.start()
                leg.publishers.append(publisher)
            log.info(
                "sensor_leg.camera_open",
                sensor_id=spec.name,
                backend=binding.backend.value,
                topic=topic,
            )
    except Exception:
        # Half-open leg → close what we already opened before re-raising;
        # a failed camera must not leak a v4l2 handle past the error.
        leg.close()
        raise
    return leg

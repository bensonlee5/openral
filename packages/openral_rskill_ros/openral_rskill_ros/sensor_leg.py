"""Real-mode camera leg: open the deploy config's sensor readers and publish to ROS.

``openral deploy run`` composes the same launch graph as ``deploy sim``,
but where the sim HAL publishes camera frames itself (``SimSensorBridge``
renders MuJoCo cameras onto ``/openral/cameras/<name>/image``), real
hardware has no camera publisher: the physical ``/dev/video*`` devices
are described in the :class:`~openral_core.RobotEnvironment`'s
``sensors:`` list (one :class:`~openral_core.SensorReaderConfig` per
camera, scaffolded by ``openral detect --deployment``) but nothing
opened them after the runner-factory removal.

This module is that leg. :func:`open_deploy_sensor_readers` builds one
reader per config via the runner's ``SENSOR_BACKEND_REGISTRY`` and
guarantees every camera ends up on the WorldState subscription topic
``<topic_prefix>/<sensor_id>/image``:

* ``gstreamer`` backend — the reader's built-in ROS tee publishes
  directly from the pipeline (``pipeline._build_ros_tee_branch``); the
  config is re-issued with ``publish_to_ros=True`` when the scaffold
  left it unset.
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
    from openral_core import RobotEnvironment, SensorReaderConfig

__all__ = ["SensorLeg", "open_deploy_sensor_readers"]

log = structlog.get_logger(__name__)

#: WorldState's camera subscription prefix (`<prefix>/<name>/image`).
DEFAULT_TOPIC_PREFIX = "/openral/cameras"

#: Publish cadence when neither the config nor its backend_params carry one.
#: Matches the WorldStateAggregator staleness-gate expectation (10 Hz cameras).
_DEFAULT_PUBLISH_RATE_HZ = 10.0


class _Closeable(Protocol):
    """Structural type for anything with a no-arg close/stop."""

    def close(self) -> None: ...  # pragma: no cover — Protocol


@dataclass
class SensorLeg:
    """Open readers + started publishers for one deploy session.

    Attributes:
        readers: Open :class:`SensorReader` instances, one per
            ``RobotEnvironment.sensors`` entry (gstreamer readers
            publish via their internal ROS tee).
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


def _publish_rate_hz(cfg: SensorReaderConfig) -> float:
    """The ROS publish cadence for ``cfg`` — explicit, else backend fps, else 10 Hz."""
    if cfg.publish_rate_hz is not None:
        return float(cfg.publish_rate_hz)
    fps = cfg.backend_params.get("fps")
    if isinstance(fps, (int, float)) and fps > 0:
        return float(fps)
    return _DEFAULT_PUBLISH_RATE_HZ


def _with_ros_tee(cfg: SensorReaderConfig, topic: str) -> SensorReaderConfig:
    """A copy of ``cfg`` with the ROS tee forced onto ``topic``.

    ``model_copy`` skips ``model_post_init`` cross-field checks, so build a
    fresh instance through the validator instead.
    """
    updates = dict(
        publish_to_ros=True,
        publish_topic=topic,
        publish_rate_hz=_publish_rate_hz(cfg),
    )
    return type(cfg)(**{**cfg.model_dump(), **updates})


def open_deploy_sensor_readers(
    env: RobotEnvironment,
    *,
    topic_prefix: str = DEFAULT_TOPIC_PREFIX,
) -> SensorLeg:
    """Open every ``env.sensors`` reader and publish each onto ROS.

    Args:
        env: The deploy config (``openral deploy run --config``). One
            reader per :class:`SensorReaderConfig`; ``sensor_id`` names
            the WorldState topic segment and MUST match the robot
            manifest's :attr:`SensorSpec.name` (``top`` / ``wrist`` on
            the SO-101) for the aggregator to pick the frames up.
        topic_prefix: WorldState's ``camera_topic_prefix``. The final
            topic is ``<topic_prefix>/<sensor_id>/image``.

    Returns:
        A :class:`SensorLeg` holding the open readers + started
        publishers. Call :meth:`SensorLeg.close` on shutdown.

    Raises:
        ROSConfigError: A config names an unknown backend, or a backend's
            optional dependency (PyGObject / opencv-python) is missing.

    Example:
        >>> from openral_core import RobotEnvironment
        >>> env = RobotEnvironment.from_yaml("deployments/so101.yaml")  # doctest: +SKIP
        >>> leg = open_deploy_sensor_readers(env)  # doctest: +SKIP
        >>> try:  # doctest: +SKIP
        ...     ...  # spin the graph
        ... finally:
        ...     leg.close()
    """
    # Deferred imports — openral_runner pulls torch-adjacent modules; keep
    # this module importable for AST/shape tests on minimal hosts.
    from openral_core import SensorReaderBackend
    from openral_runner.factory import SENSOR_BACKEND_REGISTRY

    leg = SensorLeg()
    try:
        for cfg in env.sensors:
            topic = cfg.publish_topic or f"{topic_prefix}/{cfg.sensor_id}/image"
            if cfg.backend == SensorReaderBackend.GSTREAMER:
                # Native in-pipeline tee: force it on so the frames reach ROS.
                reader_cfg = cfg if cfg.publish_to_ros else _with_ros_tee(cfg, topic)
                reader = SENSOR_BACKEND_REGISTRY[reader_cfg.backend.value](reader_cfg)
                reader.open()
                leg.readers.append(reader)
            else:
                # No native tee (opencv_thread): open the reader bare and
                # attach the polling ROS publisher pump.
                from openral_sensors.ros_publisher import SensorRosPublisher

                reader = SENSOR_BACKEND_REGISTRY[cfg.backend.value](cfg)
                reader.open()
                leg.readers.append(reader)
                publisher = SensorRosPublisher(
                    reader=reader,
                    topic=topic,
                    rate_hz=_publish_rate_hz(cfg),
                )
                publisher.start()
                leg.publishers.append(publisher)
            log.info(
                "sensor_leg.camera_open",
                sensor_id=cfg.sensor_id,
                backend=cfg.backend.value,
                topic=topic,
            )
    except Exception:
        # Half-open leg → close what we already opened before re-raising;
        # a failed camera must not leak a v4l2 handle past the error.
        leg.close()
        raise
    return leg

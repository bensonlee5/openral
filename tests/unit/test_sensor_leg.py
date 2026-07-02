"""Unit tests for the real-mode camera leg (``openral_rskill_ros.sensor_leg``).

The leg is what `openral deploy run` uses to open the deploy config's
``sensors:`` readers and publish every camera onto the WorldState image
topics (``/openral/cameras/<sensor_id>/image``). Real components per
CLAUDE.md §1.11: the GStreamer test uses a real ``videotestsrc``
pipeline (headless, no camera hardware); ROS-touching paths skip when
``rclpy`` isn't importable (CI runners without a sourced ROS install).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import SensorReaderConfig
from openral_rskill_ros.sensor_leg import (
    SensorLeg,
    _publish_rate_hz,
    _with_ros_tee,
)


def _write_env_yaml(tmp_path: Path, sensors_yaml: str) -> Path:
    """A minimal, schema-valid RobotEnvironment YAML with the given sensors."""
    out = tmp_path / "robot_env.yaml"
    out.write_text(
        "robot_id: so101_follower\n"
        "hal:\n"
        "  adapter: so101_follower\n"
        "  transport:\n"
        "    port: /dev/ttyACM0\n"
        f"sensors:\n{sensors_yaml}"
        "task:\n"
        "  id: deploy/sensor-leg-test\n"
        "  scene_id: deploy/sensor-leg-test\n"
        '  instruction: "sensor leg unit test"\n'
        "  max_steps: 30\n"
        "rate_hz: 30.0\n"
    )
    return out


def test_with_ros_tee_forces_topic_and_validates() -> None:
    """The tee-forcing copy re-runs cross-field validation with the topic set."""
    cfg = SensorReaderConfig(
        sensor_id="wrist",
        backend="gstreamer",
        backend_params={"source": "testsrc", "fps": 15},
    )
    teed = _with_ros_tee(cfg, "/openral/cameras/wrist/image")
    assert teed.publish_to_ros is True
    assert teed.publish_topic == "/openral/cameras/wrist/image"
    assert teed.publish_rate_hz == 15.0  # falls back to backend fps
    # The original is untouched (a copy, not a mutation).
    assert cfg.publish_to_ros is False


def test_publish_rate_explicit_wins_over_fps() -> None:
    """An explicit publish_rate_hz beats the backend fps and the 10 Hz default."""
    cfg = SensorReaderConfig(
        sensor_id="top",
        backend="gstreamer",
        backend_params={"source": "testsrc", "fps": 30},
        publish_to_ros=True,
        publish_topic="/openral/cameras/top/image",
        publish_rate_hz=5.0,
    )
    assert _publish_rate_hz(cfg) == 5.0


def test_publish_rate_defaults_to_10hz_without_fps() -> None:
    """No explicit rate and no fps → the WorldState-friendly 10 Hz default."""
    cfg = SensorReaderConfig(sensor_id="top", backend="opencv_thread")
    assert _publish_rate_hz(cfg) == 10.0


def test_empty_leg_close_is_idempotent() -> None:
    """close() on an empty/already-closed leg never raises."""
    leg = SensorLeg()
    leg.close()
    leg.close()
    assert leg.readers == []
    assert leg.publishers == []


def test_gstreamer_testsrc_leg_publishes_frames(tmp_path: Path) -> None:
    """Full leg over a real videotestsrc pipeline: open → ROS tee → frame → close.

    Skips without PyGObject (gi) or rclpy — the same gates the production
    factory enforces. Runs in a SUBPROCESS with the production import
    order (GStreamer backend → Gst.init → rclpy), because ``rclpy.Node()``
    segfaults inside Fast-DDS thread setup when rclpy was imported before
    ``Gst.init()`` (see ``openral_runner/backends/gstreamer/reader.py``
    PR I/8 note) — and the pytest process may already have rclpy loaded
    from an earlier test.
    """
    pytest.importorskip("gi", reason="PyGObject (gstreamer extra) not installed")
    pytest.importorskip("rclpy", reason="ROS 2 not sourced")

    env_yaml = _write_env_yaml(
        tmp_path,
        "  - sensor_id: testcam\n"
        "    backend: gstreamer\n"
        "    backend_params:\n"
        "      source: testsrc\n"
        "      width: 320\n"
        "      height: 240\n"
        "      fps: 10\n",
    )
    probe = f"""
# Production import order (mirrors scripts/runtime_node): gi + Gst.init()
# FIRST — before numpy/pydantic/rclpy. Under Fast-DDS, rclpy.Node()
# segfaults when numpy or pydantic were imported before Gst.init()
# (x86 Ubuntu 24.04 / system PyGObject / ROS Jazzy; Cyclone unaffected).
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)

import time

import rclpy
from openral_core import RobotEnvironment
from openral_rskill_ros.sensor_leg import open_deploy_sensor_readers

rclpy.init()
env = RobotEnvironment.from_yaml({str(env_yaml)!r})
leg = open_deploy_sensor_readers(env)
try:
    assert len(leg.readers) == 1, leg.readers
    # GStreamer readers publish via their in-pipeline ROS tee — no
    # polling SensorRosPublisher pump is attached for them.
    assert leg.publishers == [], leg.publishers
    deadline = time.time() + 10.0
    frame = None
    while time.time() < deadline:
        try:
            frame = leg.readers[0].read_latest(max_age_ms=2000)
        except Exception:
            frame = None
        if frame is not None:
            break
        time.sleep(0.1)
    assert frame is not None, "videotestsrc produced no frame within 10 s"

    # And the WorldState side of the contract: a BEST_EFFORT subscriber
    # on the leg's topic (same profile world_state requests) receives a
    # sensor_msgs/Image from the in-pipeline ROS tee.
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image

    got = []
    sub_node = rclpy.create_node("probe_subscriber")
    sub_node.create_subscription(
        Image,
        "/openral/cameras/testcam/image",
        got.append,
        QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        ),
    )
    deadline = time.time() + 10.0
    while time.time() < deadline and not got:
        rclpy.spin_once(sub_node, timeout_sec=0.2)
    sub_node.destroy_node()
    assert got, "no Image arrived on /openral/cameras/testcam/image within 10 s"
    assert got[0].height == 240 and got[0].width == 320, (got[0].height, got[0].width)
finally:
    leg.close()
    rclpy.try_shutdown()
assert leg.readers == []
print("SENSOR_LEG_PROBE_OK")
"""
    import os
    import subprocess
    import sys

    pkg_dir = Path(__file__).resolve().parents[2] / "packages" / "openral_rskill_ros"
    env_vars = dict(os.environ)
    env_vars["PYTHONPATH"] = f"{pkg_dir}{os.pathsep}{env_vars.get('PYTHONPATH', '')}"
    result = subprocess.run(  # reason: sys.executable with a fixed -c script
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
        env=env_vars,
        check=False,
    )
    assert result.returncode == 0, f"probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "SENSOR_LEG_PROBE_OK" in result.stdout

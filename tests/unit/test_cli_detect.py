"""Unit tests for the ``openral detect`` and ``ral skill check`` CLI commands.

Hermetic — every probe is exercised against a clean container, no
hardware required.  Larger end-to-end coverage lives in
``test_detect_probes_no_hardware.py`` / ``test_detect_assemble.py`` /
``test_detect_compatibility.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml
from openral_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


class TestBhDetect:
    def test_detect_no_write_prints_summary(self) -> None:
        result = runner.invoke(
            app,
            ["detect", "--no-write", "--include", "network", "--dds-timeout", "0"],
        )
        assert result.exit_code == 0, result.output
        assert "openral detect" in result.output
        # --no-write prints the assembled yaml to stdout.
        assert "name:" in result.output

    def test_detect_writes_full_robot_yaml(self, tmp_path: Path) -> None:
        out = tmp_path / "robot.yaml"
        result = runner.invoke(
            app,
            [
                "detect",
                "--output",
                str(out),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        data = yaml.safe_load(out.read_text())
        # Must be a complete RobotDescription, not the legacy stub.
        assert "name" in data
        assert "capabilities" in data
        assert "embodiment_kind" in data
        assert "safety" in data

    def test_detect_robot_override_forces_so101(self, tmp_path: Path) -> None:
        # No SO-101 hardware attached; the --robot override pins the manifest
        # regardless of what USB/network probing finds.
        out = tmp_path / "robot.yaml"
        result = runner.invoke(
            app,
            [
                "detect",
                "--robot",
                "so101",
                "--output",
                str(out),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        data = yaml.safe_load(out.read_text())
        assert data["name"] == "so101_follower"

    def test_detect_bad_robot_override_exits_1(self) -> None:
        result = runner.invoke(
            app,
            [
                "detect",
                "--robot",
                "not_a_robot",
                "--no-write",
                "--include",
                "network",
                "--dds-timeout",
                "0",
            ],
        )
        assert result.exit_code == 1
        assert "no committed" in result.output

    def test_detect_deployment_scaffolds_deploy_scene(self, tmp_path: Path) -> None:
        out = tmp_path / "robot.yaml"
        deploy = tmp_path / "workcell.yaml"
        result = runner.invoke(
            app,
            [
                "detect",
                "--robot",
                "so101",
                "--output",
                str(out),
                "--deployment",
                str(deploy),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        assert deploy.exists()
        # The scaffold loads back as a valid DeployScene.
        from openral_core import DeployScene

        scene = DeployScene.from_yaml(str(deploy))
        assert scene.robot_id == "so101_follower"
        assert scene.scene.id == "so101_follower_workcell"
        # safety unset → the robot manifest's envelope applies as-is.
        assert scene.safety is None
        assert scene.sensors == []
        banner = deploy.read_text()
        assert "review before" in banner
        assert "reasoner selects it at runtime" in banner

    def test_detect_interactive_binds_cameras(self, tmp_path: Path) -> None:
        """Wizard routing: every binding lands in the DeployScene — a manifest
        name yields a same-named entry (that robot sensor's binding, frame_id
        copied from the manifest); w:<name> yields a new workcell camera."""
        from openral_detect.report import V4l2CameraInfo

        out = tmp_path / "robot.yaml"
        deploy = tmp_path / "workcell.yaml"
        cams = [
            V4l2CameraInfo(device_path="/dev/video7", name="fake wrist cam"),
            V4l2CameraInfo(device_path="/dev/video8", name="fake overhead cam"),
        ]
        # Answers in device order: bind video7 → manifest "wrist";
        # video8 → new workcell camera "overhead".
        with patch("openral_detect.detect.probe_v4l2_cameras", return_value=cams):
            result = runner.invoke(
                app,
                [
                    "detect",
                    "--robot",
                    "so101",
                    "--output",
                    str(out),
                    "--deployment",
                    str(deploy),
                    "--interactive",
                    "--include",
                    "network,cameras_v4l2",
                    "--dds-timeout",
                    "0",
                    "--yes",
                ],
                input="wrist\nw:overhead\n",
            )
        assert result.exit_code == 0, result.output

        from openral_core import DeployScene, RobotDescription

        # The detect output manifest is untouched by the wizard — bindings are
        # host-specific and `deploy run` reads the canonical robots/<id>/ dir.
        desc = RobotDescription.from_yaml(str(out))
        assert all(s.deploy_binding is None for s in desc.sensors)

        scene = DeployScene.from_yaml(str(deploy))
        by_name = {s.name: s for s in scene.sensors}
        assert set(by_name) == {"wrist", "overhead"}
        # Manifest-named entry: binding for the robot's wrist cam, manifest
        # frame_id preserved.
        wrist_manifest = next(s for s in desc.sensors if s.name == "wrist")
        assert by_name["wrist"].frame_id == wrist_manifest.frame_id
        assert by_name["wrist"].deploy_binding.backend_params["device"] == "/dev/video7"
        # New name: workcell camera.
        assert by_name["overhead"].deploy_binding.backend_params["device"] == "/dev/video8"

    def test_interactive_without_deployment_warns(self) -> None:
        result = runner.invoke(
            app,
            ["detect", "--no-write", "--interactive", "--include", "network", "--dds-timeout", "0"],
        )
        assert result.exit_code == 0, result.output
        assert "--interactive has no effect without --deployment" in result.output

    def test_detect_with_report_dump(self, tmp_path: Path) -> None:
        out = tmp_path / "robot.yaml"
        report = tmp_path / "detection.json"
        result = runner.invoke(
            app,
            [
                "detect",
                "--output",
                str(out),
                "--report",
                str(report),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        assert report.exists()
        # Raw report is JSON.
        import json

        payload = json.loads(report.read_text())
        assert payload["schema_version"] == "0.1"


class TestBhSkillCheck:
    def test_skill_check_missing_robot_yaml_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["rskill", "check", "--robot", str(tmp_path / "missing.yaml")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_skill_check_against_assembled_yaml(self, tmp_path: Path) -> None:
        # Step 1: produce a robot.yaml via `openral detect`.
        out = tmp_path / "robot.yaml"
        runner.invoke(
            app,
            [
                "detect",
                "--output",
                str(out),
                "--include",
                "network",
                "--dds-timeout",
                "0",
                "--yes",
            ],
        )
        # Step 2: run `ral skill check` against an empty registry.
        empty_registry = tmp_path / "empty-registry.json"
        # Point --rskills-dir at a non-existent path so the default ("rskills/")
        # doesn't walk the in-tree rskills/ from the repo cwd.
        missing_rskills = tmp_path / "no-such-rskills"
        with patch("openral_rskill.loader.DEFAULT_REGISTRY_PATH", empty_registry):
            result = runner.invoke(
                app,
                [
                    "rskill",
                    "check",
                    "--robot",
                    str(out),
                    "--rskills-dir",
                    str(missing_rskills),
                    "--json",
                ],
            )
        # Empty registry → exit 0 (no incompat rows).
        assert result.exit_code == 0, result.output
        # JSON output parseable.
        import json

        payload = json.loads(result.output)
        assert payload["schema_version"] == "0.1"
        assert "rows" in payload

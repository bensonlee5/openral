"""The SmolVLA ONNX export ships a subprocess CLI (deadlock workaround).

``torch.onnx``'s dynamo exporter deadlocks when run inside a process that
already hosts an ``rclpy`` executor, so ``ensure_smolvla_onnx`` shells the
export as ``python -m openral_rskill.smolvla_export``. These checks pin the two
things that make that wiring work without loading torch/lerobot/GPU:

  * the module is runnable as ``-m`` and exposes the ``--repo-id`` / ``--out-dir``
    contract ``ensure_smolvla_onnx`` builds its argv against;
  * ``ensure_smolvla_onnx`` actually shells that module (argv shape), and
    surfaces a non-zero subprocess as a ``ROSConfigError`` rather than hanging.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_SRC = Path(__file__).resolve().parents[2] / "python" / "rskill" / "src"


def test_export_module_runs_as_subprocess_with_the_expected_flags() -> None:
    """``python -m openral_rskill.smolvla_export --help`` works and names the flags.

    Real subprocess (no mock): proves the ``-m`` entry point ``ensure_smolvla_onnx``
    relies on exists and parses the ``--repo-id`` / ``--out-dir`` / ``--n-cameras``
    contract. Light — argparse ``--help`` never imports torch/lerobot.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "openral_rskill.smolvla_export", "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_SRC), "PATH": ""},
        check=False,
    )
    if proc.returncode != 0 and "No module named" in proc.stderr:
        pytest.skip(f"openral_rskill not importable in this env: {proc.stderr.strip()}")
    assert proc.returncode == 0, proc.stderr
    for flag in ("--repo-id", "--out-dir", "--n-cameras"):
        assert flag in proc.stdout, f"{flag} missing from CLI help:\n{proc.stdout}"


def test_ensure_smolvla_onnx_shells_the_export_and_raises_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failing export subprocess becomes a ROSConfigError (no silent hang/fallback).

    Stubs only the process boundary (``subprocess.run``) — the acceptable double
    per CLAUDE.md §1.11 — and asserts the argv shape the deploy depends on.
    """
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    smolvla_trt = pytest.importorskip("openral_rskill.smolvla_trt")
    from openral_core.exceptions import ROSConfigError

    captured: dict[str, Any] = {}

    class _Result:
        returncode = 3
        stdout = "boom-out"
        stderr = "boom-err"

    def _fake_run(cmd: list[str], **_: Any) -> _Result:
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(smolvla_trt.subprocess, "run", _fake_run)

    with pytest.raises(ROSConfigError, match="ONNX export subprocess failed"):
        smolvla_trt.ensure_smolvla_onnx(
            "OpenRAL/rskill-smolvla-so101-pick-place-pen", cache_dir=tmp_path, n_cameras=2
        )

    cmd = captured["cmd"]
    assert cmd[:3] == [sys.executable, "-m", "openral_rskill.smolvla_export"]
    assert "--repo-id" in cmd and "OpenRAL/rskill-smolvla-so101-pick-place-pen" in cmd
    assert cmd[cmd.index("--n-cameras") + 1] == "2"

"""Robometer reward-monitor backends (ADR-0057).

Default path: :class:`RobometerInProcessReward` loads lerobot 0.6.0's native
Robometer model inside ``reward_monitor_node`` and scores clips on demand. That
keeps the heavy VLM out of the VLA runner / reasoner / HAL processes without a
second ZMQ process boundary.

Temporary fallback: :class:`RobometerReward` is the legacy sidecar client
(``OPENRAL_ROBOMETER_BACKEND=sidecar``) kept until deploy-sim validates the
in-process path. Nothing here imports torch / transformers / numpy at module
load.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openral_core import RSkillManifest
from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:
    from openral_runner.backends.reward.frame_source import Frame
    from openral_runner.backends.reward.topreward_reward import TOPRewardMonitor

# Default sidecar port — distinct from the scene-VLM (5759) and detector ports.
_DEFAULT_PORT = 5769
# |progress trend per sample| below this reads as "stalled" (no meaningful change).
_STALL_TREND_EPS = 0.002
_INPROCESS_FALLBACK_ENV = "OPENRAL_ROBOMETER_INPROCESS_FALLBACK"


def critic_score_from_assessment(
    assessment: Mapping[str, object], *, threshold: float
) -> tuple[float, float]:
    """Map a reward-monitor assessment to a generic critic ``(score, threshold)``.

    The Tier-C critic bus (ADR-0064) consumes a higher-is-better scalar per
    sample; a reward model's per-window ``progress_now`` (∈ [0, 1]) is exactly
    that. The producer's ``CriticWatchdogGroup`` decides when the score has
    *stalled* — this helper only normalises one :meth:`RobometerReward.assess`
    result into the ``openral_msgs/CriticScore`` ``(score, threshold)`` pair,
    clamping the score to ``[0, 1]`` and defaulting a missing/non-numeric
    ``progress_now`` to ``0.0`` (a conservative "no progress").

    Args:
        assessment: A :meth:`RobometerReward.assess` result. Reads
            ``progress_now``.
        threshold: The pass bar to stamp on the CriticScore — the watchdog fires
            when the score stays below it without improving.

    Returns:
        ``(score, threshold)`` ready for an ``openral_msgs/CriticScore``.

    Example:
        >>> critic_score_from_assessment({"progress_now": 0.42}, threshold=0.8)
        (0.42, 0.8)
        >>> critic_score_from_assessment({"progress_now": 1.5}, threshold=0.8)
        (1.0, 0.8)
    """
    raw = assessment.get("progress_now", 0.0)
    score = 0.0 if isinstance(raw, bool) or not isinstance(raw, (int, float)) else float(raw)
    score = max(0.0, min(1.0, score))
    return score, float(threshold)


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    """``k`` evenly-spaced indices into ``range(n)``, always including the last.

    Used to subsample a frame window to a fixed budget so the reward model's
    vision-transformer activation stays bounded on an 8 GB GPU (ADR-0058). The
    newest frame (index ``n-1``) is always kept — the reasoner reads
    ``progress_now`` from it. Returns ``list(range(n))`` when ``n <= k``.
    """
    if n <= k:
        return list(range(n))
    step = (n - 1) / (k - 1) if k > 1 else 0.0
    idx = sorted({min(n - 1, round(i * step)) for i in range(k)})
    if idx[-1] != n - 1:
        idx[-1] = n - 1
    return idx


def _find_sidecar_script() -> Path:
    """Locate ``tools/robometer_sidecar.py`` (env override or repo walk)."""
    override = os.environ.get("OPENRAL_ROBOMETER_SIDECAR")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tools" / "robometer_sidecar.py"
        if cand.exists():
            return cand
    raise ROSConfigError(
        "could not locate tools/robometer_sidecar.py; set OPENRAL_ROBOMETER_SIDECAR to its path"
    )


def _find_robometer_server_script() -> Path:
    """Locate ``tools/_robometer_server.py`` (same checkout as the sidecar wrapper)."""
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tools" / "_robometer_server.py"
        if cand.exists():
            return cand
    raise ROSConfigError("could not locate tools/_robometer_server.py")


def _load_inprocess_scorer_class() -> type:
    """Load the existing Robometer NF4 scorer without starting the ZMQ server."""
    server = _find_robometer_server_script()
    tools_dir = str(server.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    spec = importlib.util.spec_from_file_location("_openral_robometer_server", server)
    if spec is None or spec.loader is None:
        raise ROSConfigError(f"could not import Robometer server from {server}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = getattr(module, "_Scorer", None)
    if scorer is None:
        raise ROSConfigError(f"Robometer server {server} does not expose _Scorer")
    return scorer


def _bgr_frames_to_rgb_array(frames: list[Frame]) -> object:
    """Convert validated BGR888 frames to the RGB ndarray Robometer expects."""
    import numpy as np  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: PLC0415

    w, h = frames[0].width, frames[0].height
    bgr = np.frombuffer(b"".join(f.bgr for f in frames), dtype=np.uint8).reshape(
        len(frames), h, w, 3
    )
    return np.ascontiguousarray(bgr[:, :, :, ::-1])


def _validate_and_bound_frames(
    frames: list[Frame], task: str, *, max_frames: int, label: str
) -> list[Frame]:
    """Shared Robometer pre-flight: input guards + frame budget."""
    if not frames:
        raise ROSConfigError("reward score requires at least one frame")
    if not task.strip():
        raise ROSConfigError("reward score requires a non-empty task instruction")
    if len(frames) > max_frames:
        idx = _evenly_spaced_indices(len(frames), max_frames)
        print(
            f"[{label}] subsampling {len(frames)} -> {len(idx)} frames "
            f"(max_frames={max_frames}) to bound activation memory",
            flush=True,
        )
        frames = [frames[i] for i in idx]
    w, h = frames[0].width, frames[0].height
    if any(f.width != w or f.height != h for f in frames):
        raise ROSConfigError("all frames in a clip must share width/height")
    return frames


class RobometerInProcessReward:
    """In-process Robometer scorer for ``reward_monitor_node``.

    Reuses ``tools/_robometer_server.py::_Scorer`` so deploy-sim/run get the same
    native lerobot 0.6.0 + NF4 loader without the extra ZMQ process boundary.
    """

    def __init__(
        self,
        *,
        model_id: str,
        weights_source: str = "OpenRAL/rskill-robometer-4b-nf4",
        num_bins: int = 100,
        success_threshold: float = 0.5,
        max_frames: int = 8,
        device: str = "cuda",
        fallback: RobometerReward | None = None,
    ) -> None:
        """Store config; the VLM is loaded lazily on first score."""
        self._model_id = model_id
        self._weights_source = weights_source
        self._num_bins = num_bins
        self._success_threshold = success_threshold
        self._max_frames = max(1, max_frames)
        self._device = device
        self._scorer: object | None = None
        self._fallback_candidate = fallback
        self._fallback: RobometerReward | None = None

    def _ensure_ready(self) -> None:
        if self._scorer is not None or self._fallback is not None:
            return
        try:
            scorer_cls = _load_inprocess_scorer_class()
            self._scorer = scorer_cls(self._weights_source, device=self._device)
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            if os.environ.get(_INPROCESS_FALLBACK_ENV, "1").strip().lower() in {"0", "false", "no"}:
                raise ROSConfigError(f"Robometer in-process backend unavailable: {exc}") from exc
            print(
                f"[robometer] in-process backend unavailable ({type(exc).__name__}: {exc}); "
                "falling back to sidecar",
                flush=True,
            )
            self._fallback = self._fallback_candidate or RobometerReward(
                model_id=self._model_id,
                weights_source=self._weights_source,
                num_bins=self._num_bins,
                success_threshold=self._success_threshold,
                max_frames=self._max_frames,
            )

    def score(self, frames: list[Frame], task: str) -> tuple[list[float], list[float]]:
        """Score a clip in-process; sidecar fallback is temporary until deploy-sim passes."""
        frames = _validate_and_bound_frames(
            frames, task, max_frames=self._max_frames, label="robometer"
        )
        self._ensure_ready()
        if self._fallback is not None:
            return self._fallback.score(frames, task)
        assert self._scorer is not None
        progress, success = self._scorer.score(
            _bgr_frames_to_rgb_array(frames), task.strip(), self._num_bins
        )
        return [float(x) for x in progress], [float(x) for x in success]

    def assess(self, frames: list[Frame], task: str) -> dict[str, Any]:
        """Score ``frames`` and summarize the window for the Reasoner."""
        from openral_runner.backends.reward.frame_source import trend  # noqa: PLC0415

        progress, success = self.score(frames, task)
        p_trend = trend(progress)
        return {
            "progress_now": progress[-1],
            "success_now": success[-1],
            "progress_trend": p_trend,
            "success_trend": trend(success),
            "stalled": abs(p_trend) < _STALL_TREND_EPS,
            "succeeded": success[-1] >= self._success_threshold,
            "frames_seen": len(frames),
        }

    def close(self) -> None:
        """Release the model or the temporary sidecar fallback."""
        if self._fallback is not None:
            self._fallback.close()
            self._fallback = None
        self._scorer = None
        with contextlib.suppress(ImportError):
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()


class RobometerReward:
    """ZMQ client + auto-managed lifecycle for the Robometer reward sidecar."""

    def __init__(
        self,
        *,
        model_id: str,
        weights_source: str = "OpenRAL/rskill-robometer-4b-nf4",
        host: str = "127.0.0.1",
        port: int = _DEFAULT_PORT,
        auto_spawn: bool = True,
        boot_timeout_s: float = 1200.0,
        request_timeout_s: float = 180.0,
        num_bins: int = 100,
        success_threshold: float = 0.5,
        max_frames: int = 8,
    ) -> None:
        """Store config; connection to the sidecar is deferred to first use."""
        self._model_id = model_id
        self._weights_source = weights_source
        self._host = host
        self._port = port
        self._auto_spawn = auto_spawn
        self._boot_timeout_s = boot_timeout_s
        self._num_bins = num_bins
        self._success_threshold = success_threshold
        # Activation memory for the vision-transformer forward scales with the
        # number of frames (x resolution); a full 8 s x 3 fps window of 640x480
        # frames OOMs a 3.3 GB-resident model on an 8 GB GPU (ADR-0058, observed
        # in deploy-sim). Evenly subsample the window to at most this many frames
        # so the reward forward stays co-resident with the sim (and a small VLA).
        self._max_frames = max(1, max_frames)
        self._request_timeout_ms = int(request_timeout_s * 1000)
        # Lazy connection (mirrors QwenSceneVlm). `Any` because pyzmq attrs
        # aren't typed under strict.
        self._zmq: Any = None
        self._ctx: Any = None
        self._sock: Any = None
        self._child: subprocess.Popen[bytes] | None = None

    # -- wire ---------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if self._sock is not None:
            return
        try:
            import zmq  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: PLC0415 — lazy: keep zmq off the import path
        except ImportError as exc:  # pragma: no cover — env-provisioning guard
            raise ROSConfigError(
                "Robometer reward monitor needs the ZMQ + msgpack sidecar client; "
                "install it with `uv sync --group robometer` (provides pyzmq + msgpack)."
            ) from exc

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._connect()
        if not self._try_ping():
            if not self._auto_spawn:
                raise ROSConfigError(
                    f"no Robometer reward sidecar at tcp://{self._host}:{self._port} "
                    "and auto_spawn=False"
                )
            self._spawn_and_wait(self._boot_timeout_s)

    def _connect(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        sock = self._ctx.socket(self._zmq.REQ)
        sock.setsockopt(self._zmq.LINGER, 0)
        sock.setsockopt(self._zmq.RCVTIMEO, self._request_timeout_ms)
        sock.setsockopt(self._zmq.SNDTIMEO, 5000)
        sock.connect(f"tcp://{self._host}:{self._port}")
        self._sock = sock

    def _rpc(self, req: dict[str, object], *, recv_timeout_ms: int | None = None) -> dict[str, Any]:
        import msgpack  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: PLC0415 — lazy

        assert self._sock is not None
        if recv_timeout_ms is not None:
            self._sock.setsockopt(self._zmq.RCVTIMEO, recv_timeout_ms)
        try:
            self._sock.send(msgpack.packb(req, use_bin_type=True))
            reply: dict[str, Any] = msgpack.unpackb(self._sock.recv(), raw=False)
        except self._zmq.error.Again:
            self._connect()  # REQ can't recover from a missed reply; reset it
            raise
        finally:
            if recv_timeout_ms is not None:
                self._sock.setsockopt(self._zmq.RCVTIMEO, self._request_timeout_ms)
        return reply

    def _try_ping(self, *, recv_timeout_ms: int = 1000) -> bool:
        try:
            reply = self._rpc({"op": "ping"}, recv_timeout_ms=recv_timeout_ms)
        except self._zmq.error.Again:
            return False
        return bool(reply.get("ok"))

    def _spawn_and_wait(self, boot_timeout_s: float) -> None:
        import sys  # noqa: PLC0415 — lazy

        script = _find_sidecar_script()
        cmd = [
            sys.executable,
            str(script),
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--weights",
            self._weights_source,
        ]
        print(f"[robometer] spawning sidecar: {' '.join(cmd)}", flush=True)
        env = os.environ.copy()
        # torch-inductor defaults its compile pool to one worker per CPU
        # (e.g. 22 on a 22-core host) — wasteful for a 4B scorer with a handful
        # of compiled regions, and slow to spawn/reap. Cap it unless the
        # operator pinned a value.
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
        # Own session (start_new_session) so close() can signal the whole
        # process GROUP: the server forks inductor compile_worker children, and
        # a bare terminate()/kill() on the parent alone orphans them.
        self._child = subprocess.Popen(cmd, env=env, start_new_session=True)
        deadline = time.monotonic() + boot_timeout_s
        while time.monotonic() < deadline:
            if self._child.poll() is not None:
                raise ROSConfigError(
                    f"Robometer reward sidecar exited early (code {self._child.returncode})"
                )
            if self._try_ping():
                print("[robometer] sidecar ready", flush=True)
                return
            time.sleep(2.0)
        raise ROSConfigError(f"Robometer reward sidecar not ready within {boot_timeout_s}s")

    # -- public api ---------------------------------------------------------

    def score(self, frames: list[Frame], task: str) -> tuple[list[float], list[float]]:
        """Score a clip → ``(progress_series, success_series)``, per frame.

        Args:
            frames: Buffered frames (all the same ``width``/``height``), oldest
                first. Each carries raw BGR888 bytes.
            task: The natural-language task instruction.

        Returns:
            ``(progress, success)`` lists of equal length to ``frames``,
            progress normalized to the model's discrete-mode ``[0, 1]`` range.

        Raises:
            ROSConfigError: empty clip / empty task / sidecar error.
        """
        frames = _validate_and_bound_frames(
            frames, task, max_frames=self._max_frames, label="robometer"
        )
        w, h = frames[0].width, frames[0].height

        self._ensure_ready()
        reply = self._rpc(
            {
                "op": "score",
                "frames": b"".join(f.bgr for f in frames),
                "n": len(frames),
                "width": w,
                "height": h,
                "task": task.strip(),
                "num_bins": self._num_bins,
            }
        )
        if not reply.get("ok"):
            raise ROSConfigError(f"Robometer reward sidecar error: {reply.get('error')}")
        progress = [float(x) for x in reply["progress"]]
        success = [float(x) for x in reply["success"]]
        return progress, success

    def assess(self, frames: list[Frame], task: str) -> dict[str, Any]:
        """Score ``frames`` and summarize the window for the Reasoner.

        Returns a dict with ``progress_now``, ``success_now``,
        ``progress_trend``, ``success_trend``, ``stalled``, ``succeeded``
        (success_now ≥ threshold), and ``frames_seen``.
        """
        from openral_runner.backends.reward.frame_source import trend  # noqa: PLC0415

        progress, success = self.score(frames, task)
        p_trend = trend(progress)
        return {
            "progress_now": progress[-1],
            "success_now": success[-1],
            "progress_trend": p_trend,
            "success_trend": trend(success),
            "stalled": abs(p_trend) < _STALL_TREND_EPS,
            "succeeded": success[-1] >= self._success_threshold,
            "frames_seen": len(frames),
        }

    def close(self) -> None:
        """Close the socket and terminate the sidecar tree if we spawned it.

        Signals the sidecar's whole process GROUP (it runs in its own session),
        so the server's forked torch-inductor ``compile_worker`` children die
        with it instead of orphaning and pinning CPU/GPU until the next run.
        """
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        if self._child is not None and self._child.poll() is None:
            # The child is its own session leader, so its PGID == its PID.
            pgid: int | None = None
            with contextlib.suppress(ProcessLookupError, OSError):
                pgid = os.getpgid(self._child.pid)
            with contextlib.suppress(Exception):
                self._rpc({"op": "shutdown"}, recv_timeout_ms=2000)
            try:
                self._child.wait(timeout=10)
            except Exception:  # graceful RPC shutdown didn't drain in time
                # SIGKILL the whole group so the forked compile_worker children
                # die with the server. (On a clean exit the server's atexit pool
                # shutdown already reaped them; the CLI orphan-sweep is the
                # final backstop for either path.)
                if pgid is not None:
                    with contextlib.suppress(ProcessLookupError, OSError):
                        os.killpg(pgid, signal.SIGKILL)
                with contextlib.suppress(Exception):
                    self._child.wait(timeout=5)
            self._child = None


def build_reward_monitor(
    manifest: RSkillManifest,
    *,
    host: str = "127.0.0.1",
    port: int = _DEFAULT_PORT,
) -> RobometerInProcessReward | RobometerReward | TOPRewardMonitor:
    """Build a reward monitor from a ``kind: "reward"`` rSkill manifest.

    Args:
        manifest: A validated rSkill manifest with ``kind == "reward"``.
        host: Sidecar fallback host to connect to.
        port: Sidecar fallback port to connect to.

    Returns:
        Robometer defaults to the in-process backend; ``OPENRAL_ROBOMETER_BACKEND=sidecar``
        keeps the old ZMQ path as a temporary deploy-sim fallback.

    Raises:
        ROSConfigError: If the manifest is not ``kind == "reward"`` or lacks a
            ``reward`` block.
    """
    if manifest.kind != "reward":
        raise ROSConfigError(
            f"build_reward_monitor requires kind='reward', got {manifest.kind!r} "
            f"for {manifest.name!r}"
        )
    if manifest.reward is None:  # pragma: no cover — validator guarantees this
        raise ROSConfigError(f"reward manifest {manifest.name!r} has no `reward` block")
    # Backend dispatch (ADR-0057): TOPReward and Robometer run in the reward
    # monitor process; Robometer's ZMQ sidecar remains an opt-in fallback until
    # deploy-sim validates the in-process path.
    if manifest.reward.backend == "topreward":
        from openral_runner.backends.reward.topreward_reward import (  # noqa: PLC0415
            build_topreward_monitor,
        )

        return build_topreward_monitor(manifest)
    raw = manifest.weights_uri or manifest.source_repo or "OpenRAL/rskill-robometer-4b-nf4"
    # hf://org/repo[@rev] -> "org/repo[@rev]" (sidecar resolves rev); local:///path
    # -> "/path" (a pre-quantized checkpoint dir loaded directly as 4-bit).
    if raw.startswith("local://"):
        weights_source = raw.removeprefix("local://")
    else:
        weights_source = raw.removeprefix("hf://").split("@", 1)[0]
    kwargs = {
        "model_id": manifest.name,
        "weights_source": weights_source,
        "num_bins": manifest.reward.num_bins,
        "success_threshold": manifest.reward.success_threshold,
    }
    if os.environ.get("OPENRAL_ROBOMETER_BACKEND", "").strip().lower() == "sidecar":
        return RobometerReward(host=host, port=port, **kwargs)
    return RobometerInProcessReward(
        fallback=RobometerReward(host=host, port=port, **kwargs),
        **kwargs,
    )

"""VLABench scene adapter — wraps :class:`lerobot.envs.configs.VLABenchEnv`.

VLABench (ICCV 2025, OpenMOSS) is a MuJoCo + dm_control language-conditioned
manipulation benchmark on a Franka Panda arm. lerobot 0.6.0 ships a native
``vlabench`` env config; this adapter drives it through OpenRAL's unbatched
:class:`SimRollout` contract.

Env contract (``obs_type="pixels_agent_pos"``, ``n_envs=1``):
- ``obs["pixels"]`` — dict of 3 RGB views ``image`` / ``second_image`` /
  ``wrist_image`` (480×480×3), batched ``(1, H, W, 3)``.
- ``obs["agent_pos"]`` — 7-D proprio, batched ``(1, 7)``.
- action — 7-D end-effector, batched ``(1, 7)``.
- ``info["is_success"]`` — per-env success flag.

``lerobot``'s ``VLABenchEnv.create_envs`` returns a nested
``{task_group: {idx: SyncVectorEnv}}`` mapping; we extract the single
``SyncVectorEnv`` and unwrap the ``n_envs=1`` batch dimension here.

Task ID convention: ``"vlabench/<task-name>"`` (e.g. ``"vlabench/select_fruit"``).
``scene.id`` MUST be ``"vlabench"``. VLABench + its ~12 GB asset bundle are
externally provisioned (CLAUDE.md §1.9); set ``VLABENCH_ROOT`` (or
``OPENRAL_VLABENCH_ROOT``) to the ``VLABench/`` package dir that holds ``assets/``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_sim.registry import SCENES
from openral_sim.rollout import StepResult

if TYPE_CHECKING:
    from openral_core import SceneSpec, SimEnvironment, TaskSpec

    from openral_sim.rollout import Observation


_VLABENCH_SCENE_ID = "vlabench"
# Env agent_pos is 7-D (pos[3]+euler[3]+gripper[1]); the smolvla_vlabench
# normalizer stats + dataset (lerobot/vlabench_unified observation.state) are
# 7-D, so the full vector is fed through (the checkpoint config.json's [6] is
# stale metadata — the runtime normalizer buffer is 7-D).
_VLABENCH_STATE_DIM = 7
_VLABENCH_ACTION_DIM = 7
# lerobot's VLABenchEnv obs["pixels"] sub-keys → OpenRAL canonical camera keys.
# We emit camera1/2/3 (the franka_panda robot's declared sensor keys); the
# smolvla checkpoint's own preprocessor rename (image→camera1, …) then no-ops.
_VLABENCH_CAMERA_MAP = {
    "image": "camera1",
    "second_image": "camera2",
    "wrist_image": "camera3",
}


def _parse_task_id(task_id: str) -> str:
    """Parse ``"vlabench/<task-name>"`` → ``<task-name>``."""
    parts = task_id.split("/", maxsplit=1)
    expected_parts = 2
    if len(parts) != expected_parts or parts[0] != _VLABENCH_SCENE_ID:
        raise ROSConfigError(f"vlabench task id must be 'vlabench/<task-name>', got {task_id!r}")
    return parts[1]


def _unbatch(value: Any) -> NDArray[Any]:
    """Drop the leading ``n_envs=1`` batch dim from a SyncVectorEnv array."""
    arr = np.asarray(value)
    return arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 else arr


@dataclass
class _VLABenchSim:
    """Thin :class:`SimRollout` wrapper around a VLABench ``SyncVectorEnv``."""

    scene: SceneSpec
    task: TaskSpec
    _env: Any  # gymnasium SyncVectorEnv (n_envs=1), lazy-built
    _last_image: NDArray[np.uint8] | None = None

    def reset(self, seed: int | None = None) -> Observation:
        obs, _info = self._env.reset(seed=seed)
        return self._wrap_obs(obs)

    def step(self, action: NDArray[np.float32]) -> StepResult:
        # SyncVectorEnv expects a batched (1, action_dim) action.
        action_b = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, terminated, truncated, info = self._env.step(action_b)
        success = bool(_unbatch(info.get("is_success", False)))
        return StepResult(
            observation=self._wrap_obs(obs),
            reward=float(_unbatch(reward)),
            terminated=bool(_unbatch(terminated)),
            truncated=bool(_unbatch(truncated)),
            info={"is_success": success},
        )

    def _wrap_obs(self, obs: dict[str, Any]) -> Observation:
        cameras: dict[str, NDArray[np.uint8]] = {}
        pixels = obs["pixels"]
        for env_key, cam_key in _VLABENCH_CAMERA_MAP.items():
            cameras[cam_key] = _unbatch(pixels[env_key]).astype(np.uint8)
        self._last_image = cameras["camera1"]
        # Full 7-D agent_pos (pos[3] + euler[3] + gripper[1]) — matches the
        # checkpoint's 7-D normalizer stats + the vlabench_unified dataset's
        # observation.state (the config.json's [6] is stale metadata).
        state = _unbatch(obs["agent_pos"]).astype(np.float32)
        return {"images": cameras, "state": state, "task": self.task.instruction}

    def render(self) -> NDArray[np.uint8] | None:
        return None if self._last_image is None else self._last_image.copy()

    def close(self) -> None:
        self._env.close()

    def mujoco_handles(self) -> tuple[Any, Any] | None:
        # VLABench wraps a dm_control Physics behind a SyncVectorEnv; no direct
        # (MjModel, MjData) reach-through, so the runner falls back to wall-clock.
        return None


def _resolve_vlabench_root() -> None:
    """Ensure ``VLABENCH_ROOT`` points at the installed VLABench asset dir."""
    if os.environ.get("VLABENCH_ROOT"):
        return
    override = os.environ.get("OPENRAL_VLABENCH_ROOT")
    if override:
        os.environ["VLABENCH_ROOT"] = override
        return
    import VLABench

    os.environ["VLABENCH_ROOT"] = os.path.dirname(VLABench.__file__)


def _build_vlabench_scene(env_cfg: SimEnvironment) -> _VLABenchSim:
    """Lazily construct a VLABench primitive-task env via lerobot's config."""
    if env_cfg.scene.id != _VLABENCH_SCENE_ID:
        raise ROSConfigError(
            f"vlabench scene id must be {_VLABENCH_SCENE_ID!r}, got {env_cfg.scene.id!r}"
        )
    task_name = _parse_task_id(env_cfg.task.id)

    try:
        _resolve_vlabench_root()
        from lerobot.envs.configs import VLABenchEnv
    except ImportError as exc:  # pragma: no cover
        raise ROSConfigError(
            "VLABench backend not installed; provision the VLABench sim + assets "
            "(clone OpenMOSS/VLABench, `uv pip install --no-deps -e .`, "
            "`python scripts/download_assets.py`) and set VLABENCH_ROOT."
        ) from exc

    cfg = VLABenchEnv(
        task=task_name,
        obs_type="pixels_agent_pos",
        episode_length=int(env_cfg.task.max_steps or 500),
    )
    env_groups = cfg.create_envs(n_envs=1)
    # create_envs returns {task_group: {idx: SyncVectorEnv}}; take the sole env.
    group = env_groups[next(iter(env_groups))]
    env = group[next(iter(group))]
    return _VLABenchSim(scene=env_cfg.scene, task=env_cfg.task, _env=env)


SCENES.register(_VLABENCH_SCENE_ID, fixed_robot="franka_panda")(_build_vlabench_scene)

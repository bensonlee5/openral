#!/usr/bin/env python
"""Old-vs-new parity harness for the MolmoAct2 adapter migration.

Gates the deletion of the ``trust_remote_code`` load path in
``openral_sim.policies.molmoact2``. The migration swaps the model-graph
builder from a remote-code ``AutoModelForImageTextToText`` (executed via
``trust_remote_code=True``) to the in-tree
``lerobot.policies.molmoact2.molmoact2_hf_model.MolmoAct2ForConditionalGeneration``.
Both paths load the *same* NF4 prequant weights and run the same greedy
flow-matching sampler, so on a fixed observation + fixed torch seed the emitted
action chunk must be numerically identical (up to bnb 4-bit non-determinism).

Protocol (the two model classes must NEVER be resident at once):

  1. ``--capture-golden`` — builds the OLD model graph: the upstream remote-code
     ``MolmoAct2ForConditionalGeneration`` (``trust_remote_code``, needs
     ``OPENRAL_ALLOW_REMOTE_CODE=1``), injected into the migrated adapter by
     monkeypatching its ``_import_molmoact2`` seam. Everything else — the in-tree
     ``MolmoAct2Processor``, the NF4 prequant weights, and the seeded
     ``predict_action`` call — is shared with the new path, so the golden isolates
     exactly the model-class swap. Builds ONE real LIBERO observation from
     ``scenes/sim/libero_spatial.yaml`` (a real MuJoCo reset — no mocks) and saves
     both the observation and the emitted action chunk to
     ``scripts/_golden_molmoact2.npz``.

     (Why not the unmodified old adapter end-to-end? Under transformers 5.x the
     upstream *processor* no longer loads via ``AutoProcessor`` +
     ``trust_remote_code`` — the exact breakage this migration fixes — so the
     only runnable old artefact is the old model *class*, which is what parity
     gates.)
  2. default (verify) — builds the NEW model graph: lerobot's in-tree
     ``MolmoAct2ForConditionalGeneration``. Loads the SAME saved observation,
     runs the same seeded inference, and asserts
     ``np.allclose(new, golden, rtol=1e-3, atol=1e-2)``.

Both modes run under the shared GPU flock — capture golden first (separate
process), then verify. Wrap each invocation in ``flock /tmp/openral-gpu.lock -c``:

    - capture: ``OPENRAL_ALLOW_REMOTE_CODE=1 python scripts/parity_molmoact2.py
      --capture-golden``
    - verify:  ``python scripts/parity_molmoact2.py``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# LIBERO's MuJoCo reset needs a GL backend; pick EGL on a headless host.
os.environ.setdefault("MUJOCO_GL", "egl")

# Pre-stub the broken lerobot groot module before any lerobot.policies import
# (same shim tests/sim/conftest.py installs).
import openral_rskill._lerobot_compat  # noqa: F401

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Make ``tests`` importable as a package (compose_sim_env lives in tests.sim.conftest).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Heavy imports (torch / lerobot via openral_sim) come after the groot shim and
# the sys.path insert so the ``tests`` package + the lerobot compat stub resolve.
import openral_sim.policies.molmoact2 as _molmoact2_mod  # noqa: E402
import torch  # noqa: E402
from openral_sim import make_env, make_policy  # noqa: E402
from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: E402

from tests.sim.conftest import compose_sim_env  # noqa: E402

_SCENE = _REPO_ROOT / "scenes" / "sim" / "libero_spatial.yaml"
_RSKILL = "rskills/molmoact2-libero-nf4"
_GOLDEN = Path(__file__).resolve().parent / "_golden_molmoact2.npz"
_SEED = 0


def _build_real_observation() -> tuple[dict, str]:
    """Reset the real LIBERO MuJoCo env once and return ``(observation, task)``.

    The env is closed before returning so it does not sit on the GPU while the
    (heavy) MolmoAct2 policy loads and infers.
    """
    env_cfg = compose_sim_env(_SCENE, rskill_uri=_RSKILL, n_episodes=1, max_steps=5)
    env = make_env(env_cfg)
    try:
        obs = env.reset(seed=_SEED if env_cfg.seed is None else env_cfg.seed)
        task = str(obs.get("task", "") or env_cfg.task.instruction or "")
        # Deep-copy the arrays we need out of the env before it is torn down.
        images = {
            k: np.ascontiguousarray(np.asarray(v))
            for k, v in dict(obs.get("images", {})).items()
        }
        state = np.ascontiguousarray(np.asarray(obs["state"], dtype=np.float32))
    finally:
        env.close()
    return {"images": images, "state": state, "task": task}, task


def _patch_remote_model_class() -> None:
    """Swap the adapter's ``_import_molmoact2`` seam to return the OLD remote class.

    Keeps the in-tree config + processor (and the Auto-registry registration the
    real seam performs) and substitutes only the upstream remote-code
    ``MolmoAct2ForConditionalGeneration`` for the model graph — isolating the
    model-class swap the parity gate is proving.
    """
    orig_import = _molmoact2_mod._import_molmoact2

    def _remote_import() -> tuple[object, object, object]:
        _model_cls, config_cls, processor_cls = orig_import()
        remote_cls = get_class_from_dynamic_module(
            "modeling_molmoact2.MolmoAct2ForConditionalGeneration",
            "allenai/MolmoAct2-LIBERO",
        )
        print(f"[capture] OLD model class = {remote_cls.__module__}.{remote_cls.__name__}")
        return remote_cls, config_cls, processor_cls

    _molmoact2_mod._import_molmoact2 = _remote_import  # type: ignore[assignment]


def _run_policy_on_observation(
    observation: dict, task: str, *, use_remote_model: bool = False
) -> np.ndarray:
    """Load the MolmoAct2 adapter and return its full action chunk (T, action_dim).

    Seeds torch immediately before the inference so the flow-matching sampler's
    noise is identical across the old and new model classes regardless of any RNG
    consumed during model construction. ``use_remote_model`` swaps in the OLD
    upstream remote-code model class for the golden capture.
    """
    if use_remote_model:
        _patch_remote_model_class()

    env_cfg = compose_sim_env(_SCENE, rskill_uri=_RSKILL, n_episodes=1, max_steps=5)
    policy = make_policy(env_cfg)
    try:
        policy.reset()
        torch.manual_seed(_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_SEED)
        chunk = policy._predict_chunk(observation, task)
    finally:
        policy.close()
    return np.stack([np.asarray(row, dtype=np.float32) for row in chunk], axis=0)


def _capture_golden() -> int:
    observation, task = _build_real_observation()
    actions = _run_policy_on_observation(observation, task, use_remote_model=True)
    payload: dict[str, np.ndarray] = {
        "actions": actions,
        "state": observation["state"],
        "task": np.asarray(task),
        "image_keys": np.asarray(list(observation["images"].keys())),
    }
    for k, v in observation["images"].items():
        payload[f"image__{k}"] = v
    np.savez(_GOLDEN, **payload)
    print(f"[capture] saved golden -> {_GOLDEN}")
    print(f"[capture] action chunk shape={actions.shape} task={task!r}")
    print(f"[capture] action[0]={np.array2string(actions[0], precision=4)}")
    return 0


def _load_golden() -> tuple[dict, str, np.ndarray]:
    data = np.load(_GOLDEN, allow_pickle=False)
    task = str(data["task"])
    image_keys = [str(k) for k in data["image_keys"]]
    images = {k: data[f"image__{k}"] for k in image_keys}
    state = np.asarray(data["state"], dtype=np.float32)
    return {"images": images, "state": state, "task": task}, task, np.asarray(data["actions"])


def _verify() -> int:
    if not _GOLDEN.exists():
        print(f"[verify] FAIL: golden not found at {_GOLDEN}; run --capture-golden first.")
        return 2
    observation, task, golden = _load_golden()
    new = _run_policy_on_observation(observation, task)

    if new.shape != golden.shape:
        print(f"[verify] FAIL: shape mismatch new={new.shape} golden={golden.shape}")
        return 1

    max_abs = float(np.max(np.abs(new - golden)))
    sign_changed = int(np.sum(np.sign(new) != np.sign(golden)))
    new_argmax = np.argmax(np.abs(new), axis=-1)
    argmax_changed = int(np.sum(new_argmax != np.argmax(np.abs(golden), axis=-1)))
    ok = bool(np.allclose(new, golden, rtol=1e-3, atol=1e-2))

    print(f"[verify] chunk shape={new.shape}")
    print(f"[verify] max abs diff = {max_abs:.3e}")
    print(f"[verify] sign flips = {sign_changed} / {new.size} elements")
    print(f"[verify] per-step |argmax| changes = {argmax_changed} / {new.shape[0]} steps")
    print(f"[verify] golden[0]={np.array2string(golden[0], precision=4)}")
    print(f"[verify] new[0]   ={np.array2string(new[0], precision=4)}")
    if ok:
        print("[verify] PASS: new == golden within rtol=1e-3 atol=1e-2")
        return 0
    print("[verify] FAIL: new != golden — DO NOT delete the old path; root-cause the diff.")
    return 1


def main() -> int:
    """Parse args and dispatch to capture-golden or verify."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-golden",
        action="store_true",
        help="Run the CURRENT (old, remote-code) adapter and save the golden chunk.",
    )
    args = parser.parse_args()
    return _capture_golden() if args.capture_golden else _verify()


if __name__ == "__main__":
    raise SystemExit(main())

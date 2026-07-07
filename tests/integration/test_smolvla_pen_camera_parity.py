"""Camera-count parity for the SO-101 pen SmolVLA rSkill (Option B).

The ``OpenRAL/rskill-smolvla-so101-pick-place-pen`` checkpoint declares **three**
camera slots (``observation.images.camera1/2/3``) inherited from
``lerobot/smolvla_base``, but trains on **two** cameras and sets
``empty_cameras=0``. lerobot ``prepare_images`` therefore drops the unfilled
``camera3`` at inference — native inference runs on two cameras.

This test pins the deploy decision behind ``rskills/smolvla-so101-pick-place-pen``:

  * **Reference** — native torch ``VLAFlowMatching.sample_actions`` on a *real*
    dataset frame (fixed overview + arm-mounted view), fixed flow-matching
    noise, fp32 on CPU. This is exactly what ``deploy run`` does outside the
    TRT/ONNX engine.
  * **Option B** (``n_cameras=2``) — the split-ONNX export the rSkill ships.
    Must reproduce the reference bit-exactly (fp32 rounding floor). This is the
    "inside the TRT/ONNX engine vs outside" parity the deploy relies on.
  * **Option C** (``camera3`` present + attended) — what a naive 3-slot engine
    would do. Must *diverge*: the policy attends a camera native never feeds.

Skips unless the checkpoint + VLM backbone are in the local HF cache and the
dataset episode is fetchable (heavy: one SmolVLA split export, CPU fp32).
"""

from __future__ import annotations

import dataclasses
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("lerobot")

import onnxruntime as ort  # noqa: E402
from openral_rskill.smolvla_export import export_smolvla_split_onnx  # noqa: E402

_CHECKPOINT = "OpenRAL/rskill-smolvla-so101-pick-place-pen"  # shipped mirror (clean config)
_DATASET = "nota-gmbh/pick_and_place_pen_so101"
_VLM_BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
_TOL_MATCH = 2e-3  # fp32 torch↔ONNX parity floor (observed ~1.7e-06)
_TOL_DIVERGE = 1e-2  # a genuinely-attended extra camera moves the chunk well past this


def _snapshot_cached(repo_id: str) -> bool:
    d = _HF_CACHE / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return d.is_dir() and any(d.iterdir())


pytestmark = pytest.mark.skipif(
    not (_snapshot_cached(_CHECKPOINT) and _snapshot_cached(_VLM_BACKBONE)),
    reason=f"{_CHECKPOINT} (+ VLM backbone) not in local HF cache",
)


@pytest.fixture(scope="module")
def policy() -> Any:
    """Real pen checkpoint, float32/CPU, with the stray config field stripped."""
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    local = snapshot_download(_CHECKPOINT)
    # The shipped mirror already has a clean config, so the strip below is a
    # no-op there; it stays defensive (and to force device=cpu for this fp32/CPU
    # test). Keep only valid SmolVLAConfig fields.
    valid = {f.name for f in dataclasses.fields(SmolVLAConfig)}
    cfgp = Path(local) / "config.json"
    raw = json.loads(cfgp.read_text())
    dropped = {k: v for k, v in raw.items() if k in valid or k == "type"}
    dropped["device"] = "cpu"  # this test is fp32/CPU; never touch a (possibly full) GPU
    if dropped != raw:
        cfgp.write_text(json.dumps(dropped, indent=2))
    pol = SmolVLAPolicy.from_pretrained(local)
    pol.model = pol.model.float().eval().to("cpu")
    pol._openral_local = local  # type: ignore[attr-defined]  # reason: stash for stats
    return pol


def _state_stats(local: str) -> tuple[np.ndarray, np.ndarray]:
    """MEAN_STD state normalizer stats from the checkpoint's safetensors."""
    p = Path(local) / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    raw = p.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8 : 8 + n])
    body = raw[8 + n :]

    def g(name: str) -> np.ndarray:
        s, e = hdr[name]["data_offsets"]
        return np.frombuffer(body[s:e], dtype=np.float32).reshape(hdr[name]["shape"]).copy()

    return g("observation.state.mean"), g("observation.state.std")


@pytest.fixture(scope="module")
def real_obs(policy: Any) -> dict[str, Any]:
    """A real dataset frame → post-prepare native inputs + fixed noise.

    Confirms the native truth: ``prepare_images`` yields exactly two cameras.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        ds = LeRobotDataset(_DATASET, episodes=[0])
    except Exception as exc:  # offline / gated dataset → skip, never fake
        pytest.skip(f"{_DATASET} episode not fetchable: {exc}")
    item = ds[len(ds) // 2]
    cfg = policy.config
    mean, std = _state_stats(policy._openral_local)
    state_norm = ((item["observation.state"].numpy() - mean) / std).astype(np.float32)
    task = item["task"] if isinstance(item["task"], str) else item["task"][0]
    batch = {
        "observation.images.camera1": item["observation.images.fixed"].unsqueeze(0),
        "observation.images.camera2": item["observation.images.handy"].unsqueeze(0),
        "observation.state": torch.from_numpy(state_norm).unsqueeze(0),
        "task": [task],
    }
    images, img_masks = policy.prepare_images(batch)
    assert len(images) == 2, f"expected 2 cameras (empty_cameras=0), got {len(images)}"
    assert all(bool(m.flatten()[0]) for m in img_masks)
    tok = policy.model.vlm_with_expert.processor.tokenizer(
        task if task.endswith("\n") else task + "\n",
        padding="max_length",
        truncation=True,
        max_length=cfg.tokenizer_max_length,
        return_tensors="pt",
    )
    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": tok["input_ids"],
        "lang_masks": tok["attention_mask"].bool(),
        "state": policy.prepare_state(batch),
        "noise": torch.randn(
            1,
            cfg.chunk_size,
            cfg.max_action_dim,
            generator=torch.Generator().manual_seed(7),
        ),
    }


def _native(policy: Any, obs: dict[str, Any], images: list[Any], masks: list[Any]) -> np.ndarray:
    with torch.no_grad():
        return policy.model.sample_actions(
            images,
            masks,
            obs["lang_tokens"],
            obs["lang_masks"],
            obs["state"],
            noise=obs["noise"],
        ).numpy()


def _run_onnx(
    policy: Any, obs: dict[str, Any], out: Path, n_cameras: int, imgs: list[Any]
) -> np.ndarray:
    paths = export_smolvla_split_onnx(policy, out, n_cameras=n_cameras)
    vis = ort.InferenceSession(str(paths.vision_onnx), providers=["CPUExecutionProvider"])
    pol = ort.InferenceSession(str(paths.policy_onnx), providers=["CPUExecutionProvider"])
    (embs,) = vis.run(None, {"pixel_values": np.concatenate([i.numpy() for i in imgs], axis=0)})
    assert embs.shape[0] == n_cameras
    (act,) = pol.run(
        None,
        {
            "img_embs": embs.reshape(1, -1, embs.shape[-1]),
            "lang_tokens": obs["lang_tokens"].numpy(),
            "lang_masks": obs["lang_masks"].numpy(),
            "state": obs["state"].numpy(),
            "noise": obs["noise"].numpy(),
        },
    )
    return act


def test_option_b_two_camera_export_matches_native(
    policy: Any, real_obs: dict[str, Any], tmp_path: Path
) -> None:
    """The shipped 2-camera export reproduces native torch inference bit-exactly."""
    ref = _native(policy, real_obs, real_obs["images"], real_obs["img_masks"])
    act_b = _run_onnx(policy, real_obs, tmp_path / "onnx2", 2, real_obs["images"])
    assert act_b.shape == ref.shape
    max_abs = float(np.max(np.abs(act_b - ref)))
    print(f"\nOption B (2-cam ONNX) vs native: max|Δ| = {max_abs:.2e}")
    assert max_abs < _TOL_MATCH, f"2-camera export diverged from native: {max_abs:.2e}"


def test_option_c_attended_third_camera_diverges(policy: Any, real_obs: dict[str, Any]) -> None:
    """Attending a black camera3 (naive 3-slot engine) is NOT what native does.

    Torch-only (no second export): feeding the checkpoint three present cameras
    — the black ``camera3`` a 3-slot engine would demand — shifts the action
    chunk well past the parity floor, proving Option B (drop camera3) is the
    faithful export and no ``camera3`` sensor belongs in the robot manifest.
    """
    ref = _native(policy, real_obs, real_obs["images"], real_obs["img_masks"])
    black = torch.ones_like(real_obs["images"][0]) * -1.0  # SigLIP black
    imgs3 = [*real_obs["images"], black]
    masks3 = [*real_obs["img_masks"], torch.ones_like(real_obs["img_masks"][0])]
    act_c = _native(policy, real_obs, imgs3, masks3)
    max_abs = float(np.max(np.abs(act_c - ref)))
    print(f"\nOption C (attended camera3) vs native: max|Δ| = {max_abs:.2e}")
    assert max_abs > _TOL_DIVERGE, (
        f"attended camera3 unexpectedly matched native ({max_abs:.2e}); "
        "the camera-count decision needs re-checking"
    )

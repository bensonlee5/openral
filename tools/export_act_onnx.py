"""Export a LeRobot ACT policy to a single whole-model ONNX graph.

Unlike SmolVLA (whose VLM + flow-matching graph needs a *split* export — see
``openral_rskill.smolvla_export``), ACT is a plain CNN + transformer: the core
network maps normalized observations straight to the normalized action chunk,
so it exports whole-model in one shot.

Export target: ``ACTPolicy.model`` (the ``ACT`` nn.Module), reached exactly as
``predict_action_chunk`` does — it builds ``batch[OBS_IMAGES]`` as an ordered
list over ``config.image_features`` and returns ``model(batch)[0]`` of shape
``(B, chunk_size, action_dim)``. Normalization is **external** (the checkpoint's
``policy_preprocessor.json`` / ``policy_postprocessor.json`` MEAN_STD sidecars,
applied in Python by the ACT adapter), so the ONNX graph takes already-normalized
images + state and emits the normalized action chunk. The parity test
(``tests/integration/test_act_onnx.py``) compares this graph against the torch
``predict_action_chunk`` on the same normalized inputs.

Inputs are ordered by ``config.image_features`` (for so101-passing-pen that is
``[wrist, front]``) plus the 6-D state. The gabrycina/so101-passing-pen-policy
checkpoint trains at 480x640.

Usage (uses the project venv, which already carries torch + lerobot + onnx):

    uv run python tools/export_act_onnx.py \
        --repo-id gabrycina/so101-passing-pen-policy \
        --out rskills/act-so101-pen/model.onnx
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch


class _ACTExportWrapper(torch.nn.Module):
    """Flat-tensor facade over ``ACTPolicy.model`` for ONNX tracing.

    Rebuilds the dict/list batch structure ``predict_action_chunk`` feeds the
    core network, from positional image + state tensors the exporter can trace.
    """

    def __init__(self, policy: Any, image_feature_keys: list[str], state_key: str) -> None:  # reason: lerobot ACTPolicy is untyped
        super().__init__()
        from lerobot.utils.constants import OBS_IMAGES

        self.model = policy.model
        self._image_feature_keys = image_feature_keys
        self._state_key = state_key
        self._obs_images = OBS_IMAGES

    def forward(self, *images_and_state: torch.Tensor) -> torch.Tensor:
        *images, state = images_and_state
        batch: dict[str, Any] = {self._state_key: state}
        for key, img in zip(self._image_feature_keys, images, strict=True):
            batch[key] = img
        # predict_action_chunk: batch[OBS_IMAGES] = [batch[k] for k in image_features]
        batch[self._obs_images] = list(images)
        actions: torch.Tensor = self.model(batch)[0]  # (B, chunk_size, action_dim)
        return actions


def export(out_path: Path, repo_id: str, *, device: str = "cpu") -> str:
    """Export ``repo_id``'s ACT policy to ``out_path`` as ONNX; return sha256."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.constants import OBS_STATE

    policy: Any = ACTPolicy.from_pretrained(repo_id)
    policy.eval().to(device)

    cfg = policy.config
    image_feature_keys = list(cfg.image_features)
    _, h, w = next(iter(cfg.image_features.values())).shape
    state_dim = cfg.robot_state_feature.shape[0]

    wrapped = _ACTExportWrapper(policy, image_feature_keys, OBS_STATE).eval().to(device)
    dummy_images = tuple(
        torch.randn(1, 3, h, w, device=device) for _ in image_feature_keys
    )
    dummy_state = torch.randn(1, state_dim, device=device)
    example = (*dummy_images, dummy_state)

    image_input_names = [k.replace("observation.images.", "img_") for k in image_feature_keys]
    input_names = [*image_input_names, "state"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        example,
        str(out_path),
        input_names=input_names,
        output_names=["action_chunk"],
        opset_version=17,
        dynamo=False,
    )
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(  # reason: CLI export tool
        f"exported {out_path} sha256={sha}\n"
        f"  inputs: {input_names} (images 1x3x{h}x{w}, state 1x{state_dim})\n"
        f"  output: action_chunk 1x{cfg.chunk_size}x{cfg.action_feature.shape[0]}"
    )
    return sha


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="gabrycina/so101-passing-pen-policy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ns = ap.parse_args()
    export(ns.out, ns.repo_id, device=ns.device)

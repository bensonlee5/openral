---
language:
- en
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- smolvla
- lerobot
- vision-language-action
- so101_follower
datasets:
- nota-gmbh/pick_and_place_pen_so101
inference: false
---

# rskill-smolvla-so101-pick-place-pen

> **OpenRAL rSkill** — [SmolVLA](https://arxiv.org/abs/2506.01844) finetuned for
> pen pick-and-place on a **real SO-101 follower arm**, packaged for `OpenRAL`.

This package wraps
[`nota-gmbh/so101_pick_place_pen_smolvla`](https://huggingface.co/nota-gmbh/so101_pick_place_pen_smolvla)
with a `rskill.yaml` manifest that adds capability checking, license surfacing,
latency budgets, the joint-units contract, and local registry integration. It
does **not** copy model weights.

## Quick start

```python
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/smolvla-so101-pick-place-pen/rskill.yaml")
```

```bash
# Real SO-101 deploy (weights are public Apache-2.0):
uv run openral deploy run --robot so101 --rskill rskills/smolvla-so101-pick-place-pen

# Zero-copy NVMM TensorRT vision leg inside the GStreamer pipeline (ADR-0082):
OPENRAL_SMOLVLA_TRT=1 uv run openral deploy run --robot so101 \
    --rskill rskills/smolvla-so101-pick-place-pen
```

## Upstream model / training

| Field | Value |
| --- | --- |
| Source repo | [`nota-gmbh/so101_pick_place_pen_smolvla`](https://huggingface.co/nota-gmbh/so101_pick_place_pen_smolvla) |
| Base model | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| Paper | [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) — *SmolVLA* |
| Training dataset | [`nota-gmbh/pick_and_place_pen_so101`](https://huggingface.co/datasets/nota-gmbh/pick_and_place_pen_so101) (53 episodes) |
| Training | 40 000 steps, batch 64, multi-GPU |
| Architecture | SmolVLA (~0.45 B) — VLM backbone + flow-matching action expert |
| Precision | bf16 |
| License | Apache-2.0 (code + weights) |

## Supported robots / embodiments

`so101_follower` — the 6-DoF SO-101 follower arm (5 arm joints + 1 gripper).
The checkpoint drives absolute joint positions.

> **Joint units — degrees.** This checkpoint's state and action are in **degrees**
> (verified: `observation.state` normalizer spans ~[-103, +168]). openral's
> `JointState` / `Action` contract is radians, so the skill_runner converts
> deg↔rad at the policy boundary. The manifest declares
> `action_contract.joint_units: degrees` explicitly — a wrong guess would drive
> the arm into its joint limits.

## Sensors / observation contract — **two cameras, not three**

| Manifest key | Scene sensor | Checkpoint view |
| --- | --- | --- |
| `observation.images.camera1` | `top` | fixed / overview |
| `observation.images.camera2` | `wrist` | arm-mounted |

Both are 224×224+. Proprioception is the 6-D joint-position vector.

`smolvla_base` ships **three** camera slots (`camera1/2/3`) and this
checkpoint's `config.json` inherits all three, but it trained on only two
cameras and sets `empty_cameras: 0`. lerobot `prepare_images` therefore
**drops the unfilled `camera3`** at inference (present cameras only, no black
padding) — native inference runs on two cameras. The split-ONNX / TensorRT
engine is exported for the deploy's resolved camera count (the two
`sensors_required` above), so it never attends a phantom camera. No `camera3`
sensor is added to `robots/so101_follower/robot.yaml`.

**Verified numerically** (`tests/integration/test_smolvla_pen_camera_parity.py`):
on a real dataset frame, a 2-camera export reproduces native torch
`sample_actions` to `max|Δ| = 1.7e-06` (fp32); a 3-camera export (black
`camera3`, attended) diverges.

## Manifest summary

| Field | Value |
| --- | --- |
| `model_family` | `smolvla` |
| `role` | `s1` |
| `chunk_size` / `n_action_steps` | 50 / 50 |
| `action_contract` | 6-D `joint_positions`, `joint_units: degrees` |
| `latency_budget` | 400 ms/chunk |
| Actions | pick · place · pick_and_place (object: pen) |

## Known issue — upstream config field (handled automatically)

The upstream `config.json` carries a stray `pretrained_revision` key that
lerobot 0.5.1's `SmolVLAConfig` (draccus) rejects with `DecodingError`. Every
OpenRAL SmolVLA load path — the rSkill adapter, the deploy/sim factory, and the
TRT ONNX export — calls `openral_rskill._lerobot_compat.sanitize_smolvla_config`
first, which strips config keys not valid for `SmolVLAConfig` (idempotent). If
you call `SmolVLAPolicy.from_pretrained` directly, call it first too.

## License

**Apache-2.0** (code and weights). OpenRAL's packaging is Apache-2.0 (ADR-0012);
the upstream checkpoint and dataset are Apache-2.0 as published by the author.

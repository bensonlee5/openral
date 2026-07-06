---
language:
- en
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
- OpenRAL
- rskill
- act
- lerobot
- vision-language-action
- so101_follower
datasets:
- gabrycina/so101-passing-pen
inference: false
---

# rskill-act-so101-pen

> **OpenRAL rSkill** — [ACT](https://arxiv.org/abs/2304.13705) (Action Chunking
> with Transformers) finetuned for **"pass the pen"** pick-and-place on a **real
> SO-101 follower arm**, packaged for `OpenRAL`.

This package wraps
[`gabrycina/so101-passing-pen-policy`](https://huggingface.co/gabrycina/so101-passing-pen-policy)
with a `rskill.yaml` manifest that adds capability checking, license surfacing,
latency budgets, the joint-units contract, a paired reward monitor, and local
registry integration. It does **not** copy the model weights.

It is the smaller, faster, ONNX/TensorRT-friendly sibling of
[`smolvla-so101-pen`](../smolvla-so101-pen): same task and embodiment, but a
plain CNN+transformer (ResNet-18 + VAE, ~52 M params) instead of a
VLM+flow-matching policy — so the **whole model exports to a single ONNX graph**
(see *ONNX / TensorRT* below).

## Quick start

```python
from openral_rskill.loader import rSkill
pkg = rSkill.from_yaml("rskills/act-so101-pen/rskill.yaml")
```

```bash
# Real SO-101 deploy (torch baseline):
uv run openral deploy run --robot so101 --rskill rskills/act-so101-pen

# With the ONNX/TensorRT inference path (engine built + cached on first load):
OPENRAL_ACT_TRT=1 uv run openral deploy run --robot so101 --rskill rskills/act-so101-pen
```

## Upstream model / training

| Field | Value |
| --- | --- |
| Source repo | [`gabrycina/so101-passing-pen-policy`](https://huggingface.co/gabrycina/so101-passing-pen-policy) |
| Paper | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) — *Action Chunking with Transformers* |
| Training dataset | [`gabrycina/so101-passing-pen`](https://huggingface.co/datasets/gabrycina/so101-passing-pen) (~22.4k frames) |
| Architecture | ACT — ResNet-18 backbone, 4 encoder + 1 decoder layers, latent VAE, `chunk_size=100` |
| Precision | fp32 (torch); optional bf16/fp32 TensorRT engine |
| License | Apache-2.0 (code + weights) |

## Supported robots / embodiments

`so101_follower` — the 6-DoF SO-101 follower arm (5 arm joints + 1 gripper).
The checkpoint drives absolute joint positions.

> **Joint units — degrees.** This checkpoint's state and action are in **degrees**
> (verified: `observation.state` / `action` normalizer stats span ±100, MEAN_STD).
> openral's `JointState` / `Action` contract is radians, so the skill_runner
> converts deg↔rad at the policy boundary. The manifest declares
> `action_contract.joint_units: degrees` explicitly — a wrong guess would drive
> the arm into its joint limits.

## Sensors / observation contract

Two RGB streams, aliased to the checkpoint's `observation.images.*` inputs:

| Manifest key | Robot sensor | Checkpoint view |
| --- | --- | --- |
| `observation.images.camera1` | `top` | front / overview |
| `observation.images.camera2` | `wrist` | wrist |

Both are ≥224×224 (trained at 640×480). Proprioception is the 6-D
joint-position vector.

## Reward monitor

Per ADR-0077, this VLA emits no success signal of its own, so it runs paired
with a reward / progress monitor: `reward_rskill_name:
OpenRAL/rskill-robometer-4b-nf4` (Robometer-4B, NF4, ~3.6 GB). ACT (fp32,
~0.2 GB) + Robometer co-reside comfortably on an 8 GB card.

## ONNX / TensorRT

Unlike the SmolVLA sibling (whose VLM+flow-matching graph needs a bespoke
*split* export — vision encoder + unrolled flow loop, see PR #139), ACT is a
plain CNN+transformer and exports **whole-model** to one ONNX graph:

- **Export:** `tools/export_act_onnx.py` traces `ACTPolicy.select_action`
  (inputs: two RGB images + the 6-D state; output: the action chunk) to a
  single `model.onnx`.
- **Ship:** `model.onnx` is committed into this rSkill's HF repo
  (`policy_extras.act_onnx_uri`).
- **Run:** with `OPENRAL_ACT_TRT=1`, the ACT adapter loads `model.onnx` through
  the generic `ONNXRuntime` / `TensorRTRuntime` backend; TensorRT builds and
  **caches** the engine on the host on first load (same delivery shape as
  `rtdetr-v2-r50vd`). Without the flag the torch path runs unchanged.

## Manifest summary

| Field | Value |
| --- | --- |
| `model_family` | `act` |
| `role` | `s1` |
| `chunk_size` | 100 (per-step replay; `temporal_ensemble_coeff=null`) |
| `action_contract` | 6-D `joint_positions`, `joint_units: degrees` |
| `reward_rskill_name` | `OpenRAL/rskill-robometer-4b-nf4` |
| `latency_budget` | 100 ms/chunk |
| Actions | pick · place · pick_and_place · transfer (object: pen) |

## License

**Apache-2.0** (code and weights). OpenRAL's packaging is Apache-2.0 (ADR-0012);
the upstream checkpoint and dataset are Apache-2.0 as published by the author.

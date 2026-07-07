---
tags:
  - OpenRAL
  - rskill
  - smolvla
  - lerobot
  - vla
  - vlabench
  - manipulation
license: apache-2.0
language:
  - en
---

# rskill-smolvla-vlabench

> **OpenRAL rSkill — VLABench integration baseline (NOT a passing policy).**
> SmolVLA (~0.5 B) finetuned on [VLABench](https://github.com/OpenMOSS/VLABench)
> (`lerobot/vlabench_unified`), lerobot-native, runs in-process on lerobot 0.6.0
> (bf16, fits 8 GB). Wraps [`lerobot/smolvla_vlabench`](https://huggingface.co/lerobot/smolvla_vlabench).

## ⚠ Status: baseline, scores 0%

This rSkill exists to **exercise and validate the OpenRAL VLABench backend**, not
to score the benchmark. Measured **0/3 on six diverse primitive tasks**
(`select_fruit`, `select_drink`, `select_toy`, `select_book`, `add_condiment`,
`insert_flower`) — **identical to lerobot's own `lerobot-eval` reference**, which
confirms the OpenRAL wiring is faithful (state 7-D, action absolute-eef, cameras
`camera1/2/3`) and the 0% is the policy, not the integration.

The only VLABench policy above 50% is `VLABench/pi0-fast-ft-primitive-10task-deltachunk`
(51.2% primitive avg), which is **openpi/JAX** and would need conversion to lerobot
`PI0FAST` (+int8 for 8 GB) to run in-process — a dedicated, not-yet-done effort.
VLABench's composite/long-horizon suite is unsolved (<50%) by every known policy.

## Provisioning (VLABench is out-of-tree, ~12 GB)

```bash
git clone https://github.com/OpenMOSS/VLABench.git
uv pip install --no-deps -e VLABench          # works on numpy 2.x
uv pip install mujoco dm_control open3d mediapy gdown  # numpy2-compatible sim deps
# rrt_algorithms is git-only + used only for data-gen (off the VLA eval path) — stub it.
export VLABENCH_ROOT=$PWD/VLABench/VLABench
python VLABench/scripts/download_assets.py     # ~12 GB obj + scene from Google Drive
```

## Run

```bash
MUJOCO_GL=egl VLABENCH_ROOT=$VLABENCH_ROOT \
  openral benchmark scene --config scenes/benchmark/vlabench_select_fruit.yaml \
    --rskill rskills/smolvla-vlabench
```

## Contract

| Field | Value |
| --- | --- |
| `model_family` | `smolvla` (bf16, ~0.5 B) |
| cameras | `camera1/2/3` (env `image`/`second_image`/`wrist_image`) |
| state | 7-D `[pos_robot(3), euler_xyz(3), gripper(1)]` |
| action | 7-D absolute eef pose → IK (`delta_ee_6d_plus_gripper` label is nominal) |
| robot | `franka_panda` (uses the manifest's 3rd `camera3`/`front` sensor) |

## See also
- `python/sim/src/openral_sim/backends/vlabench.py` — the backend.
- [`lerobot/smolvla_vlabench`](https://huggingface.co/lerobot/smolvla_vlabench) — upstream checkpoint.

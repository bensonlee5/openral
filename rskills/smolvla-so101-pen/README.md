# rskill-smolvla-so101-pen

SmolVLA policy packaged as an OpenRAL rSkill for the SO-101 follower arm pen
pick-and-place bring-up path. It wraps the upstream
`sapanostic/so_101_smolvla_pen_placement` checkpoint and does not vendor model
weights into this repository.

## Upstream model and training

The policy is finetuned from `lerobot/smolvla_base` on the
`sapanostic/pen-placement-task` teleoperation dataset, recorded on a real SO-101
arm with side and wrist RGB cameras. The checkpoint emits 50-step absolute joint
position chunks and the manifest declares `joint_units: degrees` so OpenRAL
converts at the HAL boundary.

## Supported robots

This rSkill targets `so101_follower` only. The embodiment tag, 6-DoF state
contract, and absolute joint-position action contract match
`robots/so101_follower/robot.yaml`.

## Sensors and observation contract

The skill requires two RGB streams. `observation.images.camera1` maps to the
training side view and `observation.images.camera2` maps to the wrist view via
the manifest aliases. Both streams must provide at least 224x224 RGB frames.

## Manifest summary

Runtime is PyTorch bf16 SmolVLA with weights resolved from
`hf://sapanostic/so_101_smolvla_pen_placement`. The manifest pins chunk size,
latency budget, state dimension, action dimension, action representation, and
degree-based joint units for reproducible deploy traces.

## License

The rSkill wrapper is Apache-2.0, matching the manifest. Third-party model
weights keep their upstream license and are fetched from the referenced
Hugging Face repository at deploy time or from the local HF cache when offline.

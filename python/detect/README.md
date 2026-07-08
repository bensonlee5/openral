# openral-detect

Hardware probing + `RobotDescription` assembly for `openral detect`.

Part of [**OpenRAL**](https://github.com/OpenRAL/openral) — the open Robot
Abstraction Layer for vision-language-action robotics. This package is one
member of the OpenRAL Python workspace; see the architecture overview and the
eight-layer model in the project docs.

- **Docs:** https://openral.github.io/openral/
- **Source:** https://github.com/OpenRAL/openral
- **License:** Apache-2.0

> All OpenRAL workspace packages move in lockstep at `0.1.x` until the first
> public release.

## Usage

```bash
# Write/refresh a robot manifest.
openral detect --output robots/so101_follower/robot.yaml

# Also scaffold a deploy scene with camera bindings.
openral detect \
  --output robots/so101_follower/robot.yaml \
  --deployment scenes/deploy/so101_bench.yaml \
  --interactive

# Inspect probes without writing files.
openral detect --include usb,gpu,cameras_v4l2 --report detect.json --no-write
```

`openral detect` probes USB, DDS, GPU, V4L2, RealSense, and network interfaces;
loads a canonical manifest when the rig is known; enriches detected cameras from
the sensor catalog; and records accelerator capabilities used by
`openral rskill check`.

Bare Feetech USB detection defaults to `so101_follower` because SO-100 and
SO-101 are electrically indistinguishable on the bus. Force the older arm with:

```bash
openral detect --robot so100 --output robots/so100_follower/robot.yaml
```

`--deployment` writes a `DeployScene` shell (`robot_id` plus interactive camera
bindings). It does not choose an rSkill; `openral deploy run` lets the reasoner
pick from installed, capability-matched rSkills.

# ADR-0078: DeployScene is the deploy/workcell config

## Status

Accepted.

## Context

`openral deploy sim` used `DeployScene`, while `openral deploy run` used a
separate `RobotEnvironment`. That split duplicated robot identity and left
deploy-time safety/context inconsistent across sim and real paths.

Robot facts already belong in `robots/<robot_id>/robot.yaml`: HAL defaults,
serial/IP parameters, sensors, poses, rates, capabilities, and safety ceilings.
Workcell facts belong with the deploy scene: which robot is in the cell, scene
identity, optional sim composition, deploy-time safety tightening, and additive
allowed collision pairs.

## Decision

`DeployScene` is the single deploy/workcell artifact for both:

- `openral deploy sim --config scenes/deploy/<id>.yaml`
- `openral deploy run --config scenes/deploy/<id>.yaml`

`DeployScene` directly owns:

- `safety: SafetyEnvelope | None`
- `extra_allowed_collision_pairs: list[tuple[str, str]]`

`RobotEnvironment` is removed. `openral detect` writes/validates robot manifests;
it no longer scaffolds a second hardware deployment YAML.

### Amendment (2026-07-04): sensors declared where they are mounted

Real deploys need physical camera bindings (`/dev/video*` + reader backend) —
the capability `RobotEnvironment.sensors` used to carry. The split follows the
same robot-vs-workcell line:

- **Robot-mounted** sensors (wrist / head cameras) are declared in
  `robots/<id>/robot.yaml` as `SensorSpec` entries — they move with the robot
  and the manifest stays authoritative for frames/intrinsics.
- **Workcell-mounted** sensors (overhead / front cameras) live in
  `DeployScene.sensors: list[SensorSpec]` — they belong to the cell.

Any spec may carry an optional `deploy_binding: SensorDeployBinding`
(backend + backend_params such as device/fps, reusing the ADR-0010
`SensorReaderBackend` registry) — the runtime counterpart of `sim_placement`:
`sim_placement` says how sim renders the camera; `deploy_binding` says how
`deploy run` opens it.

Bindings are host-specific, and `deploy run` loads the robot manifest from the
canonical `robots/<robot_id>/` directory (never a detect-scaffolded local
robot.yaml) — so the **DeployScene is the operative home for bindings** in a
committed workcell. A scene entry *named like a manifest sensor* is that robot
sensor's deploy-time binding (on collision the scene entry wins —
`merge_deploy_sensors`); a *new* name declares a workcell camera. The
`openral detect --interactive` wizard routes each probed `/dev/video*`
accordingly into the `--deployment` scaffold. On `deploy run` the sensor leg
opens every bound spec of the merged set and publishes on
`/openral/cameras/<name>/image`; WorldState's `camera_names` includes both
sources. Reference workcell: `scenes/deploy/so101_bench.yaml` (+ committed
calibration in `scenes/deploy/calibration/`). Additive, backward-compatible
schema change (no version bump).

Deploy safety is tighten-only against `RobotDescription.safety`. Additive ACM
pairs are a sibling field, not part of `SafetyEnvelope`, so collision loosening
does not pass through the envelope intersection invariant. There is no
self-collision disable flag.

`composition` remains sim-only: real deployments use physical world geometry.

## Consequences

One YAML can boot the same workcell in sim or real mode. Real deploys now thread
`DeployScene.safety` and `extra_allowed_collision_pairs` into `sim_e2e.launch.py`
through `workcell_json`, where the launch computes the kernel envelope and ACM.

`thumbnail_hz` is no longer public config. `save_dir` and `max_ticks` remain
runtime/sim/benchmark concerns, not deploy/workcell fields.

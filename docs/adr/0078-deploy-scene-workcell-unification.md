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

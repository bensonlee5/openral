# ADR-0081: Box/OBB collision primitive in the safety kernel

## Status

Proposed (safety-WG review + hazard-log entry required before Accepted — see
"Process gates"). Fixes issue #84.

## Context

The C++ safety kernel (`cpp/openral_safety_kernel/`, ADR-0030/-0040) modelled
every link's collision volume as a **capsule** — a segment swept by a radius.
Capsules bound most limbs tightly, but a *blocky* link (the SO-ARM100/101
`base` is a near-cubic housing, mesh AABB ≈ `0.111 × 0.096 × 0.072 m`) forces
the enclosing capsule's circular cross-section to bulge ~3–7 cm past the block's
flat faces. That bulge over-reports proximity at poses where a flat face is what
actually faces a neighbour.

This bit in production: a live `openral deploy run` on the real SO-101 latched
`/openral/estop` on the first action of a pen-pick with

```
safety.collision kind=self a=base b=lower_arm step=0 min_distance_m=-0.0779
```

at the home pose, **before any commanded motion**. A MuJoCo mesh-distance oracle
shows the true `base↔lower_arm` clearance at that pose is **+0.162 m** (and
`base↔wrist` **+0.239 m**) — the links are nowhere near colliding. The fat base
capsule (radius 0.075 m) simply over-reports, and every `base↔<link>` pair trips
identically. A per-pair `extra_allowed_collision_pairs` ACM entry (ADR-0078)
suppressed one pair but the next base pair then tripped — whack-a-mole that ends
in disabling base self-collision entirely, which §1.1/§3 forbid.

Issue #84 established (with a MuJoCo oracle) that **no capsule decomposition**
fixes this — axis-split, PCA-grid, and VHACD convex decompositions all plateau
above 0 clearance because the failure is the circular cross-section vs the
block's flat faces, not the capsule count. The primitive itself is wrong for a
block.

## Decision

Add an **oriented box (OBB)** collision primitive alongside the capsule, end to
end:

- **Schema** — `BoxShape` (`half_extents_m`) joins the `CollisionShape`
  discriminated union (`python/core/src/openral_core/schemas.py`), usable in
  `LinkCollisionGeometry` and `WorldCollisionPrimitive`.
- **C++ hot path** — a new `Obb` struct + `box_link`/`boxes` arrays on
  `CollisionModel`; two allocation-free distance functions:
  - `box_capsule_distance` — exact for the disjoint case (closest distance from
    the capsule's segment to the box via a ternary search on the convex
    point→AABB distance, minus the capsule radius);
  - `box_box_distance` — a **conservative** separating-axis distance (the max
    gap over the 15 SAT axes lower-bounds the true Euclidean distance, so the
    kernel never *under*-reports a collision).
  `check_self_collision`, **`check_world_collision`, and `check_voxel_collision`**
  all iterate `box↔capsule` / `box↔box` (self) and `box↔world-capsule` /
  `box↔voxel` (world) pairs — so a boxed link is never invisible to the world /
  octomap check. No allocations, no exceptions across the safety boundary;
  covered by the `NoAlloc` gtest + dedicated `WorldCollisionBox` /
  `VoxelCollisionBox` gtests.
- **Param plumbing** — the kernel declares/reads `collision_box_link`,
  `collision_box_half_extents`, `collision_box_origin_xyzrpy`; the lowering
  (`collision_params_from_description`) routes `BoxShape` links to those arrays
  and omits empty primitive/allowed-pair arrays (launch_ros rejects empty typed
  arrays — an all-box robot has zero capsules, a capsule-only robot zero boxes).
- **Tuned self-collision margin** — a new `SafetyEnvelope.self_collision_margin_m`
  (manifest `safety:` field, default `0.0`) the lowering forwards. **Separate**
  from the world / voxel margins in the kernel, so loosening self-collision never
  loosens arm-vs-world collision.
- **so101 manifest** — **every** link (base + shoulder/upper_arm/lower_arm/wrist)
  becomes a `BoxShape`: the SO-101 links are rectangular 3D-printed brackets, so
  a single capsule over-reports ~7–9 cm at the arm's compact folded operating
  poses (the pen VLA's whole distribution); boxes cut that to ~1–4 cm (verified
  vs an fcl non-convex mesh oracle). The `so101_bench.yaml` ACM override is
  removed (self-collision stays fully on). `safety.self_collision_margin_m:
  -0.06` tolerates the arm's inherent grazing operating envelope (true clearance
  ≈ 0) while gross over-folds still E-stop; the real self-contact protection is
  the force/torque limits, and world/voxel collision is unaffected.
- **rSkill `starting_pose`** — the pen rSkill declares its SRDF `rest`
  starting_pose (ADR-0053) so the runner moves the arm to an in-distribution,
  collision-clear pose before the first VLA tick (belt-and-suspenders — the
  margin already tolerates the folded start).

## Consequences

- **Live-validated on the real SO-101 (2026-07-06):** the pen VLA ran 452
  action chunks continuously under TRT (bf16), unfolding the arm from its parked
  pose to reach the workspace, with **zero `safety.collision` events** — the run
  ended only on an operator e-stop. The self-collision false-positive is fixed
  end to end.
- The kernel now checks box↔box / box↔capsule / box↔world / box↔voxel pairs.
  Per-pair cost is higher than a capsule (box↔box ≈ 15-axis SAT; box↔capsule ≈
  48-iter ternary) but absolute cost is negligible (few links, allocation-free).
- **Conservatism (§3):** `box↔capsule` is exact; `box↔box` and `box↔voxel` are
  distance lower bounds; every OBB fully encloses its link mesh (unit-tested).
  Gross over-folds still fire; world/voxel margins stay conservative and positive.
- **The negative self-margin is a real safety trade** for a contact-operating
  arm — justified against the *true* (fcl non-convex) envelope clearance, force/
  torque as the primary contact protection, and the deliberate self-vs-world
  margin split. Requires safety-WG sign-off (hazard-log Entry 005).
- Teaching `openral collision lower` to emit OBBs for blocky links (rather than
  the hand-authored entries here) is the follow-up that makes the manifest
  regenerable; `mjcf_lowering.py` should likewise emit boxes for MJCF box geoms.

## Alternatives considered

- **Multi-capsule / VHACD / PCA-grid decomposition** — proven insufficient in
  issue #84 (circular section vs flat faces; all plateau above 0 clearance).
- **Per-pair ACM entries** — masks the bug pair-by-pair, converges on disabling
  base self-collision; rejected (§1.1/§3).
- **Neutral-pose auto-allow of home-colliding pairs** — safe only for accurate
  geometry; with the over-conservative capsule it would also mask pairs that
  genuinely collide elsewhere (so100 `base↔upper_arm` collides at ~17.5 % of
  reachable poses), so it can hide real collisions. Rejected.

## Process gates

- **Layer-6 hot-path contract change** — safety-WG reviewer + hazard-log entry
  required (CLAUDE.md §3) before this moves to Accepted.
- **TDD / conservatism proof** — C++ gtest (`test_collision.cpp`) for the
  distance math + box paths; `NoAlloc` gtest for allocation-freedom; an offline
  MuJoCo-oracle test (`tests/unit/test_so101_base_box_collision.py`) proving the
  home pose is collision-free, the box clears where the retired capsule fired,
  the box encloses the base mesh, and a real penetration is still caught.
- **Inspection** — `tools/viz_collision.py` overlays the kernel primitives on
  the robot meshes for visual review.

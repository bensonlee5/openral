# ADR-0079: VLABench (MuJoCo + dm_control) benchmark backend

- Status: **Accepted**
- Date: 2026-07-07
- ADR number: `0079`.
- Related: [ADR-0002](0002-eval-and-sim-environments.md) (eval & sim environments);
  [ADR-0062](0062-rlbench-benchmark-backend.md) (RLBench — the precedent for a heavy,
  externally-provisioned benchmark whose assets are never vendored);
  [ADR-0060](0060-benchmark-task-data-compatibility-gate.md) (`evaluated_tasks` gate);
  [ADR-0012](0012-open-core-licensing.md) (open-core / weight-license posture).
- Supersedes the prototype landed in `feat(sim): VLABench benchmark backend (validated
  prototype)` — this ADR promotes it to a shippable backend (auto-install plan, tests, map).

## Context

VLABench (ICCV 2025, OpenMOSS, [2412.18194](https://arxiv.org/abs/2412.18194)) is a
language-conditioned manipulation benchmark on a **Franka Panda** built on **MuJoCo +
dm_control**. lerobot 0.6.0 ships a native `VLABenchEnv` env config, so — unlike RLBench
(CoppeliaSim, proprietary, out-of-process sidecar — ADR-0062) — VLABench runs **in-process
on the py3.12 workspace**. It does, however, need bespoke provisioning that `uv sync
--group …` cannot express:

1. **No PyPI release.** VLABench installs from a git clone, editable.
2. **numpy pin conflict.** Its `setup.py` pins `numpy==1.25`; the workspace is numpy 2.x.
   The clone must be installed `--no-deps`, and its sim deps (mujoco, dm_control, open3d,
   mediapy, gdown) installed loose so uv keeps numpy 2.x. Empirically it runs clean on the
   workspace's numpy 2.2 / mujoco 3.8 once its own pins are bypassed.
3. **A git-only transitive dep.** `rrt-algorithms` (no PyPI, drags plotly) is imported on
   the env-build chain (`dm_task → skill_lib → rrt`) but only *used* in SkillLib's scripted
   data-generation methods — never on the VLA policy-eval path.
4. **A ~12 GB CC-BY asset bundle** (object + scene meshes) fetched from Google Drive.

## Decision

**1. Native in-process `SimRollout` backend.** `openral_sim.backends.vlabench` registers a
factory under scene id `vlabench`, `fixed_robot="franka_panda"`, over
`PhysicsBackend.MUJOCO` (the same value LIBERO / MetaWorld use). It drives lerobot's
`VLABenchEnv` with `obs_type="pixels_agent_pos"`, unwraps the `n_envs=1` batch dim, maps the
3 RGB views (`image` / `second_image` / `wrist_image` → `camera1/2/3`), and emits the full
7-D agent_pos state (`pos[3] + euler[3] + gripper[1]`) — the checkpoint's normalizer and the
`lerobot/vlabench_unified` dataset are 7-D (the checkpoint `config.json`'s `[6]` is stale
metadata). Task id convention: `vlabench/<task-name>`.

**2. Auto-provision the Python side via `ensure_backend_deps("vlabench")`.** `_deps` gains a
`vlabench` plan (`_vlabench_plan`): clone `OpenMOSS/VLABench` → `$OPENRAL_CACHE_HOME/repos/
VLABench` (idempotent), `uv pip install --no-deps -e` it, `uv pip install mujoco dm_control
open3d mediapy gdown` loose, then **write a raise-on-use `rrt_algorithms` stub** into
site-packages (`_vlabench_stub_rrt_step`) so the four `from rrt_algorithms.… import …` lines
resolve without the external repo and any accidental eval-path use fails loudly. The probe
(`_has_vlabench`) also checks the stub, so a stray `uv sync` that evicts it re-triggers the
plan — the same self-healing behaviour as the robosuite editable-shadow cleanup.

**3. Leave the ~12 GB asset bundle as a one-time manual fetch.** Unlike the RoboCasa CC-BY
assets (robust Box download, auto-fetched), VLABench's bundle comes from Google Drive via
`gdown`, which is too flaky to drive unattended for a 12 GB pull. The backend does a presence
check (`_check_vlabench_assets`: populated `assets/obj`) and raises `ROSConfigError` carrying
the exact `VLABENCH_ROOT=… python scripts/download_assets.py` recipe when it is missing —
the same "heavy external artefact stays manual, fail with the recipe" posture RLBench takes
for CoppeliaSim (CLAUDE.md §1.9).

**4. Ship the backend + one baseline rSkill + one primitive scene; honest scope.**
`rskills/smolvla-vlabench` wraps `lerobot/smolvla_vlabench` (Apache-2.0, bf16, fits 8 GB) and
`scenes/benchmark/vlabench_select_fruit.yaml` targets the primitive (short-horizon) suite.
Per the ADR-0060 gate, `evaluated_tasks: ["vlabench"]` covers `vlabench/<task>` ids.

## Live verification (this host)

Provisioned + verified on an 8 GB RTX 4070 Laptop (Ada, sm_89), Ubuntu 24.04, 2026-07-07.
VLABench installed editable, the 12 GB bundle unpacked under `assets/`, `rrt_algorithms`
stubbed. The OpenRAL wiring is **faithful**: `smolvla_vlabench` scores **0/3 on six diverse
primitives** (`select_fruit`, `select_drink`, `select_toy`, `select_book`, `add_condiment`,
`insert_flower`) — **identical to lerobot's own `lerobot-eval` reference** on the same tasks,
confirming the 0% is the policy, not the integration (state 7-D, action absolute-eef, cameras
`camera1/2/3` all validated). The only VLABench policy above 50% is
`VLABench/pi0-fast-ft-primitive-10task-deltachunk` (51.2% primitive avg), which is
**openpi/JAX** and needs conversion to lerobot `PI0FAST` (+int8 for 8 GB) — deferred.

## Consequences

- **Positive.** OpenRAL gains VLABench on the native in-process `SimRollout` + rSkill seams
  (no sidecar), self-provisioning on first env build. The wiring is proven correct against
  the upstream reference, so a future passing policy drops straight in.
- **Cost.** The `smolvla-vlabench` rSkill is a **0% integration baseline**, not a leaderboard
  entry (documented as such in its manifest/README). The 12 GB asset fetch is manual. The
  `rrt_algorithms` stub is a site-packages write (evicted by `uv sync`, rewritten by the
  probe) rather than a real dependency.
- **Deferred.** Converting `pi0-fast-ft` (openpi/JAX → lerobot `PI0FAST` + int8) for a
  >50% score; VLABench's composite / long-horizon suite (unsolved <50% by every known policy).

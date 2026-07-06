# ADR-0082: NVMM-native camera → in-pipeline TRT vision encoder (zero-download VLA vision path)

## Status

Proposed. Extends ADR-0011 (NVMM handoff), ADR-0037 (GStreamer perception bus —
this ADR executes its named follow-up *"migrating VLAs to consume NVMM tees
in-process"*), and the SmolVLA split-TRT runtime (`smolvla_trt.py`).

## Context

Today's deploy vision path for a VLA is CPU end to end despite TRT inference:
the camera pipeline decodes MJPEG on CPU (`v4l2src ! jpegdec ! videoconvert !
BGR ! appsink`), frames cross ROS as CPU numpy, and the TRT SmolVLA runtime
round-trips **host numpy** on every call — `_np()` pulls tensors to CPU
(`smolvla_trt.py:242`), `TensorRTRuntime.infer()` does HtoD/DtoH per call
(`runtime_tensorrt.py:437-446`, whose docstring already promises a device-
pointer path "in a later PR"), and the result is re-uploaded
(`smolvla_trt.py:256`). Per frame that is ~1.2 MB (640×480×4) decoded on CPU,
copied to GPU, with intermediates bouncing back.

Meanwhile the zero-copy machinery already exists — for the **object detector
only** (ADR-0037 amendment C4): `nvbufsurface.wrap_buffer` maps an NVMM
`GstBuffer` to a CUDA device pointer, `TrtNvmmExecutor` runs a TRT engine on
that pointer with on-GPU nvrtc preprocessing, and `TeeManager` attaches/detaches
consumer branches on a PLAYING pipeline. `TrtNvmmExecutor` even ships a
`resize_pad_pm1` kernel written for SmolVLA vision preprocessing (lerobot
`resize_with_pad` + SigLIP `[-1,1]`) that **nothing calls** (`trt_nvmm.py:74-113,
166-174`).

**Environment facts (probed 2026-07-06, live-verified):**

- The dev host has **no** NVIDIA GStreamer elements (no DeepStream, no
  `nvcodec`, no `libnvbufsurface`). The runtime for this path is the
  **`openral:x86-deepstream-latest` container** (DeepStream 9.0, GStreamer
  1.24.2, `libnvbufsurface.so` at `/opt/nvidia/deepstream/deepstream-9.0/lib/`).
- On x86 dGPU, `nvjpegdec` accepts `image/jpeg` and emits
  **`video/x-raw(memory:NVMM), format=RGB` directly** — the decoded frame is
  *born in GPU memory*. Live-verified against the real SO-101 bench camera
  (icspring, MJPG 640×480@30) inside the container:
  `v4l2src ! image/jpeg ! nvjpegdec ! NVMM(RGB) ! nvvideoconvert ! NVMM(RGBA)`
  ran 30 frames on the RTX 4070, exit 0.
- USB physics bounds the ideal: UVC DMA lands the **compressed** JPEG
  (~50–200 KB) in system RAM; that is the only PCIe crossing. The decoded
  1.2 MB frame never exists in CPU memory. CSI/GMSL cameras (Jetson/Thor) skip
  even that; the existing Tegra NVMM caps path covers them.
- The SmolVLA split export (`smolvla_export.py`) cuts exactly at the seam we
  need: `vision_encoder.onnx` maps `pixel_values → (n_cameras, T_img, hidden)`
  image embeddings; `policy_graph.onnx` consumes embeddings + language + state
  + noise. Embeddings are ~an order of magnitude smaller than the raw frame.

## Decision

Make the VLA **vision leg** NVMM-native and in-pipeline, scoped deliberately to
the vision encoder first (the policy expert stays on its current runtime):

1. **NVMM camera source on x86-DS** — `build_pipeline_string()` gains an
   x86-DeepStream tier: MJPG cameras route `v4l2src ! image/jpeg ! nvjpegdec !
   NVMM(RGB) ! nvvideoconvert ! NVMM(RGBA) ! tee`. The existing per-camera
   `openral_cam_tee` and leaky branch queues are unchanged; the CPU BGR appsink
   branch remains for ROS/world-state consumers during migration.
2. **Vision-encoder tee consumer** — a `VisionEncoderRunner` analogous to
   `DetectorRunner`/`NvmmObjectsDetector`: attaches an NVMM appsink branch via
   `TeeManager`, maps buffers with `nvbufsurface.wrap_buffer`, preprocesses with
   the dormant `resize_pad_pm1` kernel, and runs the `vision_encoder` TRT engine
   on the device pointer. Frames never touch host memory after USB.
3. **Device-side embedding handoff** — `TensorRTRuntime` gets the promised
   device-pointer input/output path; the vision engine's output device buffer is
   wrapped as a torch CUDA tensor via DLPack and fed to the policy (torch or
   policy-TRT) **without a DtoH**. Until process co-location lands (Phase 3),
   the interim boundary is the embedding — still ~10× less data than shipping
   frames, with decode + preprocess + vision encoder all on GPU.
4. **Process co-location (ADR-0037 Decision-1, previously specified, unbuilt)**
   — the GStreamer reader + vision consumer + skill runner live in one
   `runtime_node` process inside the DS container, because CUDA pointers are
   process-local. Proprio/language/noise stay host-side (bytes, not megabytes);
   the action chunk exits via the unchanged ROS → safety-kernel route.
5. **Reasoner add/remove of pipeline nodes at runtime** — per ADR-0037's model:
   activating/deactivating an rSkill (vision consumer, detector) drives
   `TeeManager.attach()/detach()` on the live pipeline — pad-blocked, no
   pipeline restart. `ReloadGstPipelineTool` stays the whole-pipeline-swap verb
   and remains a stub pending GH-126; it is **not** the per-node mechanism.

### Phasing (one PR each, in order)

| Phase | Deliverable | Proof |
|---|---|---|
| 1 | x86-DS NVMM source tier in `pipeline.py` + container smoke | live camera → NVMM appsink frames in-container — **done 2026-07-06** (`smoke_nvmm_source.py`: real bench cam → `SensorFrame.handle`, CUDA_RGBA) |
| 2 | `NvmmVisionEncoder` (multi-slot `TrtNvmmExecutor.infer_rgba_devptrs` + `resize_pad_pm1` + vision TRT on devptr) | embeddings compared vs the host-numpy path on the same cached engine — **done 2026-07-06** (real pen-skill bf16 vision engine: max-rel 1.1%, per-token cosine > 0.999, 26.6 ms/2-cam; `tests/unit/test_nvmm_vision_encoder.py`). The executor *is* the devptr I/O; a parallel `TensorRTRuntime` devptr API was dropped as redundant — the device-side embedding handoff (DLPack) lands with Phase 3. |
| 3 | in-process handle path + NVMM vision leg in deploy | live SO-101 pen deploy — **done 2026-07-06, goal SUCCEEDED** (~2100 policy steps in the DS container on the real arm; 0 e-stops, 0 collisions, 0 camera drops; no decoded pixel on the CPU). See the dated amendment below for what shipped vs the original sketch. |
| 4 | reasoner attach/detach of tee consumers (ExecuteSkill ↔ TeeManager) | live add/remove of the detector branch during a deploy run |

## Alternatives considered

- **GstCUDA (`nvcodec`) on the host** — rejected: the host has no NVIDIA
  GStreamer stack, the DS container already exists, and GstCUDA has no GPU JPEG
  decode for UVC cameras (`nvjpegdec` is DeepStream's).
- **Whole VLA (vision + policy) as one in-pipeline TRT element now** — deferred:
  the policy consumes proprio/language/noise and emits action chunks — a poor
  fit for a pad-driven element; the vision leg carries ~all the bandwidth. The
  seam is designed so a policy-TRT engine can join Phase 3's process later.
- **Reasoner edits pipeline strings (`ReloadGstPipelineTool`) for node
  add/remove** — rejected for this purpose: full-pipeline reload drops frames
  and re-negotiates the camera; `TeeManager` attach/detach is glitch-free and
  already tested.

## Consequences

- The vision path becomes: USB (compressed) → `nvjpegdec` (GPU decode, born in
  NVMM) → on-GPU preprocess → TRT vision encoder → device embeddings. The only
  per-frame PCIe crossing is the compressed JPEG; the only DtoH anywhere is the
  action chunk (a few KB).
- Deploy perception+policy moves into the `openral:x86-deepstream-latest`
  container; the plain-host CPU path remains the fallback tier (graceful
  degradation, ADR-0037 convention).
- Safety is untouched: the kernel, E-stop, and action-dispatch path are
  unchanged; a vision-leg failure degrades to the CPU path or aborts the skill,
  never bypasses a check.
- Multi-camera sync: the vision consumer must pair per-camera embeddings with
  the proprio snapshot at the inference tick; ADR-0037's aggregator tier
  already solves the N-camera pattern for the detector and is reused.
- Follow-ups: policy-graph TRT joining the co-located process; Tegra
  (`nvv4l2camerasrc`/CSI) tier reusing the same consumer unchanged.

## Amendment — 2026-07-06 (Phase 3 landed + live-validated)

What shipped differs from the original sketch in two informed ways:

- **No new co-located process was needed.** The `runtime_node` *already*
  hosts the sensor readers + WorldState aggregator + skill runner in one OS
  process (ADR-0037 Decision-1 / ADR-0018 §3); the zero-copy handle was dying
  at a pointless **intra-process ROS round trip** (reader ROS tee →
  `sensor_msgs/Image` → `_on_image` rebuilds a data-only frame). Phase 3
  therefore added a direct in-process path instead: `sensor_leg` gains an
  `_AggregatorPump` per reader writing `read_latest()` frames (handles
  intact) straight into the shared aggregator, and WorldState's new
  `direct_image_frame_sensors` parameter stops `_on_image` from
  double-writing those sensors (its ROS reconstruction keeps serving
  observability — dashboard thumbnails, detectors).
- **Buffer lifetime**: a latched NVMM `gpu_ptr` is only valid until the next
  frame swaps the Gst buffer — an async consumer racing that swap would read
  recycled memory. `StableSurfaceMirror` (reader-owned GPU double buffer)
  DtoD-copies each surface inside the appsink callback while the map is
  provably valid; worst cross-thread outcome is a torn frame, never a
  use-after-free.
- The VLA side: `obs["image_handles"]` (NVMM descriptors, VLA-slot-keyed)
  travels next to `obs["images"]`; `_SmolVLAAdapter` orders handles by the
  checkpoint's `image_features`, encodes via `NvmmVisionEncoder` (same cached
  engine as the TRT runtime), stashes embeddings via
  `set_precomputed_img_embs`, and feeds device-resident placeholder pixels to
  lerobot's plumbing. Handles without TRT, or partial camera coverage, raise
  — no silent blind fallback.
- **Live result (real SO-101, DS container, 2026-07-06):** "Pick up the pen"
  goal **SUCCEEDED** after ~2100 policy steps; 0 e-stops, 0 collisions,
  0 camera drops; one first-step latency overrun (954 ms — one-time lazy
  encoder build), in-budget after. The only per-frame PCIe crossing is the
  compressed JPEG; embeddings cross host once per chunk inference (~0.5 MB —
  device-side chaining into the policy engine remains an optional follow-up).
- **Field fixes from the container bring-up** (all landed): `jpegparse !
  nvjpegdec max-errors=-1` (a corrupt UVC MJPG frame — routine on the wrist
  cam — fatals nvjpegdec where CPU jpegdec just dropped it);
  `scenes/deploy/so101_bench.yaml` cameras moved to the spec-driven
  `jpeg: true` form (retiring the raw-pipeline workaround) and then to stable
  `/dev/v4l/by-id|by-path` device symlinks (a USB replug renumbers `/dev/videoN`
  and the bare node dies with v4l2 `-5`). The image gaps that once needed
  hand-patching are now **baked into `docker/inference/Dockerfile.x86`**: it
  builds `openral_perception_ros` (the reward/detector nodes), syncs the
  `tensorrt` + `sim` + `robometer` groups + `feetech-servo-sdk` (real so101
  HAL), COPYs `tools/` + `scenes/` and symlinks them (plus `rskills/` + `.venv`)
  under `/workspace/install` for the launch's `_REPO_ROOT`, and installs `git`
  for the Robometer sidecar's runtime venv provisioning. Skill ids still resolve
  by manifest `name` (`OpenRAL/rskill-smolvla-so101-pen`), and `HF_HUB_OFFLINE=1`
  is still needed for the gated upstream checkpoint. The first reward-enabled
  run provisions the Robometer sidecar venv (needs network or a pre-provisioned
  `OPENRAL_ROBOMETER_SIDECAR_VENV`).

## Process gates

- Phase 2 ships a bit-exactness test vs the host-numpy reference (same weights,
  same frame → same embeddings within TRT tolerance) before deploy wiring.
- Phase 3 is validated live on the SO-101 pen skill (the ADR-0081 setup) before
  the CPU vision leg is retired from the deploy default.
- No safety-WG gate: layer-1/3 data-plane change; the safety kernel contract is
  untouched.

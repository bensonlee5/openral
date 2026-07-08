# Inference deploy image

OpenRAL ships **one** deploy image. A later consolidation decision,
"Single-Dockerfile consolidation + CUDA-13/DeepStream-9 alignment,"
replaced the four-Dockerfile matrix that PR #93 introduced
(`Dockerfile.x86`, `Dockerfile.x86-ros`, `Dockerfile.x86-deepstream`,
`Dockerfile.l4t`) with a single source of truth. The OpenRAL Pro split
later moved the DeepStream + TensorRT variant out of this repo entirely
— it is now an OpenRAL Pro plugin (`openral-pro`'s `docker/Dockerfile.pro`,
which `FROM`s the image built here).

| Image | Built by | Pushed to GHCR? | License | When to use |
|---|---|---|---|---|
| `openral:x86-latest` | `just docker-build-x86` | ✅ Yes (`docker-build.yml`) | Apache-2.0 + NVIDIA CUDA runtime EULA | x86 with NVIDIA dGPU, host driver ≥ 580. The default deploy target. Carries CUDA 13, ROS 2 Jazzy, GStreamer 1.24. |
| `openral:x86-deepstream-latest` | openral-pro's `Dockerfile.pro` | ❌ **No** | Apache-2.0 **+ NVIDIA DeepStream EULA** | OpenRAL Pro only. Adds `nvvideoconvert`, NVMM caps on x86, `nvinfer`, `nvstreammux`, the TensorRT engine runtime. Local / private-registry only — see `openral-pro` for the build flow and the EULA breakdown. |

The L4T / Tegra / Jetson Orin variant, the CPU-only variant, and the
no-ROS variant from PR #93 are deliberately out of scope here. See
[`docs/decisions.md`](../../docs/decisions.md) for the trade-off rationale.

## Host driver requirements

The image's base is `nvidia/cuda:13.0.0-runtime-ubuntu24.04`. CUDA 13
needs **host NVIDIA driver ≥ 580.65**. On older drivers the image
still imports cleanly and runs non-CUDA pipelines (videotestsrc →
videoconvert → appsink, the default smoke), but every CUDA-touching
plugin will warn `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward
compatibility was attempted on non supported HW` and skip the GPU
path. `torch.cuda.is_available()` returns `False`.

| Host driver | What works | What fails |
|---|---|---|
| **≥ 580.65** | Everything: `nvh264dec/enc`, `nvjpegdec`, torch CUDA, and (OpenRAL Pro DeepStream variant) NVMM allocator + `nvvideoconvert` runtime use | — |
| **570 – 579** (e.g. 575.57 — CUDA 12.9-class) | Non-CUDA GStreamer paths (`videoconvert`, `avdec_h264`, `appsink`). 30-tick smoke against `videotestsrc` passes. `openral deploy` runs. | All `nvcodec` plugins fail to register; `torch.cuda.is_available()` returns `False`; DeepStream `nvvideoconvert` registers but `Cuda failure status=804` on first NVMM frame |
| **< 570** | nothing — base image's CUDA 13 stack stops loading entirely | everything |

The `openral doctor` command surfaces the driver version so users see the
mismatch up front.

## What's in the image

- **Base**: `nvidia/cuda:13.0.0-runtime-ubuntu24.04` (Ubuntu 24.04 noble, Py 3.12).
- **GStreamer 1.24** plugin tier (`-base`, `-good`, `-bad`, `-ugly`, `libav`, `rtsp`).
  The `nvcodec` plugin (`nvh264dec/enc`, `nvjpegdec`, `cudaupload`, etc.) is
  registered when the host driver can serve CUDA 13.
- **PyGObject** — apt-managed (`python3-gi`), spliced into the workspace venv
  at `/workspace/.venv/lib/python3.12/site-packages/gi` so it shares the same
  GLib link as ROS 2's rclpy. Mixing PyPI's PyGObject build with apt's rclpy
  segfaults at `rclpy.init()` on the gi-then-rclpy import order.
- **ROS 2 Jazzy** (`ros-jazzy-ros-base`, `ros-jazzy-sensor-msgs`,
  `ros-jazzy-rclpy`, `ros-jazzy-rmw-cyclonedds-cpp`) installed in both
  build and runtime stages. The cyclonedds rmw is preferred over Fast DDS
  because Fast DDS' SHM transport interacts badly with pydantic v2's Rust
  core + gst-cuda plugin scan.
- **uv-managed workspace venv** at `/workspace/.venv` with the OpenRAL
  Python packages plus the `sim` and `robometer` groups and
  `feetech-servo-sdk`. The `sim` group carries the lerobot /
  transformers / accelerate / bitsandbytes stack the VLA policy
  adapters import at load time (needed on real hardware too);
  `robometer` adds the reward monitor's ZMQ + msgpack sidecar client;
  `feetech-servo-sdk` is the Feetech motor driver the
  so100 / so101 real HAL needs. `uv sync` runs **without
  `--extra gstreamer`** — see the gi-splice note above. The `tensorrt`
  group (SmolVLA/ACT TRT engines) is **not** installed here — it is an
  OpenRAL Pro plugin, layered on by that repo's
  `Dockerfile.pro`.
- **colcon `install/` overlay** at `/workspace/install/` (baked by the
  builder stage). Carries every ROS / C++ package the deploy graph
  needs, mirroring `just ros2-build`:
  - `openral_msgs` — the action + message IDL the rest of the graph
    consumes (`python -c "import openral_msgs.msg"` works inside the
    image, no host `colcon build` required)
  - `opentelemetry_cpp_vendor` — builds opentelemetry-cpp from source;
    the safety kernel links against it
  - `openral_safety_kernel` — C++ deny-by-default safety process. The
    binary lands at
    `/workspace/install/openral_safety_kernel/bin/safety_kernel`
  - `openral_hal_so100`, `openral_hal_openarm` — HAL lifecycle nodes
  - `openral_world_state` — 30 Hz world-state snapshot node
  - `openral_reasoner_ros` — LLM tool dispatch
  - `openral_prompt_router` — prompt fan-in
  - `openral_safety`, `openral_safety_watchdog` — safety envelope + deadman watchdog
  - `openral_human_estop` — human e-stop forwarder
  - `openral_foxglove_bringup` — read-only allowlists imported by
    `openral_rskill_ros` launch files
  - `openral_rskill_ros` — `ExecuteSkill` action server
  - `openral_octomap_bridge` — octree → world-voxels bridge
  - `openral_perception_ros` — detector + scene-VLM + reward-monitor
    nodes (the reward monitor drives the dashboard's rSkill-card reward
    bar)

  The non-ROS trees the launch resolves from its `_REPO_ROOT`
  (`/workspace/install`) — `tools/` (autostart driver +
  Robometer sidecar scripts), `rskills/`, `scenes/`, and `.venv` —
  are COPY'd to `/workspace` and symlinked under `install/`, so
  `deploy run --config scenes/deploy/<workcell>.yaml` needs no host
  bind-mount. `git` is installed in the runtime stage for the
  Robometer sidecar's first-use venv provisioning.

  `Python3_EXECUTABLE=/workspace/.venv/bin/python` is baked into every
  ament-python package's `CTestTestfile.cmake` so the lifecycle nodes
  resolve `structlog`, `openral_*`, and the OTel SDK through the venv
  automatically — no parallel system-python install.

  Previously the perception-tee + reasoner smoke recipes (and any
  downstream consumer) bind-mounted the host's locally-built
  `install/` into the container; that requirement is gone now that
  the colcon tree ships in the image.
- **`/entrypoint.sh`** probes `/opt/ros/*/setup.bash` AND
  `/workspace/install/setup.bash`, sources both (system distro first,
  local overlay second), and exec's the user command. Probe-style
  rather than hardcoded so future arches reuse this script.
- ENV: `ROS_DISTRO=jazzy`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`,
  `ROS_DOMAIN_ID=0`, `PATH=/workspace/.venv/bin:$PATH`,
  `PYTHONUNBUFFERED=1`, `GST_DEBUG=2`.
- Default `ENTRYPOINT`: `openral deploy run` (the CLI
  ships as `openral`, no `ral` alias).

There is **no separate CUDA-13 side-load step**. PR #93's
`Dockerfile.x86-deepstream` installed `cuda-cudart-13-0` + `libnpp-13-0`
alongside the CUDA-12.6 base to make DeepStream 9 work; that hack is
gone now that the base itself is CUDA 13 — and the whole DeepStream
stage has since moved to `openral-pro` anyway.

## The DeepStream / TensorRT variant moved to OpenRAL Pro

DeepStream is **proprietary, EULA-restricted, and NOT open source**.
A 2026-05-12 decision (refined 2026-05-14) rejected bundling DeepStream
into the default image, and a later 2026-07-08 decision moved the
opt-in variant — plus the TensorRT engine runtime it depends on — into
the private `openral-pro` repo rather than keeping it here as a build
flag. If you need `nvvideoconvert` / NVMM
caps / `nvinfer` / the TensorRT-accelerated SmolVLA/ACT engines, see
`openral-pro`'s `docker/Dockerfile.pro` and its README for the build
flow and the full EULA clause-by-clause breakdown. The open-core image
built from this directory never bundles DeepStream binaries and stays
pure Apache-2.0 + NVIDIA CUDA runtime EULA.

### How the runtime sees DeepStream

The open-core pipeline builder
(`python/runner/src/openral_runner/backends/gstreamer/pipeline.py`)
returns `videoconvert` on `Platform.NVIDIA_DESKTOP` because
`nvvideoconvert` is not in the open-source `gst-plugins-bad` `nvcodec`
family. The OpenRAL Pro image patches `_build_convert` / `_build_caps`
downstream to return `nvvideoconvert` / emit `video/x-raw(memory:NVMM)`
caps on `Platform.NVIDIA_DESKTOP`. That patch must NOT be merged
upstream — it would force every default user to accept the DeepStream
EULA implicitly.

## Image sizes

Measured 2026-05-14 on the consolidation worktree (pre-OpenRAL-Pro split):

| Image | Size |
|---|---|
| `openral:x86-latest` | ~12.5 GB |
| `openral:x86-deepstream-latest` (now built by openral-pro) | ~15.1 GB |

The default image is larger than PR #93's `:x86-latest` (~11.9 GB)
because ROS Jazzy is now baked in unconditionally.

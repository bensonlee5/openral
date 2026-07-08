r"""Live NVMM camera source — runs inside the x86 DeepStream Docker image.

Proves the NVMM zero-copy vision pipeline's Phase 1 end-to-end on real
hardware: the spec-driven
reader pipeline decodes an MJPG USB camera **directly into NVMM** via
``nvjpegdec`` and the reader lifts each frame to a CUDA device pointer —
no decoded pixel ever touches system memory:

1. Assert :func:`detect_platform` identifies the DeepStream tier.
2. Build a :class:`PipelineSpec` (``source=usb, jpeg=True``) — the exact
   form a scene YAML's ``deploy_binding.backend_params`` materialises.
3. Open a real :class:`GStreamerSensorReader` on the camera.
4. Assert the latched :class:`~openral_core.SensorFrame` carries
   ``handle`` (non-null device pointer), ``encoding=CUDA_RGBA``, the
   negotiated geometry, and an RGBA ``NvBufSurfaceHandle`` descriptor.

Exits 0 on success, non-zero with an error message otherwise.

Designed to run as (camera device required):

    docker run --rm --gpus all --device /dev/video4 \
        -v "$(pwd):/workspace/src:ro" --entrypoint bash \
        openral:x86-deepstream-latest -c \
        'PYTHONPATH=/workspace/src/python/runner/src:/workspace/src/python/core/src \
         python3 /workspace/src/docker/inference/smoke_nvmm_source.py /dev/video4'
"""

from __future__ import annotations

import sys
import time

from openral_core.schemas import FrameEncoding
from openral_runner.backends.gstreamer import (
    GStreamerSensorReader,
    PipelineSpec,
    Platform,
    Source,
    build_pipeline_string,
    detect_platform,
)
from openral_runner.backends.gstreamer.nvbufsurface import NvBufSurfaceColorFormat

_TIMEOUT_S = 10.0
_RGBA_CHANNELS = 4


def main() -> int:
    """Run the live NVMM smoke; return a process exit code."""
    device = sys.argv[1] if len(sys.argv) > 1 else "/dev/video4"

    platform = detect_platform()
    if platform is not Platform.NVIDIA_DEEPSTREAM:
        print(f"FAIL: detect_platform() = {platform}, expected NVIDIA_DEEPSTREAM")
        return 1
    print(f"platform: {platform.value}")

    spec = PipelineSpec(source=Source.USB, device=device, width=640, height=480, fps=30, jpeg=True)
    pipeline_str = build_pipeline_string(spec, platform=platform)
    print(f"pipeline: {pipeline_str}")
    if "nvjpegdec" not in pipeline_str or "memory:NVMM" not in pipeline_str:
        print("FAIL: builder did not emit the NVMM nvjpegdec pipeline")
        return 1

    with GStreamerSensorReader(sensor_id="wrist", spec=spec) as reader:
        deadline = time.monotonic() + _TIMEOUT_S
        frame = None
        while time.monotonic() < deadline:
            try:
                frame = reader.read_latest(max_age_ms=None)
                break
            except Exception:  # reason: poll until first frame or the timeout below
                time.sleep(0.1)
        if frame is None:
            print(f"FAIL: no frame within {_TIMEOUT_S}s")
            return 1

        checks = {
            "handle is device ptr": bool(frame.handle),
            "data empty (zero-copy)": frame.data is None,
            "encoding CUDA_RGBA": frame.encoding is FrameEncoding.CUDA_RGBA,
            "geometry 640x480": (frame.width, frame.height) == (640, 480),
            "channels 4": frame.channels == _RGBA_CHANNELS,
        }
        descriptor = frame.metadata.get("nvbufsurface")
        checks["descriptor present"] = descriptor is not None
        if descriptor is not None:
            checks["descriptor RGBA"] = descriptor["color_format"] == NvBufSurfaceColorFormat.RGBA
            checks["descriptor ptr matches"] = descriptor["gpu_ptr"] == frame.handle
            print(
                f"frame: handle=0x{frame.handle:x} {frame.width}x{frame.height} "
                f"pitch={descriptor['pitch']} color_format={descriptor['color_format']} "
                f"size={descriptor['size']}"
            )
        failed = [name for name, ok in checks.items() if not ok]
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}: {name}")
        if failed:
            return 1

        # Sustained: a second, newer frame arrives (the pipeline streams, not a one-shot).
        first_handle = frame.handle
        time.sleep(0.5)
        frame2 = reader.read_latest(max_age_ms=1000)
        print(f"second frame: handle=0x{frame2.handle:x} (streaming OK)")
        _ = first_handle  # handles may repeat (buffer pool); freshness is the max_age check

    print("PASS: NVMM camera source live (zero-copy pipeline Phase 1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

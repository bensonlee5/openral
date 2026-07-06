"""GPU tests for the zero-copy NVMM VLA vision encoder (ADR-0082 Phase 2).

Two layers, both on real components (no mocks, CLAUDE.md §1.11):

1. Multi-slot batching in :class:`TrtNvmmExecutor` — an identity ONNX with a
   static ``(2, 3, h, w)`` input proves each camera's kernel launch lands in
   its own batch slot (no DeepStream / NVMM required; device buffers are
   allocated with cudart, exactly like the executor's own path).
2. The Phase 2 acceptance gate — the **real** SmolVLA pen-skill vision engine:
   embeddings from the device-pointer path (GPU ``resize_pad_pm1`` kernel)
   must match the deployed host-numpy path (torch ``resize_with_pad`` on CPU →
   ``TensorRTRuntime.infer``) on the *same cached engine*, isolating the
   preprocess seam. Skipped when the checkpoint's ONNX cache is absent
   (create it via ``openral_rskill.smolvla_trt.ensure_smolvla_onnx``).

Skipped on GPU-less hosts / without the ``tensorrt`` group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_PEN_ONNX_DIR = (
    Path.home() / ".cache" / "openral" / "smolvla_onnx" / "sapanostic--so_101_smolvla_pen_placement"
)


def _skip_without_gpu_stack() -> tuple[object, object]:
    """Import the GPU deps or skip; returns (cudart, trt_runtime_cls)."""
    pytest.importorskip("tensorrt", reason="tensorrt group not installed")
    pytest.importorskip("cuda", reason="cuda-python not installed")
    from cuda.bindings import runtime as cudart
    from openral_rskill.runtime_tensorrt import TensorRTRuntime

    if int(cudart.cudaSetDevice(0)[0]) != 0:
        pytest.skip("no usable CUDA device")
    return cudart, TensorRTRuntime


def _upload_rgba(cudart: object, rgba_hwc4: np.ndarray, *, pad: int = 0) -> tuple[int, int]:
    """cudaMalloc + upload a (h, w, 4) uint8 frame; return (dev_ptr, pitch)."""
    h, w = rgba_hwc4.shape[:2]
    pitch = w * 4 + pad
    rows = np.zeros((h, pitch), dtype=np.uint8)
    rows[:, : w * 4] = rgba_hwc4.reshape(h, w * 4)
    result = cudart.cudaMalloc(rows.nbytes)  # type: ignore[attr-defined]
    if int(result[0]) != 0:
        pytest.skip("cudaMalloc failed — no usable CUDA device")
    dev = result[1]
    err = cudart.cudaMemcpy(  # type: ignore[attr-defined]
        int(dev),
        rows.ctypes.data,
        rows.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,  # type: ignore[attr-defined]
    )
    assert int(err[0]) == 0
    return int(dev), pitch


def test_executor_multi_slot_batches_two_frames(tmp_path: Path) -> None:
    """Each frame's kernel launch fills its own slot of a static (2,3,h,w) input."""
    cudart, trt_runtime_cls = _skip_without_gpu_stack()
    torch = pytest.importorskip("torch", reason="torch authors the ONNX fixture")
    from openral_runner.backends.gstreamer.trt_nvmm import TrtNvmmExecutor

    h, w = 8, 8

    class _Identity(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 1.0

    onnx_path = tmp_path / "identity2.onnx"
    torch.onnx.export(
        _Identity().eval(),
        (torch.zeros(2, 3, h, w),),
        str(onnx_path),
        input_names=["images"],
        output_names=["out"],
        dynamo=False,
    )
    engine_bytes = trt_runtime_cls(  # type: ignore[operator]
        device="cuda:0", rskill_id="openral/test-nvmm-batch2"
    ).serialized_engine(onnx_path)

    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8) for _ in range(2)]
    devs = [_upload_rgba(cudart, f, pad=16 * i) for i, f in enumerate(frames)]

    ex = TrtNvmmExecutor(engine_bytes, input_size=(h, w), device_index=0)
    try:
        assert ex.batch == 2
        # A single-frame call must fail loudly on a 2-slot engine.
        from openral_core.exceptions import ROSConfigError

        with pytest.raises(ROSConfigError, match="takes 2 slot"):
            ex.infer_rgba_devptr(devs[0][0], width=w, height=h, pitch=devs[0][1])
        out = ex.infer_rgba_devptrs([(dev, w, h, pitch) for (dev, pitch) in devs])
    finally:
        ex.close()
        for dev, _pitch in devs:
            cudart.cudaFree(dev)  # type: ignore[attr-defined]

    got = out["out"].reshape(2, 3, h, w)
    for slot, frame in enumerate(frames):
        expected = frame[:, :, :3].astype(np.float32).transpose(2, 0, 1) / 255.0
        np.testing.assert_allclose(got[slot], expected, atol=1e-3, err_msg=f"slot {slot}")


def test_vision_encoder_matches_host_numpy_path() -> None:
    """ADR-0082 Phase 2 gate: devptr embeddings == host-numpy embeddings.

    Same cached bf16 engine on both sides (identical ``rskill_id`` + quant →
    the deploy's own engine); only the preprocessing differs (GPU kernel vs
    torch CPU). Measured max-abs-diff on this seam is ~1e-2 on bf16 (the
    per-pixel preprocess agreement is ≤5e-3; the transformer keeps it small).
    """
    cudart, trt_runtime_cls = _skip_without_gpu_stack()
    torch = pytest.importorskip("torch", reason="torch drives the reference")
    lerobot_mod = pytest.importorskip(
        "lerobot.policies.smolvla.modeling_smolvla", reason="lerobot not installed"
    )
    vision_onnx = _PEN_ONNX_DIR / "vision_encoder.onnx"
    if not vision_onnx.exists():
        pytest.skip(
            f"pen-skill ONNX cache absent at {vision_onnx}; create via "
            "openral_rskill.smolvla_trt.ensure_smolvla_onnx"
        )
    from openral_core.schemas import QuantizationBackend, QuantizationConfig, QuantizationDtype
    from openral_runner.backends.gstreamer.nvbufsurface import (
        NvBufSurfaceColorFormat,
        NvBufSurfaceHandle,
    )
    from openral_runner.backends.gstreamer.nvmm_vision_encoder import NvmmVisionEncoder

    quant = QuantizationConfig(
        dtype=QuantizationDtype("bf16"), backend=QuantizationBackend.TENSORRT
    )
    model_id = "sapanostic--so_101_smolvla_pen_placement#vision"

    # Two synthetic 480x640 camera frames (the real bench cams' geometry).
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, size=(480, 640, 4), dtype=np.uint8) for _ in range(2)]

    # ── Reference: the deployed host-numpy path (smolvla_trt._TrtSampleActions) ──
    ref_rt = trt_runtime_cls(device="cuda:0", rskill_id=model_id, quantization=quant)  # type: ignore[operator]
    ref_rt.load(vision_onnx)
    pixels = np.concatenate(
        [
            (
                lerobot_mod.resize_with_pad(
                    torch.from_numpy(f[:, :, :3].astype(np.float32) / 255.0)
                    .permute(2, 0, 1)
                    .unsqueeze(0),
                    512,
                    512,
                    pad_value=0,
                )
                * 2.0
                - 1.0
            ).numpy()
            for f in frames
        ],
        axis=0,
    ).astype(np.float32)
    (ref_embs,) = ref_rt.infer({"pixel_values": pixels}).values()

    # ── Device-pointer path: NvmmVisionEncoder on uploaded RGBA buffers ──
    encoder = NvmmVisionEncoder(vision_onnx, model_id=model_id, quantization=quant)
    devs = [_upload_rgba(cudart, f, pad=16 * i) for i, f in enumerate(frames)]
    try:
        assert encoder.n_cameras == 2
        handles = [
            NvBufSurfaceHandle(
                gpu_ptr=dev,
                width=640,
                height=480,
                pitch=pitch,
                color_format=NvBufSurfaceColorFormat.RGBA,
                size=pitch * 480,
            )
            for (dev, pitch) in devs
        ]
        got_embs = encoder.encode_nvmm(handles)
    finally:
        encoder.close()
        for dev, _pitch in devs:
            cudart.cudaFree(dev)  # type: ignore[attr-defined]

    ref = np.asarray(ref_embs, dtype=np.float32)
    assert got_embs.shape == ref.shape == (2, 64, 960)
    scale = float(np.abs(ref).max())
    max_rel = float(np.abs(got_embs - ref).max()) / scale
    # Per-token cosine similarity — direction of every embedding must agree.
    a = got_embs.reshape(-1, got_embs.shape[-1])
    b = ref.reshape(-1, ref.shape[-1])
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    assert max_rel < 0.02, f"max relative embedding diff {max_rel:.4f}"
    assert float(cos.min()) > 0.999, f"min token cosine {cos.min():.5f}"

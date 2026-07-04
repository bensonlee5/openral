"""GPU round-trip test for the clean-room NVMM TRT executor (ADR-0037 PR5b).

No DeepStream / NVMM required: an identity ONNX lets us assert the CUDA
RGBA->NCHW kernel read a real device buffer correctly through the engine,
including pitch-padded rows. The source RGBA buffer is allocated with cudart
(cuda-python), matching the executor's nvrtc + cuda-python path (no pycuda).
Skipped when tensorrt / cuda-python / a CUDA GPU is unavailable (GPU-less CI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _export_identity_onnx(path: Path, *, h: int, w: int) -> None:
    """Export y = x on (1,3,h,w) so the engine output equals the preprocessed input."""
    torch = pytest.importorskip("torch", reason="torch authors the ONNX fixture")

    class _Identity(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 1.0

    torch.onnx.export(
        _Identity().eval(),
        (torch.zeros(1, 3, h, w),),
        str(path),
        input_names=["images"],
        output_names=["out"],
        dynamo=False,
    )


@pytest.mark.parametrize("pad", [0, 16])  # contiguous and pitch-padded rows
def test_executor_reads_gpu_rgba_through_kernel(tmp_path: Path, pad: int) -> None:
    pytest.importorskip("tensorrt", reason="tensorrt group not installed")
    pytest.importorskip("cuda", reason="cuda-python not installed")
    from cuda.bindings import runtime as cudart
    from openral_rskill.runtime_tensorrt import TensorRTRuntime
    from openral_runner.backends.gstreamer.trt_nvmm import TrtNvmmExecutor

    def _rt(result: tuple[object, ...]) -> tuple[object, ...]:
        if int(result[0]) != 0:  # type: ignore[call-overload]
            pytest.skip(f"no usable CUDA device: {cudart.cudaGetErrorString(result[0])[1]!r}")
        return result[1:]

    # cudaSetDevice initializes the primary context; skip on a GPU-less host.
    if int(cudart.cudaSetDevice(0)[0]) != 0:
        pytest.skip("no usable CUDA device")

    h, w = 8, 8
    onnx_path = tmp_path / "identity.onnx"
    _export_identity_onnx(onnx_path, h=h, w=w)
    engine_bytes = TensorRTRuntime(
        device="cuda:0", rskill_id="openral/test-nvmm"
    ).serialized_engine(onnx_path)

    # Known RGBA frame, optionally pitch-padded (extra bytes per row beyond w*4).
    pitch = w * 4 + pad
    ref = np.random.default_rng(0).integers(0, 256, size=(h, w, 4), dtype=np.uint8)
    rgba = np.zeros((h, pitch), dtype=np.uint8)
    rgba[:, : w * 4] = ref.reshape(h, w * 4)

    # Allocate the source buffer on-device with cudart and upload the frame, so
    # the executor's kernel (same primary context) can read it by pointer.
    (src,) = _rt(cudart.cudaMalloc(rgba.nbytes))
    _rt(
        cudart.cudaMemcpy(
            int(src),
            rgba.ctypes.data,
            rgba.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
    )

    ex = TrtNvmmExecutor(engine_bytes, input_size=(h, w), device_index=0)
    try:
        out = ex.infer_rgba_devptr(int(src), width=w, height=h, pitch=pitch)
    finally:
        ex.close()
        cudart.cudaFree(int(src))

    nchw = out["out"].reshape(3, h, w)
    expected = ref[:, :, :3].astype(np.float32).transpose(2, 0, 1) / 255.0
    np.testing.assert_allclose(nchw, expected, atol=1e-3)


def test_resize_pad_geometry_matches_lerobot_reference() -> None:
    """Pure geometry math equals lerobot ``resize_with_pad`` output shapes."""
    lerobot_mod = pytest.importorskip(
        "lerobot.policies.smolvla.modeling_smolvla", reason="lerobot not installed"
    )
    torch = pytest.importorskip("torch", reason="torch drives the reference")
    from openral_runner.backends.gstreamer.trt_nvmm import resize_pad_geometry

    for src_h, src_w in [(480, 640), (720, 1280), (512, 512), (100, 60), (513, 511)]:
        ref = lerobot_mod.resize_with_pad(torch.zeros(1, 3, src_h, src_w), 512, 512, pad_value=0)
        resized_h, resized_w, pad_h, pad_w = resize_pad_geometry(src_h, src_w, 512, 512)
        assert ref.shape == (1, 3, 512, 512)
        assert (pad_h + resized_h, pad_w + resized_w) == (512, 512), (src_h, src_w)
        # lerobot pads top/left with pad_value; assert our pad box matches where
        # the reference is exactly pad_value*2-1... reference here is pre-norm 0.
        if pad_h:
            assert float(ref[0, 0, pad_h - 1, -1]) == 0.0
            assert float(ref[0, 0, pad_h, -1]) == 0.0  # image content is zeros too


@pytest.mark.parametrize("pad", [0, 16])  # contiguous and pitch-padded rows
def test_executor_resize_pad_pm1_matches_torch(tmp_path: Path, pad: int) -> None:
    """resize_pad_pm1 kernel == lerobot resize_with_pad + [-1,1] norm, on GPU."""
    pytest.importorskip("tensorrt", reason="tensorrt group not installed")
    pytest.importorskip("cuda", reason="cuda-python not installed")
    torch = pytest.importorskip("torch", reason="torch drives the reference")
    lerobot_mod = pytest.importorskip(
        "lerobot.policies.smolvla.modeling_smolvla", reason="lerobot not installed"
    )
    from cuda.bindings import runtime as cudart
    from openral_rskill.runtime_tensorrt import TensorRTRuntime
    from openral_runner.backends.gstreamer.trt_nvmm import TrtNvmmExecutor

    if int(cudart.cudaSetDevice(0)[0]) != 0:
        pytest.skip("no usable CUDA device")

    def _rt(result: tuple[object, ...]) -> tuple[object, ...]:
        if int(result[0]) != 0:  # type: ignore[call-overload]
            pytest.skip(f"no usable CUDA device: {cudart.cudaGetErrorString(result[0])[1]!r}")
        return result[1:]

    net_h, net_w = 32, 32
    src_h, src_w = 24, 40  # aspect != 1 so both resize and top-pad paths run
    onnx_path = tmp_path / "identity.onnx"
    _export_identity_onnx(onnx_path, h=net_h, w=net_w)
    engine_bytes = TensorRTRuntime(
        device="cuda:0", rskill_id="openral/test-nvmm-resize"
    ).serialized_engine(onnx_path)

    rng = np.random.default_rng(3)
    rgba = rng.integers(0, 256, size=(src_h, src_w, 4), dtype=np.uint8)
    pitch = src_w * 4 + pad
    padded = np.zeros((src_h, pitch), dtype=np.uint8)
    padded[:, : src_w * 4] = rgba.reshape(src_h, src_w * 4)

    (src,) = _rt(cudart.cudaMalloc(padded.nbytes))
    _rt(
        cudart.cudaMemcpy(
            int(src),
            padded.ctypes.data,
            padded.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
    )
    ex = TrtNvmmExecutor(
        engine_bytes,
        input_size=(net_h, net_w),
        device_index=0,
        preprocess="resize_pad_pm1",
    )
    try:
        out = ex.infer_rgba_devptr(int(src), width=src_w, height=src_h, pitch=pitch)
    finally:
        ex.close()
        cudart.cudaFree(int(src))

    img = torch.from_numpy(rgba[:, :, :3].astype(np.float32) / 255.0)
    img = img.permute(2, 0, 1).unsqueeze(0)
    ref = lerobot_mod.resize_with_pad(img, net_w, net_h, pad_value=0) * 2.0 - 1.0

    got = out["out"].reshape(3, net_h, net_w)
    np.testing.assert_allclose(got, ref[0].numpy(), atol=5e-3)

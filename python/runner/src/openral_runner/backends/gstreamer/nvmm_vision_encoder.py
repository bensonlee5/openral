"""Zero-copy NVMM VLA vision encoder (ADR-0082 Phase 2).

Runs a SmolVLA **vision-encoder TRT engine** (the split export's
``vision_encoder.onnx``, ADR-0037 follow-up) directly on NVMM camera frames:
one :class:`~openral_runner.backends.gstreamer.trt_nvmm.TrtNvmmExecutor` in
``resize_pad_pm1`` mode preprocesses each camera's RGBA device pointer into
its batch slot (the exact lerobot ``resize_with_pad`` + SigLIP ``[-1, 1]``
normalization, on-GPU) and the engine emits the ``(n_cameras, T_img, hidden)``
image embeddings. No decoded pixel ever touches host memory; only the
embeddings (~an order of magnitude smaller than the frames) are copied back —
the device-side embedding handoff to the policy is ADR-0082 Phase 3.

Mirrors :class:`~openral_runner.backends.gstreamer.nvmm_detector.NvmmObjectsDetector`:
compose ``TensorRTRuntime.serialized_engine`` (build + per-host cache) with the
executor. Requires the ``tensorrt`` group (``cuda-python`` + ``tensorrt``) +
``nvrtc``; deploys in the lean DeepStream ds-on image (no pycuda).
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_runner.backends.gstreamer.trt_nvmm import TrtNvmmExecutor

log = structlog.get_logger(__name__)

__all__ = ["NvmmVisionEncoder"]


def _ms_since(start_ns: int) -> float:
    """Return elapsed milliseconds from a ``perf_counter_ns`` start."""
    return (perf_counter_ns() - start_ns) / 1_000_000.0


class NvmmVisionEncoder:
    """Zero-copy NVMM vision encoder wrapping a TRT engine built from ONNX.

    Args:
        onnx_path: Path to the split-export ``vision_encoder.onnx``
            (:func:`openral_rskill.smolvla_trt.ensure_smolvla_onnx`). Its
            static input shape fixes ``n_cameras`` and the network size.
        model_id: Identifier for the engine cache / logs (e.g. the checkpoint
            repo id).
        device_index: CUDA device ordinal.
        quantization: Build-time TRT quantization (``QuantizationConfig``;
            bf16 is the validated SmolVLA deploy precision). ``None`` = fp32.

    Raises:
        ROSConfigError: On missing deps, a missing/invalid ONNX, or an engine
            whose input is not a static ``(n, 3, H, W)`` image batch.

    Example:
        >>> # Exercised live (real vision engine, GPU) in
        >>> # tests/unit/test_nvmm_vision_encoder.py; doctest skipped because
        >>> # tensorrt / cuda-python are optional at doctest time.
        >>> pass
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        model_id: str,
        device_index: int = 0,
        quantization: Any = None,  # noqa: ANN401  # reason: QuantizationConfig | None — avoid import at signature
    ) -> None:
        """Build (or cache-load) the vision engine and its NVMM executor."""
        from openral_rskill.runtime_tensorrt import TensorRTRuntime  # noqa: PLC0415

        p = Path(onnx_path)
        if not p.exists():
            raise ROSConfigError(f"NvmmVisionEncoder: vision ONNX not found at '{p}'.")

        started_ns = perf_counter_ns()
        runtime = TensorRTRuntime(
            device=f"cuda:{device_index}", rskill_id=model_id, quantization=quantization
        )
        engine_bytes = runtime.serialized_engine(p)
        engine_ms = _ms_since(started_ns)

        # The executor derives n_cameras from the engine's static batch dim; the
        # network size comes from the ONNX input (read cheaply, no session).
        import onnx  # noqa: PLC0415  # reason: deferred heavy dep

        dims = onnx.load(str(p), load_external_data=False).graph.input[0].type.tensor_type.shape
        net_h, net_w = (int(dims.dim[2].dim_value), int(dims.dim[3].dim_value))
        executor_started_ns = perf_counter_ns()
        self._executor = TrtNvmmExecutor(
            engine_bytes,
            input_size=(net_h, net_w),
            device_index=device_index,
            preprocess="resize_pad_pm1",
        )
        self._model_id = model_id
        self._last_timings_ms = {
            "engine_ms": engine_ms,
            "executor_init_ms": _ms_since(executor_started_ns),
            "init_total_ms": _ms_since(started_ns),
        }
        log.debug(
            "nvmm_vision_encoder.created",
            model_id=model_id,
            n_cameras=self._executor.batch,
            input_size=(net_h, net_w),
            timings_ms=self._last_timings_ms,
        )

    @property
    def n_cameras(self) -> int:
        """Camera slots the vision engine was exported for."""
        return self._executor.batch

    @property
    def last_timings_ms(self) -> dict[str, float]:
        """Most recent init/encode timing breakdown in milliseconds."""
        return dict(self._last_timings_ms)

    def encode_nvmm(self, handles: list[Any]) -> NDArray[np.float32]:  # NvBufSurfaceHandle list
        """Encode one NVMM frame per camera into image embeddings, zero-copy.

        Args:
            handles: One :class:`NvBufSurfaceHandle` per camera, in the
                checkpoint's camera order (the order lerobot's
                ``prepare_images`` produces — the graph bakes that ordering).

        Returns:
            ``(n_cameras, T_img, hidden)`` float32 image embeddings.

        Raises:
            ROSConfigError: On a camera-count mismatch.
            ROSRuntimeError: On a CUDA / engine failure.
        """
        started_ns = perf_counter_ns()
        outputs = self._executor.infer_rgba_devptrs(
            [(h.gpu_ptr, h.width, h.height, h.pitch) for h in handles]
        )
        (embeddings,) = outputs.values()
        self._last_timings_ms = {"encode_ms": _ms_since(started_ns)}
        return np.asarray(embeddings, dtype=np.float32)

    def close(self) -> None:
        """Release the executor's device buffers. Idempotent."""
        self._executor.close()

"""Device-resident ACT inference — NVMM RGBA handles → GPU preprocess → TRT (ADR-0082).

The DeepStream camera leg delivers each frame as an NVMM device pointer, never
touching host memory. This executor runs the ACT **device** engine (built from
``tools/export_act_onnx.py --preprocess device``: ``[0,1]`` RGB image inputs +
raw 6-D state → raw action, with image/state normalize + action unnormalize all
folded into the graph) straight on those pointers:

* the nvrtc RGBA→NCHW/255 kernel (reused verbatim from
  :class:`~openral_runner.backends.gstreamer.trt_nvmm.TrtNvmmExecutor`) fills a
  per-camera device buffer — one kernel launch per camera, still zero host copy;
* the 6-D proprio state is a negligible host→device copy;
* only the ``(1, chunk, action_dim)`` action chunk is copied back to the host.

Unlike :class:`TrtNvmmExecutor` (one image input — the SmolVLA vision tower),
ACT is monolithic with **two image inputs + a state input**, so it needs its own
multi-input binder; the low-level kernel + CUDA-error plumbing is shared.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

from openral_runner.backends.gstreamer.trt_nvmm import (
    _BLOCK_X,
    _BLOCK_Y,
    _KERNEL_NAME,
    _KERNEL_SRC,
    TrtNvmmExecutor,
)

log = structlog.get_logger(__name__)

# Reuse the CUDA-error-check helpers (they are pure static methods).
_rt = TrtNvmmExecutor._rt
_dr = TrtNvmmExecutor._dr
_nv = TrtNvmmExecutor._nv

_NCHW_RANK = 4
_RGB_CHANNELS = 3


class ActNvmmExecutor:
    """Run the ACT device engine on NVMM RGBA device pointers, no host vision copy.

    Args:
        engine_bytes: Serialized TRT engine (the ACT ``device`` engine).
        image_input_names: Engine image-input tensor names, in the order the
            checkpoint expects (``config.image_features``, e.g.
            ``["img_wrist", "img_front"]``). Each is a ``(1, 3, H, W)`` fp32 input.
        state_input_name: Engine state-input tensor name (``"state"``).
        height, width: Network image size; the NVMM frames must match it
            (the camera caps scale to it).
        device_index: CUDA device ordinal.

    Raises:
        ROSConfigError: Missing GPU deps, bad engine, or an input shape mismatch.
        ROSRuntimeError: A CUDA / nvrtc call fails during setup.
    """

    def __init__(
        self,
        engine_bytes: bytes,
        *,
        image_input_names: list[str],
        state_input_name: str,
        height: int,
        width: int,
        device_index: int = 0,
    ) -> None:
        """Compile the kernel, deserialize the engine, and bind static I/O buffers."""
        try:
            from cuda.bindings import driver as cuda, nvrtc, runtime as cudart  # noqa: PLC0415,I001

            import tensorrt as trt  # noqa: PLC0415
        except ImportError as exc:
            raise ROSConfigError(
                "ActNvmmExecutor needs cuda-python + tensorrt + nvrtc "
                "(the DeepStream runtime image / `uv sync --group tensorrt`)."
            ) from exc

        self._cuda = cuda
        self._cudart = cudart
        self._h, self._w = height, width
        self._device_index = device_index
        self._image_input_names = list(image_input_names)
        self._state_input_name = state_input_name
        self._in_bufs: dict[str, Any] = {}
        self._state_dev: Any = None
        self._state_dim = 0
        self._outputs: dict[str, tuple[Any, Any]] = {}
        self._module: Any = None
        self._func: Any = None
        self._stream: Any = None
        self._closed = False
        try:
            self._build(engine_bytes, cuda, cudart, nvrtc, trt)
        except BaseException:
            self._free_resources()
            raise
        log.debug(
            "act_nvmm.ready", images=self._image_input_names, outputs=list(self._outputs)
        )

    def _build(
        self,
        engine_bytes: bytes,
        cuda: Any,  # noqa: ANN401  # reason: cuda.bindings.driver is untyped
        cudart: Any,  # noqa: ANN401  # reason: cuda.bindings.runtime is untyped
        nvrtc: Any,  # noqa: ANN401  # reason: cuda.bindings.nvrtc is untyped
        trt: Any,  # noqa: ANN401  # reason: tensorrt is untyped
    ) -> None:
        # ── nvrtc-compile the RGBA→NCHW/255 kernel to a SASS CUBIN for this device ──
        _rt(cudart.cudaSetDevice(self._device_index), cudart)
        (props,) = _rt(cudart.cudaGetDeviceProperties(self._device_index), cudart)
        cc = f"{props.major}{props.minor}"
        (prog,) = _nv(
            nvrtc.nvrtcCreateProgram(_KERNEL_SRC, _KERNEL_NAME + b".cu", 0, [], []), nvrtc
        )
        try:
            opts = [f"--gpu-architecture=sm_{cc}".encode()]
            _nv(nvrtc.nvrtcCompileProgram(prog, len(opts), opts), nvrtc, prog)
            (size,) = _nv(nvrtc.nvrtcGetCUBINSize(prog), nvrtc)
            cubin = b" " * size
            _nv(nvrtc.nvrtcGetCUBIN(prog, cubin), nvrtc)
        finally:
            nvrtc.nvrtcDestroyProgram(prog)
        _dr(cuda.cuInit(0), cuda)
        (self._module,) = _dr(cuda.cuModuleLoadData(cubin), cuda)
        (self._func,) = _dr(cuda.cuModuleGetFunction(self._module, _KERNEL_NAME), cuda)

        # ── deserialize the engine + bind static I/O buffers ──
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise ROSConfigError("ActNvmmExecutor: failed to deserialize TRT engine bytes.")
        self._engine = engine
        self._context = engine.create_execution_context()

        slot_bytes = 4 * 3 * self._h * self._w  # fp32 CHW
        for name in self._image_input_names:
            shape = tuple(int(d) for d in engine.get_tensor_shape(name))
            if (
                len(shape) != _NCHW_RANK
                or shape[1] != _RGB_CHANNELS
                or (shape[2], shape[3]) != (self._h, self._w)
            ):
                raise ROSConfigError(
                    f"ActNvmmExecutor: image input {name!r} shape {shape} != "
                    f"(1,3,{self._h},{self._w})."
                )
            (dev,) = _rt(cudart.cudaMalloc(slot_bytes), cudart)
            self._in_bufs[name] = dev
            self._context.set_tensor_address(name, int(dev))

        sshape = tuple(int(d) for d in engine.get_tensor_shape(self._state_input_name))
        self._state_dim = sshape[-1]
        (self._state_dev,) = _rt(cudart.cudaMalloc(4 * self._state_dim), cudart)
        self._context.set_tensor_address(self._state_input_name, int(self._state_dev))

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(int(d) for d in self._context.get_tensor_shape(name))
                dt = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
                host = np.empty(shape, dtype=dt)
                (dev,) = _rt(cudart.cudaMalloc(host.nbytes), cudart)
                self._context.set_tensor_address(name, int(dev))
                self._outputs[name] = (host, dev)

        (self._stream,) = _rt(cudart.cudaStreamCreate(), cudart)

    def _fill_image(
        self,
        dst_dev: Any,  # noqa: ANN401  # reason: untyped cuda device pointer
        src_ptr: int,
        *,
        width: int,
        height: int,
        pitch: int,
    ) -> None:
        """Enqueue the RGBA→NCHW/255 kernel writing one camera's device buffer."""
        if (height, width) != (self._h, self._w):
            raise ROSConfigError(
                f"ActNvmmExecutor: frame {height}x{width} != network {self._h}x{self._w}; "
                "the NVMM camera caps must scale to the network size."
            )
        p_dst = np.array([int(dst_dev)], dtype=np.uint64)
        p_src = np.array([int(src_ptr)], dtype=np.uint64)
        scalar_arrs = [np.array([s], dtype=np.int32) for s in (height, width, pitch)]
        kargs = np.array(
            [p_dst.ctypes.data, p_src.ctypes.data] + [a.ctypes.data for a in scalar_arrs],
            dtype=np.uint64,
        )
        gx = (width + _BLOCK_X - 1) // _BLOCK_X
        gy = (height + _BLOCK_Y - 1) // _BLOCK_Y
        _dr(
            self._cuda.cuLaunchKernel(
                self._func, gx, gy, 1, _BLOCK_X, _BLOCK_Y, 1, 0, int(self._stream),
                kargs.ctypes.data, 0,
            ),
            self._cuda,
        )

    def infer(self, frames: dict[str, tuple[int, int, int, int]], state_raw: Any) -> Any:  # noqa: ANN401  # reason: numpy array in/out
        """Run one step: NVMM frames + raw state → raw action chunk (host numpy).

        Args:
            frames: ``engine_image_input_name -> (src_ptr, width, height, pitch)``
                for every image input.
            state_raw: Raw 6-D proprio state (any array-like), pre-unit-conversion —
                exactly what the host path feeds the checkpoint's preprocessor.

        Returns:
            The single engine output as a ``numpy.ndarray`` (``(1, chunk, dim)``).

        Raises:
            ROSConfigError: A required image input has no frame / wrong size.
            ROSRuntimeError: A CUDA call or ``execute_async_v3`` fails.
        """
        cudart = self._cudart
        for name in self._image_input_names:
            if name not in frames:
                raise ROSConfigError(f"ActNvmmExecutor: no NVMM frame for image input {name!r}.")
            src_ptr, width, height, pitch = frames[name]
            self._fill_image(self._in_bufs[name], src_ptr, width=width, height=height, pitch=pitch)

        state = np.ascontiguousarray(state_raw, dtype=np.float32).reshape(1, self._state_dim)
        _rt(
            cudart.cudaMemcpyAsync(
                int(self._state_dev), state.ctypes.data, state.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, int(self._stream),
            ),
            cudart,
        )
        if not self._context.execute_async_v3(int(self._stream)):
            raise ROSRuntimeError("ActNvmmExecutor: execute_async_v3 returned False.")
        for _name, (host, dev) in self._outputs.items():
            _rt(
                cudart.cudaMemcpyAsync(
                    host.ctypes.data, int(dev), host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, int(self._stream),
                ),
                cudart,
            )
        _rt(cudart.cudaStreamSynchronize(int(self._stream)), cudart)
        del state  # kept alive across the async HtoD until the sync above
        (out,) = self._outputs.values()
        return np.array(out[0])

    def _free_resources(self) -> None:
        """Free device buffers, unload the kernel module, destroy the stream. Idempotent."""
        cudart = getattr(self, "_cudart", None)
        cuda = getattr(self, "_cuda", None)
        if cudart is not None:
            for _host, dev in getattr(self, "_outputs", {}).values():
                cudart.cudaFree(dev)
            for dev in getattr(self, "_in_bufs", {}).values():
                cudart.cudaFree(dev)
            if getattr(self, "_state_dev", None) is not None:
                cudart.cudaFree(self._state_dev)
            if getattr(self, "_stream", None) is not None:
                cudart.cudaStreamDestroy(self._stream)
        if cuda is not None and getattr(self, "_module", None) is not None:
            cuda.cuModuleUnload(self._module)
        self._outputs = {}
        self._in_bufs = {}
        self._state_dev = None
        self._stream = None
        self._module = None
        self._func = None

    def close(self) -> None:
        """Free all GPU resources. Idempotent."""
        if self._closed:
            return
        self._free_resources()
        self._closed = True

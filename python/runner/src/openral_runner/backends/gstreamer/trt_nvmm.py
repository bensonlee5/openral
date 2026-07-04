"""Clean-room zero-copy TensorRT executor for NVMM frames (ADR-0037 PR5b).

Runs a TensorRT engine **directly on a CUDA device pointer** (the
``NvBufSurface.dataPtr`` of an NVMM GStreamer buffer) with no GPU->CPU copy. A
small CUDA kernel converts the pitch-padded RGBA frame to the planar
float32 NCHW the engine expects, writing straight into the engine's input
device buffer; the engine then runs on the CUDA primary context of the
selected device.

Implementation uses **nvrtc** (runtime kernel compilation, no ``nvcc`` binary;
compiled straight to a SASS CUBIN for the local device's ``sm_<cc>`` so no
driver-side PTX JIT runs) plus **cuda-python** (``cuda.bindings`` — driver +
runtime) and ``tensorrt``, mirroring the cudart malloc/memcpy/stream/execute
patterns of
:meth:`openral_rskill.runtime_tensorrt.TensorRTRuntime.infer`. This deliberately
avoids ``pycuda`` + ``SourceModule(nvcc)`` so the NVMM aggregator tier is
deployable in the lean DeepStream ``ds-on`` runtime image, which ships
``libnvrtc`` + ``cuda-python`` but has no ``g++`` / ``nvcc`` / CUDA dev headers
(so ``pycuda`` cannot install or run there). Validated against real DeepStream
NVMM buffers in the ds-on container.

This is written from scratch (NumPy / cuda-python / nvrtc / TensorRT public
APIs only). The GN/Jabra videotech files are reference-only and were not copied
(CLAUDE.md §9).

Importing this module requires the ``tensorrt`` group (``cuda-python`` +
``tensorrt``); ``nvrtc`` ships with the CUDA toolkit (host) or
``libnvrtc.so`` (ds-on image). No ``pycuda`` and no shared CUDA context: the
executor operates on the device's primary context (``cudaSetDevice`` +
``cuCtxGetCurrent``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from openral_core.exceptions import ROSConfigError, ROSRuntimeError

log = structlog.get_logger(__name__)

__all__ = ["TrtNvmmExecutor", "resize_pad_geometry"]

# Clean-room kernel: pitch-padded RGBA uint8 -> planar RGB float32 NCHW, /255.
# Generic layout+normalize; no model-specific logic. nvrtc compiles this at
# runtime (no nvcc), so it ships as bytes with explicit C linkage.
_KERNEL_SRC = rb"""
extern "C" __global__ void rgba_to_nchw_norm(
        float* dst, const unsigned char* src,
        int height, int width, int src_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const unsigned char* px = src + (size_t)y * src_pitch + (size_t)x * 4;
        long plane = (long)height * width;
        long idx = (long)y * width + x;
        dst[idx]            = (float)px[0] / 255.0f;  // R
        dst[plane + idx]    = (float)px[1] / 255.0f;  // G
        dst[2 * plane + idx]= (float)px[2] / 255.0f;  // B
    }
}
"""

_KERNEL_NAME = b"rgba_to_nchw_norm"

# Clean-room kernel: pitch-padded RGBA uint8 (any size) -> aspect-preserving
# bilinear resize (align_corners=False, border-replicate at edges) -> top-left
# zero-pad -> planar RGB float32 NCHW in [-1, 1]. Replicates lerobot SmolVLA
# preprocessing exactly: ``resize_with_pad(img, W, H, pad_value=0)`` followed
# by ``img * 2 - 1`` (padding lands at -1). Bilinear indices are clamped after
# the fractional weights are taken from the unclamped coordinate — the same
# border-replicate behaviour as ``torch.nn.functional.interpolate``.
_RESIZE_KERNEL_SRC = rb"""
extern "C" __global__ void rgba_resize_pad_norm(
        float* dst, const unsigned char* src,
        int src_h, int src_w, int src_pitch,
        int dst_h, int dst_w,
        int resized_h, int resized_w,
        int pad_h, int pad_w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    long plane = (long)dst_h * dst_w;
    long idx = (long)y * dst_w + x;
    if (x < pad_w || y < pad_h) {
        dst[idx] = -1.0f;
        dst[plane + idx] = -1.0f;
        dst[2 * plane + idx] = -1.0f;
        return;
    }
    float sx = ((float)(x - pad_w) + 0.5f) * ((float)src_w / (float)resized_w) - 0.5f;
    float sy = ((float)(y - pad_h) + 0.5f) * ((float)src_h / (float)resized_h) - 0.5f;
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    int x1 = min(max(x0 + 1, 0), src_w - 1);
    int y1 = min(max(y0 + 1, 0), src_h - 1);
    x0 = min(max(x0, 0), src_w - 1);
    y0 = min(max(y0, 0), src_h - 1);
    const unsigned char* p00 = src + (size_t)y0 * src_pitch + (size_t)x0 * 4;
    const unsigned char* p01 = src + (size_t)y0 * src_pitch + (size_t)x1 * 4;
    const unsigned char* p10 = src + (size_t)y1 * src_pitch + (size_t)x0 * 4;
    const unsigned char* p11 = src + (size_t)y1 * src_pitch + (size_t)x1 * 4;
    for (int c = 0; c < 3; ++c) {
        float v0 = (float)p00[c] * (1.0f - fx) + (float)p01[c] * fx;
        float v1 = (float)p10[c] * (1.0f - fx) + (float)p11[c] * fx;
        float v = v0 * (1.0f - fy) + v1 * fy;
        dst[c * plane + idx] = v * (2.0f / 255.0f) - 1.0f;
    }
}
"""

_RESIZE_KERNEL_NAME = b"rgba_resize_pad_norm"
_BLOCK_X = 16
_BLOCK_Y = 16


def resize_pad_geometry(
    src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[int, int, int, int]:
    """Compute ``(resized_h, resized_w, pad_h, pad_w)`` for resize-with-pad.

    Pure integer math mirroring lerobot's ``resize_with_pad``: scale by
    ``ratio = max(src_w / dst_w, src_h / dst_h)`` (truncating like ``int()``),
    pad on the **top and left**. Unit-tested hardware-free against the torch
    reference geometry.

    Args:
        src_h: Source frame height in pixels (> 0).
        src_w: Source frame width in pixels (> 0).
        dst_h: Network input height (> 0).
        dst_w: Network input width (> 0).

    Returns:
        ``(resized_h, resized_w, pad_h, pad_w)`` — the bilinear-resized frame
        size and the top/left padding that centres it at the bottom-right.

    Example:
        >>> resize_pad_geometry(480, 640, 512, 512)
        (384, 512, 128, 0)
    """
    if min(src_h, src_w, dst_h, dst_w) <= 0:
        raise ROSConfigError(
            f"resize_pad_geometry: all dims must be > 0, got "
            f"src {src_h}x{src_w} -> dst {dst_h}x{dst_w}"
        )
    ratio = max(src_w / dst_w, src_h / dst_h)
    resized_h = int(src_h / ratio)
    resized_w = int(src_w / ratio)
    return resized_h, resized_w, max(0, dst_h - resized_h), max(0, dst_w - resized_w)


class TrtNvmmExecutor:
    """Run a TRT engine on a device-pointer RGBA frame with no host copy.

    Args:
        engine_bytes: Serialized TRT engine (from
            :meth:`~openral_rskill.runtime_tensorrt.TensorRTRuntime.serialized_engine`).
        input_size: ``(height, width)`` of the engine's image input. In
            ``"scale"`` mode the NVMM branch caps must scale frames to this;
            in ``"resize_pad_pm1"`` mode any frame size is accepted.
        device_index: CUDA device ordinal; selected via ``cudaSetDevice``,
            which initializes and makes the device's primary context current.
        preprocess: GPU preprocessing applied to the RGBA frame:

            - ``"scale"`` (default): frame already matches ``input_size``;
              RGBA -> planar float NCHW, ``/255`` (detector convention).
            - ``"resize_pad_pm1"``: aspect-preserving bilinear resize with
              top-left padding into ``input_size``, normalized to ``[-1, 1]``
              — the exact lerobot SmolVLA ``resize_with_pad`` + SigLIP
              normalization, for VLA vision-encoder engines.

    Raises:
        ROSConfigError: If ``cuda-python`` / ``tensorrt`` are unavailable, or
            the engine fails to deserialize / has no single image input.
        ROSRuntimeError: If a CUDA / nvrtc call fails during setup.

    Example:
        >>> # Full GPU round-trip exercised in tests/unit/test_trt_nvmm_executor.py;
        >>> # doctest skipped here because tensorrt / cuda-python are optional at doctest time.
        >>> pass
    """

    def __init__(
        self,
        engine_bytes: bytes,
        *,
        input_size: tuple[int, int],
        device_index: int = 0,
        preprocess: str = "scale",
    ) -> None:
        """Compile the kernel, deserialize the engine, and allocate I/O buffers.

        Operates on the device's CUDA primary context (made current by
        ``cudaSetDevice``); no shared pycuda context is involved.
        """
        if preprocess not in ("scale", "resize_pad_pm1"):
            raise ROSConfigError(
                f"TrtNvmmExecutor: unknown preprocess {preprocess!r}; "
                "valid: 'scale', 'resize_pad_pm1'."
            )
        self._preprocess = preprocess
        try:
            # Deferred optional GPU deps (see module docstring); I001 keeps the
            # tensorrt + cuda.bindings imports grouped without an isort reflow.
            from cuda.bindings import driver as cuda, nvrtc, runtime as cudart  # noqa: PLC0415,I001

            import tensorrt as trt  # noqa: PLC0415
        except ImportError as exc:
            raise ROSConfigError(
                "TrtNvmmExecutor needs cuda-python + tensorrt + nvrtc. "
                "Install the tensorrt group: `uv sync --group tensorrt` "
                "(provides cuda-python + tensorrt; nvrtc ships with the CUDA "
                "toolkit / the ds-on runtime image)."
            ) from exc

        self._cuda = cuda
        self._cudart = cudart
        self._h, self._w = input_size
        self._device_index = device_index
        # Buffers/handles tracked so a partial __init__ (or close()) can free
        # everything; a failed __init__ never returns an instance for close().
        self._outputs: dict[str, tuple[Any, Any]] = {}
        self._in_dev: Any = None
        self._stream: Any = None
        self._module: Any = None
        self._func: Any = None
        self._closed = False
        try:
            self._build_io(engine_bytes, cuda, cudart, nvrtc, trt)
        except BaseException:
            self._free_resources()
            raise
        log.debug("trt_nvmm.executor_ready", input_size=input_size, outputs=list(self._outputs))

    @staticmethod
    def _rt(result: tuple[Any, ...], cudart: Any) -> tuple[Any, ...]:  # noqa: ANN401  # reason: cuda.bindings.runtime is untyped
        """Raise on a non-success cudart return tuple; return the trailing values."""
        if int(result[0]) != 0:
            msg = cudart.cudaGetErrorString(result[0])[1]
            decoded = msg.decode(errors="ignore") if isinstance(msg, bytes) else str(msg)
            raise ROSRuntimeError(f"TrtNvmmExecutor: cudart error: {decoded}")
        return result[1:]

    @staticmethod
    def _dr(result: tuple[Any, ...], cuda: Any) -> tuple[Any, ...]:  # noqa: ANN401  # reason: cuda.bindings.driver is untyped
        """Raise on a non-success driver return tuple; return the trailing values."""
        if int(result[0]) != 0:
            msg = cuda.cuGetErrorName(result[0])[1]
            decoded = msg.decode(errors="ignore") if isinstance(msg, bytes) else str(msg)
            raise ROSRuntimeError(f"TrtNvmmExecutor: cuda-driver error: {decoded}")
        return result[1:]

    @staticmethod
    def _nv(result: tuple[Any, ...], nvrtc: Any, prog: Any = None) -> tuple[Any, ...]:  # noqa: ANN401  # reason: cuda.bindings.nvrtc is untyped
        """Raise on a non-success nvrtc return tuple (+ compile log); return the rest."""
        if int(result[0]) != 0:
            extra = ""
            if prog is not None:
                (size,) = nvrtc.nvrtcGetProgramLogSize(prog)[1:]
                buf = b" " * size
                nvrtc.nvrtcGetProgramLog(prog, buf)
                extra = " " + buf.decode(errors="ignore")
            msg = nvrtc.nvrtcGetErrorString(result[0])[1]
            decoded = msg.decode(errors="ignore") if isinstance(msg, bytes) else str(msg)
            raise ROSRuntimeError(f"TrtNvmmExecutor: nvrtc error: {decoded}{extra}")
        return result[1:]

    def _compile_kernel(self, cuda: Any, cudart: Any, nvrtc: Any) -> None:  # noqa: ANN401  # reason: cuda.bindings modules are untyped
        """nvrtc-compile the RGBA->NCHW kernel to a CUBIN and load it into the primary context.

        ``cudaSetDevice`` initializes and makes the device's primary context
        current; ``cuInit`` then ensures the driver API is initialized so the
        module/kernel calls run against that implicit current context.

        We compile straight to a SASS **CUBIN** for the device's exact
        ``sm_<cc>`` (not PTX) and load that. Because the arch is always the
        local device, no PTX->SASS JIT happens at load time, which sidesteps the
        driver-side ``CUDA_ERROR_UNSUPPORTED_PTX_VERSION`` that occurs when the
        nvrtc toolkit emits a newer PTX ISA than the installed driver's JIT
        accepts (e.g. an nvrtc/driver version skew on a dev host).
        """
        rt, dr, nv = self._rt, self._dr, self._nv
        rt(cudart.cudaSetDevice(self._device_index), cudart)
        (props,) = rt(cudart.cudaGetDeviceProperties(self._device_index), cudart)
        cc = f"{props.major}{props.minor}"  # e.g. "89" for an Ada RTX 4070
        src, self._kernel_name = (
            (_KERNEL_SRC, _KERNEL_NAME)
            if self._preprocess == "scale"
            else (_RESIZE_KERNEL_SRC, _RESIZE_KERNEL_NAME)
        )
        (prog,) = nv(
            nvrtc.nvrtcCreateProgram(src, self._kernel_name + b".cu", 0, [], []), nvrtc
        )
        try:
            opts = [f"--gpu-architecture=sm_{cc}".encode()]
            nv(nvrtc.nvrtcCompileProgram(prog, len(opts), opts), nvrtc, prog)
            (size,) = nv(nvrtc.nvrtcGetCUBINSize(prog), nvrtc)
            cubin = b" " * size
            nv(nvrtc.nvrtcGetCUBIN(prog, cubin), nvrtc)
        finally:
            nvrtc.nvrtcDestroyProgram(prog)
        dr(cuda.cuInit(0), cuda)
        (self._module,) = dr(cuda.cuModuleLoadData(cubin), cuda)
        (self._func,) = dr(cuda.cuModuleGetFunction(self._module, self._kernel_name), cuda)

    def _build_io(
        self,
        engine_bytes: bytes,
        cuda: Any,  # noqa: ANN401  # reason: cuda.bindings.driver is untyped
        cudart: Any,  # noqa: ANN401  # reason: cuda.bindings.runtime is untyped
        nvrtc: Any,  # noqa: ANN401  # reason: cuda.bindings.nvrtc is untyped
        trt: Any,  # noqa: ANN401  # reason: tensorrt is untyped
    ) -> None:
        """Compile the kernel, deserialize the engine, and allocate persistent I/O buffers."""
        self._compile_kernel(cuda, cudart, nvrtc)

        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise ROSConfigError("TrtNvmmExecutor: failed to deserialize TRT engine bytes.")
        self._engine = engine
        self._context = engine.create_execution_context()

        input_name: str | None = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if input_name is not None:
                    raise ROSConfigError(
                        "TrtNvmmExecutor: engine has >1 input; detector expects one image input."
                    )
                input_name = name
        if input_name is None:
            raise ROSConfigError("TrtNvmmExecutor: engine has no input tensor.")
        self._context.set_input_shape(input_name, (1, 3, self._h, self._w))

        in_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(input_name)))
        in_bytes = int(in_dtype.itemsize) * 1 * 3 * self._h * self._w
        (self._in_dev,) = self._rt(cudart.cudaMalloc(in_bytes), cudart)
        self._context.set_tensor_address(input_name, int(self._in_dev))

        # Output shapes resolve only after set_input_shape above.
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(int(d) for d in self._context.get_tensor_shape(name))
                dt = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
                host = np.empty(shape, dtype=dt)
                (dev,) = self._rt(cudart.cudaMalloc(host.nbytes), cudart)
                self._context.set_tensor_address(name, int(dev))
                self._outputs[name] = (host, dev)

        (self._stream,) = self._rt(cudart.cudaStreamCreate(), cudart)

    def _free_resources(self) -> None:
        """Free device buffers, unload the module, destroy the stream. Idempotent.

        Best-effort: a non-zero CUDA return during teardown is logged
        (``log.warning``) but does not abort the rest of the cleanup, so one bad
        handle cannot leak the others. The ``cuda.bindings`` module refs are read
        via ``getattr(..., None)`` and the corresponding frees skipped if absent
        (interpreter-shutdown ordering), so this can never raise ``AttributeError``.
        """
        cudart = getattr(self, "_cudart", None)
        cuda = getattr(self, "_cuda", None)
        if cudart is not None:
            for _host, dev in getattr(self, "_outputs", {}).values():
                err = cudart.cudaFree(dev)
                if int(err[0]) != 0:
                    log.warning("trt_nvmm.cuda_free_failed", which="output", err=int(err[0]))
        self._outputs = {}
        if cudart is not None and getattr(self, "_in_dev", None) is not None:
            err = cudart.cudaFree(self._in_dev)
            if int(err[0]) != 0:
                log.warning("trt_nvmm.cuda_free_failed", which="input", err=int(err[0]))
            self._in_dev = None
        if cudart is not None and getattr(self, "_stream", None) is not None:
            err = cudart.cudaStreamDestroy(self._stream)
            if int(err[0]) != 0:
                log.warning("trt_nvmm.cuda_stream_destroy_failed", err=int(err[0]))
            self._stream = None
        if cuda is not None and getattr(self, "_module", None) is not None:
            err = cuda.cuModuleUnload(self._module)
            if int(err[0]) != 0:
                log.warning("trt_nvmm.cuda_module_unload_failed", err=int(err[0]))
            self._module = None
        self._func = None

    def output_shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        """Return ``(name, shape)`` for each engine output (for output identification)."""
        return [(name, tuple(host.shape)) for name, (host, _dev) in self._outputs.items()]

    def infer_rgba_devptr(
        self, src_ptr: int, *, width: int, height: int, pitch: int
    ) -> dict[str, Any]:
        """Run inference on a device-pointer RGBA frame; return host output arrays.

        Args:
            src_ptr: CUDA device pointer to the RGBA frame (``NvBufSurface.dataPtr``).
            width: Frame width. In ``"scale"`` mode must equal the configured
                network width; in ``"resize_pad_pm1"`` mode any size.
            height: Frame height (same rule as ``width``).
            pitch: Row pitch in bytes (NVMM frames are pitch-padded).

        Returns:
            Map of engine-output name -> ``numpy.ndarray``.

        Raises:
            ROSConfigError: If ``(height, width)`` violates the preprocess
                mode's size rule.
            ROSRuntimeError: If a CUDA call or ``execute_async_v3`` fails.
        """
        cuda = self._cuda
        cudart = self._cudart

        # Pack kernel args as host pointers to single-element numpy scalars; the
        # arg array passes their addresses to the driver. Keep every array alive
        # (referenced locally) until the launch is enqueued.
        p_dst = np.array([int(self._in_dev)], dtype=np.uint64)
        p_src = np.array([int(src_ptr)], dtype=np.uint64)
        if self._preprocess == "scale":
            if (height, width) != (self._h, self._w):
                raise ROSConfigError(
                    f"TrtNvmmExecutor: frame {height}x{width} != configured "
                    f"{self._h}x{self._w}; the NVMM branch caps must scale to "
                    "the network size (or use preprocess='resize_pad_pm1')."
                )
            scalars = [height, width, pitch]
            gx = (width + _BLOCK_X - 1) // _BLOCK_X
            gy = (height + _BLOCK_Y - 1) // _BLOCK_Y
        else:  # resize_pad_pm1 — kernel iterates over the network-sized output
            resized_h, resized_w, pad_h, pad_w = resize_pad_geometry(
                height, width, self._h, self._w
            )
            scalars = [
                height, width, pitch,
                self._h, self._w,
                resized_h, resized_w,
                pad_h, pad_w,
            ]
            gx = (self._w + _BLOCK_X - 1) // _BLOCK_X
            gy = (self._h + _BLOCK_Y - 1) // _BLOCK_Y
        scalar_arrs = [np.array([s], dtype=np.int32) for s in scalars]
        kargs = np.array(
            [p_dst.ctypes.data, p_src.ctypes.data]
            + [a.ctypes.data for a in scalar_arrs],
            dtype=np.uint64,
        )
        self._dr(
            cuda.cuLaunchKernel(
                self._func,
                gx,
                gy,
                1,
                _BLOCK_X,
                _BLOCK_Y,
                1,
                0,
                int(self._stream),
                kargs.ctypes.data,
                0,
            ),
            cuda,
        )
        if not self._context.execute_async_v3(int(self._stream)):
            raise ROSRuntimeError("TrtNvmmExecutor: execute_async_v3 returned False.")
        for _name, (host, dev) in self._outputs.items():
            self._rt(
                cudart.cudaMemcpyAsync(
                    host.ctypes.data,
                    int(dev),
                    host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    int(self._stream),
                ),
                cudart,
            )
        self._rt(cudart.cudaStreamSynchronize(int(self._stream)), cudart)
        return {name: np.array(host) for name, (host, _dev) in self._outputs.items()}

    def close(self) -> None:
        """Free device buffers, unload the kernel module, destroy the stream. Idempotent."""
        if self._closed:
            return
        self._free_resources()
        self._closed = True
        log.debug("trt_nvmm.executor_closed")

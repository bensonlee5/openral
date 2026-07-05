"""TensorRT runtime for SmolVLA — swaps ``sample_actions`` for TRT engines.

Composes the split ONNX export (:mod:`openral_rskill.smolvla_export`) with two
:class:`~openral_rskill.runtime_tensorrt.TensorRTRuntime` engines (vision +
policy) and attaches a drop-in replacement for
``VLAFlowMatching.sample_actions`` on a loaded ``SmolVLAPolicy``. Everything
around that seam — processor pipeline, ``prepare_images`` / ``prepare_state``,
action queues, un-padding, post-processing, :class:`ChunkedExecutor` — keeps
running the upstream lerobot code unchanged, which is exactly the boundary the
parity study validated (ONNX == torch fp32 to 1e-06; TRT bf16 within 1.6% of
fp32 on a real training-set sample, vs 0.4% for the deployed torch bf16).

ONNX graphs are exported once per checkpoint into a local cache
(``~/.cache/openral/smolvla_onnx/<repo>/``, ~7 min CPU) and TRT engines are
built once per host/precision via :class:`EngineCache` (~5 min GPU). Both are
reused on every subsequent boot.

Precision: default **bf16** (fp16 is numerically unusable for the policy
graph — flow matching amplifies fp16 layernorm overflow to ~60% action error;
bf16 measured at 1.6% vs fp32 on real data). ``fp32`` is available for debug.

Heavy deps (torch, lerobot, tensorrt) deferred to call time, matching the
sibling runtime modules.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import structlog
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_core.schemas import QuantizationBackend, QuantizationConfig, QuantizationDtype

from openral_rskill.smolvla_export import SmolVLAOnnxPaths, export_smolvla_split_onnx

log = structlog.get_logger(__name__)

__all__ = ["attach_trt_sample_actions", "ensure_smolvla_onnx", "maybe_attach_trt_from_env"]

_DEFAULT_ONNX_CACHE = Path.home() / ".cache" / "openral" / "smolvla_onnx"
_PRECISIONS = ("bf16", "fp32")
_ENV_ENABLE = "OPENRAL_SMOLVLA_TRT"
_ENV_PRECISION = "OPENRAL_SMOLVLA_TRT_PRECISION"


def _slug(repo_id: str) -> str:
    """Filesystem-safe cache-directory name for a HF repo id."""
    return re.sub(r"[^A-Za-z0-9._-]", "--", repo_id)


def _device_index(device: str) -> int:
    """Parse a torch device string (``"cuda:1"`` / ``"cuda"`` / ``"cpu"``) to an ordinal."""
    if ":" in device:
        return int(device.rsplit(":", 1)[1])
    return 0


def maybe_attach_trt_from_env(
    policy: Any,  # noqa: ANN401  # reason: lerobot SmolVLAPolicy; deferred import
    repo_id: str,
    *,
    device: str = "cuda:0",
) -> bool:
    """Attach the TRT runtime iff ``OPENRAL_SMOLVLA_TRT`` is truthy.

    The single opt-in seam shared by both SmolVLA load paths (the
    ``openral_rskill`` rSkill adapter and the ``openral_sim`` deploy/sim policy
    factory), so the env knob behaves identically wherever a SmolVLA policy is
    built. Reads ``OPENRAL_SMOLVLA_TRT_PRECISION`` (default ``bf16``).

    Args:
        policy: A loaded ``SmolVLAPolicy``.
        repo_id: Checkpoint id (keys the ONNX + engine caches).
        device: torch device string; the CUDA ordinal is parsed from it.

    Returns:
        ``True`` if the TRT runtime was attached, ``False`` if the env knob is
        off (caller keeps the torch path, e.g. ``torch.compile``).

    Raises:
        ROSConfigError / ROSRuntimeError: Propagated from
            :func:`attach_trt_sample_actions` — no silent fallback (§1.4).
    """
    if os.environ.get(_ENV_ENABLE, "0").lower() not in ("1", "true"):
        return False
    precision = os.environ.get(_ENV_PRECISION, "bf16")
    attach_trt_sample_actions(
        policy, repo_id, precision=precision, device_index=_device_index(device)
    )
    log.info("smolvla_trt.enabled_from_env", repo_id=repo_id, precision=precision, device=device)
    return True


def ensure_smolvla_onnx(
    repo_id: str, *, cache_dir: Path | None = None
) -> SmolVLAOnnxPaths:
    """Return cached split-ONNX graphs for ``repo_id``, exporting on first use.

    The export needs a **separate fp32/CPU copy** of the checkpoint
    (:func:`export_smolvla_split_onnx` converts its policy in place), so the
    caller's deploy policy is never touched. One-time cost ~7 min CPU;
    subsequent calls just stat the files.

    Args:
        repo_id: HF checkpoint id (must resolve from cache when offline).
        cache_dir: Override for the ONNX cache root (tests).

    Returns:
        :class:`SmolVLAOnnxPaths` pointing into the cache directory.

    Raises:
        ROSConfigError: If the checkpoint or lerobot deps are unavailable.
    """
    root = (cache_dir or _DEFAULT_ONNX_CACHE) / _slug(repo_id)
    vision = root / "vision_encoder.onnx"
    policy_graph = root / "policy_graph.onnx"
    if vision.exists() and policy_graph.exists():
        # n_cameras / tokens-per-camera are static graph facts; recover them
        # cheaply from the vision graph's input/output shapes via onnx.
        import onnx  # noqa: PLC0415  # reason: deferred heavy dep

        model = onnx.load(str(vision), load_external_data=False)
        vin = model.graph.input[0].type.tensor_type.shape.dim
        vout = model.graph.output[0].type.tensor_type.shape.dim
        return SmolVLAOnnxPaths(
            vision_onnx=vision,
            policy_onnx=policy_graph,
            n_cameras=int(vin[0].dim_value),
            image_tokens_per_camera=int(vout[1].dim_value),
        )

    try:
        from lerobot.policies.smolvla.modeling_smolvla import (  # noqa: PLC0415  # reason: deferred heavy dep
            SmolVLAPolicy,
        )
    except ImportError as exc:  # pragma: no cover - import guard
        raise ROSConfigError(
            "ensure_smolvla_onnx requires 'lerobot' (uv add lerobot)."
        ) from exc

    log.info("smolvla_trt.exporting_onnx", repo_id=repo_id, out_dir=str(root))
    t0 = time.perf_counter()
    export_policy = SmolVLAPolicy.from_pretrained(repo_id)
    paths = export_smolvla_split_onnx(export_policy, root)
    del export_policy
    log.info(
        "smolvla_trt.onnx_exported",
        repo_id=repo_id,
        seconds=round(time.perf_counter() - t0, 1),
        n_cameras=paths.n_cameras,
    )
    return paths


class _TrtSampleActions:
    """``VLAFlowMatching.sample_actions``-compatible callable backed by TRT."""

    def __init__(
        self,
        policy: Any,  # noqa: ANN401  # reason: lerobot SmolVLAPolicy; deferred import
        paths: SmolVLAOnnxPaths,
        repo_id: str,
        *,
        precision: str,
        device_index: int,
    ) -> None:
        from openral_rskill.runtime_tensorrt import (  # noqa: PLC0415  # reason: deferred heavy dep
            TensorRTRuntime,
        )

        self._policy = policy
        self._paths = paths
        quant = QuantizationConfig(
            dtype=QuantizationDtype(precision),
            backend=QuantizationBackend.TENSORRT,
        )
        device = f"cuda:{device_index}"
        repo_tag = _slug(repo_id)
        t0 = time.perf_counter()
        self._vision = TensorRTRuntime(
            device=device, rskill_id=f"{repo_tag}#vision", quantization=quant
        )
        self._vision.load(paths.vision_onnx)
        self._policy_rt = TensorRTRuntime(
            device=device, rskill_id=f"{repo_tag}#policy", quantization=quant
        )
        self._policy_rt.load(paths.policy_onnx)
        log.info(
            "smolvla_trt.engines_ready",
            precision=precision,
            n_cameras=paths.n_cameras,
            seconds=round(time.perf_counter() - t0, 1),
        )

    def __call__(
        self,
        images: list[Any],
        img_masks: list[Any],
        lang_tokens: Any,  # noqa: ANN401  # reason: torch tensors; deferred import
        lang_masks: Any,  # noqa: ANN401
        state: Any,  # noqa: ANN401
        noise: Any = None,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401  # reason: upstream ActionSelectKwargs (RTC), rejected below
    ) -> Any:  # noqa: ANN401  # reason: returns a torch tensor
        import numpy as np  # noqa: PLC0415  # reason: deferred with torch
        import torch  # noqa: PLC0415  # reason: deferred heavy dep

        if any(kwargs.get(k) is not None for k in kwargs):
            raise ROSRuntimeError(
                f"smolvla_trt: RTC/extra sample_actions kwargs {sorted(kwargs)} "
                "are not supported by the TRT runtime."
            )
        if len(images) != self._paths.n_cameras:
            raise ROSRuntimeError(
                f"smolvla_trt: got {len(images)} camera images but the engine "
                f"was exported for {self._paths.n_cameras}. All configured "
                "cameras must be present (the graph bakes present-masks)."
            )
        for m in img_masks:
            if not bool(m.all()):
                raise ROSRuntimeError(
                    "smolvla_trt: an absent/padded camera (img_mask=False) is "
                    "not supported — the exported graph bakes all-present masks."
                )
        cfg = self._policy.config
        if lang_tokens.shape[1] != cfg.tokenizer_max_length:
            raise ROSRuntimeError(
                f"smolvla_trt: lang tokens length {lang_tokens.shape[1]} != "
                f"exported static length {cfg.tokenizer_max_length}; set "
                "pad_language_to='max_length' in the checkpoint's processor."
            )
        if noise is None:
            noise = torch.randn(
                state.shape[0], cfg.chunk_size, cfg.max_action_dim, dtype=torch.float32
            )

        def _np(t: Any, dtype: Any) -> Any:  # noqa: ANN401  # reason: torch->numpy bridge
            return np.ascontiguousarray(t.detach().to("cpu", torch.float32).numpy(), dtype=dtype)

        pixels = np.concatenate([_np(img, np.float32) for img in images], axis=0)
        (embs,) = self._vision.infer({"pixel_values": pixels}).values()
        img_embs = embs.reshape(1, -1, embs.shape[-1])
        (actions,) = self._policy_rt.infer(
            {
                "img_embs": np.ascontiguousarray(img_embs, dtype=np.float32),
                "lang_tokens": lang_tokens.detach().cpu().numpy().astype(np.int64),
                "lang_masks": lang_masks.detach().cpu().numpy().astype(bool),
                "state": _np(state, np.float32),
                "noise": _np(noise, np.float32),
            }
        ).values()
        return torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(state.device)


def attach_trt_sample_actions(
    policy: Any,  # noqa: ANN401  # reason: lerobot SmolVLAPolicy; deferred import
    repo_id: str,
    *,
    precision: str = "bf16",
    device_index: int = 0,
    cache_dir: Path | None = None,
) -> None:
    """Replace ``policy.model.sample_actions`` with the TRT-backed runner.

    Loud and explicit (CLAUDE.md §1.4): logs the runtime posture, never falls
    back silently — any TRT failure raises. The original bound method is kept
    at ``policy.model._openral_torch_sample_actions`` for debugging.

    Args:
        policy: A loaded ``SmolVLAPolicy`` (any device/dtype; not modified
            beyond the method swap).
        repo_id: Checkpoint id — keys the ONNX cache and engine cache.
        precision: ``"bf16"`` (default, validated) or ``"fp32"`` (debug).
            ``fp16`` is deliberately rejected (60% action error, measured).
        device_index: CUDA device ordinal.
        cache_dir: ONNX cache override (tests).

    Raises:
        ROSConfigError: Unknown precision, missing deps, or export failure.
        ROSRuntimeError: Engine build/load failure.

    Example:
        >>> # Exercised end-to-end (real checkpoint + engines) in
        >>> # tests/integration/test_smolvla_trt_runtime.py; doctest skipped
        >>> # because tensorrt + GPU + weights are optional at doctest time.
        >>> pass
    """
    if precision not in _PRECISIONS:
        raise ROSConfigError(
            f"smolvla_trt: precision {precision!r} not in {_PRECISIONS}. "
            "fp16 is deliberately unsupported: flow matching amplifies fp16 "
            "layernorm overflow to ~60% action error (measured 2026-07-04)."
        )
    paths = ensure_smolvla_onnx(repo_id, cache_dir=cache_dir)
    runner = _TrtSampleActions(
        policy, paths, repo_id, precision=precision, device_index=device_index
    )
    policy.model._openral_torch_sample_actions = policy.model.sample_actions
    policy.model.sample_actions = runner
    log.info(
        "smolvla_trt.attached",
        repo_id=repo_id,
        precision=precision,
        vision_onnx=str(paths.vision_onnx),
        policy_onnx=str(paths.policy_onnx),
    )

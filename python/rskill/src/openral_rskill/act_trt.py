"""ACT whole-model ONNX / TensorRT runtime — the ACT analogue of ``smolvla_trt``.

Where SmolVLA needs a bespoke *split* export (vision encoder + unrolled flow
loop, see :mod:`openral_rskill.smolvla_export`), ACT is a plain CNN+transformer:
one ONNX graph maps the normalized observation straight to the normalized action
chunk (``tools/export_act_onnx.py``). This module attaches that graph to a loaded
``ACTPolicy`` by swapping ``predict_action_chunk`` — the single method
``select_action`` calls — so the action queue, chunk replay, and the external
MEAN_STD post-processor all keep working unchanged.

The graph runs through the generic :class:`~openral_rskill.runtime.Runtime`
backends: :class:`~openral_rskill.runtime_tensorrt.TensorRTRuntime` when
``tensorrt`` is importable on a CUDA host (it builds + caches the engine on first
load, same as ``rtdetr-v2-r50vd``), otherwise
:class:`~openral_rskill.runtime_onnx.ONNXRuntime`.

Opt-in seam: ``OPENRAL_ACT_TRT=1`` (loud, no silent fallback — §1.4).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
from openral_core.exceptions import ROSConfigError

# Reuse the identical slug helper from the SmolVLA runtime rather than duplicating.
from openral_rskill.runtime import Runtime
from openral_rskill.smolvla_trt import _slug

log = structlog.get_logger(__name__)

_ENV_ENABLE = "OPENRAL_ACT_TRT"
# Local ONNX override — point the runtime at an unpublished model.onnx (a
# pre-publish deploy test, or a freshly re-exported graph) without editing the
# manifest. Wins over policy_extras.act_onnx_uri.
_ENV_ONNX = "OPENRAL_ACT_ONNX"
_DEFAULT_ONNX_CACHE = Path.home() / ".cache" / "openral" / "act_onnx"

__all__ = [
    "attach_act_onnx_runtime",
    "ensure_act_onnx",
    "maybe_attach_act_trt_from_env",
]


def ensure_act_onnx(repo_id: str, *, onnx_uri: str | None, cache_dir: Path | None = None) -> Path:
    """Resolve the ACT ONNX graph for ``repo_id`` to a local path.

    Resolution order:

    1. ``onnx_uri`` is a local file that exists → use it directly (tests / dev).
    2. ``onnx_uri`` is ``hf://<repo>/<file>`` → ``hf_hub_download`` it (the
       shipped ``model.onnx``, cached per host).

    A missing ``onnx_uri`` is a hard error: the graph must be exported once with
    ``tools/export_act_onnx.py`` and published to the rSkill's HF repo (declared
    as ``policy_extras.act_onnx_uri`` in the manifest). No silent local export.

    Args:
        repo_id: Checkpoint id (keys the local cache slug; diagnostics).
        onnx_uri: Local path or ``hf://repo/file`` from the manifest.
        cache_dir: Unused today; reserved for a future local-export cache.

    Returns:
        Path to a readable ``.onnx`` file.

    Raises:
        ROSConfigError: ``onnx_uri`` is missing or cannot be resolved.
    """
    if not onnx_uri:
        raise ROSConfigError(
            f"act_trt: no ONNX for {repo_id!r}. Export it once with "
            "`python tools/export_act_onnx.py --repo-id <ckpt> --out model.onnx`, "
            "publish it to the rSkill HF repo, and set "
            "policy_extras.act_onnx_uri in the manifest."
        )
    local = Path(onnx_uri)
    if local.is_file():
        return local
    if onnx_uri.startswith("hf://"):
        repo_and_file = onnx_uri[len("hf://") :]
        repo, _, filename = repo_and_file.rpartition("/")
        if not repo or not filename:
            raise ROSConfigError(f"act_trt: malformed act_onnx_uri {onnx_uri!r}.")
        from huggingface_hub import hf_hub_download  # noqa: PLC0415  # reason: deferred

        return Path(hf_hub_download(repo_id=repo, filename=filename))
    raise ROSConfigError(
        f"act_trt: unresolvable act_onnx_uri {onnx_uri!r} (not a file, not hf://)."
    )


def _build_runtime(onnx_path: Path, repo_id: str, device: str) -> Any:  # noqa: ANN401  # reason: Runtime Protocol, concrete backend chosen at runtime
    """Load ``onnx_path`` into a TensorRT backend if available, else ONNX Runtime.

    TensorRT is preferred on CUDA (build-on-first-load engine, cached per host
    keyed on ``repo_id``); it is silently unavailable on hosts without the wheel,
    where ONNX Runtime (CUDA or CPU by ``device``) is the correct, always-present
    backend.
    """
    if device.startswith("cuda"):
        try:
            import tensorrt  # noqa: F401, PLC0415  # reason: probe only

            from openral_rskill.runtime_tensorrt import TensorRTRuntime  # noqa: PLC0415

            rt: Any = TensorRTRuntime(device=device, rskill_id=repo_id)
            rt.load(onnx_path)
            log.info("act_trt.backend", backend="tensorrt", onnx=str(onnx_path), device=device)
            return rt
        except ImportError:
            log.info("act_trt.tensorrt_unavailable", falling_back_to="onnxruntime")

    from openral_rskill.runtime_onnx import ONNXRuntime  # noqa: PLC0415

    rt = ONNXRuntime(device=device)
    rt.load(onnx_path)
    log.info("act_trt.backend", backend="onnxruntime", onnx=str(onnx_path), device=device)
    return rt


class _OnnxActChunk:
    """Callable replacement for ``ACTPolicy.predict_action_chunk``.

    Feeds the normalized batch through a :class:`Runtime` backend and returns the
    ``(B, chunk_size, action_dim)`` normalized action chunk as a torch tensor on
    the policy device — a drop-in for the torch method ``select_action`` calls.
    """

    def __init__(
        self,
        runtime: Runtime,
        image_feature_keys: list[str],
        state_key: str,
        device: str,
    ) -> None:
        import torch  # noqa: PLC0415  # reason: deferred heavy dep

        self._rt = runtime
        self._image_feature_keys = image_feature_keys
        self._state_key = state_key
        self._device = device
        self._torch = torch

    def __call__(self, batch: dict[str, Any]) -> Any:  # noqa: ANN401  # reason: torch tensor
        feed: dict[str, Any] = {
            "state": batch[self._state_key].detach().cpu().numpy(),
        }
        for key in self._image_feature_keys:
            feed[key.replace("observation.images.", "img_")] = batch[key].detach().cpu().numpy()
        out = self._rt.infer(feed)
        actions = out["action_chunk"]
        return self._torch.from_numpy(actions).to(self._device)


def attach_act_onnx_runtime(
    policy: Any,  # noqa: ANN401  # reason: lerobot ACTPolicy; deferred import
    repo_id: str,
    *,
    onnx_uri: str | None,
    device: str = "cuda:0",
    cache_dir: Path | None = None,
) -> None:
    """Swap ``policy.predict_action_chunk`` for the ONNX/TRT-backed runner.

    Loud and explicit (§1.4): logs the backend, never falls back silently — any
    failure raises. The original bound method is kept at
    ``policy._openral_torch_predict_action_chunk`` for debugging.

    Args:
        policy: A loaded ``ACTPolicy`` (unmodified beyond the method swap).
        repo_id: Checkpoint id — diagnostics + cache slug.
        onnx_uri: Local path or ``hf://repo/file`` (from the manifest).
        device: torch device string; selects the runtime backend + tensor device.
        cache_dir: ONNX cache override (reserved).

    Raises:
        ROSConfigError: ONNX unresolvable.
        ROSRuntimeError: Backend build/load failure (propagated).
    """
    onnx_path = ensure_act_onnx(repo_id, onnx_uri=onnx_uri, cache_dir=cache_dir)
    runtime = _build_runtime(onnx_path, repo_id, device)
    image_feature_keys = list(policy.config.image_features)
    runner = _OnnxActChunk(runtime, image_feature_keys, "observation.state", device)
    policy._openral_torch_predict_action_chunk = policy.predict_action_chunk
    policy.predict_action_chunk = runner
    log.info(
        "act_trt.attached",
        repo_id=repo_id,
        onnx=str(onnx_path),
        cameras=image_feature_keys,
        device=device,
        slug=_slug(repo_id),
    )


def maybe_attach_act_trt_from_env(
    policy: Any,  # noqa: ANN401  # reason: lerobot ACTPolicy; deferred import
    repo_id: str,
    *,
    onnx_uri: str | None,
    device: str = "cuda:0",
) -> bool:
    """Attach the ONNX/TRT runtime iff ``OPENRAL_ACT_TRT`` is truthy.

    The single opt-in seam for the ACT load path (mirrors
    :func:`openral_rskill.smolvla_trt.maybe_attach_trt_from_env`).

    Returns:
        ``True`` if attached, ``False`` if the env knob is off (caller keeps the
        torch path, e.g. ``torch.compile``).

    Raises:
        ROSConfigError / ROSRuntimeError: Propagated from
            :func:`attach_act_onnx_runtime` — no silent fallback (§1.4).
    """
    if os.environ.get(_ENV_ENABLE, "0").lower() not in ("1", "true"):
        return False
    onnx_uri = os.environ.get(_ENV_ONNX) or onnx_uri
    attach_act_onnx_runtime(policy, repo_id, onnx_uri=onnx_uri, device=device)
    log.info("act_trt.enabled_from_env", repo_id=repo_id, device=device)
    return True

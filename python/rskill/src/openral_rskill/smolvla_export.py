"""Split ONNX export for SmolVLA: vision encoder + policy graph (ADR-0037 follow-up).

A whole-model export of ``SmolVLAPolicy._get_action_chunk`` produces a graph
that ONNXRuntime and the TensorRT parser both reject. The blockers are not the
transformer math — they are two upstream implementation details plus host-side
preprocessing:

1. ``SmolVLMVisionEmbeddings.forward`` computes patch position ids with a
   boolean-mask ``index_put`` (``position_ids[mask] = pos_ids[mask]``). For
   SmolVLA the input is always a full-size square image (512x512 after
   ``resize_with_pad``), where that computation reduces *exactly* to
   ``arange(num_patches)``.
2. ``apply_rope`` writes its result with in-place slice assignment
   (``res[..., :d_half] = ...``), which exports as ``index_put``. The same
   math expressed as ``torch.cat`` is export-clean and numerically identical.
3. Tokenization, image resize/normalize, and noise sampling are host-side.

So the model is exported as **two static-shape graphs** split at the embedding
boundary, with both blockers patched only for the duration of the export:

- ``vision_encoder.onnx``: pixels ``(n_cameras, 3, H, W)`` -> SigLIP tower +
  connector -> image embeddings ``(n_cameras, T_img, hidden)``. All cameras
  ride one pass as the **batch** axis: the tower is per-sample throughout
  (patch conv, within-image attention, per-sample pixel-shuffle connector),
  so batching is mathematically identical to per-camera passes, and the host
  reshape ``(N, T, hidden) -> (1, N*T, hidden)`` reproduces ``embed_prefix``'s
  per-camera concat order exactly (no special tokens interleave when
  ``add_image_special_tokens=False``, which the export enforces).
- ``policy_graph.onnx``: image embeddings (all cameras concatenated), language
  tokens/masks, projected-state input, and flow-matching noise -> the full
  action chunk ``(1, chunk_size, max_action_dim)``. Internally: prefix pass
  filling the KV cache + the ``num_steps`` Euler denoise loop, which has a
  constant trip count and therefore unrolls at trace time.

Noise is a graph *input* (not sampled in-graph) so a chunk is replayable from
the trace (CLAUDE.md §1.8) and so torch/ONNX/TRT parity can be asserted
bit-comparably.

Heavy dependencies (``torch``, ``lerobot``, ``transformers``) are deferred to
call time — importing this module stays clean on hosts without the VLA groups,
matching ``runtime_onnx.py`` / ``runtime_tensorrt.py``.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

log = structlog.get_logger(__name__)

__all__ = ["SmolVLAOnnxPaths", "export_smolvla_split_onnx"]


@dataclass(frozen=True)
class SmolVLAOnnxPaths:
    """Filesystem locations of the two exported graphs.

    Attributes:
        vision_onnx: SigLIP vision-encoder graph; batch axis = camera count.
        policy_onnx: Prefix + unrolled flow-matching policy graph.
        n_cameras: Camera count baked into both graphs (vision batch size).
        image_tokens_per_camera: Vision-graph output tokens per camera; the
            policy graph's ``img_embs`` input length is
            ``n_cameras * image_tokens_per_camera``.
    """

    vision_onnx: Path
    policy_onnx: Path
    n_cameras: int
    image_tokens_per_camera: int


def _export_safe_apply_rope(x: Any, positions: Any, max_wavelength: int = 10_000) -> Any:  # noqa: ANN401  # reason: torch tensors; torch is a deferred import
    """``lerobot`` ``apply_rope`` with ``cat`` instead of slice assignment.

    Numerically identical to
    :func:`lerobot.policies.smolvla.smolvlm_with_expert.apply_rope`; the
    in-place ``res[..., :d_half] = ...`` writes there export as ``index_put``
    nodes that the TensorRT ONNX parser rejects.
    """
    import torch  # noqa: PLC0415  # reason: deferred heavy dep

    d_half = x.shape[-1] // 2
    dtype = x.dtype
    x = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        d_half, dtype=torch.float32, device=x.device
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    radians = radians[..., None, :]
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    x1, x2 = x.split(d_half, dim=-1)
    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return res.to(dtype)


def _static_vision_embeddings_forward(
    self: Any,  # noqa: ANN401  # reason: transformers SmolVLMVisionEmbeddings; deferred import
    pixel_values: Any,  # noqa: ANN401  # reason: torch tensor; torch is a deferred import
    patch_attention_mask: Any = None,  # noqa: ANN401  # reason: unused upstream-signature slot
) -> Any:  # noqa: ANN401  # reason: torch tensor; torch is a deferred import
    """``SmolVLMVisionEmbeddings.forward`` specialized to full-size square input.

    Upstream supports variable-resolution images via fractional-coordinate
    bucketing and a boolean-mask ``index_put``. When the input is exactly
    ``image_size`` x ``image_size`` with a full attention mask (always true for
    SmolVLA, which feeds ``resize_with_pad``-ed 512x512 frames), the bucketed
    position ids reduce to ``arange(num_patches)``. Verified by the parity
    test against the unpatched torch forward.
    """
    import torch  # noqa: PLC0415  # reason: deferred heavy dep

    _, _, height, width = pixel_values.shape
    if height != self.image_size or width != self.image_size:
        raise ROSConfigError(
            f"smolvla_export: static vision export expects {self.image_size}x"
            f"{self.image_size} input, got {height}x{width}."
        )
    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(2).transpose(1, 2)
    position_ids = torch.arange(self.num_patches, device=pixel_values.device)
    return embeddings + self.position_embedding(position_ids)


@contextmanager
def _export_safe_patches() -> Iterator[None]:
    """Swap in export-clean equivalents of the two blocking upstream functions.

    Patches are process-global while the context is open (module attribute and
    class attribute), and always restored — export runs single-threaded.
    """
    from lerobot.policies.smolvla import (  # noqa: PLC0415  # reason: deferred heavy dep
        smolvlm_with_expert as swe,
    )
    from transformers.models.smolvlm import (  # noqa: PLC0415  # reason: deferred heavy dep
        modeling_smolvlm as msv,
    )

    orig_rope = swe.apply_rope
    orig_vis = msv.SmolVLMVisionEmbeddings.forward
    swe.apply_rope = _export_safe_apply_rope
    msv.SmolVLMVisionEmbeddings.forward = _static_vision_embeddings_forward  # type: ignore[method-assign]  # reason: deliberate export-scoped monkeypatch, restored in finally
    try:
        yield
    finally:
        swe.apply_rope = orig_rope
        msv.SmolVLMVisionEmbeddings.forward = orig_vis  # type: ignore[method-assign]  # reason: restores the original upstream method


def _vision_encoder_module(model: Any) -> Any:  # noqa: ANN401  # reason: lerobot VLAFlowMatching; deferred import
    """Wrap ``vlm_with_expert.embed_image`` as a single-input ``nn.Module``."""
    import torch  # noqa: PLC0415  # reason: deferred heavy dep

    class _VisionEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vlm_with_expert = model.vlm_with_expert

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.vlm_with_expert.embed_image(pixel_values)
            return out

    return _VisionEncoder().eval()


def _policy_graph_module(model: Any) -> Any:  # noqa: ANN401  # reason: lerobot VLAFlowMatching; deferred import
    """Wrap prefix-fill + the unrolled denoise loop as a five-input ``nn.Module``.

    The image half of ``VLAFlowMatching.embed_prefix`` is inlined here with
    ``embed_image`` elided (the vision graph already produced embeddings); the
    language/state half and every denoise step delegate to the upstream
    modules, so there is no reimplementation of transformer internals. The
    inlined part is the scale-and-mask bookkeeping only, asserted equivalent
    by the parity test.
    """
    import torch  # noqa: PLC0415  # reason: deferred heavy dep
    from lerobot.policies.smolvla.modeling_smolvla import (  # noqa: PLC0415  # reason: deferred heavy dep
        make_att_2d_masks,
        pad_tensor,
    )

    if model.add_image_special_tokens:
        raise ROSConfigError(
            "smolvla_export: add_image_special_tokens=True is not supported by "
            "the split export (the deployed SmolVLA checkpoints use False)."
        )

    class _PolicyGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(
            self,
            img_embs: torch.Tensor,
            lang_tokens: torch.Tensor,
            lang_masks: torch.Tensor,
            state: torch.Tensor,
            noise: torch.Tensor,
        ) -> torch.Tensor:
            m = self.model
            bsize = state.shape[0]
            device = state.device

            # ── embed_prefix, with embed_image elided ────────────────────
            img_emb = img_embs * math.sqrt(img_embs.shape[-1])
            img_pad = torch.ones(bsize, img_emb.shape[1], dtype=torch.bool, device=device)
            lang_emb = m.vlm_with_expert.embed_language_tokens(lang_tokens)
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
            state_emb = m.state_proj(state)
            if state_emb.ndim == 2:  # noqa: PLR2004  # reason: (B, F) -> (B, 1, F), mirrors embed_prefix
                state_emb = state_emb[:, None, :]
            state_pad = torch.ones(bsize, state_emb.shape[1], dtype=torch.bool, device=device)

            prefix_embs = torch.cat([img_emb, lang_emb, state_emb], dim=1)
            prefix_pad_masks = torch.cat([img_pad, lang_masks, state_pad], dim=1)
            # att_masks: images+language are one bidirectional block (0),
            # state starts a new causal block (1) — mirrors embed_prefix.
            # Built with cat, not slice assignment (which exports as index_put).
            att_masks = torch.cat(
                [
                    torch.zeros(
                        1,
                        prefix_embs.shape[1] - state_emb.shape[1],
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.ones(1, state_emb.shape[1], dtype=torch.bool, device=device),
                ],
                dim=1,
            )

            if prefix_pad_masks.shape[1] < m.prefix_length:
                prefix_embs = pad_tensor(prefix_embs, m.prefix_length, pad_value=0)
                prefix_pad_masks = pad_tensor(prefix_pad_masks, m.prefix_length, pad_value=0)
                att_masks = pad_tensor(att_masks, m.prefix_length, pad_value=0)
            prefix_att_masks = att_masks.expand(bsize, -1)

            # ── sample_actions: prefix pass filling the KV cache ─────────
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = m.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
                fill_kv_cache=True,
            )

            # ── constant-trip-count Euler loop: unrolls at trace time ────
            num_steps = int(m.config.num_steps)
            dt = -1.0 / num_steps
            x_t = noise
            for step in range(num_steps):
                timestep = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=device)
                v_t = m.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=x_t,
                    timestep=timestep,
                )
                x_t = x_t + dt * v_t
            return x_t

    return _PolicyGraph().eval()


def export_smolvla_split_onnx(
    policy: Any,  # noqa: ANN401  # reason: lerobot SmolVLAPolicy; deferred import
    out_dir: Path | str,
    *,
    n_cameras: int | None = None,
) -> SmolVLAOnnxPaths:
    """Export a loaded ``SmolVLAPolicy`` as two static-shape ONNX graphs.

    The policy is exported **in float32 on CPU** (a temporary dtype/device
    move that is restored afterwards is deliberately *not* attempted — pass a
    policy you own for the duration; a bf16 CUDA policy is converted in place).
    Build fp16/bf16 precision at TensorRT engine-build time instead
    (:meth:`openral_rskill.runtime_tensorrt.TensorRTRuntime.serialized_engine`).

    Args:
        policy: A loaded ``lerobot`` ``SmolVLAPolicy`` (weights resolved; the
            caller owns HF-cache/offline concerns).
        out_dir: Directory receiving ``vision_encoder.onnx`` and
            ``policy_graph.onnx`` (+ external-data sidecars when large).
        n_cameras: Camera count baked into the vision graph's batch axis and
            the policy graph's ``img_embs`` input length. Defaults to
            ``len(policy.config.image_features)``.

    Returns:
        :class:`SmolVLAOnnxPaths` with both graph paths.

    Raises:
        ROSConfigError: If the policy uses unsupported options
            (``add_image_special_tokens``) or the export produces an invalid
            graph.

    Example:
        >>> # Exercised end-to-end (real checkpoint, ORT parity vs torch) in
        >>> # tests/integration/test_smolvla_onnx_split.py; doctest skipped
        >>> # because lerobot + weights are optional at doctest time.
        >>> pass
    """
    import torch  # noqa: PLC0415  # reason: deferred heavy dep

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = policy.config
    if n_cameras is None:
        n_cameras = len(cfg.image_features)
    if n_cameras < 1:
        raise ROSConfigError("smolvla_export: policy has no image features.")

    model = policy.model.float().eval().to("cpu")
    height, width = cfg.resize_imgs_with_padding

    vision_path = out_dir / "vision_encoder.onnx"
    policy_path = out_dir / "policy_graph.onnx"

    with torch.no_grad(), _export_safe_patches():
        pixels = torch.zeros(n_cameras, 3, height, width, dtype=torch.float32)
        t0 = time.perf_counter()
        vision = _vision_encoder_module(model)
        img_emb_example = vision(pixels)
        tokens_per_camera = int(img_emb_example.shape[1])
        hidden = int(img_emb_example.shape[2])
        _export_onnx(
            vision,
            (pixels,),
            vision_path,
            input_names=["pixel_values"],
            output_names=["image_embeddings"],
        )
        log.info(
            "smolvla_export.vision_done",
            path=str(vision_path),
            seconds=round(time.perf_counter() - t0, 1),
            tokens_per_camera=tokens_per_camera,
        )

        img_embs = torch.zeros(1, n_cameras * tokens_per_camera, hidden)
        lang_tokens = torch.zeros(1, cfg.tokenizer_max_length, dtype=torch.int64)
        lang_masks = torch.ones(1, cfg.tokenizer_max_length, dtype=torch.bool)
        state = torch.zeros(1, cfg.max_state_dim)
        noise = torch.zeros(1, cfg.chunk_size, cfg.max_action_dim)
        t0 = time.perf_counter()
        _export_onnx(
            _policy_graph_module(model),
            (img_embs, lang_tokens, lang_masks, state, noise),
            policy_path,
            input_names=["img_embs", "lang_tokens", "lang_masks", "state", "noise"],
            output_names=["actions"],
        )
        log.info(
            "smolvla_export.policy_done",
            path=str(policy_path),
            seconds=round(time.perf_counter() - t0, 1),
        )

    return SmolVLAOnnxPaths(
        vision_onnx=vision_path,
        policy_onnx=policy_path,
        n_cameras=n_cameras,
        image_tokens_per_camera=tokens_per_camera,
    )


def _export_onnx(
    module: Any,  # noqa: ANN401  # reason: torch nn.Module; deferred import
    args: tuple[Any, ...],
    path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Run the dynamo ONNX exporter with static shapes and save to ``path``.

    The dynamo exporter (not the legacy tracer) is required: the legacy
    tracer fails on SmolVLA's rope before the patches even apply.
    ``ONNXProgram.optimize()`` is deliberately **not** called: on the policy
    graph it constant-folds away a shape tensor that three Reshape nodes still
    reference, producing a graph ORT rejects at load ("Node input 'val_...' is
    not a graph input, initializer, or output of a previous node"). ORT applies
    its own graph optimizations at session init and TensorRT constant-folds
    during engine build, so the raw graph loses nothing.
    """
    import torch  # noqa: PLC0415  # reason: deferred heavy dep

    onnx_program = torch.onnx.export(
        module,
        args,
        dynamo=True,
        input_names=input_names,
        output_names=output_names,
    )
    if onnx_program is None:  # pragma: no cover - defensive; dynamo=True always returns
        raise ROSConfigError(f"smolvla_export: dynamo export returned None for {path.name}")
    onnx_program.save(str(path))


def _cli_export(argv: list[str] | None = None) -> int:
    """Subprocess entry point: load a checkpoint and write its split ONNX graphs.

    Run as ``python -m openral_rskill.smolvla_export --repo-id <id> --out-dir
    <dir> [--n-cameras N]``.

    This exists because the ``torch.onnx`` **dynamo** exporter deadlocks when
    invoked inside a process that already runs an ``rclpy`` executor: the
    exporter's internal threadpool and the ROS 2 executor threads contend, and
    every thread parks in ``futex_wait`` right after the "Translate ✅" step —
    the policy graph never finishes. The same export completes fine in a fresh
    process, so :func:`openral_rskill.smolvla_trt.ensure_smolvla_onnx` shells
    this module instead of exporting in-process. Standalone callers (tests, the
    engine prebuild) reach it through that same path.

    Args:
        argv: Command-line arguments (``None`` = ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    import argparse  # noqa: PLC0415  # reason: only needed on the subprocess path

    parser = argparse.ArgumentParser(prog="openral_rskill.smolvla_export")
    parser.add_argument("--repo-id", required=True, help="HF checkpoint id (offline-cache ok).")
    parser.add_argument("--out-dir", required=True, help="Directory for the two ONNX graphs.")
    parser.add_argument(
        "--n-cameras", type=int, default=None, help="Camera slots to bake (default: config count)."
    )
    ns = parser.parse_args(argv)

    from lerobot.policies.smolvla.modeling_smolvla import (  # noqa: PLC0415  # reason: deferred heavy dep
        SmolVLAPolicy,
    )

    from openral_rskill._lerobot_compat import sanitize_smolvla_config  # noqa: PLC0415

    sanitize_smolvla_config(ns.repo_id)
    policy = SmolVLAPolicy.from_pretrained(ns.repo_id)
    export_smolvla_split_onnx(policy, ns.out_dir, n_cameras=ns.n_cameras)
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(_cli_export())

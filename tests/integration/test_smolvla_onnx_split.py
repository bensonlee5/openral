"""Split-ONNX export parity for SmolVLA (vision encoder + policy graph).

Loads the **real** ``sapanostic/so_101_smolvla_pen_placement`` checkpoint from
the local HF cache (the upstream repo went gated after this host cached it, so
the test sets ``HF_HUB_OFFLINE=1`` and skips when the snapshot is absent),
exports both graphs via :func:`openral_rskill.smolvla_export.export_smolvla_split_onnx`,
and asserts ONNXRuntime output parity against the untouched torch
``VLAFlowMatching.sample_actions`` with the same injected flow-matching noise.

Everything runs on CPU in float32 — no GPU, no network. Per-stage wall times
are printed so export/runtime regressions are visible in the test log.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("lerobot")

import onnxruntime as ort  # noqa: E402
from openral_rskill.smolvla_export import export_smolvla_split_onnx  # noqa: E402

_CHECKPOINT = "sapanostic/so_101_smolvla_pen_placement"
_INSTRUCTION = "Grab pen and put into cup"
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _snapshot_cached(repo_id: str) -> bool:
    """True when a full local snapshot of ``repo_id`` exists in the HF cache."""
    d = _HF_CACHE / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return d.is_dir() and any(d.iterdir())


_VLM_BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

pytestmark = pytest.mark.skipif(
    not (_snapshot_cached(_CHECKPOINT) and _snapshot_cached(_VLM_BACKBONE)),
    reason=f"{_CHECKPOINT} (+ VLM backbone) not in the local HF cache; repo is gated upstream",
)


@pytest.fixture(scope="module")
def policy() -> Any:
    """The real pen-placement SmolVLA policy, float32 on CPU, offline."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    pol = SmolVLAPolicy.from_pretrained(_CHECKPOINT)
    pol.model = pol.model.float().eval().to("cpu")
    return pol


def _deterministic_inputs(policy: Any) -> dict[str, torch.Tensor]:
    """Deterministic, real-domain inputs shared by the torch and ONNX paths.

    Images are smooth per-channel gradients in the post-``prepare_images``
    domain ([-1, 1] SigLIP range); state is in the normalized joint domain;
    noise is a seeded standard normal, passed to both paths explicitly.
    """
    cfg = policy.config
    h, w = cfg.resize_imgs_with_padding
    yy = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1)
    xx = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w)
    img1 = (yy * xx).expand(1, 3, h, w).contiguous()
    img2 = (0.5 * yy - 0.25 * xx).expand(1, 3, h, w).contiguous()

    tok = policy.model.vlm_with_expert.processor.tokenizer(
        _INSTRUCTION,
        padding="max_length",
        truncation=True,
        max_length=cfg.tokenizer_max_length,
        return_tensors="pt",
    )
    state = torch.linspace(-0.8, 0.8, cfg.max_state_dim).unsqueeze(0)
    noise = torch.randn(
        1, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(7)
    )
    return {
        "images": [img1, img2],
        "lang_tokens": tok["input_ids"],
        "lang_masks": tok["attention_mask"].bool(),
        "state": state,
        "noise": noise,
    }


def test_split_onnx_matches_torch_sample_actions(policy: Any, tmp_path: Path) -> None:
    """vision.onnx -> policy.onnx equals torch ``sample_actions`` on real weights."""
    inp = _deterministic_inputs(policy)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    paths = export_smolvla_split_onnx(policy, tmp_path)
    timings["export_both_graphs_s"] = time.perf_counter() - t0

    # ── torch reference (unpatched upstream code path) ────────────────────
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in inp["images"]]
    t0 = time.perf_counter()
    with torch.no_grad():
        ref = policy.model.sample_actions(
            inp["images"],
            img_masks,
            inp["lang_tokens"],
            inp["lang_masks"],
            inp["state"],
            noise=inp["noise"],
        )
    timings["torch_sample_actions_s"] = time.perf_counter() - t0

    # ── split ONNX path ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    vision = ort.InferenceSession(str(paths.vision_onnx), providers=["CPUExecutionProvider"])
    policy_sess = ort.InferenceSession(str(paths.policy_onnx), providers=["CPUExecutionProvider"])
    timings["ort_session_init_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    embs = [
        vision.run(None, {"pixel_values": img.numpy()})[0] for img in inp["images"]
    ]
    timings["ort_vision_per_cam_s"] = (time.perf_counter() - t0) / len(embs)
    img_embs = np.concatenate(embs, axis=1)
    assert img_embs.shape[1] == len(embs) * paths.image_tokens_per_camera

    t0 = time.perf_counter()
    (actions,) = policy_sess.run(
        None,
        {
            "img_embs": img_embs,
            "lang_tokens": inp["lang_tokens"].numpy(),
            "lang_masks": inp["lang_masks"].numpy(),
            "state": inp["state"].numpy(),
            "noise": inp["noise"].numpy(),
        },
    )
    timings["ort_policy_graph_s"] = time.perf_counter() - t0

    print("\nsmolvla split-onnx timings:", {k: round(v, 3) for k, v in timings.items()})

    assert actions.shape == tuple(ref.shape)
    max_abs = float(np.max(np.abs(actions - ref.numpy())))
    print(f"max |onnx - torch| over the action chunk: {max_abs:.2e}")
    assert max_abs < 2e-3, f"split-ONNX diverged from torch: max abs diff {max_abs:.2e}"

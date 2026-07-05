"""End-to-end TRT runtime for SmolVLA: attach + sample_actions parity.

Loads the real pen checkpoint (bf16, CUDA — the deployed configuration),
captures the torch ``sample_actions`` chunk on deterministic inputs with fixed
noise, attaches the TRT bf16 runtime, and asserts the TRT chunk agrees within
the bf16-precision envelope.

Opt-in via ``OPENRAL_SMOLVLA_TRT_TEST=1``: the first run exports ONNX (~7 min
CPU) and builds engines (~5 min GPU); both are cached (ONNX cache + EngineCache)
so subsequent runs are seconds.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorrt")
pytest.importorskip("lerobot")

_CHECKPOINT = "sapanostic/so_101_smolvla_pen_placement"
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _cached(repo_id: str) -> bool:
    d = _HF_CACHE / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return d.is_dir() and any(d.iterdir())


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("OPENRAL_SMOLVLA_TRT_TEST") != "1",
        reason="opt-in: set OPENRAL_SMOLVLA_TRT_TEST=1 (first run builds engines, ~12 min)",
    ),
    pytest.mark.skipif(not _cached(_CHECKPOINT), reason="pen checkpoint not in HF cache"),
]


@pytest.fixture(scope="module")
def policy() -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    pol = SmolVLAPolicy.from_pretrained(_CHECKPOINT)
    pol.model = pol.model.to("cuda").eval()  # native bf16 — deployed config
    return pol


def test_trt_sample_actions_matches_torch_bf16(policy: Any) -> None:
    from openral_rskill.smolvla_trt import attach_trt_sample_actions

    cfg = policy.config
    h, w = cfg.resize_imgs_with_padding
    dev = "cuda"
    yy = torch.linspace(-1, 1, h, device=dev).view(1, 1, h, 1)
    xx = torch.linspace(-1, 1, w, device=dev).view(1, 1, 1, w)
    images = [
        (yy * xx).expand(1, 3, h, w).contiguous(),
        (0.5 * yy - 0.25 * xx).expand(1, 3, h, w).contiguous(),
    ]
    img_masks = [torch.ones(1, dtype=torch.bool, device=dev) for _ in images]
    tok = policy.model.vlm_with_expert.processor.tokenizer(
        "Pick up the pen and place it in the pen holder",
        padding="max_length",
        truncation=True,
        max_length=cfg.tokenizer_max_length,
        return_tensors="pt",
    )
    lang_tokens = tok["input_ids"].to(dev)
    lang_masks = tok["attention_mask"].bool().to(dev)
    state = torch.linspace(-0.8, 0.8, cfg.max_state_dim, device=dev).unsqueeze(0)
    noise = torch.randn(
        1, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(7)
    ).to(dev)

    with torch.no_grad():
        ref = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        ).float()

    attach_trt_sample_actions(policy, _CHECKPOINT, precision="bf16")
    assert policy.model.sample_actions is not policy.model._openral_torch_sample_actions

    with torch.no_grad():
        trt = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        ).float()

    assert trt.shape == ref.shape
    diff = float((trt - ref).abs().max())
    print(f"TRT bf16 vs torch bf16 (same noise): max abs {diff:.4f}")
    # Both are bf16-class computations with different rounding trajectories;
    # on synthetic inputs the envelope is wide (the real-sample study measured
    # 0.035 vs fp32 — synthetic inputs exaggerate ~10-30x, see PR #139).
    assert diff < 1.0, diff
    assert torch.isfinite(trt).all()

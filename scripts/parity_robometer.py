"""Parity gate: native lerobot Robometer (new) vs upstream robometer (old).

Proves the migration of the Robometer reward sidecar from the pinned ``robometer``
git package + ``transformers==4.57.1`` to lerobot 0.6.0's in-tree
``RobometerRewardModel`` is behaviour-preserving, BEFORE any old code is deleted.

Both paths score the SAME fixed clip of REAL deploy frames + the SAME task and
emit per-frame progress[]/success[]. They are never co-resident (one GPU); each
runs in its own interpreter:

  # 0. one-time, in the MAIN ./.venv (has cv2): extract the shared frames
  ./.venv/bin/python scripts/parity_robometer.py --prep-frames

  # 1. golden, in the SIDECAR venv (upstream robometer + transformers 4.57.1):
  SIDE=~/.cache/openral/robometer-sidecar/.venv/bin/python
  flock /tmp/openral-gpu.lock -c \
    "$SIDE scripts/parity_robometer.py --capture-golden"

  # 2. verify, in the MAIN ./.venv (lerobot 0.6.0, native path):
  flock /tmp/openral-gpu.lock -c \
    "./.venv/bin/python scripts/parity_robometer.py --verify"

Passes when the new per-frame progress/success match the golden within
``rtol=1e-3, atol=1e-2``.
"""
# ruff: noqa: PLC0415 — imports are deliberately lazy: the old path imports
# `robometer` (only in the sidecar venv) and the new path imports lerobot (only
# in ./.venv); neither can live at module top level.

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tools"))

import _robometer_quant as q  # noqa: E402 — sets determinism env before torch

q.set_cublas_workspace_env()

import numpy as np  # noqa: E402

_FRAMES_NPZ = _REPO / "scripts" / "_parity_frames.npz"
_GOLDEN_NPZ = _REPO / "scripts" / "_golden_robometer.npz"
_WEIGHTS = "OpenRAL/rskill-robometer-4b-nf4"
_MP4 = "/home/allopart/workspace/_deploy_videos/libero_object/libero_object_simulator.mp4"
_TASK = "pick up the alphabet soup and place it in the basket"
_N_FRAMES = 8
_RES = 256


def prep_frames() -> None:
    """Extract a fixed 8-frame, 256x256 RGB clip from a real deploy MP4 (cv2)."""
    import cv2

    cap = cv2.VideoCapture(_MP4)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, total - 1, _N_FRAMES).round().astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {i} of {_MP4}")
        bgr = cv2.resize(bgr, (_RES, _RES), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    frames_rgb = np.stack(frames).astype(np.uint8)
    np.savez(_FRAMES_NPZ, frames=frames_rgb, task=np.array(_TASK))
    print(f"[parity] wrote {_FRAMES_NPZ} frames={frames_rgb.shape} task={_TASK!r}")


def load_frames() -> tuple[np.ndarray, str]:
    """Load the shared fixed clip + task both paths score (byte-identical input)."""
    if not _FRAMES_NPZ.exists():
        raise SystemExit(f"{_FRAMES_NPZ} missing — run `--prep-frames` first (in ./.venv)")
    d = np.load(_FRAMES_NPZ, allow_pickle=True)
    return d["frames"], str(d["task"])


def _seed() -> None:
    import torch

    q.apply_determinism()
    torch.manual_seed(0)


def score_new(frames_rgb: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
    """New native lerobot path — the shipped sidecar ``_Scorer``."""
    _seed()
    import _robometer_server as srv

    scorer = srv._Scorer(_WEIGHTS, device="cuda")
    prog, succ = scorer.score(frames_rgb, task, num_bins=100)
    return np.asarray(prog, dtype=np.float32), np.asarray(succ, dtype=np.float32)


def score_old(frames_rgb: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
    """Old upstream ``robometer`` path (NF4 pre-quant meta-load), inlined."""
    _seed()
    import torch
    import yaml
    from huggingface_hub import snapshot_download
    from robometer.configs.experiment_configs import ExperimentConfig
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs
    from robometer.models.rbm import RBM
    from robometer.utils.setup_utils import setup_batch_collator
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    device = "cuda"
    local = snapshot_download(_WEIGHTS)
    raw = yaml.safe_load(open(os.path.join(local, "config.yaml")))  # noqa: SIM115
    valid = {f.name for f in fields(ExperimentConfig)}
    exp = ExperimentConfig(**{k: v for k, v in raw.items() if k in valid})
    base_id = getattr(exp.model, "base_model_id", "Qwen/Qwen3-VL-4B-Instruct")

    config = AutoConfig.from_pretrained(local)
    processor = AutoProcessor.from_pretrained(local)
    tokenizer = AutoTokenizer.from_pretrained(local)
    for c in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
        if c is not None:
            c._attn_implementation = "sdpa"

    with torch.device("meta"):
        model = RBM(config, processor, tokenizer, base_model=None, base_model_id=base_id,
                    model_config=exp.model)
    q.install_linear4bit_shells(model, torch.bfloat16)
    state = load_file(os.path.join(local, "model.safetensors"), device=device)
    consumed = q.install_prequantized(model, state, device)
    leftover = {k: v for k, v in state.items() if k not in consumed}
    model.load_state_dict(leftover, strict=False, assign=True)
    q.assign_meta_buffers(model, state, device)
    model.eval()
    collator = setup_batch_collator(processor, tokenizer, exp, is_eval=True)

    t = int(frames_rgb.shape[0])
    traj = Trajectory(frames=frames_rgb, frames_shape=tuple(frames_rgb.shape), task=task,
                      id="0", metadata={"subsequence_length": t}, video_embeddings=None)
    batch = collator([ProgressSample(trajectory=traj, sample_type="progress")])
    inp = batch["progress_inputs"]
    for k, v in inp.items():
        if hasattr(v, "to"):
            inp[k] = v.to(device)
    with torch.no_grad():
        res = compute_batch_outputs(model, tokenizer, inp, sample_type="progress",
                                    is_discrete_mode=True, num_bins=100)
    prog = np.asarray(res["progress_pred"][0], dtype=np.float32)
    succ_raw = res.get("outputs_success", {}).get("success_probs")
    succ = np.asarray(succ_raw[0] if isinstance(succ_raw, list) else succ_raw, dtype=np.float32)
    return prog, succ


def capture_golden() -> None:
    """Run the OLD upstream path and save the golden per-frame series (sidecar venv)."""
    frames, task = load_frames()
    prog, succ = score_old(frames, task)
    np.savez(_GOLDEN_NPZ, progress=prog, success=succ, task=np.array(task))
    print(f"[parity] golden progress: {np.round(prog, 4).tolist()}")
    print(f"[parity] golden success : {np.round(succ, 4).tolist()}")
    print(f"[parity] wrote {_GOLDEN_NPZ}")


def verify() -> int:
    """Run the NEW native path and assert it matches the golden within tolerance."""
    if not _GOLDEN_NPZ.exists():
        raise SystemExit(f"{_GOLDEN_NPZ} missing — run `--capture-golden` first (sidecar venv)")
    frames, task = load_frames()
    gold = np.load(_GOLDEN_NPZ, allow_pickle=True)
    pg, sg = gold["progress"], gold["success"]
    pn, sn = score_new(frames, task)
    print(f"[parity] new progress: {np.round(pn, 4).tolist()}")
    print(f"[parity] new success : {np.round(sn, 4).tolist()}")
    print(f"[parity] gold progress: {np.round(pg, 4).tolist()}")
    print(f"[parity] gold success : {np.round(sg, 4).tolist()}")
    if pn.shape != pg.shape or sn.shape != sg.shape:
        print(f"[parity] FAIL shape mismatch progress {pn.shape} vs {pg.shape}, "
              f"success {sn.shape} vs {sg.shape}")
        return 1
    dp = float(np.max(np.abs(pn - pg)))
    ds = float(np.max(np.abs(sn - sg)))
    ok_p = np.allclose(pn, pg, rtol=1e-3, atol=1e-2)
    ok_s = np.allclose(sn, sg, rtol=1e-3, atol=1e-2)
    print(f"[parity] max|Δprogress|={dp:.5f}  max|Δsuccess|={ds:.5f}  "
          f"(rtol=1e-3 atol=1e-2)")
    if ok_p and ok_s:
        print("[parity] PASS")
        return 0
    print(f"[parity] FAIL  progress_ok={ok_p} success_ok={ok_s}")
    return 1


def main() -> int:
    """Dispatch --prep-frames / --capture-golden / --verify."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep-frames", action="store_true")
    ap.add_argument("--capture-golden", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.prep_frames:
        prep_frames()
        return 0
    if args.capture_golden:
        capture_golden()
        return 0
    if args.verify:
        return verify()
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

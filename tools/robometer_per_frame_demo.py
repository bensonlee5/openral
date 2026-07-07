#!/usr/bin/env python
"""Prove the lerobot 0.6.0 native Robometer yields a per-frame progress curve on 8 GB.

Robometer (``lerobot.rewards.robometer.RobometerRewardModel``, Qwen3-VL-4B) is a
*windowed* progress+success reward: given a short clip of frames + the task
instruction it returns, per frame, a **progress (0-1)** and a **success (0-1)**
score. In deploy the reasoner scores a *trailing* window every tick and reads the
last-frame value (``success_now``); this demo reproduces exactly that — a sliding
window over a real LIBERO episode — so the curve matches deploy behaviour.

The only thing we add over stock lerobot is NF4 quantization of the Qwen3-VL
backbone (loaded from ``OpenRAL/rskill-robometer-4b-nf4``), because a 4 B bf16 VLM
does not fit the 8 GB dev GPU. This runs the migrated in-tree scorer
(``tools/_robometer_server._Scorer`` — plain transformers, no ``robometer`` git
package, no ``transformers==4.57.1`` pin).

Run (single GPU — take the shared lock)::

    flock /tmp/openral-gpu.lock -c \
      "./.venv/bin/python tools/robometer_per_frame_demo.py \
         --media-dir rskills/robometer-4b/media --out rskills/robometer-4b/media/progress.png"

ponytail: no test framework — one real clip, real NF4 weights, real asserts.
"""

from __future__ import annotations

# expandable_segments must be set before the first CUDA allocation (fits the
# NF4 4 B backbone + an 8-frame Qwen3-VL forward on 8 GB).
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import pathlib
import sys

import cv2
import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
_MODEL_MAX_FRAMES = 8  # RobometerConfig.max_frames — the native scoring window


def _load_frames(mp4: pathlib.Path, n: int, res: int) -> np.ndarray:
    """Evenly sample ``n`` RGB frames from ``mp4`` as a (n, res, res, 3) uint8 clip."""
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, total - 1, n).round().astype(int)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {i} of {mp4}")
        bgr = cv2.resize(bgr, (res, res), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out).astype(np.uint8)


def _trailing_window_progress(
    scorer: object, clip: np.ndarray, task: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame progress/success via a trailing window (deploy semantics).

    For frame ``i`` score the window ``clip[max(0, i-7) : i+1]`` and take the
    LAST value — the same trailing-window read the reasoner uses live.
    """
    prog, succ = [], []
    for i in range(len(clip)):
        lo = max(0, i + 1 - _MODEL_MAX_FRAMES)
        window = clip[lo : i + 1]
        p, s = scorer.score(window, task, num_bins=100)  # type: ignore[attr-defined]
        prog.append(float(p[-1]))
        succ.append(float(s[-1]))
    return np.asarray(prog), np.asarray(succ)


def _render_overlay(frame: np.ndarray, progress: float, success: float, task: str) -> np.ndarray:
    """RGB frame with a Robometer progress bar + success readout burned in."""
    h, w = frame.shape[:2]
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
    y0 = h - 84
    cv2.rectangle(canvas, (0, y0 - 8), (w, h), (28, 28, 28), -1)
    x1, x2, by = 12, w - 12, y0 + 44
    cv2.rectangle(canvas, (x1, by), (x2, by + 14), (70, 70, 70), -1)
    fillx = x1 + int((x2 - x1) * float(np.clip(progress, 0, 1)))
    cv2.rectangle(canvas, (x1, by), (fillx, by + 14), (80, 200, 90), -1)
    cv2.putText(
        canvas,
        f"progress {progress:0.2f}   success {success:0.2f}",
        (x1, y0 + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cap = task if len(task) < 46 else task[:43] + "..."
    cv2.putText(
        canvas, cap, (x1, y0 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 200, 235), 1, cv2.LINE_AA
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _write_media(
    frames: np.ndarray, progress: np.ndarray, success: np.ndarray, task: str, media_dir: pathlib.Path
) -> None:
    """Write progress.mp4 + start/mid/end stills with the Robometer overlay."""
    import imageio.v2 as imageio

    media_dir.mkdir(parents=True, exist_ok=True)
    overlaid = [
        _render_overlay(frames[i], float(progress[i]), float(success[i]), task)
        for i in range(len(frames))
    ]
    imageio.mimwrite(
        str(media_dir / "progress.mp4"), overlaid, fps=6, codec="libx264", quality=8,
        macro_block_size=1,
    )
    n = len(frames)
    for label, i in (("start", 1), ("mid", n // 2), ("end", n - 2)):
        imageio.imwrite(str(media_dir / f"frame_{label}.png"), overlaid[i])


def _sparkline(x: np.ndarray) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(7, int(round(v * 7)))] for v in x)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default=str(_REPO / "example_videos" / "libero_spatial.mp4"))
    p.add_argument("--weights", default="OpenRAL/rskill-robometer-4b-nf4")
    p.add_argument("--task", default="pick up the black bowl and place it on the plate")
    p.add_argument("--num-frames", type=int, default=20)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--out", default="tools/_robometer_progress.png")
    p.add_argument("--media-dir", default=None, help="if set, write progress.mp4 + 3 stills here")
    args = p.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA GPU")
        return 0

    clip = _load_frames(pathlib.Path(args.clip), args.num_frames, args.res)
    sys.path.insert(0, str(_REPO / "tools"))
    import _robometer_server as srv

    scorer = srv._Scorer(args.weights, device="cuda")
    progress, success = _trailing_window_progress(scorer, clip, args.task)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    head = float(progress[: max(1, len(progress) // 5)].mean())
    tail = float(progress[-max(1, len(progress) // 5) :].mean())
    print(f"\ntask: {args.task!r}")
    print(f"frames: {len(clip)}   peak VRAM: {peak_gb:.2f} GB   backbone: Qwen3-VL-4B (NF4)")
    print(f"per-frame progress [{progress.min():.2f}..{progress.max():.2f}]  {_sparkline(progress)}")
    print(f"per-frame success  [{success.min():.2f}..{success.max():.2f}]  {_sparkline(success)}")
    print(f"first-20% progress mean={head:.3f}   last-20% mean={tail:.3f}")

    assert progress.shape == (len(clip),), f"one value per frame expected, got {progress.shape}"
    assert progress.min() >= 0.0 and progress.max() <= 1.0, "progress must be in [0, 1]"
    assert success.min() >= 0.0 and success.max() <= 1.0, "success must be in [0, 1]"
    assert tail > head, f"progress should rise on a success clip (head={head:.3f} tail={tail:.3f})"
    assert peak_gb < 8.0, f"must fit 8 GB, peaked {peak_gb:.2f} GB"

    if args.media_dir:
        _write_media(clip, progress, success, args.task, pathlib.Path(args.media_dir))
        print(f"wrote progress.mp4 + start/mid/end stills to {args.media_dir}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 3))
        plt.plot(progress, lw=2, color="#50c85a", label="progress")
        plt.plot(success, lw=1.5, color="#4a90d9", ls="--", label="success")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("frame")
        plt.ylabel("score")
        plt.title(f"Robometer per-frame progress — {args.task}", fontsize=8)
        plt.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        plt.savefig(args.out, dpi=110)
        print(f"saved {args.out}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print("\nPASS: native Robometer produces a valid rising per-frame progress curve within 8 GB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

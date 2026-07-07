"""Launch the Robometer reward-monitor sidecar (ADR-0057).

The Robometer reward model runs out-of-process (its own process, optionally its
own venv) for env isolation and 8 GB memory tuning (``PYTORCH_ALLOC_CONF``). As
of lerobot 0.6.0 it is loaded through lerobot's in-tree
``lerobot.rewards.robometer.RobometerRewardModel`` — a vanilla
``AutoModelForImageTextToText`` (Qwen3-VL-4B) with plain ``transformers``. There
is NO longer a pinned ``robometer`` git package and NO ``transformers==4.57.1``
force-pin: the sidecar boots straight into ``_robometer_server.py`` with the
current interpreter (the node's own env, which ships lerobot + qwen-vl-utils +
pyzmq/msgpack via ``uv sync --group robometer``).

An operator may still point the sidecar at a separate, pre-provisioned venv with
``$OPENRAL_ROBOMETER_SIDECAR_VENV`` (or ``--venv``) — it just needs the same
deps. The node-side client is
:class:`openral_runner.backends.reward.robometer_reward.RobometerReward`, which
auto-spawns this script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_VENV_ENV = "OPENRAL_ROBOMETER_SIDECAR_VENV"


def _resolve_python(override: str | None) -> str:
    """Return the interpreter to run the server with.

    Defaults to the current interpreter (the node's env already ships lerobot +
    the sidecar deps). An explicit venv override is honored for operators who
    want a fully isolated env.
    """
    override = override or os.environ.get(_VENV_ENV)
    if override:
        py = Path(override) / "bin" / "python"
        if not py.exists():
            raise SystemExit(f"{_VENV_ENV} points at {override} but {py} does not exist")
        return str(py)
    return sys.executable


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--weights", default="OpenRAL/rskill-robometer-4b-nf4")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5769)
    p.add_argument("--venv", default=None, help=f"Reuse this venv (or set {_VENV_ENV}).")
    args = p.parse_args()

    py = _resolve_python(args.venv)
    server = Path(__file__).resolve().parent / "_robometer_server.py"

    env = os.environ.copy()
    # Drop PYTHONPATH/PYTHONHOME so the sidecar boots from its interpreter's own
    # site-packages (ROS 2 / colcon populate PYTHONPATH and would shadow lerobot).
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    # Empirically required to fit NF4 + the forward in 8 GB (ADR-0058 Phase 2).
    # torch >= 2.9 renamed the knob; set both so old + new torch both honor it.
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [
        py,
        str(server),
        "--weights",
        args.weights,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print(
        f"[robometer-sidecar] launching server: weights={args.weights} port={args.port}",
        flush=True,
    )
    os.execvpe(py, cmd, env)


if __name__ == "__main__":
    raise SystemExit(main())

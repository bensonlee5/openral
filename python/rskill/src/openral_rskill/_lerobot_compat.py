"""Compatibility shim for ``lerobot.policies`` import side-effects.

Importing ``lerobot.policies`` (any submodule) eagerly initialises
``lerobot.policies.groot``. Upstream ``lerobot==0.5.0/0.5.1`` shipped a
``GR00TN15Config`` dataclass that failed to construct on Python 3.12 with
``transformers>=5.3`` (``TypeError: non-default argument 'backbone_cfg'
follows default argument``), which crashed the whole package import.

For that broken combination this module installs a stub
``lerobot.policies.groot.modeling_groot`` so the package initialises.

On ``lerobot>=0.6.0`` GR00T-N1.7 is first-party and imports cleanly, so the
shim self-disables: it must NOT install the stub, because a stub
``GrootPolicy`` would shadow the real class and break the in-process GR00T
backend (``openral_sim.policies.gr00t``). Importing this module is therefore a
harmless no-op on a modern lerobot.
"""

from __future__ import annotations

import importlib
import sys
import types

_STUB_NAME = "lerobot.policies.groot.modeling_groot"


def _install_stub() -> None:
    if _STUB_NAME in sys.modules:
        return
    try:
        # lerobot >= 0.6.0: the real modeling_groot imports fine — use it, so the
        # genuine GrootPolicy (with from_pretrained) is what callers resolve.
        importlib.import_module(_STUB_NAME)
        return
    except Exception:
        # Only the broken 0.5.x combination (or a groot-deps-less install) lands
        # here; install the empty stub so `lerobot.policies` can still initialise.
        stub = types.ModuleType(_STUB_NAME)
        stub.GrootPolicy = type("GrootPolicy", (), {})  # type: ignore[attr-defined]
        sys.modules[_STUB_NAME] = stub


_install_stub()

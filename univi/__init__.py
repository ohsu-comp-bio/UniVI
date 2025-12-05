# univi/__init__.py

from __future__ import annotations

from typing import Any, List

__version__ = "0.2.2"

from .config import ModalityConfig, UniVIConfig, TrainingConfig
from .models import UniVIMultiModalVAE
from . import matching

# Public API that is safe to import eagerly:
__all__ = [
    "__version__",
    "ModalityConfig",
    "UniVIConfig",
    "TrainingConfig",
    "UniVIMultiModalVAE",
    "matching",
    # Lazy exports (resolved via __getattr__)
    "UniVITrainer",
    "write_univi_latent",
    "pipeline",
    "diagnostics",
]


def __getattr__(name: str) -> Any:
    """
    Lazy exports keep `import univi` fast/light and avoid heavy deps unless needed.
    """
    if name == "UniVITrainer":
        from .trainer import UniVITrainer
        return UniVITrainer

    if name == "write_univi_latent":
        from .utils.io import write_univi_latent
        return write_univi_latent

    # Note: these return the *module* objects
    if name == "pipeline":
        from . import pipeline as _pipeline
        return _pipeline

    if name == "diagnostics":
        from . import diagnostics as _diagnostics
        return _diagnostics

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    # Makes autocomplete / dir(univi) show lazy symbols too.
    return sorted(list(globals().keys()) + __all__)


# univi/__init__.py

__version__ = "0.2.1"

from .config import ModalityConfig, UniVIConfig, TrainingConfig
from .models import UniVIMultiModalVAE
from . import matching

def __getattr__(name):
    # Lazy imports keep `import univi` fast/light and avoid heavy deps unless needed.
    if name == "UniVITrainer":
        from .trainer import UniVITrainer
        return UniVITrainer
    if name == "write_univi_latent":
        from .utils.io import write_univi_latent
        return write_univi_latent
    if name == "pipeline":
        from . import pipeline
        return pipeline
    if name == "diagnostics":
        from . import diagnostics
        return diagnostics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ModalityConfig",
    "UniVIConfig",
    "TrainingConfig",
    "UniVITrainer",
    "UniVIMultiModalVAE",
    "matching",
    "write_univi_latent",
    "pipeline",
    "diagnostics",
]

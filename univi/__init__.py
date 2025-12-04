# univi/__init__.py

__version__ = "0.1.6"

from .config import ModalityConfig, UniVIConfig, TrainingConfig
from .models import UniVIMultiModalVAE
from . import matching
from .utils import write_univi_latent

def __getattr__(name):
    # Lazy import so `import univi` works even without torch installed
    if name == "UniVITrainer":
        from .trainer import UniVITrainer
        return UniVITrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ModalityConfig",
    "UniVIConfig",
    "TrainingConfig",
    "UniVITrainer",
    "UniVIMultiModalVAE",
    "matching",
    "write_univi_latent",
]

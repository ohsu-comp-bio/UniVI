# univi/__init__.py
#
# Keep exports stable so existing notebooks/scripts keep working.

from .config import ModalityConfig, UniVIConfig, TrainingConfig
from .trainer import UniVITrainer
from .models import UniVIMultiModalVAE
from . import matching

__all__ = [
    "ModalityConfig",
    "UniVIConfig",
    "TrainingConfig",
    "UniVITrainer",
    "UniVIMultiModalVAE",
    "matching",
]

# univi/__init__.py
"""
UniVI: a scalable multi-modal variational autoencoder toolkit
for multimodal single-cell data.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    # Reads the version from installed package metadata (pyproject.toml)
    __version__ = _version("univi")
except PackageNotFoundError:
    # Fallback for editable installs or unusual environments
    __version__ = "0.0.0"

from .models import UniVIMultiModalVAE
from .config import ModalityConfig, UniVIConfig, TrainingConfig
from . import matching

__all__ = [
    "UniVIMultiModalVAE",
    "ModalityConfig",
    "UniVIConfig",
    "TrainingConfig",
    "matching",
    "__version__",
]

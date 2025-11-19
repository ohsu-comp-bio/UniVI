# univi/config.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

# ----------------- Model configs ----------------- #

@dataclass
class ModalityConfig:
    """
    Per-modality configuration for UniVI.

    - name:       "rna", "adt", "atac", etc.
    - input_dim:  number of features (n_vars) for this modality
    - encoder_hidden / decoder_hidden: list of hidden layer sizes
    - likelihood: "gaussian", "nb", "zinb", "lognormal", ...
    """
    name: str
    input_dim: int
    encoder_hidden: List[int]
    decoder_hidden: List[int]
    likelihood: str = "gaussian"


@dataclass
class UniVIConfig:
    """
    Global model configuration (architecture + regularization).
    """
    latent_dim: int
    modalities: List[ModalityConfig]

    # β-VAE and alignment strengths
    beta: float = 1.0           # weight on KL term
    gamma: float = 1.0          # weight on alignment term (formerly gamma_align)

    # encoder / decoder options
    encoder_dropout: float = 0.0
    decoder_dropout: float = 0.0
    encoder_batchnorm: bool = True
    decoder_batchnorm: bool = False

    # KL and alignment annealing (epoch indices)
    kl_anneal_start: int = 0
    kl_anneal_end: int = 0
    align_anneal_start: int = 0
    align_anneal_end: int = 0


# ----------------- Training config ----------------- #

@dataclass
class TrainingConfig:
    """
    Hyperparameters for the training loop.
    """
    n_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    log_every: int = 10
    grad_clip: Optional[float] = None
    num_workers: int = 0
    seed: int = 0

    # early stopping
    early_stopping: bool = False
    patience: int = 20
    min_delta: float = 0.0

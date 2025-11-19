# univi/trainer.py

from __future__ import annotations
from dataclasses import asdict
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import scipy.sparse as sp

from .config import TrainingConfig
from .utils.logging import get_logger


class UniVITrainer:
    """
    Trainer for UniVIMultiModalVAE.

    - Handles train / val loops
    - Supports early stopping on validation loss (if val_loader is given)
    - Tracks β and γ per epoch (annealed weights from the model)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        train_cfg: Optional[TrainingConfig] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # config
        self.cfg = train_cfg or TrainingConfig()
        self.device = device or self.cfg.device

        # move model to device
        self.model.to(self.device)

        # optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.logger = get_logger("UniVITrainer")

        # early stopping state
        self.best_val_loss: float = float("inf")
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.epochs_no_improve: int = 0
        self.best_epoch: Optional[int] = None

        # history dict we can return
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "beta": [],   # epoch-wise β (from train)
            "gamma": [],  # epoch-wise γ (from train)
        }

        # log config once
        self._log_config()

    # ------------------- public API -------------------

    def fit(self) -> Dict[str, List[float]]:
        """
        Train for cfg.n_epochs with optional early stopping.
        Returns a history dict with train/val losses + beta/gamma.
        """
        n_epochs = self.cfg.n_epochs

        epoch_iter = tqdm(
            range(1, n_epochs + 1),
            desc="Training UniVI",
            leave=True,
        )

        for epoch in epoch_iter:
            # ---- training epoch ----
            train_loss, train_beta, train_gamma = self._run_one_epoch(epoch, train=True)
            self.history["train_loss"].append(train_loss)
            self.history["beta"].append(train_beta)
            self.history["gamma"].append(train_gamma)

            if self.val_loader is not None:
                # ---- validation epoch ----
                val_loss, _, _ = self._run_one_epoch(epoch, train=False)
                self.history["val_loss"].append(val_loss)

                epoch_iter.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    beta=f"{train_beta:.3f}",
                    gamma=f"{train_gamma:.3f}",
                )

                self._maybe_early_stop(val_loss, epoch)
                if self._should_stop():
                    self.logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(best val loss = {self.best_val_loss:.4f})"
                    )
                    break
            else:
                epoch_iter.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    beta=f"{train_beta:.3f}",
                    gamma=f"{train_gamma:.3f}",
                )

        # restore best model if we used validation + early stopping
        if self.val_loader is not None and self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            if self.best_epoch is not None:
                self.logger.info(
                    f"Restored best model from epoch {self.best_epoch} "
                    f"(val loss = {self.best_val_loss:.4f})"
                )

        return self.history

    # ------------------- core loops -------------------

    def _run_one_epoch(self, epoch: int, train: bool = True) -> tuple[float, float, float]:
        """
        Run one epoch; returns (avg_loss, avg_beta, avg_gamma).
        """
        if train:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader

        total_loss = 0.0
        total_beta = 0.0
        total_gamma = 0.0
        n_batches = 0

        for batch in loader:
            # batch is expected to be a dict: modality_name -> tensor
            x_dict = {
                k: v.to(self.device) if v is not None else None
                for k, v in batch.items()
            }

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                out = self.model(x_dict, epoch=epoch)
                loss = out["loss"]

                beta_val = float(out.get("beta", 0.0))
                gamma_val = float(out.get("gamma", 0.0))

                if train:
                    loss.backward()
                    if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()

            total_loss += loss.item()
            total_beta += beta_val
            total_gamma += gamma_val
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_beta = total_beta / max(n_batches, 1)
        avg_gamma = total_gamma / max(n_batches, 1)

        mode = "Train" if train else "Val"
        if epoch % self.cfg.log_every == 0 or epoch == 1:
            self.logger.info(
                f"[Epoch {epoch:03d}] {mode} loss: {avg_loss:.4f} "
                f"(beta={avg_beta:.3f}, gamma={avg_gamma:.3f})"
            )
        return avg_loss, avg_beta, avg_gamma

    # ------------------- early stopping helpers -------------------

    def _maybe_early_stop(self, val_loss: float, epoch: int) -> None:
        if not self.cfg.early_stopping:
            return

        improved = (self.best_val_loss - val_loss) > self.cfg.min_delta
        if improved:
            self.best_val_loss = val_loss
            self.best_state_dict = {
                k: v.cpu().clone()
                for k, v in self.model.state_dict().items()
            }
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            self.logger.info(
                f"[Epoch {epoch:03d}] New best val loss: {val_loss:.4f}"
            )
        else:
            self.epochs_no_improve += 1

    def _should_stop(self) -> bool:
        if not self.cfg.early_stopping:
            return False
        return self.epochs_no_improve >= self.cfg.patience

    # ------------------- convenience: encode a single modality -------------------

    def encode_modality(
        self,
        adata,
        modality: str,
        layer: Optional[str] = None,
        X_key: str = "X",
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        Encode a single AnnData object for one modality into z.

        Parameters
        ----------
        adata : AnnData
            AnnData object for the chosen modality (same features as used in training).
        modality : str
            Name of the modality (must match cfg.modalities[*].name).
        layer : str or None
            If not None, use adata.layers[layer]; otherwise use .X.
        X_key : str
            Not used for now (kept for future compatibility).
        batch_size : int
            Batch size for encoding.

        Returns
        -------
        z_all : np.ndarray
            Latent embedding of shape [n_cells, latent_dim].
        """
        from anndata import AnnData  # just for type hints / sanity

        assert modality in self.model.modality_names, (
            f"Unknown modality '{modality}'. "
            f"Available: {self.model.modality_names}"
        )

        self.model.eval()

        X = adata.layers[layer] if layer is not None else adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        zs = []
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                xb = X[start:start + batch_size]
                xb_t = torch.as_tensor(xb, dtype=torch.float32, device=self.device)

                # encode only the chosen modality
                mu_dict, logvar_dict = self.model.encode_modalities({modality: xb_t})
                mu_z, logvar_z = self.model.mixture_of_experts(mu_dict, logvar_dict)
                # use posterior mean as embedding (more stable than a single sample)
                zs.append(mu_z.cpu().numpy())

        z_all = np.vstack(zs)
        return z_all

    # ------------------- misc -------------------

    def _log_config(self) -> None:
        cfg_dict = asdict(self.cfg)
        self.logger.info("TrainingConfig:")
        for k, v in cfg_dict.items():
            self.logger.info(f"  {k}: {v}")

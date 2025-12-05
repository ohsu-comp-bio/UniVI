# univi/trainer.py

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import scipy.sparse as sp

from .config import TrainingConfig
from .utils.logging import get_logger

BatchType = Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]


class UniVITrainer:
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

        self.cfg = train_cfg or TrainingConfig()
        self.device = device or self.cfg.device

        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        self.logger = get_logger("UniVITrainer")

        self.best_val_loss = float("inf")
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.epochs_no_improve = 0
        self.best_epoch: Optional[int] = None

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "beta": [],
            "gamma": [],
        }

        self._log_config()

    def fit(self) -> Dict[str, List[float]]:
        epoch_iter = tqdm(range(1, int(self.cfg.n_epochs) + 1), desc="Training UniVI", leave=True)

        for epoch in epoch_iter:
            tr_loss, tr_beta, tr_gamma = self._run_one_epoch(epoch, train=True)
            self.history["train_loss"].append(tr_loss)
            self.history["beta"].append(tr_beta)
            self.history["gamma"].append(tr_gamma)

            if self.val_loader is not None:
                va_loss, _, _ = self._run_one_epoch(epoch, train=False)
                self.history["val_loss"].append(va_loss)

                epoch_iter.set_postfix(
                    train_loss="%.4f" % tr_loss,
                    val_loss="%.4f" % va_loss,
                    beta="%.3f" % tr_beta,
                    gamma="%.3f" % tr_gamma,
                )

                self._maybe_early_stop(va_loss, epoch)
                if self._should_stop():
                    self.logger.info("Early stopping at epoch %d (best val loss=%.4f)" % (epoch, self.best_val_loss))
                    break
            else:
                epoch_iter.set_postfix(
                    train_loss="%.4f" % tr_loss,
                    beta="%.3f" % tr_beta,
                    gamma="%.3f" % tr_gamma,
                )

        if self.val_loader is not None and self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            if self.best_epoch is not None:
                self.logger.info("Restored best model from epoch %d (val loss=%.4f)" % (self.best_epoch, self.best_val_loss))

        return self.history

    def _split_batch(self, batch: BatchType) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x_dict, y = batch
        else:
            x_dict, y = batch, None

        if not isinstance(x_dict, dict):
            raise TypeError("Expected batch to be dict or (dict, y). Got %r" % type(x_dict))

        x_out: Dict[str, torch.Tensor] = {}
        for k, v in x_dict.items():
            if v is None:
                x_out[k] = None
            elif torch.is_tensor(v):
                x_out[k] = v.to(self.device)
            else:
                x_out[k] = torch.as_tensor(v, dtype=torch.float32, device=self.device)

        if y is not None:
            if not torch.is_tensor(y):
                y = torch.as_tensor(y, dtype=torch.long)
            y = y.long().to(self.device)

        return x_out, y

    def _as_float(self, v: Any) -> float:
        if v is None:
            return 0.0
        if isinstance(v, (float, int)):
            return float(v)
        if torch.is_tensor(v):
            return float(v.detach().cpu().item())
        try:
            return float(v)
        except Exception:
            return 0.0

    def _forward_model(self, x_dict: Dict[str, torch.Tensor], epoch: int, y: Optional[torch.Tensor]):
        # Prefer newest signature
        try:
            return self.model(x_dict, epoch=epoch, y=y)
        except TypeError:
            pass
        # epoch only
        try:
            return self.model(x_dict, epoch=epoch)
        except TypeError:
            pass
        # y only
        if y is not None:
            try:
                return self.model(x_dict, y=y)
            except TypeError:
                pass
        # oldest
        return self.model(x_dict)

    def _run_one_epoch(self, epoch: int, train: bool = True) -> Tuple[float, float, float]:
        if train:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            if self.val_loader is None:
                raise ValueError("val_loader is None but train=False")
            loader = self.val_loader

        total_loss = 0.0
        total_beta = 0.0
        total_gamma = 0.0
        n_batches = 0

        for batch in loader:
            x_dict, y = self._split_batch(batch)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                out = self._forward_model(x_dict, epoch=epoch, y=y)
                loss = out["loss"]

                if train:
                    loss.backward()
                    if self.cfg.grad_clip is not None and float(self.cfg.grad_clip) > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip))
                    self.optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_beta += self._as_float(out.get("beta", 0.0))
            total_gamma += self._as_float(out.get("gamma", 0.0))
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_beta = total_beta / max(1, n_batches)
        avg_gamma = total_gamma / max(1, n_batches)

        if epoch % int(self.cfg.log_every) == 0 or epoch == 1:
            self.logger.info("[Epoch %03d] %s loss=%.4f (beta=%.3f, gamma=%.3f)" %
                             (epoch, "Train" if train else "Val", avg_loss, avg_beta, avg_gamma))

        return avg_loss, avg_beta, avg_gamma

    def _maybe_early_stop(self, val_loss: float, epoch: int) -> None:
        if not self.cfg.early_stopping:
            return

        improved = (self.best_val_loss - float(val_loss)) > float(self.cfg.min_delta)
        if improved:
            self.best_val_loss = float(val_loss)
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.best_epoch = int(epoch)
            self.epochs_no_improve = 0
            self.logger.info("[Epoch %03d] New best val loss: %.4f" % (epoch, val_loss))
        else:
            self.epochs_no_improve += 1

    def _should_stop(self) -> bool:
        if not self.cfg.early_stopping:
            return False
        return int(self.epochs_no_improve) >= int(self.cfg.patience)

    def encode_modality(
        self,
        adata,
        modality: str,
        layer: Optional[str] = None,
        X_key: str = "X",
        batch_size: int = 512,
    ) -> np.ndarray:
        names = getattr(self.model, "modality_names", None)
        if names is not None and modality not in names:
            raise ValueError("Unknown modality %r. Available: %s" % (modality, names))

        self.model.eval()
        X = adata.layers[layer] if layer is not None else adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        zs = []
        with torch.no_grad():
            for start in range(0, X.shape[0], int(batch_size)):
                xb = torch.as_tensor(X[start:start + int(batch_size)], dtype=torch.float32, device=self.device)
                mu_dict, logvar_dict = self.model.encode_modalities({modality: xb})
                mu_z, _ = self.model.mixture_of_experts(mu_dict, logvar_dict)
                zs.append(mu_z.detach().cpu().numpy())

        return np.vstack(zs)

    def _log_config(self) -> None:
        cfg_dict = asdict(self.cfg)
        self.logger.info("TrainingConfig:")
        for k, v in cfg_dict.items():
            self.logger.info("  %s: %r" % (k, v))


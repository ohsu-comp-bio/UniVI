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
        use_amp: bool = False,
        amp_dtype: str = "fp16",  # "fp16" or "bf16"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.cfg = train_cfg or TrainingConfig()
        self.device = device or self.cfg.device

        self.use_amp = bool(use_amp)
        self.amp_dtype = str(amp_dtype).lower().strip()
        if self.amp_dtype not in ("fp16", "bf16"):
            raise ValueError(f"amp_dtype must be 'fp16' or 'bf16', got {amp_dtype!r}")

        self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        self._scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self.use_amp and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            self._scaler = torch.cuda.amp.GradScaler(enabled=(self.amp_dtype == "fp16"))

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

    # ------------------------------ training ------------------------------

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
                    self.logger.info(
                        "Early stopping at epoch %d (best val loss=%.4f)" % (epoch, self.best_val_loss)
                    )
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
                self.logger.info(
                    "Restored best model from epoch %d (val loss=%.4f)"
                    % (self.best_epoch, self.best_val_loss)
                )

        return self.history

    # ------------------------------ batch handling ------------------------------

    def _split_batch(self, batch: BatchType) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Accepts either:
          - x_dict
          - (x_dict, y)

        Moves tensors to device. Does NOT force-cast x modalities (keeps float32, float16, etc).
        Ensures y is long if provided.
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x_dict, y = batch
        else:
            x_dict, y = batch, None

        if not isinstance(x_dict, dict):
            raise TypeError(f"Expected batch to be dict or (dict, y). Got {type(x_dict)!r}")

        x_out: Dict[str, torch.Tensor] = {}
        for k, v in x_dict.items():
            if v is None:
                x_out[k] = None
                continue

            if torch.is_tensor(v):
                x_out[k] = v.to(self.device, non_blocking=True)
            else:
                # fallback: assume array-like; keep float32
                x_out[k] = torch.as_tensor(v, dtype=torch.float32, device=self.device)

        if y is not None:
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)
            y = y.long().to(self.device, non_blocking=True)

        return x_out, y

    # ------------------------------ forward wrappers ------------------------------

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

    @staticmethod
    def _as_float(v: Any) -> float:
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

    def _amp_context(self):
        if not (self.use_amp and torch.cuda.is_available() and str(self.device).startswith("cuda")):
            return torch.autocast(device_type="cuda", enabled=False)

        dtype = torch.float16 if self.amp_dtype == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)

    # ------------------------------ epoch loop ------------------------------

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
                with self._amp_context():
                    out = self._forward_model(x_dict, epoch=epoch, y=y)
                    loss = out["loss"]

                if train:
                    if self._scaler is not None and self._scaler.is_enabled():
                        self._scaler.scale(loss).backward()
                        if self.cfg.grad_clip is not None and float(self.cfg.grad_clip) > 0:
                            self._scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip))
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                    else:
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
            self.logger.info(
                "[Epoch %03d] %s loss=%.4f (beta=%.3f, gamma=%.3f)"
                % (epoch, "Train" if train else "Val", avg_loss, avg_beta, avg_gamma)
            )

        return avg_loss, avg_beta, avg_gamma

    # ------------------------------ early stopping ------------------------------

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

    # ------------------------------ encoding utility ------------------------------

    def encode_modality(
        self,
        adata: AnnData,
        modality: str,
        layer: Optional[str] = None,
        X_key: str = "X",
        obs_key: Optional[str] = None,
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        Encode a single modality AnnData into latent means (mu_z).

        Supports:
          - matrix modalities via X/layers/obsm (default, using layer/X_key)
          - categorical-from-obs modalities if obs_key is provided (returns (B,1) codes)
        """
        names = getattr(self.model, "modality_names", None)
        if names is not None and modality not in names:
            raise ValueError("Unknown modality %r. Available: %s" % (modality, names))

        self.model.eval()

        # Pull features
        if obs_key is not None:
            if obs_key not in adata.obs:
                raise KeyError(f"obs_key={obs_key!r} not found in adata.obs.")
            col = adata.obs[obs_key]
            if hasattr(col, "cat"):
                vals = col.cat.codes.to_numpy()
            else:
                vals = np.asarray(col.values)
            X = vals.astype(np.float32).reshape(-1, 1)
        else:
            if X_key != "X":
                if X_key not in adata.obsm:
                    raise KeyError(f"X_key={X_key!r} not in adata.obsm.")
                X = adata.obsm[X_key]
            else:
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

    # ------------------------------ logging ------------------------------

    def _log_config(self) -> None:
        cfg_dict = asdict(self.cfg)
        self.logger.info("TrainingConfig:")
        for k, v in cfg_dict.items():
            self.logger.info("  %s: %r" % (k, v))


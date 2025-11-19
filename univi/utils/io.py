# univi/utils/io.py

from __future__ import annotations
from typing import Any, Dict, Optional
import json
import torch


def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    payload: Dict[str, Any] = {
        "model_state": model_state,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str):
    return torch.load(path, map_location="cpu")


def save_config_json(config: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

"""Device selection shared by every entry point that takes ``device=``."""

from __future__ import annotations

import torch

__all__ = ["resolve_device"]


def resolve_device(device, default="cpu") -> torch.device:
    """*device* as a :class:`torch.device`, or *default* when it is ``None``."""
    return torch.device(default if device is None else device)

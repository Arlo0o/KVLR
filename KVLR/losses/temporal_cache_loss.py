from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def temporal_cache_loss(
    features: Tensor | None,
    target_shape: tuple[int, int, int] | None = None,
    token_policy: Tensor | None = None,
) -> Tensor | None:
    """
    Simple temporal smoothness regularizer for LIGHT/REUSE regions.
    """
    if features is None or target_shape is None or features.ndim != 3:
        return None
    T, H, W = target_shape
    B, N, C = features.shape
    if N != T * H * W or T < 2:
        return None
    feat = features.reshape(B, T, H * W, C)
    diff = (feat[:, 1:] - feat[:, :-1]).float()
    if token_policy is not None and token_policy.shape == (B, N):
        mask = (token_policy.reshape(B, T, H * W)[:, 1:] <= 1).float().unsqueeze(-1)
        diff = diff * mask
        denom = mask.sum().clamp_min(1.0)
        return (diff.square().sum() / denom).to(features.dtype)
    return F.mse_loss(feat[:, 1:].float(), feat[:, :-1].detach().float()).to(features.dtype)

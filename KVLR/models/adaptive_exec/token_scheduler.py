from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TokenExecutionPolicy:
    REUSE: int = 0
    LIGHT: int = 1
    FULL: int = 2


class TokenScheduler:
    """
    Converts a criticality score into token execution modes.

    The scheduler is budget-first: within each sample, the most critical tokens
    receive FULL execution, the next band receives LIGHT execution, and the
    remaining tokens are marked REUSE.
    """

    def __init__(self, full_ratio: float = 0.2, light_ratio: float = 0.3):
        if full_ratio < 0 or light_ratio < 0 or full_ratio + light_ratio > 1.0:
            raise ValueError("full_ratio and light_ratio must be non-negative and sum to <= 1")
        self.full_ratio = float(full_ratio)
        self.light_ratio = float(light_ratio)
        self.policy = TokenExecutionPolicy()

    def __call__(self, criticality: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if criticality.ndim != 2:
            raise ValueError("criticality must have shape [B, N]")

        B, N = criticality.shape
        modes = torch.full_like(criticality, self.policy.REUSE, dtype=torch.long)
        order = torch.argsort(criticality, dim=1, descending=True)

        n_full = int(round(N * self.full_ratio))
        n_light = int(round(N * self.light_ratio))
        if n_full > 0:
            modes.scatter_(1, order[:, :n_full], self.policy.FULL)
        if n_light > 0:
            start = n_full
            end = min(N, n_full + n_light)
            modes.scatter_(1, order[:, start:end], self.policy.LIGHT)

        stats = self.compute_stats(modes)
        return modes, stats

    def compute_stats(self, modes: Tensor) -> dict[str, Tensor]:
        policy = self.policy
        total = modes.numel()
        full = (modes == policy.FULL).float().sum() / max(total, 1)
        light = (modes == policy.LIGHT).float().sum() / max(total, 1)
        reuse = (modes == policy.REUSE).float().sum() / max(total, 1)
        return {
            "full_ratio": full,
            "light_ratio": light,
            "reuse_ratio": reuse,
            "active_ratio": full + light,
        }

from __future__ import annotations

import torch
from torch import Tensor


class TemporalRefreshScheduler:
    """
    Frame-level refresh policy for video reuse/cache experiments.
    """

    REFRESH = 1
    KEEP_CACHE = 0

    def __init__(
        self,
        high_threshold: float = 0.65,
        mid_threshold: float = 0.35,
        refresh_high: int = 1,
        refresh_mid: int = 2,
        refresh_low: int = 4,
    ):
        self.high_threshold = float(high_threshold)
        self.mid_threshold = float(mid_threshold)
        self.refresh_high = max(int(refresh_high), 1)
        self.refresh_mid = max(int(refresh_mid), 1)
        self.refresh_low = max(int(refresh_low), 1)

    def __call__(
        self,
        criticality: Tensor,
        target_shape: tuple[int, int, int],
        denoise_step: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        T, H, W = target_shape
        frame_score = criticality.reshape(criticality.shape[0], T, H, W).mean(dim=(2, 3))
        interval = torch.full_like(frame_score, self.refresh_low, dtype=torch.long)
        interval = torch.where(
            frame_score >= self.mid_threshold,
            torch.full_like(interval, self.refresh_mid),
            interval,
        )
        interval = torch.where(
            frame_score >= self.high_threshold,
            torch.full_like(interval, self.refresh_high),
            interval,
        )
        refresh = (denoise_step % interval.clamp(min=1) == 0).long()
        stats = {"refresh_ratio": refresh.float().mean()}
        return refresh, stats

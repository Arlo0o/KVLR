from __future__ import annotations

import torch
from torch import Tensor


def budget_loss(
    adaptive_stats: dict | None,
    target_active_ratio: float = 0.5,
    target_refresh_ratio: float = 0.5,
    full_weight: float = 1.0,
    light_weight: float = 0.35,
    refresh_weight: float = 0.25,
) -> tuple[Tensor | None, dict[str, Tensor]]:
    """
    Penalize execution cost using policy statistics. Returns None when no
    adaptive stats are available.
    """
    if not adaptive_stats:
        return None, {}
    full = adaptive_stats.get("full_ratio")
    light = adaptive_stats.get("light_ratio")
    refresh = adaptive_stats.get("refresh_ratio")
    if full is None or light is None:
        return None, {}
    active_cost = full_weight * full + light_weight * light
    soft_active = adaptive_stats.get("soft_active_ratio")
    if soft_active is not None:
        # Differentiable proxy for learnable criticality heads. The hard token
        # ratios remain useful for reporting, while this term can shape the
        # policy before committing to hard top-k budgets.
        active_cost = 0.5 * active_cost + 0.5 * soft_active
    active_target = full.new_tensor(float(target_active_ratio))
    loss = (active_cost - active_target).abs()
    losses = {"budget_active": loss}
    if refresh is not None:
        refresh_target = refresh.new_tensor(float(target_refresh_ratio))
        refresh_loss = refresh_weight * (refresh - refresh_target).abs()
        losses["budget_refresh"] = refresh_loss
        loss = loss + refresh_loss
    return loss, losses

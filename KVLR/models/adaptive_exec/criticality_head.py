from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class CriticalityWeights:
    motion: float = 1.0
    tool: float = 0.7
    gate_confidence: float = 0.3
    fine_motion: float = 0.4
    skip: float = 0.5


class HeuristicCriticalityHead(nn.Module):
    """
    Converts structured action-control signals into an interpretable token
    criticality map. This is intentionally heuristic for the MVP: it gives an
    inspectable baseline before replacing the score with a learned head.
    """

    def __init__(self, weights: Optional[dict] = None, learnable: bool = False):
        super().__init__()
        base = CriticalityWeights()
        if weights:
            for key, value in weights.items():
                if hasattr(base, key):
                    setattr(base, key, float(value))
        self.weights = base
        self.learnable = bool(learnable)
        if self.learnable:
            init = torch.tensor(
                [base.motion, base.tool, base.gate_confidence, base.fine_motion, base.skip],
                dtype=torch.float32,
            )
            self.raw_weights = nn.Parameter(init)
        else:
            self.register_parameter("raw_weights", None)

    def forward(
        self,
        skeleton_tensor: Tensor,
        target_shape: tuple[int, int, int],
        outer_gate: Optional[Tensor] = None,
        inner_gate: Optional[Tensor] = None,
        skip_prob: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            skeleton_tensor: [B, T, H, W, 9] KVA-Field tensor.
            target_shape: latent token grid (T', H', W').
            outer_gate: optional [B*T, E, H_g, W_g] routing probabilities/logits.
            inner_gate: optional [B*T, S, H_g, W_g] averaged sub-expert probabilities.
            skip_prob: optional [B*T, 1, H_g, W_g] skip probability.

        Returns:
            criticality: [B, T' * H' * W'] normalized to [0, 1].
        """
        if skeleton_tensor.ndim != 5 or skeleton_tensor.shape[-1] < 9:
            raise ValueError("skeleton_tensor must have shape [B, T, H, W, 9]")

        B, _, _, _, _ = skeleton_tensor.shape
        T_target, H_target, W_target = target_shape

        semantic = skeleton_tensor[..., 0:3].float().norm(dim=-1)
        velocity = skeleton_tensor[..., 5:8].float().norm(dim=-1)
        accel = skeleton_tensor[..., 8].float().abs()
        motion_energy = velocity + accel
        tool_mask = (semantic > 0).to(skeleton_tensor.dtype)

        weights = self._current_weights(skeleton_tensor.device, skeleton_tensor.dtype)
        score = weights["motion"] * motion_energy + weights["tool"] * tool_mask
        score = score[:, None]  # [B, 1, T, H, W]
        score = F.adaptive_avg_pool3d(score, (T_target, H_target, W_target)).squeeze(1)

        gate_conf = self._pool_bt_map(
            outer_gate, B, T_target, H_target, W_target, reducer="max"
        )
        if gate_conf is not None:
            score = score + weights["gate_confidence"] * gate_conf

        fine_prob = self._pool_bt_map(
            inner_gate[:, 0:1] if inner_gate is not None and inner_gate.shape[1] > 0 else None,
            B,
            T_target,
            H_target,
            W_target,
            reducer="mean",
        )
        if fine_prob is not None:
            score = score + weights["fine_motion"] * fine_prob

        skip = self._pool_bt_map(
            skip_prob, B, T_target, H_target, W_target, reducer="mean"
        )
        if skip is not None:
            score = score - weights["skip"] * skip

        score = score.flatten(1)
        score_min = score.amin(dim=1, keepdim=True)
        score_max = score.amax(dim=1, keepdim=True)
        return (score - score_min) / (score_max - score_min + 1e-6)

    def _current_weights(self, device, dtype) -> dict[str, Tensor]:
        if self.raw_weights is None:
            return {
                "motion": torch.tensor(self.weights.motion, device=device, dtype=dtype),
                "tool": torch.tensor(self.weights.tool, device=device, dtype=dtype),
                "gate_confidence": torch.tensor(self.weights.gate_confidence, device=device, dtype=dtype),
                "fine_motion": torch.tensor(self.weights.fine_motion, device=device, dtype=dtype),
                "skip": torch.tensor(self.weights.skip, device=device, dtype=dtype),
            }
        values = F.softplus(self.raw_weights).to(device=device, dtype=dtype)
        return {
            "motion": values[0],
            "tool": values[1],
            "gate_confidence": values[2],
            "fine_motion": values[3],
            "skip": values[4],
        }

    @staticmethod
    def _pool_bt_map(
        value: Optional[Tensor],
        batch_size: int,
        target_t: int,
        target_h: int,
        target_w: int,
        reducer: str,
    ) -> Optional[Tensor]:
        if value is None:
            return None
        if value.ndim != 4:
            return None
        bt, channels, height, width = value.shape
        if bt % batch_size != 0:
            return None
        frames = bt // batch_size
        value = value.float().reshape(batch_size, frames, channels, height, width)
        if reducer == "max":
            value = value.max(dim=2).values
        else:
            value = value.mean(dim=2)
        value = value[:, None]
        value = F.adaptive_avg_pool3d(value, (target_t, target_h, target_w)).squeeze(1)
        return value

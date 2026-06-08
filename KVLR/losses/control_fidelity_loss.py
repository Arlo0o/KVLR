from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _get(output: dict, key: str) -> Tensor | None:
    if not isinstance(output, dict):
        return None
    inter = output.get("intermediates", {})
    return inter.get(key)


def _as_pred(output):
    if isinstance(output, dict):
        if "pred" in output:
            return output["pred"]
        if "sample" in output:
            return output["sample"]
    return output


def control_fidelity_loss(student_output: dict, teacher_output: dict) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Proxy action-fidelity loss for few-step student training.
    It keeps the student control pathway close to the teacher control pathway.
    """
    pred = _as_pred(student_output)
    zero = pred.new_tensor(0.0)
    losses = {}

    for key in ("action_features", "control_features", "motion_energy", "tool_mask"):
        student_value = _get(student_output, key)
        teacher_value = _get(teacher_output, key)
        if student_value is None or teacher_value is None:
            continue
        if student_value.shape != teacher_value.shape:
            continue
        losses[f"ctrl_{key}"] = F.mse_loss(student_value.float(), teacher_value.detach().float())

    if not losses:
        return zero, {}
    total = torch.stack([v for v in losses.values()]).sum()
    return total, losses

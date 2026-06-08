from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def _as_pred(output) -> Tensor:
    if isinstance(output, dict):
        if "pred" in output:
            return output["pred"]
        if "sample" in output:
            return output["sample"]
    return output


def prediction_distill_loss(student_output, teacher_output, reduction: str = "mean") -> Tensor:
    """MSE distillation on velocity/noise/flow predictions."""
    student_pred = _as_pred(student_output)
    teacher_pred = _as_pred(teacher_output).detach()
    return F.mse_loss(student_pred.float(), teacher_pred.float(), reduction=reduction)

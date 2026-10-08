from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _get_nested(output: dict, *keys):
    cur = output
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _as_pred(output):
    if isinstance(output, dict):
        if "pred" in output:
            return output["pred"]
        if "sample" in output:
            return output["sample"]
    return output


def _kl_probs(student: Tensor | None, teacher: Tensor | None) -> Tensor | None:
    if student is None or teacher is None:
        return None
    student = student.float()
    teacher = teacher.detach().float()
    if student.shape != teacher.shape:
        return None
    student_log = (student.clamp_min(1e-8) / student.clamp_min(1e-8).sum(dim=1, keepdim=True)).log()
    teacher_prob = teacher.clamp_min(1e-8)
    teacher_prob = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(student_log, teacher_prob, reduction="batchmean")


def route_distill_loss(student_output: dict, teacher_output: dict) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Distill outer modality routing, inner motion-scale routing, and skip behavior.
    Missing keys are skipped so the loss can be used during incremental adoption.
    """
    pred = _as_pred(student_output)
    zero = pred.new_tensor(0.0)

    losses = {}
    outer = _kl_probs(
        _get_nested(student_output, "intermediates", "outer_gate"),
        _get_nested(teacher_output, "intermediates", "outer_gate"),
    )
    if outer is not None:
        losses["route_outer"] = outer

    inner = _kl_probs(
        _get_nested(student_output, "intermediates", "inner_gate"),
        _get_nested(teacher_output, "intermediates", "inner_gate"),
    )
    if inner is not None:
        losses["route_inner"] = inner

    s_skip = _get_nested(student_output, "intermediates", "skip_prob")
    t_skip = _get_nested(teacher_output, "intermediates", "skip_prob")
    if s_skip is not None and t_skip is not None and s_skip.shape == t_skip.shape:
        losses["route_skip"] = F.binary_cross_entropy(
            s_skip.float().clamp(1e-6, 1.0 - 1e-6),
            t_skip.detach().float().clamp(0.0, 1.0),
        )

    if not losses:
        return zero, {}
    total = torch.stack([v for v in losses.values()]).sum()
    return total, losses

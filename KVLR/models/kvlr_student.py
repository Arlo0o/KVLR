from __future__ import annotations

import torch

from KVLR.models.mmdit.model import Flux
from KVLR.registry import MODELS


@MODELS.register_module("kvlr_teacher")
def KVLRTeacher(**kwargs):
    """
    Registry alias for the unchanged full KVLR teacher. The alias is useful in
    distillation configs because it makes the teacher/student roles explicit
    without changing the underlying architecture.
    """
    return Flux(**kwargs)


@MODELS.register_module("kvlr_student")
def KVLRStudent(
    adaptive_exec: dict | None = None,
    num_distill_steps: int = 4,
    torch_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):
    """
    Few-step KVLR student. Architecturally it is intentionally close to the
    teacher; the difference is its adaptive execution policy and few-step
    sampling/training config.
    """
    adaptive_exec = dict(adaptive_exec or {})
    adaptive_exec.setdefault("enabled", True)
    model = Flux(adaptive_exec=adaptive_exec, torch_dtype=torch_dtype, **kwargs)
    model.num_distill_steps = int(num_distill_steps)
    return model

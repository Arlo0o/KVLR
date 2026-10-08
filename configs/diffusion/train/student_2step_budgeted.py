_base_ = ["student_4step_budgeted.py"]

model = dict(
    num_distill_steps=2,
)

distillation = dict(
    num_steps=2,
)

sampling_option = dict(
    num_steps=2,
    method="distill",
)

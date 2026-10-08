_base_ = ["768px_action_student_4step.py"]

# Stage-3/budgeted student inference: same 4-step I2V sampling interface, but
# uses the learnable criticality/budgeted adaptive-execution setting.
adaptive_exec = dict(
    enabled=True,
    learnable_criticality=True,
    full_ratio=0.20,
    light_ratio=0.30,
    light_scale=0.35,
    reuse_fallback_scale=0.0,
    cache_enabled=True,
    refresh_high=1,
    refresh_mid=2,
    refresh_low=4,
    refresh_high_threshold=0.65,
    refresh_mid_threshold=0.35,
)

model = dict(
    num_distill_steps=4,
    adaptive_exec=adaptive_exec,
)

sampling_option = dict(
    resolution="768px",
    num_steps=4,
    method="i2v",
)

profile = dict(
    output_json="./outputs/efficiency_profile/student_4step_budgeted_profile.json",
)

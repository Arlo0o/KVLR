_base_ = ["stage3_action.py"]

bucket_config = {
    "_delete_": True,
    "256px": {
        17: (1.0, 1),
    },
}

adaptive_exec = dict(
    enabled=True,
    criticality_head="heuristic",
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
    type="kvlr_student",
    num_distill_steps=4,
    adaptive_exec=adaptive_exec,
)

distillation = dict(
    enabled=True,
    teacher_ckpt=None,
    num_steps=4,
    distill_target="velocity",
    return_intermediates=True,
)

loss_weights = dict(
    flow=0.0,
    distill=1.0,
    route=0.5,
    ctrl=0.5,
    budget=0.0,
    temp=0.0,
)

budget = dict(
    target_active_ratio=0.50,
    target_refresh_ratio=0.50,
)

sampling_option = dict(
    num_steps=4,
    method="distill",
)

epochs = 20
ckpt_every = 50
keep_n_latest = 100

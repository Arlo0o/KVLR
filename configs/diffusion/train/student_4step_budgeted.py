_base_ = ["student_4step.py"]

bucket_config = {
    "_delete_": True,
    "256px": {
        17: (1.0, 1),
    },
}

adaptive_exec = dict(
    enabled=True,
    learnable_criticality=True,
    full_ratio=0.20,
    light_ratio=0.30,
    light_scale=0.35,
    reuse_fallback_scale=0.0,
    cache_enabled=True,
)

model = dict(
    adaptive_exec=adaptive_exec,
)

loss_weights = dict(
    distill=1.0,
    route=0.5,
    ctrl=0.5,
    budget=0.05,
    temp=0.1,
)

budget = dict(
    target_active_ratio=0.50,
    target_refresh_ratio=0.50,
)

epochs = 50
ckpt_every = 50
keep_n_latest = 100

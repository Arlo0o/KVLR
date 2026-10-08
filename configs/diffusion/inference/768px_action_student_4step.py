_base_ = ["768px_action.py"]

# Few-step KVLR student inference. We keep method="i2v" by default so surgical
# reference-image conditioning follows the existing inference path, while
# num_steps=4 provides the few-step setting.
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

sampling_option = dict(
    resolution="768px",
    num_steps=4,
    method="i2v",
)

profile = dict(
    enabled=True,
    batch_size=1,
    num_frames=None,
    height=None,
    width=None,
    num_warmup=2,
    num_iters=5,
    txt_len=512,
    guidance_repeats=3,
    load_ckpt=True,
    output_json="./outputs/efficiency_profile/student_4step_profile.json",
)

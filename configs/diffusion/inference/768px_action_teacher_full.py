_base_ = ["768px_action.py"]

# Full original KVLR/action-MoE inference setting for efficiency comparison.
# This keeps the teacher architecture and action pathway unchanged.
model = dict(
    type="kvlr_teacher",
)

sampling_option = dict(
    resolution="768px",
    num_steps=50,
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
    output_json="./outputs/efficiency_profile/teacher_full_profile.json",
)

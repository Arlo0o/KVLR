_base_ = ["768px_action_student_4step_budgeted.py"]

# Optional stronger compression setting. We keep method="i2v" for reference-image
# conditioning and reduce the denoising depth to two steps.
model = dict(
    num_distill_steps=2,
)

sampling_option = dict(
    resolution="768px",
    num_steps=2,
    method="i2v",
)

profile = dict(
    output_json="./outputs/efficiency_profile/student_2step_budgeted_profile.json",
)

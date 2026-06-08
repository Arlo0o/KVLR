import os

_base_ = [
    "256px.py",
    "plugins/sp.py",
]

sampling_option = dict(
    resolution="768px",
)

dataset = dict(
    type="surgical_action_video_text",
    pth_search_dirs=[
        os.path.join(os.environ.get("KASA_ACTION_ROOT", "./data/kasa/annotations/label_results"), "knotting_results", "knotting_results"),
        os.path.join(os.environ.get("KASA_ACTION_ROOT", "./data/kasa/annotations/label_results"), "needleGrasping_results", "needleGrasping_results"),
        os.path.join(os.environ.get("KASA_ACTION_ROOT", "./data/kasa/annotations/label_results"), "needlePuncture_results", "needlePuncture_results"),
    ],
    focal_length=587.544,
    line_width=3,
)

use_action_condition = True
use_skeleton_maps = True
use_soft_moe_skeleton = True

action_injection_double_layers = [0, 3, 6, 9, 12, 15, 18]
action_injection_single_layers = [0, 10, 20, 30, 37]

soft_moe_num_experts = 5
soft_moe_t_emb_dim = 256
moe_config = dict(
    top_k=2,
    capacity_pred_loss_weight=0.01,
    kp_alb_loss_weight=0.01,
    src_loss_weight=0.005,
    use_capacity_predictor=True,
    num_sub_experts=3,
    sub_expert_top_k=1,
    sub_lb_loss_weight=0.005,
    capacity_schedule_stage_I_frac=0.40,
    capacity_schedule_stage_II_frac=0.75,
)

trainable_layers = "action_only"
action_dropout = 0.0
prefetch_factor = 2
num_workers = 8
num_bucket_build_workers = 16

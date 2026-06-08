_base_ = ["image.py"]

# new config
grad_ckpt_settings = (100, 100)

plugin = "hybrid"
plugin_config = dict(
    tp_size=1,
    pp_size=1,
    # NOTE: For 768px training on a single 8xA800-80GB node, using sp_size=8
    # reduces per-GPU sequence length and memory pressure (ring-attn KV/activation/comm buffers),
    # which helps avoid CUDA OOM that can surface as NCCL errors.
    sp_size=8,
    sequence_parallelism_mode="ring_attn",
    enable_sequence_parallelism=True,
    static_graph=True,
    zero_stage=2,
    reduce_bucket_size_in_m=32,  # Further reduced for stage2
    overlap_allgather=False,
)

bucket_config = {
    "_delete_": True,
    "256px": {
        # 1: (1.0, 6),   # Reduced from 130 to 6 for 80GB VRAM
        # 5: (1.0, 3),   # Reduced from 14 to 3
        # 9: (1.0, 2),   # Reduced from 14 to 2
        # 13: (1.0, 2),  # Reduced from 14 to 2
        17: (1.0, 2),  # Reduced from 14 to 2
        # 21: (1.0, 2),  # Reduced from 14 to 2
        # 25: (1.0, 1),  # Reduced from 14 to 1
        # 29: (1.0, 1),  # Reduced from 14 to 1
        33: (1.0, 1),  # Reduced from 14 to 1
        # 37: (1.0, 1),  # Reduced from 10 to 1
        # 41: (1.0, 1),  # Reduced from 10 to 1
        # 45: (1.0, 1),  # Reduced from 10 to 1
        # 49: (1.0, 1),  # Reduced from 10 to 1
        # 53: (1.0, 1),  # Reduced from 10 to 1
        # 57: (1.0, 1),  # Reduced from 10 to 1
        # 61: (1.0, 1),  # Reduced from 10 to 1
        # 65: (1.0, 1),  # Reduced from 10 to 1
        # 73: (1.0, 1),  # Reduced from 7 to 1
        # 77: (1.0, 1),  # Reduced from 7 to 1
        # 81: (1.0, 1),  # Reduced from 7 to 1
        # 85: (1.0, 1),  # Reduced from 7 to 1
        # 89: (1.0, 1),  # Reduced from 7 to 1
        # 93: (1.0, 1),  # Reduced from 7 to 1
        # 97: (1.0, 1),  # Reduced from 7 to 1
        # 101: (1.0, 1), # Reduced from 6 to 1
        # 105: (1.0, 1), # Reduced from 6 to 1
        # 109: (1.0, 1), # Reduced from 6 to 1
        # 113: (1.0, 1), # Reduced from 6 to 1
        # 117: (1.0, 1), # Reduced from 6 to 1
        # 121: (1.0, 1), # Reduced from 6 to 1
        # 125: (1.0, 1), # Reduced from 6 to 1
        # 129: (1.0, 1), # Reduced from 6 to 1
    },
    "768px": {
        # 1: (1.0, 6),   # Reduced from 130 to 6 for 80GB VRAM
        # 5: (1.0, 3),   # Reduced from 14 to 3
        # 9: (1.0, 2),   # Reduced from 14 to 2
        # 13: (1.0, 2),  # Reduced from 14 to 2
        17: (1.0, 2),  # Reduced from 14 to 2
        # 21: (1.0, 2),  # Reduced from 14 to 2
        # 25: (1.0, 1),  # Reduced from 14 to 1
        # 29: (1.0, 1),  # Reduced from 14 to 1
        33: (1.0, 1),  # Reduced from 14 to 1
        # 37: (1.0, 1),  # Reduced from 10 to 1
        # 41: (1.0, 1),  # Reduced from 10 to 1
        # 45: (1.0, 1),  # Reduced from 10 to 1
        # 49: (1.0, 1),  # Reduced from 10 to 1
        # 53: (1.0, 1),  # Reduced from 10 to 1
        # 57: (1.0, 1),  # Reduced from 10 to 1
        # 61: (1.0, 1),  # Reduced from 10 to 1
        # 65: (1.0, 1),  # Reduced from 10 to 1
        # 73: (1.0, 1),  # Reduced from 7 to 1
        # 77: (1.0, 1),  # Reduced from 7 to 1
        # 81: (1.0, 1),  # Reduced from 7 to 1
        # 85: (1.0, 1),  # Reduced from 7 to 1
        # 89: (1.0, 1),  # Reduced from 7 to 1
        # 93: (1.0, 1),  # Reduced from 7 to 1
        # 97: (1.0, 1),  # Reduced from 7 to 1
        # 101: (1.0, 1), # Reduced from 6 to 1
        # 105: (1.0, 1), # Reduced from 6 to 1
        # 109: (1.0, 1), # Reduced from 6 to 1
        # 113: (1.0, 1), # Reduced from 6 to 1
        # 117: (1.0, 1), # Reduced from 6 to 1
        # 121: (1.0, 1), # Reduced from 6 to 1
        # 125: (1.0, 1), # Reduced from 6 to 1
        # 129: (1.0, 1), # Reduced from 6 to 1
    },
}

model = dict(grad_ckpt_settings=grad_ckpt_settings)
lr = 1e-5
optim = dict(lr=lr)
ckpt_every = 800
keep_n_latest = 20

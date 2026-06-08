# Checkpoint root (override via env var if needed)
import os

KVLR_CKPT_ROOT = os.getenv(
    "KVLR_CKPT_ROOT",
    "./checkpoints/base",
)
KVLR_HF_CACHE_DIR = os.getenv("KVLR_HF_CACHE_DIR", os.path.join(KVLR_CKPT_ROOT, "cache"))
KVLR_OUTPUTS_DIR = os.getenv("KVLR_OUTPUTS_DIR", "outputs")

# Dataset settings
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    fps_max=30,  # the desired fps for training - changed from 24 to 30Hz
    vmaf=True,  # load vmaf scores into text
    validate_videos=True,
    validation_num_probe_frames=3,
)

grad_ckpt_settings = (8, 100)  # set the grad checkpoint settings
bucket_config = {
    "256px": {1: (1.0, 50)},
    "768px": {1: (0.5, 11)},
    "1024px": {1: (0.5, 7)},
}

# Define model components
model = dict(
    type="flux",
    from_pretrained=None,
    strict_load=False,
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=True,
    grad_ckpt_settings=grad_ckpt_settings,
    # model architecture
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
)
dropout_ratio = {  # probability for dropout text embedding
    "t5": 0.31622777,
    "clip": 0.31622777,
}
ae = dict(
    type="hunyuan_vae",
    from_pretrained=os.path.join(KVLR_CKPT_ROOT, "hunyuan_vae.safetensors"),
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)
is_causal_vae = True
t5 = dict(
    type="text_embedder",
    from_pretrained=os.path.join(KVLR_CKPT_ROOT, "google", "t5-v1_1-xxl"),
    # cache_dir=KVLR_HF_CACHE_DIR,
    max_length=512,
    shardformer=True,
)
clip = dict(
    type="text_embedder",
    from_pretrained=os.path.join(KVLR_CKPT_ROOT, "openai", "clip-vit-large-patch14"),
    # cache_dir=KVLR_HF_CACHE_DIR,
    max_length=77,
)

# Optimization settings
lr = 1e-5
eps = 1e-15
optim = dict(
    cls="HybridAdam",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
    adamw_mode=True,
)
warmup_steps = 0
update_warmup_steps = True

grad_clip = 1.0
accumulation_steps = 1
ema_decay = None

# Acceleration settings - optimized for 80GB VRAM
prefetch_factor = 1  # Reduced from 2 to save memory
num_workers = 4      # Reduced from 12 to save memory 
num_bucket_build_workers = 8  # Reduced from 64 to save memory
dtype = "bf16"
plugin = "zero2"
grad_checkpoint = True
plugin_config = dict(
    reduce_bucket_size_in_m=64,  # Reduced from 128 to save memory
    overlap_allgather=False,
)
# Significantly reduced memory pre-allocation for 80GB VRAM
pin_memory_cache_pre_alloc_numels = [(80 + 10) * 1024 * 1024] * 12 + [
    (20 + 5) * 1024 * 1024
] * 2
async_io = False

# Other settings
seed = 42
outputs = KVLR_OUTPUTS_DIR
epochs = 300
log_every = 10
ckpt_every = 100
keep_n_latest = 20
wandb_project = "KVLR"

save_master_weights = True
load_master_weights = True

# For debugging
# record_time = True
# record_barrier = True

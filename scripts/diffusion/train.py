import gc
import math
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import random
import subprocess
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pprint import pformat

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
gc.disable()


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from colossalai.booster import Booster
from colossalai.utils import set_seed
from peft import LoraConfig
from tqdm import tqdm

from KVLR.acceleration.checkpoint import (
    GLOBAL_ACTIVATION_MANAGER,
    set_grad_checkpoint,
)
from KVLR.acceleration.parallel_states import get_data_parallel_group
from KVLR.datasets.aspect import bucket_to_shapes
from KVLR.datasets.dataloader import prepare_dataloader
from KVLR.datasets.pin_memory_cache import PinMemoryCache
from KVLR.models.mmdit.distributed import MMDiTPolicy
from KVLR.registry import DATASETS, MODELS, build_module
from KVLR.utils.ckpt import (
    CheckpointIO,
    load_checkpoint,
    model_sharding,
    record_model_param_shape,
    rm_checkpoints,
)
from KVLR.losses import (
    budget_loss,
    control_fidelity_loss,
    prediction_distill_loss,
    route_distill_loss,
    temporal_cache_loss,
)
from KVLR.utils.config import (
    config_to_name,
    create_experiment_workspace,
    parse_configs,
)
from KVLR.utils.logger import create_logger
from KVLR.utils.misc import (
    NsysProfiler,
    Timers,
    all_reduce_mean,
    create_tensorboard_writer,
    is_log_process,
    is_pipeline_enabled,
    log_cuda_max_memory,
    log_cuda_memory,
    log_model_params,
    print_mem,
    to_torch_dtype,
)
from KVLR.utils.optimizer import create_lr_scheduler, create_optimizer
from KVLR.utils.sampling import (
    get_res_lin_function,
    pack,
    prepare,
    prepare_ids,
    time_shift,
)
from KVLR.utils.train import (
    create_colossalai_plugin,
    dropout_condition,
    get_batch_loss,
    prepare_visual_condition_causal,
    prepare_visual_condition_uncausal,
    set_eps,
    set_lr,
    setup_device,
    update_ema,
    warmup_ae,
)

torch.backends.cudnn.benchmark = False  # True leads to slow down in conv3d


if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)


# ===========================================================================
# Action conditioning helpers
# ===========================================================================

def _init_action_proj(hidden_size):
    """
    Create an action projection layer with Xavier-uniform weight and zero bias.

    Cold-start is guaranteed by zero_conv inside SkeletonEncoder (action_encoded
    = 0 at step 0), so action_proj must NOT also be zero-initialized.  If both
    layers are zero, the chain rule produces 0^T @ 0 = 0 in both directions,
    permanently blocking gradient flow from the main MSE loss into the skeleton
    encoder.  Xavier-uniform weight ensures that once zero_conv's bias receives
    a gradient and starts producing non-zero output, the signal can propagate
    back through action_proj into the full encoder.
    """
    proj = torch.nn.Linear(hidden_size, hidden_size)
    torch.nn.init.xavier_uniform_(proj.weight)
    torch.nn.init.zeros_(proj.bias)
    return proj


def _setup_action_conditioning(model, cfg, device, dtype):
    """
    Dynamically add action conditioning modules to an existing MMDiTModel.

    Called after the DiT is loaded from a pre-trained Stage-2 checkpoint so
    the action branch can be grafted on without modifying the checkpoint format.

    All action projection layers use Xavier-uniform weight with zero bias.
    Cold-start is guaranteed by zero_conv inside SkeletonEncoder (not by
    action_proj), so action_proj must NOT be zero-initialized to avoid a
    double-zero gradient deadlock that permanently blocks learning.

    Args:
        model:  The (possibly wrapped) MMDiTModel to modify in-place.
        cfg:    Config object with action conditioning settings.
        device: torch.device for new parameters.
        dtype:  torch.dtype for new parameters.
    """
    from KVLR.models.action_encoder import SkeletonEncoder, ModalityAwareSoftMoEEncoder

    hidden_size = model.hidden_size
    moe_config = cfg.get("moe_config", {}) or {}
    use_soft_moe = cfg.get("use_soft_moe_skeleton", False)

    if use_soft_moe:
        skeleton_encoder = ModalityAwareSoftMoEEncoder(
            in_channels=9,
            model_dim=hidden_size,
            num_experts=cfg.get("soft_moe_num_experts", 5),
            t_emb_dim=cfg.get("soft_moe_t_emb_dim", 256),
            top_k=moe_config.get("top_k", 2),
            capacity_pred_loss_weight=moe_config.get("capacity_pred_loss_weight", 0.01),
            kp_alb_loss_weight=moe_config.get("kp_alb_loss_weight", 0.01),
            src_loss_weight=moe_config.get("src_loss_weight", 0.005),
            use_capacity_predictor=moe_config.get("use_capacity_predictor", True),
            num_sub_experts=moe_config.get("num_sub_experts", 3),
            sub_expert_top_k=moe_config.get("sub_expert_top_k", 1),
            sub_lb_loss_weight=moe_config.get("sub_lb_loss_weight", 0.005),
        )
    else:
        skeleton_encoder = SkeletonEncoder(in_channels=9, model_dim=hidden_size)

    skeleton_encoder = skeleton_encoder.to(device=device, dtype=dtype)

    injection_double = cfg.get("action_injection_double_layers", [0, 3, 6, 9, 12, 15, 18])
    injection_single = cfg.get("action_injection_single_layers", [])

    action_proj_double = torch.nn.ModuleList([
        _init_action_proj(hidden_size)
        for _ in injection_double
    ]).to(device=device, dtype=dtype)

    action_proj_single = torch.nn.ModuleList([
        _init_action_proj(hidden_size)
        for _ in injection_single
    ]).to(device=device, dtype=dtype) if injection_single else torch.nn.ModuleList([])

    model.skeleton_encoder = skeleton_encoder
    model.action_proj_double = action_proj_double
    model.action_proj_single = action_proj_single
    model._action_injection_double_layers = injection_double
    model._action_injection_single_layers = injection_single
    model.use_action_condition = True
    model.use_soft_moe_skeleton = use_soft_moe
    model.use_skeleton_maps = True


def _resolve_model_ckpt_path(path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "running_states.json")):
        model_dir = os.path.join(path, "model")
        if os.path.isdir(model_dir):
            return model_dir
    return path


def _build_distillation_teacher(cfg, device, dtype, logger):
    distill_cfg = cfg.get("distillation", None)
    if not distill_cfg or not distill_cfg.get("enabled", False):
        return None

    teacher_cfg = cfg.get("teacher_model", None)
    if teacher_cfg is None:
        teacher_cfg = cfg.model.copy()
        teacher_cfg["type"] = "kvlr_teacher"
        teacher_cfg.pop("num_distill_steps", None)
    else:
        teacher_cfg = teacher_cfg.copy()
    teacher_cfg["adaptive_exec"] = dict(enabled=False)

    # If teacher_ckpt is a full Stage-3 action checkpoint, load it after
    # grafting the action branch so action weights are present in the module.
    teacher_ckpt = distill_cfg.get("teacher_ckpt", None)
    if teacher_ckpt is not None:
        teacher_cfg["from_pretrained"] = ""

    logger.info("Building frozen KVLR teacher for distillation...")
    teacher = build_module(teacher_cfg, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)

    if cfg.get("use_action_condition", False):
        _setup_action_conditioning(teacher, cfg, device, dtype)
        teacher.eval().requires_grad_(False)

    resolved = _resolve_model_ckpt_path(teacher_ckpt)
    if resolved is not None:
        logger.info("Loading teacher checkpoint from %s", resolved)
        load_checkpoint(teacher, resolved, device_map=device, strict=distill_cfg.get("teacher_strict_load", False))

    return teacher


def _unwrap_for_intermediates(model):
    """Best-effort unwrap for ColossalAI/DDP wrappers."""
    current = model
    seen = set()
    for _ in range(8):
        obj_id = id(current)
        if obj_id in seen:
            break
        seen.add(obj_id)

        if hasattr(current, "unwrap"):
            try:
                unwrapped = current.unwrap()
            except TypeError:
                unwrapped = None
            if unwrapped is not None and unwrapped is not current:
                current = unwrapped
                continue

        moved = False
        for attr in ("module", "_module"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                current = child
                moved = True
                break
        if not moved:
            break
    return current


def _get_last_intermediates(model):
    model = _unwrap_for_intermediates(model)
    getter = getattr(model, "get_last_intermediates", None)
    if getter is None:
        return {}
    try:
        intermediates = getter()
    except Exception:
        return {}
    return intermediates if isinstance(intermediates, dict) else {}


def _normalize_distill_output(model, output):
    """
    Return a stable {"pred": ..., "intermediates": ...} dict.

    Some distributed wrappers keep the original tensor-returning forward even
    when return_intermediates=True reaches the module.  The model still records
    its latest routing/control state internally, so recover it here.
    """
    if isinstance(output, dict):
        pred = output.get("pred", output.get("sample", None))
        if pred is None:
            return output
        normalized = dict(output)
        normalized["pred"] = pred
        if not isinstance(normalized.get("intermediates", None), dict):
            normalized["intermediates"] = _get_last_intermediates(model)
        return normalized
    return {"pred": output, "intermediates": _get_last_intermediates(model)}


def _update_moe_capacity(model, progress: float, moe_cfg: dict):
    """
    Update Soft MoE capacity schedule based on training progress.

    Stage I  (dense):     progress < stage_I_frac   — all experts contribute (warm-up)
    Stage II (annealing): stage_I_frac ≤ progress < stage_II_frac — linear blend
    Stage III (target):   progress ≥ stage_II_frac  — fully sparse top-k routing

    Args:
        model:    The MMDiTModel (or wrapper).
        progress: Float in [0, 1] = current_step / total_steps.
        moe_cfg:  dict with capacity schedule params.
    """
    unwrapped = model.unwrap() if hasattr(model, 'unwrap') else model
    enc = getattr(unwrapped, 'skeleton_encoder', None)
    if enc is None or not hasattr(enc, '_capacity_stage'):
        return

    stage_I_frac  = moe_cfg.get("capacity_schedule_stage_I_frac",  0.40)
    stage_II_frac = moe_cfg.get("capacity_schedule_stage_II_frac", 0.75)

    if progress < stage_I_frac:
        enc._capacity_stage = 'dense'
        enc._annealing_alpha = 0.0
    elif progress < stage_II_frac:
        enc._capacity_stage = 'annealing'
        # Linear ramp 0→1 over Stage II
        enc._annealing_alpha = (progress - stage_I_frac) / max(stage_II_frac - stage_I_frac, 1e-8)
    else:
        enc._capacity_stage = 'target'
        enc._annealing_alpha = 1.0


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs()

    # == get dtype & device ==
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    device, coordinator = setup_device()
    if cfg.get("adaptive_exec", None) is not None:
        cfg.model["adaptive_exec"] = cfg.adaptive_exec
    grad_ckpt_buffer_size = cfg.get("grad_ckpt_buffer_size", 0)
    if grad_ckpt_buffer_size > 0:
        GLOBAL_ACTIVATION_MANAGER.setup_buffer(grad_ckpt_buffer_size, dtype)
    checkpoint_io = CheckpointIO()
    set_seed(cfg.get("seed", 1024))
    PinMemoryCache.force_dtype = dtype
    pin_memory_cache_pre_alloc_numels = cfg.get("pin_memory_cache_pre_alloc_numels", None)
    PinMemoryCache.pre_alloc_numels = pin_memory_cache_pre_alloc_numels

    # == init ColossalAI booster ==
    plugin_type = cfg.get("plugin", "zero2")
    plugin_config = cfg.get("plugin_config", {})
    plugin_kwargs = {}
    if plugin_type == "hybrid":
        plugin_kwargs["custom_policy"] = MMDiTPolicy
    plugin = create_colossalai_plugin(
        plugin=plugin_type,
        dtype=cfg.get("dtype", "bf16"),
        grad_clip=cfg.get("grad_clip", 0),
        **plugin_config,
        **plugin_kwargs,
    )
    booster = Booster(plugin=plugin)

    seq_align = plugin_config.get("sp_size", 1)

    # == init exp_dir ==
    exp_name, exp_dir = create_experiment_workspace(
        cfg.get("outputs", "./outputs"),
        model_name=config_to_name(cfg),
        config=cfg.to_dict(),
        exp_name=cfg.get("exp_name", None),  # useful for automatic restart to specify the exp_name
    )

    if is_log_process(plugin_type, plugin_config):
        print(f"changing {exp_dir} to share")
        os.system(f"chgrp -R share {exp_dir}")

    # == init logger, tensorboard & wandb ==
    logger = create_logger(exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    tb_writer = None
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(
                project=cfg.get("wandb_project", "KVLR"),
                name=exp_name,
                config=cfg.to_dict(),
                dir=exp_dir,
            )
    num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    tp_size = cfg["plugin_config"].get("tp_size", 1)
    sp_size = cfg["plugin_config"].get("sp_size", 1)
    pp_size = cfg["plugin_config"].get("pp_size", 1)
    num_groups = num_gpus // (tp_size * sp_size * pp_size)
    logger.info("Number of GPUs: %s", num_gpus)
    logger.info("Number of groups: %s", num_groups)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    cache_pin_memory = pin_memory_cache_pre_alloc_numels is not None
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
        cache_pin_memory=cache_pin_memory,
        num_groups=num_groups,
    )
    print_mem("before prepare_dataloader")
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    print_mem("after prepare_dataloader")
    num_steps_per_epoch = len(dataloader)
    dataset.to_efficient()

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")

    # == build model model ==
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).train()
    if cfg.get("grad_checkpoint", True):
        set_grad_checkpoint(model)
    log_cuda_memory("diffusion")
    log_model_params(model)

    # == action conditioning setup (Stage 3) ==
    if cfg.get("use_action_condition", False):
        logger.info("Setting up action conditioning modules...")
        _setup_action_conditioning(model, cfg, device, dtype)
        logger.info(
            "Action conditioning added: skeleton_encoder + %d double + %d single projections.",
            len(model._action_injection_double_layers),
            len(model._action_injection_single_layers),
        )
        log_model_params(model)

    teacher_model = _build_distillation_teacher(cfg, device, dtype, logger)

    # == freeze DiT for action-only training ==
    trainable_layers = cfg.get("trainable_layers", "all")
    if trainable_layers == "action_only":
        model.requires_grad_(False)
        for module_name in [
            "skeleton_encoder",
            "action_proj_double",
            "action_proj_single",
            "criticality_head",
        ]:
            m = getattr(model, module_name, None)
            if m is not None:
                m.requires_grad_(True)
        # use_reentrant=True gradient checkpointing requires at least one input
        # with requires_grad=True to preserve the grad_fn chain through each block.
        # When DiT is frozen, img/txt have requires_grad=False, so checkpoint
        # silently drops grad_fn after every block, breaking action injection gradients.
        # enable_input_require_grads() sets requires_grad=True on img and txt inputs
        # so the grad chain survives through all checkpointed blocks.
        model.enable_input_require_grads()
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "trainable_layers=action_only: DiT frozen, %d action parameters trainable.",
            n_trainable,
        )

    # == build EMA model ==
    use_lora = cfg.get("lora_config", None) is not None
    if cfg.get("ema_decay", None) is not None and not use_lora:
        ema = deepcopy(model).cpu().eval().requires_grad_(False)
        ema_shape_dict = record_model_param_shape(ema)
        logger.info("EMA model created.")
    else:
        ema = ema_shape_dict = None
        logger.info("No EMA model created.")
    log_cuda_memory("EMA")

    # == enable LoRA ==
    if use_lora:
        lora_config = LoraConfig(**cfg.get("lora_config", None))
        model = booster.enable_lora(
            model=model,
            lora_config=lora_config,
            pretrained_dir=cfg.get("lora_checkpoint", None),
        )
        log_cuda_memory("lora")
        log_model_params(model)

    if not cfg.get("cached_video", False):
        # == buildn autoencoder ==
        model_ae = build_module(cfg.ae, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
        del model_ae.decoder
        log_cuda_memory("autoencoder")
        log_model_params(model_ae)
        model_ae.encode = torch.compile(model_ae.encoder, dynamic=True)

    if not cfg.get("cached_text", False):
        # == build text encoder (t5) ==
        model_t5 = build_module(cfg.t5, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
        log_cuda_memory("t5")
        log_model_params(model_t5)

        # == build text encoder (clip) ==
        model_clip = build_module(cfg.clip, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
        log_cuda_memory("clip")
        log_model_params(model_clip)

    # == setup optimizer ==
    optimizer = create_optimizer(model, cfg.optim)

    # == setup lr scheduler ==
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer,
        num_steps_per_epoch=num_steps_per_epoch,
        epochs=cfg.get("epochs", 1000),
        warmup_steps=cfg.get("warmup_steps", None),
        use_cosine_scheduler=cfg.get("use_cosine_scheduler", False),
    )
    log_cuda_memory("optimizer")

    # == prepare null vectors for dropout ==
    if cfg.get("cached_text", False):
        null_t5_path = cfg.get("null_t5_path", os.environ.get("NULL_T5_PATH", "./checkpoints/cache/null_t5.pt"))
        null_clip_path = cfg.get("null_clip_path", os.environ.get("NULL_CLIP_PATH", "./checkpoints/cache/null_clip.pt"))
        null_txt = torch.load(null_t5_path, map_location=device)
        null_vec = torch.load(null_clip_path, map_location=device)
    else:
        null_txt = model_t5("")
        null_vec = model_clip("")

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    # == boosting ==
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosted model for distributed training")
    log_cuda_memory("boost")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    log_step = acc_step = 0
    running_loss = 0.0
    timers = Timers(record_time=cfg.get("record_time", False), record_barrier=cfg.get("record_barrier", False))
    nsys = NsysProfiler(
        warmup_steps=cfg.get("nsys_warmup_steps", 2),
        num_steps=cfg.get("nsys_num_steps", 2),
        enabled=cfg.get("nsys", False),
    )
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    load_master_weights = cfg.get("load_master_weights", False)
    save_master_weights = cfg.get("save_master_weights", False)
    start_epoch = cfg.get("start_epoch", None)
    start_step = cfg.get("start_step", None)
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint from %s", cfg.load)

        lr_scheduler_to_load = lr_scheduler
        if cfg.get("update_warmup_steps", False):
            lr_scheduler_to_load = None
        ret = checkpoint_io.load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler_to_load,
            sampler=(
                None if start_step is not None else sampler
            ),  # if specify start step, set last_micro_batch_access_index of a new sampler instead
            include_master_weights=load_master_weights,
        )
        start_epoch = start_epoch if start_epoch is not None else ret[0]
        start_step = start_step if start_step is not None else ret[1]
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, ret[0], ret[1])

        # load optimizer and scheduler will overwrite some of the hyperparameters, so we need to reset them
        set_lr(optimizer, lr_scheduler, cfg.optim.lr, cfg.get("initial_lr", None))
        set_eps(optimizer, cfg.optim.eps)

        if cfg.get("update_warmup_steps", False):
            assert (
                cfg.get("warmup_steps", None) is not None
            ), "you need to set warmup_steps in order to pass --update-warmup-steps True"
            # set_warmup_steps(lr_scheduler, cfg.warmup_steps)
            lr_scheduler.step(start_epoch * num_steps_per_epoch + start_step)
            logger.info("The learning rate starts from %s", optimizer.param_groups[0]["lr"])
    if start_step is not None:
        # if start step exceeds data length, go to next epoch
        if start_step > num_steps_per_epoch:
            start_epoch = (
                start_epoch + start_step // num_steps_per_epoch
                if start_epoch is not None
                else start_step // num_steps_per_epoch
            )
            start_step = start_step % num_steps_per_epoch
    else:
        start_step = 0
    sampler.set_step(start_step)
    start_epoch = start_epoch if start_epoch is not None else 0
    logger.info("Starting from epoch %s step %s", start_epoch, start_step)

    # == sharding EMA model ==
    if ema is not None:
        model_sharding(ema)
        ema = ema.to(device)
        log_cuda_memory("sharding EMA")

    # == warmup autoencoder ==
    if cfg.get("warmup_ae", False):
        shapes = bucket_to_shapes(cfg.get("bucket_config", None), batch_size=cfg.ae.batch_size)
        warmup_ae(model_ae, shapes, device, dtype)

    # =======================================================
    # 5. training iter
    # =======================================================
    sigma_min = cfg.get("sigma_min", 1e-5)
    accumulation_steps = cfg.get("accumulation_steps", 1)
    ckpt_every = cfg.get("ckpt_every", 0)

    if cfg.get("is_causal_vae", False):
        prepare_visual_condition = prepare_visual_condition_causal
    else:
        prepare_visual_condition = prepare_visual_condition_uncausal

    @torch.no_grad()
    def prepare_inputs(batch):
        inp = dict()
        x = batch.pop("video")
        y = batch.pop("text")
        skeleton_maps_raw = batch.pop("skeleton_maps", None)  # action conditioning
        bs = x.shape[0]

        # == encode video ==
        with nsys.range("encode_video"), timers["encode_video"]:
            # == prepare condition ==
            if cfg.get("condition_config", None) is not None:
                # condition for i2v & v2v
                x_0, cond = prepare_visual_condition(x, cfg.condition_config, model_ae)
                cond = pack(cond, patch_size=cfg.get("patch_size", 2))
                inp["cond"] = cond
            else:
                if cfg.get("cached_video", False):
                    x_0 = batch.pop("video_latents").to(device=device, dtype=dtype)
                else:
                    x_0 = model_ae.encode(x)

        # == prepare timestep ==
        # follow SD3 time shift, shift_alpha = 1 for 256px and shift_alpha = 3 for 1024px
        shift_alpha = get_res_lin_function()((x_0.shape[-1] * x_0.shape[-2]) // 4)
        # add temporal influence
        shift_alpha *= math.sqrt(x_0.shape[-3])  # for image, T=1 so no effect
        t = torch.sigmoid(torch.randn((bs), device=device))
        t = time_shift(shift_alpha, t).to(dtype)

        # Capture latent shape (5D: [B, C, T, H, W]) before pack() converts to 3D
        _latent_thw = (x_0.shape[2], x_0.shape[3], x_0.shape[4]) if x_0.ndim == 5 else None

        if cfg.get("cached_text", False):
            # == encode text ==
            t5_embedding = batch.pop("text_t5").to(device=device, dtype=dtype)
            clip_embedding = batch.pop("text_clip").to(device=device, dtype=dtype)
            with nsys.range("encode_text"), timers["encode_text"]:
                inp_ = prepare_ids(x_0, t5_embedding, clip_embedding)
                inp.update(inp_)
                x_0 = pack(x_0, patch_size=cfg.get("patch_size", 2))
        else:
            # == encode text ==
            with nsys.range("encode_text"), timers["encode_text"]:
                inp_ = prepare(
                    model_t5,
                    model_clip,
                    x_0,
                    prompt=y,
                    seq_align=seq_align,
                    patch_size=cfg.get("patch_size", 2),
                )
                inp.update(inp_)
                x_0 = pack(x_0, patch_size=cfg.get("patch_size", 2))

        # == dropout ==
        if cfg.get("dropout_ratio", None) is not None:
            cur_null_txt = null_txt
            num_pad_null_txt = inp["txt"].shape[1] - cur_null_txt.shape[1]
            if num_pad_null_txt > 0:
                cur_null_txt = torch.cat([cur_null_txt] + [cur_null_txt[:, -1:]] * num_pad_null_txt, dim=1)
            inp["txt"] = dropout_condition(
                cfg.dropout_ratio.get("t5", 0.0),
                inp["txt"],
                cur_null_txt,
            )
            inp["y_vec"] = dropout_condition(
                cfg.dropout_ratio.get("clip", 0.0),
                inp["y_vec"],
                null_vec,
            )

        # == prepare noise vector ==
        x_1 = torch.randn_like(x_0, dtype=torch.float32).to(device, dtype)
        t_rev = 1 - t
        x_t = t_rev[:, None, None] * x_0 + (1 - (1 - sigma_min) * t_rev[:, None, None]) * x_1
        inp["img"] = x_t
        inp["timesteps"] = t.to(dtype)
        inp["guidance"] = torch.full((x_t.shape[0],), cfg.get("guidance", 4), device=x_t.device, dtype=x_t.dtype)

        # == action conditioning (skeleton maps) ==
        # CRITICAL: All collective ops (broadcast, all_reduce) MUST be called by ALL
        # ranks regardless of local data.  The old code placed dist.broadcast() inside
        # `if skeleton_maps_raw is not None` — a DATA-DEPENDENT condition that differs
        # across DP groups when some batches lack PTH files.  This caused an NCCL
        # deadlock: ranks WITH skeleton_maps called broadcast while ranks WITHOUT it
        # skipped the call, so both sides blocked forever.
        #
        # Fix: (1) all_reduce(MIN) to sync skeleton_maps availability across ALL ranks.
        #      If ANY rank lacks skeleton_maps, ALL ranks discard them so the computation
        #      graph (and thus gradient all_reduce) stays consistent.
        #      (2) action_dropout broadcast is now guaranteed to be reached by all ranks
        #      (only called when all ranks have skeleton_maps after the sync).
        
        if cfg.get("use_action_condition", False):
            # print(f"========== DEBUG Rank {dist.get_rank() if dist.is_initialized() else 0}:   batch  : skeleton_maps_raw {'OK' if skeleton_maps_raw is not None else 'NO (None)'} ==========")
            # if dist.is_initialized() and dist.get_world_size() > 1:
            #     _has_skmap = torch.tensor(
            #         [1.0 if skeleton_maps_raw is not None else 0.0], device=device
            #     )
            #     dist.all_reduce(_has_skmap, op=dist.ReduceOp.MIN)
            #     print(f"========== DEBUG Rank {dist.get_rank()}: multi-card all_reduce   {_has_skmap.item()} ==========")
            #     if _has_skmap.item() < 0.5:
            #         skeleton_maps_raw = None
       
            # action_dropout: batch-level CFG dropout — all ranks participate.
            # Now safe because the all_reduce above guarantees all ranks agree on
            # whether skeleton_maps_raw is None.
            if skeleton_maps_raw is not None:
                action_dropout = cfg.get("action_dropout", 0.0)
                if action_dropout > 0.01:
                    _drop_flag = torch.zeros(1, device=device)
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        if dist.get_rank() == 0:
                            _drop_flag.fill_(1.0 if random.random() < action_dropout else 0.0)
                        dist.broadcast(_drop_flag, src=0)
                    else:
                        _drop_flag.fill_(1.0 if random.random() < action_dropout else 0.0)
                    if _drop_flag.item() > 0.5:
                        skeleton_maps_raw = None
        # print(f"==========22============  :   batch  : skeleton_maps_raw {'OK' if skeleton_maps_raw is not None else 'NO (None)'} ==========")
        if skeleton_maps_raw is not None and _latent_thw is not None:
            # img_shape: (T_lat, H_lat // patch_size, W_lat // patch_size)
            # SkeletonEncoder will resample skeleton maps to this token-grid resolution.
            patch_size = cfg.get("patch_size", 2)
            T_lat, H_lat, W_lat = _latent_thw
            img_shape = (T_lat, H_lat // patch_size, W_lat // patch_size)
            for k, v in skeleton_maps_raw.items():
                if isinstance(v, torch.Tensor):
                    skeleton_maps_raw[k] = v.to(device=device, dtype=dtype)
            inp["skeleton_maps"] = skeleton_maps_raw
            inp["img_shape"] = img_shape

        return inp, x_0, x_1

    def run_iter(inp, x_0, x_1):
        if is_pipeline_enabled(plugin_type, plugin_config):
            if teacher_model is not None:
                raise NotImplementedError("Distillation training currently supports non-pipeline execution only.")
            inp["target"] = (1 - sigma_min) * x_1 - x_0  # follow MovieGen, modify V_t accordingly
            with nsys.range("forward-backward"), timers["forward-backward"]:
                data_iter = iter([inp])
                if cfg.get("no_i2v_ref_loss", False):
                    loss_fn = (
                        lambda out, input_: get_batch_loss(out, input_["target"], input_.pop("masks", None))
                        / accumulation_steps
                    )
                else:
                    loss_fn = (
                        lambda out, input_: F.mse_loss(out.float(), input_["target"].float(), reduction="mean")
                        / accumulation_steps
                    )
                loss = booster.execute_pipeline(data_iter, model, loss_fn, optimizer)["loss"]
                loss = loss * accumulation_steps if loss is not None else loss
                loss_item = all_reduce_mean(loss.data.clone().detach())
        else:
            with nsys.range("forward"), timers["forward"]:
                v_t = (1 - sigma_min) * x_1 - x_0
                if teacher_model is not None:
                    raw_model_out = model(**inp, return_intermediates=True)
                    model_out = _normalize_distill_output(model, raw_model_out)
                    model_pred = model_out["pred"]
                    with torch.no_grad():
                        raw_teacher_out = teacher_model(**inp, return_intermediates=True)
                        teacher_out = _normalize_distill_output(teacher_model, raw_teacher_out)
                else:
                    model_out = None
                    teacher_out = None
                    model_pred = model(**inp)  # B, T, L

                if cfg.get("no_i2v_ref_loss", False):
                    flow_loss = get_batch_loss(model_pred, v_t, inp.pop("masks", None))
                else:
                    flow_loss = F.mse_loss(model_pred.float(), v_t.float(), reduction="mean")

                if teacher_model is not None:
                    loss_weights = cfg.get("loss_weights", {}) or {}
                    default_flow_weight = 0.0 if cfg.get("distillation", {}).get("replace_flow_loss", True) else 1.0
                    loss = loss_weights.get("flow", default_flow_weight) * flow_loss

                    distill = prediction_distill_loss(model_out, teacher_out)
                    loss = loss + loss_weights.get("distill", 1.0) * distill

                    route, _route_parts = route_distill_loss(model_out, teacher_out)
                    loss = loss + loss_weights.get("route", 0.0) * route

                    ctrl, _ctrl_parts = control_fidelity_loss(model_out, teacher_out)
                    loss = loss + loss_weights.get("ctrl", 0.0) * ctrl

                    adaptive_stats = model_out.get("intermediates", {}).get("adaptive_stats", {})
                    budget, _budget_parts = budget_loss(
                        adaptive_stats,
                        target_active_ratio=cfg.get("budget", {}).get("target_active_ratio", 0.5),
                        target_refresh_ratio=cfg.get("budget", {}).get("target_refresh_ratio", 0.5),
                    )
                    if budget is not None:
                        loss = loss + loss_weights.get("budget", 0.0) * budget

                    temp = temporal_cache_loss(
                        model_out.get("intermediates", {}).get("action_features"),
                        target_shape=model_out.get("intermediates", {}).get("target_shape"),
                        token_policy=model_out.get("intermediates", {}).get("token_policy"),
                    )
                    if temp is not None:
                        loss = loss + loss_weights.get("temp", 0.0) * temp
                else:
                    loss = flow_loss

                # == MoE auxiliary losses (action conditioning) ==
                # _last_aux_loss is a SCALAR tensor (weighted sum of: KP-ALB + cap_pred BCE + SRC + sub_lb).
                # It must NOT be treated as a dict — the old .values() pattern was wrong and would crash.
                if cfg.get("use_soft_moe_skeleton", False) and inp.get("skeleton_maps") is not None:
                    _unwrapped = model.unwrap() if hasattr(model, "unwrap") else model
                    _enc = getattr(_unwrapped, "skeleton_encoder", None)
                    if _enc is not None:
                        _aux = getattr(_enc, "_last_aux_loss", None)
                        if _aux is not None and isinstance(_aux, torch.Tensor) and _aux.requires_grad:
                            # loss = loss + _aux
                            loss = loss + _aux.to(loss.dtype)

            loss_item = all_reduce_mean(loss.data.clone().detach()).item()

            # == backward & update ==
            # NOTE: dist.barrier() was removed here.  It is unnecessary (booster.backward
            # already synchronises gradients) and DANGEROUS: if any rank diverges (e.g.
            # exception in forward, different skeleton_maps availability), ranks that
            # reach the barrier block forever waiting for those that don't.
            with nsys.range("backward"), timers["backward"]:
                ctx = (
                    booster.no_sync(model, optimizer)
                    if cfg.get("plugin", "zero2") in ("zero1", "zero1-seq") and (step + 1) % accumulation_steps != 0
                    else nullcontext()
                )
                with ctx:
                    booster.backward(loss=(loss / accumulation_steps), optimizer=optimizer)

        # == sync CapacityPredictor EMA thresholds across ranks ==
        # action_encoder.py's update_threshold() intentionally skips dist.all_reduce
        # because action_dropout can cause some ranks to skip the encoder forward entirely,
        # making it unsafe to call all_reduce inside the encoder. Instead, we sync here
        # after backward, where all ranks are guaranteed to participate.
        if cfg.get("use_soft_moe_skeleton", False) and dist.is_initialized() and dist.get_world_size() > 1:
            _uw_cp = model.unwrap() if hasattr(model, "unwrap") else model
            _enc_cp = getattr(_uw_cp, "skeleton_encoder", None)
            _cap_pred = getattr(_enc_cp, "capacity_predictor", None) if _enc_cp is not None else None
            if _cap_pred is not None:
                dist.all_reduce(_cap_pred.expert_threshold, op=dist.ReduceOp.AVG)

        with nsys.range("optim"), timers["optim"]:
            if (step + 1) % accumulation_steps == 0:
                booster.checkpoint_io.synchronize()
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        # == update EMA ==
        if ema is not None:
            with nsys.range("update_ema"), timers["update_ema"]:
                update_ema(
                    ema,
                    model.unwrap(),
                    optimizer=optimizer,
                    decay=cfg.get("ema_decay", 0.9999),
                )

        return loss_item

    # =======================================================
    # 6. training loop
    # =======================================================
    dist.barrier()
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not is_log_process(plugin_type, plugin_config),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            pbar_iter = iter(pbar)

            # prefetch one for non-blocking data loading
            def fetch_data():
                step, batch = next(pbar_iter)
                # print(f"==debug== rank{dist.get_rank()} {dataloader_iter.get_cache_info()}")
                pinned_video = batch["video"]
                batch["video"] = pinned_video.to(device, dtype, non_blocking=True)
                return batch, step, pinned_video

            batch_, step_, pinned_video_ = fetch_data()

            for _ in range(start_step, num_steps_per_epoch):
                nsys.step()
                # == load data ===
                with nsys.range("load_data"), timers["load_data"]:
                    batch, step, pinned_video = batch_, step_, pinned_video_

                    if step + 1 < num_steps_per_epoch:
                        # only fetch new data if not last step
                        batch_, step_, pinned_video_ = fetch_data()

                # == run iter ==
                with nsys.range("iter"), timers["iter"]:
                    # Update MoE capacity schedule before forward pass
                    if cfg.get("use_soft_moe_skeleton", False):
                        _global_step = epoch * num_steps_per_epoch + step
                        _total_steps = cfg_epochs * num_steps_per_epoch
                        _progress = _global_step / max(_total_steps, 1)
                        _update_moe_capacity(model, _progress, cfg.get("moe_config", {}) or {})

                    inp, x_0, x_1 = prepare_inputs(batch)
                    if cache_pin_memory:
                        dataloader_iter.remove_cache(pinned_video)
                    loss = run_iter(inp, x_0, x_1)

                # == update log info ==
                if loss is not None:
                    running_loss += loss

                # == log config ==
                global_step = epoch * num_steps_per_epoch + step
                actual_update_step = (global_step + 1) // accumulation_steps
                log_step += 1
                acc_step += 1

                # == logging ==
                if (global_step + 1) % accumulation_steps == 0:
                    if actual_update_step % cfg.get("log_every", 1) == 0:
                        if is_log_process(plugin_type, plugin_config):
                            avg_loss = running_loss / log_step
                            # progress bar
                            pbar.set_postfix(
                                {
                                    "loss": avg_loss,
                                    "global_grad_norm": optimizer.get_grad_norm(),
                                    "step": step,
                                    "global_step": global_step,
                                    # "actual_update_step": actual_update_step,
                                    "lr": optimizer.param_groups[0]["lr"],
                                }
                            )
                            # tensorboard
                            if tb_writer is not None:
                                tb_writer.add_scalar("loss", loss, actual_update_step)
                                # == MoE aux loss breakdown (diagnostic, detached) ==
                                if cfg.get("use_soft_moe_skeleton", False):
                                    _uw = model.unwrap() if hasattr(model, "unwrap") else model
                                    _enc2 = getattr(_uw, "skeleton_encoder", None)
                                    if _enc2 is not None:
                                        _kp = getattr(_enc2, "_last_kp_alb_loss", None)
                                        _src = getattr(_enc2, "_last_src_loss", None)
                                        _aux2 = getattr(_enc2, "_last_aux_loss", None)
                                        if _kp is not None:
                                            tb_writer.add_scalar("moe/kp_alb_loss", float(_kp), actual_update_step)
                                        if _src is not None:
                                            tb_writer.add_scalar("moe/src_loss", float(_src), actual_update_step)
                                        if _aux2 is not None:
                                            tb_writer.add_scalar("moe/total_aux_loss", float(_aux2.detach() if hasattr(_aux2, "detach") else _aux2), actual_update_step)
                            # wandb
                            if cfg.get("wandb", False):
                                wandb_dict = {
                                    "iter": global_step,
                                    "acc_step": acc_step,
                                    "epoch": epoch,
                                    "loss": loss,
                                    "avg_loss": avg_loss,
                                    "lr": optimizer.param_groups[0]["lr"],
                                    "eps": optimizer.param_groups[0]["eps"],
                                    "global_grad_norm": optimizer.get_grad_norm(),  # test grad norm
                                }
                                if cfg.get("record_time", False):
                                    wandb_dict.update(timers.to_dict())
                                wandb.log(wandb_dict, step=actual_update_step)

                        running_loss = 0.0
                        log_step = 0

                # == checkpoint saving ==
                # uncomment below 3 lines to forcely clean cache
                with nsys.range("clean_cache"), timers["clean_cache"]:
                    if ckpt_every > 0 and actual_update_step > 0 and actual_update_step % ckpt_every == 0 and coordinator.is_master():
                        subprocess.run("sudo drop_cache", shell=True)

                with nsys.range("checkpoint"), timers["checkpoint"]:
                    if ckpt_every > 0 and actual_update_step > 0 and actual_update_step % ckpt_every == 0:
                        # mannual garbage collection
                        gc.collect()

                        save_dir = checkpoint_io.save(
                            booster,
                            exp_dir,
                            model=model,
                            ema=ema,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            sampler=sampler,
                            epoch=epoch,
                            step=step + 1,
                            global_step=global_step + 1,
                            batch_size=cfg.get("batch_size", None),
                            lora=use_lora,
                            actual_update_step=actual_update_step,
                            ema_shape_dict=ema_shape_dict,
                            async_io=cfg.get("async_io", False),
                            include_master_weights=save_master_weights,
                        )

                        if is_log_process(plugin_type, plugin_config):
                            os.system(f"chgrp -R share {save_dir}")

                        logger.info(
                            "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                            epoch,
                            step + 1,
                            actual_update_step,
                            save_dir,
                        )

                        # remove old checkpoints
                        rm_checkpoints(exp_dir, keep_n_latest=cfg.get("keep_n_latest", -1))
                        logger.info("Removed old checkpoints and kept %s latest ones.", cfg.get("keep_n_latest", -1))
                # uncomment below 3 lines to benchmark checkpoint
                # if ckpt_every > 0 and actual_update_step % ckpt_every == 0:
                #     booster.checkpoint_io._sync_io()
                #     checkpoint_io._sync_io()
                # == terminal timer ==
                if cfg.get("record_time", False):
                    print(timers.to_str(epoch, step))

        sampler.reset()
        start_step = 0
    log_cuda_max_memory("final")


if __name__ == "__main__":
    main()

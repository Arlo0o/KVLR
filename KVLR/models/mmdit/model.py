# Modified from Flux
#
# Copyright 2024 Black Forest Labs

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from KVLR.acceleration.checkpoint import auto_grad_checkpoint
from KVLR.models.mmdit.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    LigerEmbedND,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from KVLR.registry import MODELS
from KVLR.utils.ckpt import load_checkpoint


def _make_action_proj(hidden_size: int) -> nn.Linear:
    """
    Create an action projection layer with Xavier-uniform weight and zero bias.

    Cold-start is guaranteed by zero_conv inside SkeletonEncoder (action_encoded
    = 0 at step 0), so action_proj must NOT also be zero-initialized — double-zero
    blocks gradient flow permanently.
    """
    proj = nn.Linear(hidden_size, hidden_size)
    nn.init.xavier_uniform_(proj.weight)
    nn.init.zeros_(proj.bias)
    return proj


@dataclass
class MMDiTConfig:
    model_type = "MMDiT"
    from_pretrained: str
    cache_dir: str
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    cond_embed: bool = False
    fused_qkv: bool = True
    grad_ckpt_settings: tuple[int, int] | None = None
    use_liger_rope: bool = False
    patch_size: int = 2
    # Action conditioning fields (all have defaults → fully backward compatible)
    use_action_condition: bool = False
    use_skeleton_maps: bool = False
    use_soft_moe_skeleton: bool = False
    soft_moe_num_experts: int = 5
    soft_moe_t_emb_dim: int = 256
    action_injection_double_layers: Optional[list] = None   # None = no injection
    action_injection_single_layers: Optional[list] = None
    moe_config: Optional[dict] = None
    adaptive_exec: Optional[dict] = None

    def get(self, attribute_name, default=None):
        return getattr(self, attribute_name, default)

    def __contains__(self, attribute_name):
        return hasattr(self, attribute_name)


class MMDiTModel(nn.Module):
    config_class = MMDiTConfig

    def __init__(self, config: MMDiTConfig):
        super().__init__()

        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = self.in_channels
        self.patch_size = config.patch_size

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            )

        pe_dim = config.hidden_size // config.num_heads
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {config.axes_dim} but expected positional dim {pe_dim}"
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        pe_embedder_cls = LigerEmbedND if config.use_liger_rope else EmbedND
        self.pe_embedder = pe_embedder_cls(
            dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim
        )

        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if config.guidance_embed
            else nn.Identity()
        )
        self.cond_in = (
            nn.Linear(
                self.in_channels + self.patch_size**2, self.hidden_size, bias=True
            )
            if config.cond_embed
            else nn.Identity()
        )
        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    fused_qkv=config.fused_qkv,
                )
                for _ in range(config.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    fused_qkv=config.fused_qkv,
                )
                for _ in range(config.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.initialize_weights()

        # ── Action Conditioning modules (ControlNet-style + Soft MoE) ────────
        if config.use_action_condition:
            from KVLR.models.action_encoder import SkeletonEncoder, ModalityAwareSoftMoEEncoder
            self._action_injection_double_layers = config.action_injection_double_layers or []
            self._action_injection_single_layers = config.action_injection_single_layers or []

            moe_cfg = config.moe_config or {}
            if config.use_soft_moe_skeleton:
                self.skeleton_encoder = ModalityAwareSoftMoEEncoder(
                    in_channels=9,
                    model_dim=self.hidden_size,
                    num_experts=config.soft_moe_num_experts,
                    t_emb_dim=config.soft_moe_t_emb_dim,
                    top_k=moe_cfg.get("top_k", 2),
                    capacity_pred_loss_weight=moe_cfg.get("capacity_pred_loss_weight", 0.01),
                    kp_alb_loss_weight=moe_cfg.get("kp_alb_loss_weight", 0.01),
                    src_loss_weight=moe_cfg.get("src_loss_weight", 0.005),
                    use_capacity_predictor=moe_cfg.get("use_capacity_predictor", True),
                    num_sub_experts=moe_cfg.get("num_sub_experts", 3),
                    sub_expert_top_k=moe_cfg.get("sub_expert_top_k", 1),
                    sub_lb_loss_weight=moe_cfg.get("sub_lb_loss_weight", 0.005),
                )
            elif config.use_skeleton_maps:
                self.skeleton_encoder = SkeletonEncoder(in_channels=9, model_dim=self.hidden_size)
            else:
                self.skeleton_encoder = None

            # Action projections: Xavier-uniform weight + zero bias.
            # Cold-start is guaranteed by zero_conv inside SkeletonEncoder
            # (action_encoded = 0 at step 0), so action_proj must NOT also be
            # zero-initialized — double-zero blocks gradient flow permanently.
            n_double = len(self._action_injection_double_layers)
            n_single = len(self._action_injection_single_layers)
            self.action_proj_double = nn.ModuleList([
                _make_action_proj(self.hidden_size)
                for _ in range(n_double)
            ])
            self.action_proj_single = nn.ModuleList([
                _make_action_proj(self.hidden_size)
                for _ in range(n_single)
            ]) if n_single > 0 else nn.ModuleList([])

            self.use_action_condition = True
            self.use_soft_moe_skeleton = config.use_soft_moe_skeleton
        else:
            self.use_action_condition = False
            self.skeleton_encoder = None

        self._last_action_intermediates = {}
        self._last_adaptive_stats = {}
        self._last_token_policy = None
        self._last_refresh_policy = None
        self.adaptive_exec_enabled = False
        self.adaptive_cache = None
        adaptive_cfg = config.adaptive_exec or {}
        if adaptive_cfg.get("enabled", False):
            from KVLR.models.adaptive_exec import (
                FeatureCache,
                HeuristicCriticalityHead,
                TemporalRefreshScheduler,
                TokenExecutionPolicy,
                TokenScheduler,
            )

            self.adaptive_exec_enabled = True
            self.adaptive_policy = TokenExecutionPolicy()
            self.criticality_head = HeuristicCriticalityHead(
                adaptive_cfg.get("criticality_weights", None),
                learnable=adaptive_cfg.get("learnable_criticality", False),
            )
            self.token_scheduler = TokenScheduler(
                full_ratio=adaptive_cfg.get("full_ratio", 0.2),
                light_ratio=adaptive_cfg.get("light_ratio", 0.3),
            )
            self.temporal_refresh = TemporalRefreshScheduler(
                high_threshold=adaptive_cfg.get("refresh_high_threshold", 0.65),
                mid_threshold=adaptive_cfg.get("refresh_mid_threshold", 0.35),
                refresh_high=adaptive_cfg.get("refresh_high", 1),
                refresh_mid=adaptive_cfg.get("refresh_mid", 2),
                refresh_low=adaptive_cfg.get("refresh_low", 4),
            )
            self.adaptive_cache = FeatureCache(enabled=adaptive_cfg.get("cache_enabled", True))
            self.adaptive_light_scale = float(adaptive_cfg.get("light_scale", 0.35))
            self.adaptive_reuse_fallback = float(adaptive_cfg.get("reuse_fallback_scale", 0.0))

        if self.config.grad_ckpt_settings:
            self.forward = self.forward_selective_ckpt
        else:
            self.forward = self.forward_ckpt
        self._input_requires_grad = False

    def initialize_weights(self):
        if self.config.cond_embed:
            nn.init.zeros_(self.cond_in.weight)
            nn.init.zeros_(self.cond_in.bias)

    def prepare_block_inputs(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,  # t5 encoded vec
        txt_ids: Tensor,
        timesteps: Tensor,
        y_vec: Tensor,  # clip encoded vec
        cond: Tensor = None,
        guidance: Tensor | None = None,
    ):
        """
        obtain the processed:
            img: projected noisy img latent,
            txt: text context (from t5),
            vec: clip encoded vector,
            pe: the positional embeddings for concatenated img and txt
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        if self.config.cond_embed:
            if cond is None:
                raise ValueError("Didn't get conditional input for conditional model.")
            img = img + self.cond_in(cond)

        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.config.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y_vec)

        txt = self.txt_in(txt)

        # concat: 4096 + t*h*2/4
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        if self._input_requires_grad:
            # we only apply lora to double/single blocks, thus we only need to enable grad for these inputs
            img.requires_grad_()
            txt.requires_grad_()

        return img, txt, vec, pe

    def enable_input_require_grads(self):
        """Fit peft lora. This method should not be called manually."""
        self._input_requires_grad = True

    def clear_adaptive_cache(self):
        if self.adaptive_cache is not None:
            self.adaptive_cache.invalidate()

    def get_last_intermediates(self) -> dict:
        ret = dict(self._last_action_intermediates)
        if self._last_adaptive_stats:
            ret["adaptive_stats"] = self._last_adaptive_stats
        if self._last_token_policy is not None:
            ret["token_policy"] = self._last_token_policy
        if self._last_refresh_policy is not None:
            ret["refresh_policy"] = self._last_refresh_policy
        if self.adaptive_cache is not None:
            ret["cache_stats"] = self.adaptive_cache.stats()
        return ret

    def _make_action_intermediates(
        self,
        skeleton_tensor: Tensor,
        encoded: Tensor,
        target_shape: tuple[int, int, int],
    ) -> dict:
        semantic = skeleton_tensor[..., 0:3].float().norm(dim=-1)
        motion = skeleton_tensor[..., 5:8].float().norm(dim=-1) + skeleton_tensor[..., 8].float().abs()
        tool = (semantic > 0).to(skeleton_tensor.dtype)
        motion = F.adaptive_avg_pool3d(motion[:, None], target_shape).squeeze(1)
        tool = F.adaptive_max_pool3d(tool[:, None], target_shape).squeeze(1)

        enc_info = {}
        if self.skeleton_encoder is not None and hasattr(self.skeleton_encoder, "get_last_intermediates"):
            enc_info = self.skeleton_encoder.get_last_intermediates()

        action_features = encoded.flatten(2).transpose(1, 2)
        info = {
            "action_features": action_features,
            "motion_energy": motion.flatten(1),
            "tool_mask": tool.flatten(1),
            "target_shape": target_shape,
        }
        info.update({k: v for k, v in enc_info.items() if v is not None})
        return info

    def _compute_adaptive_policy(
        self,
        skeleton_tensor: Tensor,
        action_info: dict,
        target_shape: tuple[int, int, int],
        denoise_step: int = 0,
    ) -> None:
        self._last_adaptive_stats = {}
        self._last_token_policy = None
        self._last_refresh_policy = None
        if not self.adaptive_exec_enabled:
            return

        criticality = self.criticality_head(
            skeleton_tensor=skeleton_tensor,
            target_shape=target_shape,
            outer_gate=action_info.get("outer_gate"),
            inner_gate=action_info.get("inner_gate"),
            skip_prob=action_info.get("skip_prob"),
        )
        token_policy, token_stats = self.token_scheduler(criticality)
        refresh_policy, refresh_stats = self.temporal_refresh(
            criticality,
            target_shape=target_shape,
            denoise_step=denoise_step,
        )
        self._last_token_policy = token_policy
        self._last_refresh_policy = refresh_policy
        self._last_adaptive_stats = {**token_stats, **refresh_stats}
        self._last_adaptive_stats["soft_active_ratio"] = criticality.mean()
        action_info["criticality"] = criticality
        action_info["token_policy"] = token_policy
        action_info["refresh_policy"] = refresh_policy
        action_info["adaptive_stats"] = self._last_adaptive_stats

    def _apply_adaptive_action_residual(
        self,
        residual: Tensor,
        cache_key: str,
    ) -> Tensor:
        if not self.adaptive_exec_enabled or self._last_token_policy is None:
            return residual

        policy = self._last_token_policy.to(residual.device)
        full = (policy == self.adaptive_policy.FULL).unsqueeze(-1).to(residual.dtype)
        light = (policy == self.adaptive_policy.LIGHT).unsqueeze(-1).to(residual.dtype)
        reuse = (policy == self.adaptive_policy.REUSE).unsqueeze(-1).to(residual.dtype)

        cached = None
        if self.adaptive_cache is not None and not self.training:
            cached = self.adaptive_cache.get(cache_key)
            if cached is not None and cached.shape != residual.shape:
                cached = None
        if cached is None:
            cached = residual.detach() * self.adaptive_reuse_fallback
        else:
            cached = cached.to(residual.device, residual.dtype)

        adapted = residual * full + residual * self.adaptive_light_scale * light + cached * reuse
        if self.adaptive_cache is not None and not self.training:
            self.adaptive_cache.put(cache_key, value=adapted)
        return adapted

    def _action_encoder_dtype_device(self, fallback: Tensor) -> tuple[torch.dtype, torch.device]:
        if self.skeleton_encoder is not None:
            for param in self.skeleton_encoder.parameters(recurse=True):
                if param.is_floating_point():
                    return param.dtype, param.device
            for buffer in self.skeleton_encoder.buffers(recurse=True):
                if buffer.is_floating_point():
                    return buffer.dtype, buffer.device
        return fallback.dtype, fallback.device

    def _encode_skeleton_maps(
        self,
        skeleton_maps: dict,
        img_shape: tuple,
        timesteps: Tensor,
    ) -> Tensor:
        """
        Encode skeleton_maps dict into [B, L_img, model_dim] action features.

        Args:
            skeleton_maps: dict with keys 'semantic','depth','rotation','vel_u','vel_v','vel_z','accel_mag'
                           each value: [B, T, H, W, C] tensor
            img_shape: (T', H', W') target latent grid shape (after patchify)
            timesteps: [B] flow-matching timesteps in [0, 1]

        Returns:
            encoded: [B, L_img, model_dim] where L_img = T' * H' * W'
        """
        target_dtype, target_device = self._action_encoder_dtype_device(skeleton_maps["depth"])

        def _map_tensor(name: str) -> Tensor:
            return skeleton_maps[name].to(device=target_device, dtype=target_dtype)

        skeleton_maps = dict(skeleton_maps)
        skeleton_maps["semantic"] = _map_tensor("semantic")

        # Concatenate 9 channels: [B, T, H, W, 9]
        skeleton_tensor = torch.cat([
            skeleton_maps['semantic'].float() / 255.0,  # [B,T,H,W,3] uint8 → float
            _map_tensor("depth"),                        # [B,T,H,W,1]
            _map_tensor("rotation"),                     # [B,T,H,W,1]
            _map_tensor("vel_u"),                        # [B,T,H,W,1]
            _map_tensor("vel_v"),                        # [B,T,H,W,1]
            _map_tensor("vel_z"),                        # [B,T,H,W,1]
            _map_tensor("accel_mag"),                    # [B,T,H,W,1]
        ], dim=-1).to(device=target_device, dtype=target_dtype)  # [B, T, H, W, 9]

        T_target, H_target, W_target = img_shape

        import torch.utils.checkpoint as cp
        # if self.training and not skeleton_tensor.requires_grad:
        #     skeleton_tensor.requires_grad_(True)
            
        def _run_encoder(st, ts):
            if self.use_soft_moe_skeleton:
                return self.skeleton_encoder(st, target_shape=(T_target, H_target, W_target), timesteps=ts)
            else:
                return self.skeleton_encoder(st, target_shape=(T_target, H_target, W_target))

        ts_arg = timesteps.to(device=target_device).float().clamp(0.0, 1.0)
        
        if getattr(self, "grad_checkpointing", False) and self.training:
            encoded = cp.checkpoint(_run_encoder, skeleton_tensor, ts_arg, use_reentrant=False)
        else:
            encoded = _run_encoder(skeleton_tensor, ts_arg)
        self._last_action_intermediates = self._make_action_intermediates(
            skeleton_tensor=skeleton_tensor,
            encoded=encoded,
            target_shape=(T_target, H_target, W_target),
        )
        self._compute_adaptive_policy(
            skeleton_tensor=skeleton_tensor,
            action_info=self._last_action_intermediates,
            target_shape=(T_target, H_target, W_target),
            denoise_step=int(getattr(self, "_adaptive_denoise_step", 0)),
        )
        # [B, model_dim, T', H', W'] → [B, L_img, model_dim]
        return encoded.flatten(2).transpose(1, 2)

    def forward_ckpt(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y_vec: Tensor,
        cond: Tensor = None,
        guidance: Tensor | None = None,
        skeleton_maps: dict | None = None,   # NEW: action conditioning
        img_shape: tuple | None = None,       # NEW: (T', H', W') latent grid shape
        **kwargs,
    ) -> Tensor:
        return_intermediates = kwargs.pop("return_intermediates", False)
        self._adaptive_denoise_step = int(kwargs.pop("denoise_step", 0))
        self._last_action_intermediates = {}
        self._last_adaptive_stats = {}
        self._last_token_policy = None
        self._last_refresh_policy = None
        img, txt, vec, pe = self.prepare_block_inputs(
            img, img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance
        )

        # Action encoding (computed once, injected at selected blocks)
        action_encoded = None
        
        has_skmap = skeleton_maps is not None
        skmap_keys = list(skeleton_maps.keys()) if has_skmap else []
        # print(f"========= [forward_ckpt] use_action: {self.use_action_condition}, has_skmaps: {has_skmap}, img_shape: {img_shape}, keys: {skmap_keys} =========")
        
        if self.use_action_condition and skeleton_maps is not None and img_shape is not None:
            action_encoded = self._encode_skeleton_maps(skeleton_maps, img_shape, timesteps)

        # DoubleStreamBlocks + action injection
        proj_d_idx = 0
        for i, block in enumerate(self.double_blocks):
            img, txt = auto_grad_checkpoint(block, img, txt, vec, pe)
            if action_encoded is not None and i in self._action_injection_double_layers:
                residual = self.action_proj_double[proj_d_idx](action_encoded)
                residual = self._apply_adaptive_action_residual(residual, f"double:{i}")
                img = img + residual
                proj_d_idx += 1

        # SingleStreamBlocks + action injection (img part only)
        txt_len = txt.shape[1]
        img = torch.cat((txt, img), 1)
        proj_s_idx = 0
        for i, block in enumerate(self.single_blocks):
            img = auto_grad_checkpoint(block, img, vec, pe)
            if action_encoded is not None and i in self._action_injection_single_layers:
                residual = self.action_proj_single[proj_s_idx](action_encoded)
                residual = self._apply_adaptive_action_residual(residual, f"single:{i}")
                img_part = img[:, txt_len:, :] + residual
                img = torch.cat([img[:, :txt_len, :], img_part], dim=1)
                proj_s_idx += 1
        img = img[:, txt_len:, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        if return_intermediates:
            return {"pred": img, "intermediates": self.get_last_intermediates()}
        return img

    def forward_selective_ckpt(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y_vec: Tensor,
        cond: Tensor = None,
        guidance: Tensor | None = None,
        skeleton_maps: dict | None = None,   # NEW: action conditioning
        img_shape: tuple | None = None,       # NEW: (T', H', W') latent grid shape
        **kwargs,
    ) -> Tensor:
        return_intermediates = kwargs.pop("return_intermediates", False)
        self._adaptive_denoise_step = int(kwargs.pop("denoise_step", 0))
        self._last_action_intermediates = {}
        self._last_adaptive_stats = {}
        self._last_token_policy = None
        self._last_refresh_policy = None
        img, txt, vec, pe = self.prepare_block_inputs(
            img, img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance
        )

        # Action encoding (computed once, injected at selected blocks)
        action_encoded = None
        
        has_skmap = skeleton_maps is not None
        skmap_keys = list(skeleton_maps.keys()) if has_skmap else []
        # print(f"========= [forward_selective_ckpt] use_action: {self.use_action_condition}, has_skmaps: {has_skmap}, img_shape: {img_shape}, keys: {skmap_keys} =========")
        
        if self.use_action_condition and skeleton_maps is not None and img_shape is not None:
            action_encoded = self._encode_skeleton_maps(skeleton_maps, img_shape, timesteps)

        ckpt_depth_double = self.config.grad_ckpt_settings[0]
        proj_d_idx = 0
        for i, block in enumerate(self.double_blocks[:ckpt_depth_double]):
            img, txt = auto_grad_checkpoint(block, img, txt, vec, pe)
            if action_encoded is not None and i in self._action_injection_double_layers:
                residual = self.action_proj_double[proj_d_idx](action_encoded)
                residual = self._apply_adaptive_action_residual(residual, f"double:{i}")
                img = img + residual
                proj_d_idx += 1

        for i, block in enumerate(self.double_blocks[ckpt_depth_double:], start=ckpt_depth_double):
            img, txt = block(img, txt, vec, pe)
            if action_encoded is not None and i in self._action_injection_double_layers:
                residual = self.action_proj_double[proj_d_idx](action_encoded)
                residual = self._apply_adaptive_action_residual(residual, f"double:{i}")
                img = img + residual
                proj_d_idx += 1

        ckpt_depth_single = self.config.grad_ckpt_settings[1]
        txt_len = txt.shape[1]
        img = torch.cat((txt, img), 1)
        proj_s_idx = 0
        for i, block in enumerate(self.single_blocks[:ckpt_depth_single]):
            img = auto_grad_checkpoint(block, img, vec, pe)
            if action_encoded is not None and i in self._action_injection_single_layers:
                residual = self.action_proj_single[proj_s_idx](action_encoded)
                residual = self._apply_adaptive_action_residual(residual, f"single:{i}")
                img_part = img[:, txt_len:, :] + residual
                img = torch.cat([img[:, :txt_len, :], img_part], dim=1)
                proj_s_idx += 1

        for i, block in enumerate(self.single_blocks[ckpt_depth_single:], start=ckpt_depth_single):
            img = block(img, vec, pe)
            if action_encoded is not None and i in self._action_injection_single_layers:
                residual = self.action_proj_single[proj_s_idx](action_encoded)
                residual = self._apply_adaptive_action_residual(residual, f"single:{i}")
                img_part = img[:, txt_len:, :] + residual
                img = torch.cat([img[:, :txt_len, :], img_part], dim=1)
                proj_s_idx += 1

        img = img[:, txt_len:, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        if return_intermediates:
            return {"pred": img, "intermediates": self.get_last_intermediates()}
        return img


@MODELS.register_module("flux")
def Flux(
    cache_dir: str = None,
    from_pretrained: str = None,
    device_map: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    strict_load: bool = False,
    **kwargs,
) -> MMDiTModel:
    config = MMDiTConfig(
        from_pretrained=from_pretrained,
        cache_dir=cache_dir,
        **kwargs,
    )
    low_precision_init = from_pretrained is not None and len(from_pretrained) > 0
    if low_precision_init:
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
    with torch.device(device_map):
        model = MMDiTModel(config)
    if low_precision_init:
        torch.set_default_dtype(default_dtype)
    else:
        model = model.to(torch_dtype)
    if from_pretrained:
        model = load_checkpoint(
            model,
            from_pretrained,
            cache_dir=cache_dir,
            device_map=device_map,
            strict=strict_load,
        )
    return model

"""
Action Encoder for surgical video action conditioning.

Follows ControlNet design principles:
- Separate encoding branch (NOT concatenated to latents)
- Zero-initialized output layer (cold start mechanism)
- Progressive temporal downsampling
- Spatial broadcasting to match latent dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMS_norm(nn.Module):
    """
    RMS normalization (Root Mean Square normalization).
    Same as used in Wan2_2_VAE for numerical stability.

    More stable than GroupNorm for near-zero inputs.
    """
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        # Store parameters for dynamic reshaping
        self.dim = dim
        self.channel_first = channel_first
        self.scale = dim**0.5

        # Store gamma and bias as 1D tensors, reshape in forward
        self.gamma = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        # Use F.normalize exactly as VAE does - it's more stable
        # F.normalize internally uses clamp to prevent division by zero
        normalized = F.normalize(x, dim=(1 if self.channel_first else -1))

        # Dynamically reshape gamma to match input dimensions
        if self.channel_first:
            # x shape: (B, C, *spatial_dims)
            # gamma shape: (C,) -> (1, C, 1, 1, ...) to broadcast
            gamma_shape = [1, self.dim] + [1] * (x.ndim - 2)
            gamma = self.gamma.view(*gamma_shape)
            if self.bias is not None:
                bias = self.bias.view(*gamma_shape)
            else:
                bias = 0.0
        else:
            gamma = self.gamma
            bias = self.bias if self.bias is not None else 0.0

        return normalized * self.scale * gamma + bias


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    This is the ControlNet cold start mechanism.

    Args:
        module: PyTorch module to zero-initialize

    Returns:
        module: Same module with zero-initialized parameters
    """
    for p in module.parameters():
        # CRITICAL FIX: p.detach().zero_() doesn't work because detach() creates a copy
        # We must use p.data.zero_() to modify the parameter in-place
        p.data.zero_()
    return module


class ActionEncoder(nn.Module):
    """
    Encodes action features to match Wan latent dimensions.
    Follows ControlNet design: separate branch with zero-initialized output.

    Input: [B, T, 128] - action features per frame
    Output: [B, C=3072, T', H', W'] - matches VAE latent shape

    Architecture (4 stages):
    1. Temporal feature extraction (MLP)
    2. Progressive temporal downsampling (Conv1D)
    3. Spatial broadcasting to latent grid
    4. Zero-initialized 3D convolution (CRITICAL!)
    """

    def __init__(
        self,
        action_dim: int = 128,
        model_dim: int = 3072,
        hidden_dims: Tuple[int, ...] = (256, 512, 1024, 2048),
    ):
        """
        Initialize ActionEncoder.

        Args:
            action_dim: Input action feature dimension (default: 128)
            model_dim: Output model dimension (default: 3072 for Wan TI2V-5B)
            hidden_dims: Hidden dimensions for progressive downsampling
        """
        super().__init__()

        self.action_dim = action_dim
        self.model_dim = model_dim
        self.hidden_dims = hidden_dims

        # Stage 1: Temporal encoding
        self.temporal_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )

        # Stage 2: Progressive downsampling
        # [B, 256, T] → [B, 512, T/2] → [B, 1024, T/4] → [B, 2048, T/8]
        self.temporal_blocks = nn.ModuleList()

        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            self.temporal_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                    RMS_norm(out_dim, channel_first=True, images=False),
                    nn.SiLU(),
                )
            )
            in_dim = out_dim

        # Stage 3: Spatial broadcast
        self.spatial_broadcast = nn.Sequential(
            nn.Linear(hidden_dims[-1], model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # Stage 4: Zero convolution (CRITICAL - ControlNet principle!)
        self.zero_conv = zero_module(
            nn.Conv3d(model_dim, model_dim, kernel_size=1)
        )

    def forward(
        self,
        action_features: torch.Tensor,
        target_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Encode action features to match target latent shape.

        Args:
            action_features: [B, T, action_dim] action features
            target_shape: (T', H', W') target latent shape

        Returns:
            encoded: [B, model_dim, T', H', W'] encoded action features
        """
        B, T, C = action_features.shape
        T_target, H_target, W_target = target_shape

        # Stage 1: Temporal encoding
        x = self.temporal_encoder(action_features)  # [B, T, 256]

        # Stage 2: Progressive downsampling
        x = x.transpose(1, 2)  # [B, 256, T]
        for block in self.temporal_blocks:
            x = block(x)  # [B, 2048, T/8]

        # Stage 3: Spatial broadcast
        x = x.transpose(1, 2)  # [B, T/8, 2048]
        x = self.spatial_broadcast(x)  # [B, T/8, model_dim]

        # Interpolate to target temporal dimension
        x = F.interpolate(
            x.transpose(1, 2),  # [B, model_dim, T/8]
            size=T_target,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # [B, T_target, model_dim]

        # Broadcast to spatial dimensions
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B, T_target, model_dim, 1, 1]
        x = x.expand(-1, -1, -1, H_target, W_target)  # [B, T_target, model_dim, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, model_dim, T_target, H, W]

        # Stage 4: Zero convolution (ensures cold start)
        x = self.zero_conv(x)  # [B, model_dim, T_target, H, W]

        return x


class SkeletonEncoder(nn.Module):
    """
    Encodes pixel-level skeleton maps to match Wan latent dimensions.
    Follows ControlNet design with VAE-style patchify + spatial-temporal architecture.

    Input: [B, T, H, W, C] - skeleton maps per frame
           C=9: semantic(3) + depth(1) + rotation(1) + vel_u(1) + vel_v(1) + vel_z(1) + accel_mag(1)
    Output: [B, model_dim=3072, T', H', W'] - matches VAE latent shape

    Architecture (5 stages):
    0. Patchify (patch_size=2) - spatial 2x downsample, channel 4x increase (mimics VAE)
    1. Spatial encoding (2D CNN) - process each frame independently, 8x downsample
    2. Temporal encoding (3D CNN) - capture temporal dynamics, 4x downsample
    3. Channel projection - match model dimension
    4. Zero-initialized 3D convolution (CRITICAL!)

    Total downsampling: spatial 16x (2x patchify + 8x CNN), temporal 4x
    """

    def __init__(
        self,
        in_channels: int = 9,  # semantic(3) + depth(1) + rotation(1) + vel_u(1) + vel_v(1) + vel_z(1) + accel_mag(1)
        model_dim: int = 3072,
        hidden_dims: Tuple[int, ...] = (128, 256, 512),  # 3 stages for 8x spatial downsample
        temporal_downsample: Tuple[bool, ...] = (True, True, False, False),
        patch_size: int = 2,  # Patchify patch size (mimics VAE)
    ):
        """
        Initialize SkeletonEncoder.

        Args:
            in_channels: Input channels (default: 9 for semantic3+depth1+rotation1+vel_u1+vel_v1+vel_z1+accel_mag1)
            model_dim: Output model dimension (default: 3072 for Wan TI2V-5B)
            hidden_dims: Hidden dimensions for spatial encoding (3 stages for 8x downsample)
            temporal_downsample: Whether to downsample temporally at each stage
            patch_size: Patchify patch size (default: 2, mimics VAE)
        """
        super().__init__()

        self.in_channels = in_channels
        self.model_dim = model_dim
        self.hidden_dims = hidden_dims
        self.temporal_downsample = temporal_downsample
        self.patch_size = patch_size

        # Stage 0: Patchify (mimics VAE)
        # Will be applied in forward(): [B, C, T, H, W] -> [B, C*4, T, H/2, W/2]

        # Stage 1: Spatial encoding (2D CNN)
        # After patchify: [B*T, C*4, H/2, W/2] → [B*T, 512, H/16, W/16]
        # 3 stages of 2x downsample: (H/2)/8 = H/16
        self.spatial_encoder = nn.ModuleList()
        in_ch = in_channels * (patch_size ** 2)  # After patchify: 9*4=36 channels
        for i, out_ch in enumerate(hidden_dims):
            self.spatial_encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    RMS_norm(out_ch, images=False),
                    nn.SiLU(),
                )
            )
            in_ch = out_ch

        # Stage 2: Temporal encoding (3D CNN)
        # Capture temporal dynamics: [B, 512, T, H/16, W/16] → [B, 512, T/4, H/16, W/16]
        self.temporal_encoder = nn.ModuleList()
        for i, downsample in enumerate(temporal_downsample):
            stride = (2, 1, 1) if downsample else (1, 1, 1)
            self.temporal_encoder.append(
                nn.Sequential(
                    nn.Conv3d(
                        hidden_dims[-1], hidden_dims[-1],
                        kernel_size=3, stride=stride, padding=1
                    ),
                    RMS_norm(hidden_dims[-1], images=False),
                    nn.SiLU(),
                )
            )

        # Stage 3: Channel projection
        # Match model dimension: [B, 512, T/4, H/16, W/16] → [B, 3072, T/4, H/16, W/16]
        self.channel_proj = nn.Sequential(
            nn.Conv3d(hidden_dims[-1], model_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(model_dim, model_dim, kernel_size=1),
        )

        # Stage 4: Zero-initialized output (ControlNet principle)
        self.zero_conv = zero_module(
            nn.Conv3d(model_dim, model_dim, kernel_size=1)
        )

    def forward(
        self,
        skeleton_maps: torch.Tensor,
        target_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Encode skeleton maps to match target latent shape.

        Args:
            skeleton_maps: [B, T, H, W, C] skeleton maps (C=7)
            target_shape: (T', H', W') target latent shape

        Returns:
            encoded: [B, model_dim, T', H', W'] encoded skeleton features
        """
        B, T, H, W, C = skeleton_maps.shape
        T_target, H_target, W_target = target_shape

        # # CRITICAL: Check for NaN/Inf in input
        # if torch.isnan(skeleton_maps).any() or torch.isinf(skeleton_maps).any():
        #     import logging
        #     logging.warning(f"[SkeletonEncoder] NaN or Inf detected in input, returning zeros")
        #     return torch.zeros(B, self.model_dim, T_target, H_target, W_target,
        #                      dtype=skeleton_maps.dtype, device=skeleton_maps.device)

        # # CRITICAL: Check for all-zero input (prevents GroupNorm division by zero)
        # if skeleton_maps.abs().max() < 1e-8:
        #     import logging
        #     logging.warning(f"[SkeletonEncoder] All-zero skeleton maps detected, returning zeros")
        #     return torch.zeros(B, self.model_dim, T_target, H_target, W_target,
        #                      dtype=skeleton_maps.dtype, device=skeleton_maps.device)
        # Removed NaN/Inf and all-zero early returns here to prevent DDP/ZeRO graph deadlocks

        # Stage 0: Patchify (mimics VAE)
        # [B, T, H, W, C] -> [B, C, T, H, W] -> [B, C*4, T, H/2, W/2]
        x = skeleton_maps.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]

        # Apply patchify using einops rearrange (same as VAE)
        from einops import rearrange
        x = rearrange(
            x,
            "b c t (h q) (w r) -> b (c r q) t h w",
            q=self.patch_size,
            r=self.patch_size,
        )  # [B, C*4, T, H/2, W/2] = [B, 28, T, H/2, W/2]

        # Stage 1: Spatial encoding (2D CNN)
        # Rearrange to [B*T, C*4, H/2, W/2] for 2D convolution
        _, C_patch, T_new, H_new, W_new = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C_patch, H_new, W_new)

        for layer in self.spatial_encoder:
            x = layer(x)  # [B*T, 512, H/16, W/16]

        # Rearrange to [B, C, T, H, W] for 3D convolution
        _, C_hidden, H_enc, W_enc = x.shape
        x = x.reshape(B, T, C_hidden, H_enc, W_enc).permute(0, 2, 1, 3, 4)

        # Stage 2: Temporal encoding (3D CNN)
        for layer in self.temporal_encoder:
            x = layer(x)  # [B, 512, T/4, H/16, W/16]

        # Stage 3: Channel projection
        x = self.channel_proj(x)  # [B, model_dim, T/4, H/16, W/16]

        # Interpolate to target shape if needed
        if x.shape[2:] != target_shape:
            x = F.interpolate(
                x, size=target_shape, mode='trilinear', align_corners=False
            )

        # Stage 4: Zero convolution (ensures cold start)
        x = self.zero_conv(x)  # [B, model_dim, T', H', W']

        # CRITICAL: Check for NaN/Inf in output
        if torch.isnan(x).any() or torch.isinf(x).any():
            import logging
            logging.warning(f"[SkeletonEncoder] NaN or Inf detected in output, returning zeros")
            return torch.zeros_like(x)

        return x


class MotionEncoder(nn.Module):
    """
    Encodes pose sequence to motion features matching latent dimensions.

    Input: [B, T, 9] - pose parameters per frame
    Output: [B, 3072, T', H', W'] - matches VAE latent shape

    Architecture:
    1. Pose embedding (MLP) - extract pose features
    2. Temporal encoding (1D CNN) - capture motion dynamics
    3. Spatial broadcasting - expand to spatial grid
    4. Zero-initialized output (ControlNet principle)
    """

    def __init__(
        self,
        pose_dim: int = 9,
        model_dim: int = 3072,
        hidden_dims: Tuple[int, ...] = (128, 256, 512),
    ):
        super().__init__()

        self.pose_dim = pose_dim
        self.model_dim = model_dim
        self.hidden_dims = hidden_dims

        # Stage 1: Pose embedding
        self.pose_embedding = nn.Sequential(
            nn.Linear(pose_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )

        # Stage 2: Temporal encoding (1D CNN)
        self.temporal_encoder = nn.ModuleList()
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            self.temporal_encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                    RMS_norm(out_dim, channel_first=True, images=False),
                    nn.SiLU(),
                )
            )
            in_dim = out_dim

        # Stage 3: Project to model dimension
        self.motion_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # Stage 4: Zero-initialized output
        self.zero_conv = zero_module(
            nn.Conv3d(model_dim, model_dim, kernel_size=1)
        )

    def forward(
        self,
        pose_sequence: torch.Tensor,
        target_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Args:
            pose_sequence: [B, T, 9] pose parameters
            target_shape: (T', H', W') target latent shape

        Returns:
            motion_features: [B, model_dim, T', H', W']
        """
        B, T, _ = pose_sequence.shape
        T_target, H_target, W_target = target_shape

        # Stage 1: Pose embedding
        x = self.pose_embedding(pose_sequence)  # [B, T, 128]

        # Stage 2: Temporal encoding
        x = x.transpose(1, 2)  # [B, 128, T]
        for block in self.temporal_encoder:
            x = block(x)  # [B, 512, T/4]
        x = x.transpose(1, 2)  # [B, T/4, 512]

        # Stage 3: Motion projection
        x = self.motion_proj(x)  # [B, T/4, model_dim]

        # Interpolate to target temporal dimension
        x = F.interpolate(
            x.transpose(1, 2),  # [B, model_dim, T/4]
            size=T_target,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # [B, T_target, model_dim]

        # Broadcast to spatial dimensions
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B, T_target, model_dim, 1, 1]
        x = x.expand(-1, -1, -1, H_target, W_target)
        x = x.permute(0, 2, 1, 3, 4)  # [B, model_dim, T_target, H, W]

        # Stage 4: Zero convolution
        x = self.zero_conv(x)

        return x


class MotionModulationEncoder(nn.Module):
    """
    Encodes pose sequence to frame-wise modulation parameters.

    CRITICAL DESIGN PRINCIPLE:
    - Motion controls temporal DISTRIBUTION, not spatial CONTENT
    - Output is [B, T', 6] with NO channel dimension
    - Each frame gets 6 scalars: [shift1, scale1, gate1, shift2, scale2, gate2]
    - These affect LayerNorm statistics, NOT feature content
    - Applied uniformly to all spatial positions within each frame

    This is FUNDAMENTALLY DIFFERENT from broadcasting features:
    - Current MotionEncoder: Broadcasts 3072 feature channels → spatial pollution
    - This encoder: Outputs 6 modulation scalars → controls normalization

    Input: [B, T, 10] - pose parameters per frame (quaternion[4] + translation[3] + joints[3])
    Output: [B, T', 6] - modulation parameters per frame (NO spatial dimensions!)
    """

    def __init__(
        self,
        pose_dim: int = 10,  # quaternion[4] + translation[3] + joints[3]
        hidden_dims: Tuple[int, ...] = (128, 256, 512),
        modulation_dim: int = 6,  # 6 modulation parameters per frame
    ):
        super().__init__()

        self.pose_dim = pose_dim
        self.hidden_dims = hidden_dims
        self.modulation_dim = modulation_dim

        # Stage 1: Pose embedding
        self.pose_embedding = nn.Sequential(
            nn.Linear(pose_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )

        # Stage 2: Temporal encoding (1D CNN)
        # Captures motion dynamics across time
        self.temporal_encoder = nn.ModuleList()
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            self.temporal_encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                    RMS_norm(out_dim, channel_first=True, images=False),
                    nn.SiLU(),
                )
            )
            in_dim = out_dim

        # Stage 3: Project to modulation parameters
        # Output: 6 scalars per frame (NOT per channel!)
        self.modulation_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.SiLU(),
            nn.Linear(hidden_dims[-1] // 2, modulation_dim),
        )

        # Stage 4: Zero-initialized output (CRITICAL for cold start!)
        self.zero_linear = zero_module(nn.Linear(modulation_dim, modulation_dim))

    def forward(
        self,
        pose_sequence: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        """
        Args:
            pose_sequence: [B, T, 10] pose parameters
            target_frames: T' target frame count after temporal compression

        Returns:
            modulation_params: [B, T', 6] frame-wise modulation parameters
                              NO spatial dimensions! NO channel dimension!
        """
        B, T, _ = pose_sequence.shape

        # CRITICAL: Check for NaN/Inf in input
        if torch.isnan(pose_sequence).any() or torch.isinf(pose_sequence).any():
            import logging
            logging.warning(f"[MotionModulationEncoder] NaN or Inf detected in input, returning zeros")
            return torch.zeros(B, target_frames, self.modulation_dim,
                             dtype=pose_sequence.dtype, device=pose_sequence.device)

        # CRITICAL: Check for all-zero input (prevents GroupNorm division by zero)
        if pose_sequence.abs().max() < 1e-8:
            import logging
            logging.warning(f"[MotionModulationEncoder] All-zero pose sequence detected, returning zeros")
            return torch.zeros(B, target_frames, self.modulation_dim,
                             dtype=pose_sequence.dtype, device=pose_sequence.device)

        # Stage 1: Pose embedding
        x = self.pose_embedding(pose_sequence)  # [B, T, 128]

        # Stage 2: Temporal encoding
        x = x.transpose(1, 2)  # [B, 128, T]
        for block in self.temporal_encoder:
            x = block(x)  # [B, 512, T/4]
        x = x.transpose(1, 2)  # [B, T/4, 512]

        # Stage 3: Modulation projection
        x = self.modulation_proj(x)  # [B, T/4, 6]

        # Interpolate to target temporal dimension if needed
        if x.size(1) != target_frames:
            x = F.interpolate(
                x.transpose(1, 2),  # [B, 6, T/4]
                size=target_frames,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [B, T', 6]

        # Stage 4: Zero-initialized output (ensures cold start)
        x = self.zero_linear(x)  # [B, T', 6]

        # CRITICAL: Check for NaN/Inf in output
        if torch.isnan(x).any() or torch.isinf(x).any():
            import logging
            logging.warning(f"[MotionModulationEncoder] NaN or Inf detected in output, returning zeros")
            return torch.zeros_like(x)

        return x  # [B, T', 6] - NO spatial dimensions!


# ===========================================================================
# ModalityAwareSoftMoEEncoder — 5-branch modality encoder + timestep gating
# Replaces MoESkeletonEncoder (EC sparse routing) with physics-grounded
# modality branches and timestep-conditioned soft dense routing.
# ===========================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal embedding for a scalar timestep t in [0, 1].

    Uses DiT-standard formula: emb = [sin(t * 10000^(2i/D)), cos(t * 10000^(2i/D))]

    Args:
        dim: Output embedding dimension (must be even).
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for sinusoidal embedding"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] timestep values in [0, 1]

        Returns:
            emb: [B, dim] sinusoidal embeddings
        """
        half = self.dim // 2
        # Frequencies: 10000^(2i/D) for i = 0 .. half-1
        freqs = torch.pow(
            10000.0,
            torch.arange(half, device=t.device, dtype=torch.float32) / half,
        )  # [half]
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, dim]
        return emb


class ActionInputNetwork(nn.Module):
    """
    Diagram "Action Input c_action": encodes full 9-channel skeleton maps into
    a shared representation c_action used exclusively by the Gating Network.

    The modality experts do NOT receive c_action — each expert processes its
    own modality channels via ModalityBranch.  c_action exists solely to give
    the Gating Network a holistic view of ALL modalities so it can produce
    content-aware gate weights.

    Flow: Patchify(2x2) -> 2-layer Conv2d(stride=2)
    Total spatial downsample: 8x (2x patchify + 2x conv1 + 2x conv2)

    Input:  [B, T, H, W, 9]
    Output: c_action [B*T, out_channels, H/8, W/8]
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 256, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        patch_ch = in_channels * (patch_size ** 2)  # 9*4=36
        self.encoder = nn.Sequential(
            nn.Conv2d(patch_ch, 128, kernel_size=3, stride=2, padding=1),
            RMS_norm(128, images=False),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),
            RMS_norm(out_channels, images=False),
            nn.SiLU(),
        )

    def forward(self, skeleton_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skeleton_maps: [B, T, H, W, C] skeleton maps (C=9)

        Returns:
            c_action: [B*T, out_channels, H/8, W/8]
        """
        from einops import rearrange
        B, T, H, W, C = skeleton_maps.shape

        # [B, T, H, W, C] -> [B, C, T, H, W]
        x = skeleton_maps.permute(0, 4, 1, 2, 3)

        # Patchify: [B, C, T, H, W] -> [B, C*p^2, T, H/p, W/p]
        x = rearrange(
            x,
            "b c t (h q) (w r) -> b (c r q) t h w",
            q=self.patch_size,
            r=self.patch_size,
        )

        # Reshape to [B*T, C*p^2, H/p, W/p] for 2D conv
        _, Cp, _, Hp, Wp = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, Cp, Hp, Wp)

        return self.encoder(x)  # [B*T, out_channels, H/8, W/8]


class ModalityBranch(nn.Module):
    """
    Specialized Expert for a single modality.

    Each ModalityBranch is a "Specialized Expert" in the MoE design — it ONLY
    receives and processes its own modality's channels:
      Expert_semantic:  ch[0:3]  (3ch RGB bone-type colour)
      Expert_depth:     ch[3:4]  (1ch Z-depth)
      Expert_rotation:  ch[4:5]  (1ch joint rotation)
      Expert_velocity:  ch[5:8]  (3ch vel_u, vel_v, vel_z)
      Expert_accel:     ch[8:9]  (1ch acceleration magnitude)

    Architecture: Patchify(2x2) -> conv1(stride=2) -> conv2(stride=2) -> conv3(stride=1, residual)
    Total spatial downsample: 8x (2x patchify + 2x conv1 + 2x conv2)
    conv3 adds a residual refinement stage at the final resolution without
    further downsampling, deepening per-expert capacity.

    Input:  [B, T, H, W, C]  per-modality skeleton maps
    Output: [B*T, out_channels, H/8, W/8]
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int = 2,
        mid_channels: int = 64,
        out_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.out_channels = out_channels

        patch_ch = in_channels * (patch_size ** 2)  # channels after patchify

        self.conv1 = nn.Sequential(
            nn.Conv2d(patch_ch, mid_channels, kernel_size=3, stride=2, padding=1),
            RMS_norm(mid_channels, images=False),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1),
            RMS_norm(out_channels, images=False),
            nn.SiLU(),
        )
        # Residual refinement at final resolution (stride=1, no spatial downsampling).
        # x + conv3(x): identity shortcut valid because in/out channels match.
        # Deepens per-expert representation without changing output shape.
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            RMS_norm(out_channels, images=False),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, C] single-modality maps

        Returns:
            [B*T, out_channels, H/8, W/8]
        """
        from einops import rearrange
        B, T, H, W, C = x.shape

        # Patchify: [B, C, T, H, W] -> [B, C*p^2, T, H/p, W/p]
        x = x.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        x = rearrange(
            x,
            "b c t (h q) (w r) -> b (c r q) t h w",
            q=self.patch_size,
            r=self.patch_size,
        )  # [B, C*p^2, T, H/p, W/p]

        # Reshape to [B*T, C*p^2, H/p, W/p] for 2D conv
        _, C_patch, _, H_p, W_p = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C_patch, H_p, W_p)

        # Stride-2 downsampling layers
        x = self.conv1(x)      # [B*T, mid_channels, H/(p*2), W/(p*2)]
        x = self.conv2(x)      # [B*T, out_channels, H/(p*4), W/(p*4)]
        # Residual refinement: deepens features without changing spatial resolution
        x = x + self.conv3(x)  # [B*T, out_channels, H/(p*4), W/(p*4)]
        return x


class SubExpertGating(nn.Module):
    """
    Per-spatial-token gating across sub-experts within one modality branch.

    Produces per-position logits [B*T, num_sub, H', W'] used to select which
    sub-expert (Fine / Transport / Skip) handles each spatial token.

    Initialisation (DiffMoE practice, consistent with TokenLevelGating):
      weight: N(0, 0.006) — small random perturbation breaks argmax ties so that
              different spatial positions route to different sub-experts from step 1,
              giving all three sub-experts gradient signal from the very first backward.
              Zero-weight init (all logits = 0) causes torch.argmax to deterministically
              return index 0 for every token, starving Transport (idx=1) and Skip (idx=2)
              of main-loss gradients throughout Stage I.
      bias:   zeros — keeps the prior probability ≈ 1/num_sub (uniform) after softmax,
              while the weight noise alone supplies the tie-breaking diversity.
    """
    def __init__(self, channels: int, num_sub_experts: int = 3):
        super().__init__()
        self.num_sub = num_sub_experts
        self.gate = nn.Conv2d(channels, num_sub_experts, kernel_size=1, bias=True)
        nn.init.normal_(self.gate.weight, std=0.006)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, H, W]
        return self.gate(x)  # [B*T, num_sub, H, W]


class HierarchicalModalityExpert(nn.Module):
    """
    Replaces ModalityBranch with 3 internal sub-experts per spatial token.

    Sub-expert 0 — Fine Manipulation:  1×1 + 3×3 dilated (dilation=2), targets suturing/grasping
    Sub-expert 1 — Transport Motion:   7×7 depthwise + global avg-pool context, targets fast tool sweeps
    Sub-expert 2 — Zero/Skip:          identity (x), zero-cost for static instrument regions

    Inner routing: SubExpertGating (top-1 per token, hard routing)
    Load-balance loss: stored in self._last_sub_lb_loss after each forward pass.

    Spatial stride (same as original ModalityBranch → output at H/8, W/8):
      rearrange patchify (2×): H → H/2, W → W/2
      patchify_proj  (stride=2): H/2 → H/4
      sub-expert ops (stride=1): H/4  (all sub-experts)
      down           (stride=2): H/4 → H/8
    Total: 8× spatial downsampling ✓ (identical to ModalityBranch)

    Input:  [B, T, H, W, C]  per-modality skeleton maps
    Output: [B*T, out_channels, H/8, W/8]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        patch_size: int = 2,
        num_sub_experts: int = 3,
        sub_top_k: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_sub = num_sub_experts
        self.sub_top_k = sub_top_k
        mid = out_channels

        patch_ch = in_channels * (patch_size ** 2)  # channels after rearrange patchify

        # Shared patchify projection (equivalent to ModalityBranch conv1):
        # [B*T, patch_ch, H/p, W/p] -> [B*T, mid, H/(p*2), W/(p*2)]
        self.patchify_proj = nn.Conv2d(patch_ch, mid, kernel_size=3, stride=2, padding=1)
        self.patchify_norm = RMS_norm(mid, images=False)

        # ── Sub-expert 0: Fine Manipulation ─────────────────────────────────
        # 1×1 pointwise + 3×3 dilated(dilation=2): large effective receptive field
        # without extra spatial cost; suited for suturing / grasping precision tasks.
        self.fine_conv1 = nn.Conv2d(mid, mid, kernel_size=1)
        self.fine_conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=2, dilation=2)
        self.fine_norm  = RMS_norm(mid, images=False)

        # ── Sub-expert 1: Transport Motion ──────────────────────────────────
        # 7×7 depthwise (local motion pattern) + global avg-pool context residual;
        # suited for fast tool sweeps and large-displacement frames.
        self.trans_dw   = nn.Conv2d(mid, mid, kernel_size=7, padding=3, groups=mid)
        self.trans_pw   = nn.Conv2d(mid, mid, kernel_size=1)
        self.trans_pool = nn.AdaptiveAvgPool2d(1)   # global context [B*T, mid, 1, 1]
        self.trans_ctx  = nn.Conv2d(mid, mid, kernel_size=1)
        self.trans_norm = RMS_norm(mid, images=False)

        # ── Sub-expert 2: Skip — identity (no parameters) ───────────────────

        # ── Inner routing ────────────────────────────────────────────────────
        self.sub_gating = SubExpertGating(mid, num_sub_experts)

        # ── Second stride-2 downsample (ModalityBranch conv2 equivalent) ────
        # H/(p*2) → H/(p*4) = H/8 with p=2
        self.down      = nn.Conv2d(mid, out_channels, kernel_size=3, stride=2, padding=1)
        self.down_norm = RMS_norm(out_channels, images=False)

        # _last_sub_lb_loss: set during forward (not a persistent buffer; lives on
        # the same device as the forward inputs, created fresh each forward call).
        self._last_sub_lb_loss: torch.Tensor = torch.tensor(0.0)
        self._last_sub_gate_probs = None
        self._last_sub_routing_mask = None
        self._last_skip_prob = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, C]  single-modality skeleton maps

        Returns:
            [B*T, out_channels, H/8, W/8]
        """
        from einops import rearrange
        B, T, H, W, C = x.shape

        # Rearrange patchify (same as ModalityBranch): H → H/p, W → W/p
        x = x.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        x = rearrange(
            x,
            "b c t (h q) (w r) -> b (c r q) t h w",
            q=self.patch_size,
            r=self.patch_size,
        )  # [B, C*p^2, T, H/p, W/p]

        _, C_patch, _, H_p, W_p = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C_patch, H_p, W_p)
        # [B*T, C*p^2, H/p, W/p]

        # Patchify projection with stride-2 → [B*T, mid, H/(p*2), W/(p*2)]
        x = F.silu(self.patchify_norm(self.patchify_proj(x)))

        # ── Sub-expert outputs (all at same spatial resolution) ──────────────

        # ── Inner gating: top-1 per spatial token ────────────────────────────
        gate_logits = self.sub_gating(x)           # [B*T, num_sub, H', W']
        gate_probs  = gate_logits.softmax(dim=1)   # [B*T, num_sub, H', W']

        top_idx  = gate_probs.argmax(dim=1, keepdim=True)   # [B*T, 1, H', W']
        top_prob = gate_probs.gather(1, top_idx)             # [B*T, 1, H', W']
        self._last_sub_gate_probs = gate_probs
        self._last_sub_routing_mask = torch.zeros_like(gate_probs).scatter_(1, top_idx, 1.0)
        self._last_skip_prob = gate_probs[:, 2:3] if gate_probs.shape[1] > 2 else None

        # In-place routed fusion reduces peak activation memory.
        # The convolutional sub-experts still run densely, then their outputs are
        # immediately fused according to the selected token route.
        
        mask0 = (top_idx == 0).to(x.dtype)
        mask1 = (top_idx == 1).to(x.dtype)
        mask2 = (top_idx == 2).to(x.dtype)

        # Initialize with the skip branch.
        h_fused = x * mask2

        # Fuse sub-expert 0.
        h0 = F.silu(self.fine_norm(self.fine_conv2(self.fine_conv1(x))))
        h_fused.add_(h0 * mask0)
        del h0

        # Fuse sub-expert 1.
        h1_local = self.trans_dw(x)
        h1_ctx   = self.trans_ctx(self.trans_pool(x)).expand_as(x)
        h1 = F.silu(self.trans_norm(self.trans_pw(h1_local) + h1_ctx))
        del h1_local, h1_ctx
        h_fused.add_(h1 * mask1)
        del h1

        # Apply the selected route probability.
        h_fused.mul_(top_prob.to(x.dtype))
        # [B*T, mid, H', W']

        # ── Inner load-balance loss (same formula as outer compute_load_balance_loss) ──
        # f_i = fraction of tokens hard-routed to sub-expert i
        # P_i = mean softmax probability for sub-expert i
        # L_lb_inner = num_sub * Σ(f_i * P_i)
        top_hard = gate_probs.argmax(dim=1)                  # [B*T, H', W']
        f = torch.stack([
            (top_hard == i).float().mean()
            for i in range(self.num_sub)
        ]).to(x.device).to(x.dtype)                                       # [num_sub]
        P = gate_probs.mean(dim=[0, 2, 3]).to(x.device).to(x.dtype)      # [num_sub]
        self._last_sub_lb_loss = self.num_sub * (f * P).sum()

        # ── Stride-2 downsample → [B*T, out_channels, H/8, W/8] ────────────
        out = F.silu(self.down_norm(self.down(h_fused)))

        return out


class ContentAwareTimestepGating(nn.Module):
    """
    Content-aware timestep gating: G(pool(c_action), tau(t)).

    Combines global-average-pooled c_action features with sinusoidal timestep
    embeddings to produce per-expert soft gate weights.

    Comparison:
      DiffMoE:      logits = x @ W_g.T              (content only)
      Old design:   logits = MLP(tau(t))             (timestep only)
      This design:  logits = MLP([pool(c); tau(t)])  (content + timestep)

    Input:
      c_action_pooled: [B, feature_dim]  — global average pooled c_action
      t:               [B]               — flow-matching timestep in [0, 1]
    Output:
      g: [B, num_experts]                — softmax-normalised gate weights
    """

    def __init__(self, feature_dim: int = 256, t_emb_dim: int = 256, num_experts: int = 5):
        super().__init__()
        self.t_emb = SinusoidalPositionEmbedding(dim=t_emb_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim + t_emb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_experts),
        )
        # DiffMoE practice: gate output layer uses very small variance init
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.006)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, c_action_pooled: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_action_pooled: [B, feature_dim] global-average-pooled c_action
            t: [B] flow-matching timesteps in [0, 1]

        Returns:
            g: [B, num_experts] soft gate weights summing to 1
        """
        t_emb = self.t_emb(t.to(c_action_pooled.device)).to(c_action_pooled.dtype) # [B, t_emb_dim]
        combined = torch.cat([c_action_pooled, t_emb], dim=-1) # [B, feature_dim + t_emb_dim]
        logits = self.gate_mlp(combined)                       # [B, num_experts]
        return logits.softmax(dim=-1)


class TokenLevelGating(nn.Module):
    """
    Token-level gating: produces per-spatial-position gate logits.

    Unlike ContentAwareTimestepGating which collapses spatial dims via global
    average pooling (giving identical weights to all positions), this module
    preserves spatial resolution so each (h, w) position independently selects
    its expert mixture.

    Input:
      c_action: [B*T, 256, H', W']  — shared feature map from ActionInputNetwork
      t:        [B]                   — flow-matching timestep in [0, 1]
      T:        int                   — number of frames

    Output:
      gate_logits: [B*T, num_experts, H', W']  (raw, NOT softmaxed)
    """

    def __init__(self, feature_dim: int = 256, t_emb_dim: int = 256, num_experts: int = 5):
        super().__init__()
        self.t_emb = SinusoidalPositionEmbedding(dim=t_emb_dim)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(feature_dim + t_emb_dim, 128, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(128, num_experts, kernel_size=1),
        )
        # DiffMoE practice: gate output layer uses very small variance init
        nn.init.normal_(self.gate_conv[-1].weight, std=0.006)
        nn.init.zeros_(self.gate_conv[-1].bias)

    def forward(self, c_action: torch.Tensor, t: torch.Tensor, T: int) -> torch.Tensor:
        """
        Args:
            c_action: [B*T, 256, H', W'] feature map from ActionInputNetwork
            t: [B] flow-matching timesteps in [0, 1]
            T: number of frames per sample

        Returns:
            gate_logits: [B*T, num_experts, H', W'] raw logits
        """
        BT, C, H, W = c_action.shape
        B = BT // T

        # Compute timestep embedding [B, t_emb_dim]
        t_emb = self.t_emb(t.float().to(c_action.device)).to(c_action.dtype)  # [B, t_emb_dim]

        # Expand to [B*T, t_emb_dim, H', W']
        t_broadcast = (
            t_emb
            .unsqueeze(1)                        # [B, 1, t_emb_dim]
            .expand(B, T, -1)                    # [B, T, t_emb_dim]
            .reshape(BT, -1)                     # [B*T, t_emb_dim]
            .unsqueeze(-1).unsqueeze(-1)         # [B*T, t_emb_dim, 1, 1]
            .expand(-1, -1, H, W)                # [B*T, t_emb_dim, H', W']
        )

        # Concatenate and compute gate logits
        combined = torch.cat([c_action, t_broadcast], dim=1)  # [B*T, 256+t_emb_dim, H', W']
        gate_logits = self.gate_conv(combined)  # [B*T, num_experts, H', W']
        return gate_logits


class CapacityPredictor(nn.Module):
    """
    DiffMoE-inspired capacity predictor for efficient inference routing.

    During training: predicts which experts will be selected (binary) via BCE loss
    against the actual routing mask. This trains the predictor to anticipate routing.

    During inference: uses the predicted capacities with EMA-updated per-expert
    thresholds for dynamic routing without needing the full top-k computation.

    Input:  c_action_detached [B*T, 256, H', W']
    Output: pred [B*T, num_experts, H', W'] (raw logits)
    """

    def __init__(self, feature_dim: int = 256, num_experts: int = 5):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(feature_dim, num_experts, kernel_size=1),
        )
        self.register_buffer('expert_threshold', torch.zeros(num_experts))
        self.register_buffer('ema_decay', torch.tensor(0.95))

    @torch.no_grad()
    def update_threshold(self, pred: torch.Tensor, routing_mask: torch.Tensor):
        """
        EMA update of per-expert quantile thresholds.

        Args:
            pred: [B*T, num_experts, H', W'] raw logits from predictor
            routing_mask: [B*T, num_experts, H', W'] binary routing decisions

        Design notes:
          - Always updates regardless of frac_routed value (including 0.0 and 1.0).
            The old guard `if frac_routed > 0 and frac_routed < 1` caused stagnation:
            when an expert received 0 tokens in a batch its threshold was never
            updated, drifting upward and starving that expert in future batches
            (positive-feedback expert starvation loop).
          - frac_routed is clamped to [1e-4, 1-1e-4] only for the quantile call
            to avoid torch.quantile edge-cases at exactly 0.0 or 1.0; the EMA
            update itself always runs.
          - Cross-rank threshold sync (dist.all_reduce) is intentionally NOT done
            here: action_dropout can cause some ranks to skip the encoder forward
            entirely, so calling all_reduce inside update_threshold risks deadlock.
            Sync is instead done in trainer.py train_step after scaled_loss.backward(),
            where all ranks are guaranteed to participate.
        """
        pred_sigmoid = torch.sigmoid(pred)
        num_experts = pred.shape[1]

        for i in range(num_experts):
            expert_pred = pred_sigmoid[:, i].flatten()
            expert_mask = routing_mask[:, i].flatten()

            frac_routed = expert_mask.float().mean()
            # Clamp only to keep torch.quantile numerically stable, NOT to skip update
            q = (1.0 - frac_routed).clamp(1e-4, 1.0 - 1e-4)
            quantile_val = torch.quantile(expert_pred.float(), q)
            self.expert_threshold[i] = (
                self.ema_decay * self.expert_threshold[i]
                + (1 - self.ema_decay) * quantile_val
            )

    def forward(self, c_action_detached: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_action_detached: [B*T, 256, H', W'] (detached from main graph)

        Returns:
            pred: [B*T, num_experts, H', W'] raw logits
        """
        return self.predictor(c_action_detached)


def sparse_topk_fusion(
    expert_outputs: list,
    gate_logits: torch.Tensor,
    top_k: int = 2,
):
    """
    Sparse top-k fusion of expert outputs at each spatial position.

    Args:
        expert_outputs: list of 5 tensors, each [B*T, 256, H', W']
        gate_logits: [B*T, num_experts, H', W'] raw logits
        top_k: number of experts to select per spatial position

    Returns:
        h_fused: [B*T, 256, H', W'] fused output
        routing_mask: [B*T, num_experts, H', W'] binary mask (1 where expert selected)
        gate_weights: [B*T, num_experts, H', W'] re-normalized sparse weights
    """
    expert_dtype = expert_outputs[0].dtype
    expert_device = expert_outputs[0].device

    # 1. Softmax over experts
    scores = gate_logits.softmax(dim=1).to(device=expert_device, dtype=expert_dtype)  # [B*T, E, H', W']

    # 2. Top-k selection
    topk_vals, topk_idx = scores.topk(top_k, dim=1)  # [B*T, top_k, H', W']

    # 3. Build routing mask
    routing_mask = torch.zeros_like(scores).scatter_(1, topk_idx, 1.0)  # [B*T, E, H', W']

    # 4. Sparse scores with re-normalization
    sparse_scores = scores * routing_mask  # zero out non-selected
    sparse_sum = sparse_scores.sum(dim=1, keepdim=True).clamp(min=1e-8)
    gate_weights = sparse_scores / sparse_sum  # re-normalize to sum=1

    # 5. Weighted sum
    stacked = torch.stack(expert_outputs, dim=1)  # [B*T, E, 256, H', W']
    h_fused = (stacked * gate_weights.unsqueeze(2)).sum(dim=1)  # [B*T, 256, H', W']

    return h_fused.to(dtype=expert_dtype), routing_mask, gate_weights


def compute_load_balance_loss(
    gate_logits: torch.Tensor,
    routing_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Load balance loss to prevent expert collapse.

    L_lb = E * sum_i(f_i * P_i)
      f_i = fraction of tokens routed to expert i (from routing_mask)
      P_i = mean softmax probability for expert i (from gate_logits)

    Args:
        gate_logits: [B*T, num_experts, H', W'] raw logits
        routing_mask: [B*T, num_experts, H', W'] binary routing decisions

    Returns:
        Scalar loss tensor
    """
    num_experts = gate_logits.shape[1]
    scores = gate_logits.softmax(dim=1)  # [B*T, E, H', W']

    # f_i: fraction of tokens routed to expert i
    # f = routing_mask.float().mean(dim=(0, 2, 3))  # [E]
    f = routing_mask.to(gate_logits.dtype).mean(dim=(0, 2, 3))  # [E]

    # P_i: mean softmax probability for expert i
    P = scores.mean(dim=(0, 2, 3))  # [E]

    # L_lb = E * sum(f_i * P_i)
    loss = num_experts * (f * P).sum()
    return loss


def compute_kinematic_prior_loss(
    gate_logits: torch.Tensor,
    routing_mask: torch.Tensor,
    skeleton_maps: torch.Tensor,
) -> torch.Tensor:
    """
    Kinematic-Prior Adaptive Load Balancing (KP-ALB) loss.

    Replaces uniform load balance loss with a physics-grounded target
    distribution π_i(K) derived directly from the 9-channel kinematic
    skeleton tensor K. Each expert's target utilization is proportional
    to its corresponding physical signal magnitude.

    L_KP-ALB = Σ_{i∈M} (f_i · P_i - π_i(K))²

    Channel layout for π computation:
      ch 0-2: semantic energy  → L2 norm of RGB (tool presence/color)
      ch 3:   depth energy     → |Z| (distance from camera)
      ch 4:   rotation energy  → |α| (joint rotation magnitude)
      ch 5-7: velocity energy  → L2 norm of (vel_u, vel_v, vel_z)
      ch 8:   accel energy     → |a| (acceleration magnitude)

    Args:
        gate_logits:   [B*T, 5, H', W'] raw gate logits
        routing_mask:  [B*T, 5, H', W'] binary top-k routing decisions
        skeleton_maps: [B, T, H, W, 9]  original 9-channel input

    Returns:
        Scalar loss tensor (MSE between f·P and π)
    """
    B, T, H, W, _ = skeleton_maps.shape
    H_prime, W_prime = gate_logits.shape[2], gate_logits.shape[3]

    # Reshape to [B*T, 9, H, W] and downsample to gate spatial resolution
    skel = skeleton_maps.reshape(B * T, H, W, 9).permute(0, 3, 1, 2).contiguous()
    skel_down = F.adaptive_avg_pool2d(skel, (H_prime, W_prime))  # [B*T, 9, H', W']

    # Physical energy per modality (order must match expert assignment)
    energy_sem = skel_down[:, 0:3].norm(dim=1)       # semantic: RGB color magnitude
    energy_dep = skel_down[:, 3].abs()               # depth: camera-Z magnitude
    energy_rot = skel_down[:, 4].abs()               # rotation: joint angle magnitude
    energy_vel = skel_down[:, 5:8].norm(dim=1)       # velocity: 3D speed magnitude
    energy_acc = skel_down[:, 8].abs()               # accel: acceleration magnitude

    # π_i(K): normalized physical prior [B*T, 5, H', W']
    pi = torch.stack([energy_sem, energy_dep, energy_rot, energy_vel, energy_acc], dim=1)
    pi = pi / (pi.sum(dim=1, keepdim=True) + 1e-8)

    # Global routing statistics
    scores = gate_logits.softmax(dim=1)
    # f = routing_mask.float().mean(dim=(0, 2, 3))     # [5] fraction of tokens routed
    f = routing_mask.to(gate_logits.dtype).mean(dim=(0, 2, 3)) 
    P = scores.mean(dim=(0, 2, 3))                   # [5] mean softmax probability

    # Global physical prior target (spatial+batch average)
    # target_pi = pi.mean(dim=(0, 2, 3))               # [5]
    target_pi = pi.mean(dim=(0, 2, 3)).to(gate_logits.dtype)   

    # MSE between actual routing product f·P and physical target
    loss_kp = F.mse_loss(f * P, target_pi.detach())
    return loss_kp


def compute_spatiotemporal_consistency_loss(
    cap_pred: torch.Tensor,
    skeleton_maps: torch.Tensor,
) -> torch.Tensor:
    """
    Spatiotemporal Routing Consistency (SRC) loss.

    Enforces temporal smoothness in routing probability distributions
    strictly within regions where surgical instruments are present.
    Prevents routing flicker at highly articulate instrument tips.

    L_SRC = 1/(T-1) Σ_{t=1}^{T-1} Σ_{h,w} M_tool^(t) ⊙ ||R_t - R_{t-1}||²

    R_t ∈ R^{H×W×N} is the continuous routing probability at frame t,
    generated by the capacity predictor (sigmoid of raw logits).
    M_tool^(t) ∈ {0,1}^{H×W} is the binary tool presence mask derived
    from semantic channels (ch 0-2) of the skeleton maps.

    Args:
        cap_pred:      [B*T, N, H', W'] raw logits from capacity predictor
        skeleton_maps: [B, T, H, W, 9]  original 9-channel input

    Returns:
        Scalar loss tensor (0.0 if T < 2)
    """
    B, T, H, W, _ = skeleton_maps.shape
    if T < 2:
        return cap_pred.new_tensor(0.0)

    _, N, H_prime, W_prime = cap_pred.shape

    # Routing probability [B, T, N, H', W']
    R = torch.sigmoid(cap_pred).view(B, T, N, H_prime, W_prime)

    # Tool presence mask from semantic channels (max-pool preserves sparse activations)
    skel = skeleton_maps.reshape(B * T, H, W, 9).permute(0, 3, 1, 2).contiguous()
    skel_down = F.adaptive_max_pool2d(skel, (H_prime, W_prime))  # [B*T, 9, H', W']
    # tool_present = (skel_down[:, 0:3].sum(dim=1) > 0).float()    # [B*T, H', W']
    tool_present = (skel_down[:, 0:3].sum(dim=1) > 0).to(cap_pred.dtype)    # [B*T, H', W']
    M_tool = tool_present.view(B, T, 1, H_prime, W_prime)        # [B, T, 1, H', W']

    # Squared temporal differences masked to tool regions
    temporal_diff = (R[:, 1:] - R[:, :-1]) ** 2                  # [B, T-1, N, H', W']
    masked_diff = temporal_diff * M_tool[:, 1:]                   # [B, T-1, N, H', W']

    # Normalize by number of active tool pixels × N
    loss_src = masked_diff.sum() / (M_tool[:, 1:].sum() * N + 1e-8)
    return loss_src


class ModalityAwareSoftMoEEncoder(nn.Module):
    """
    Modality-Aware Sparse Mixture-of-Experts Skeleton Encoder with token-level routing.

    Each of the 5 experts is structurally specialized for one physical modality:
      Expert 0 (semantic):  ch[0:3]  RGB bone-type colour-coding
      Expert 1 (depth):     ch[3:4]  Camera-frame Z-depth
      Expert 2 (rotation):  ch[4:5]  Joint rotation angle
      Expert 3 (velocity):  ch[5:8]  vel_u, vel_v, vel_z
      Expert 4 (accel):     ch[8:9]  3D acceleration magnitude

    Shared Expert (always-active, DiffMoE n_shared_experts concept):
      Sees all 9 channels simultaneously. Captures cross-modal interactions
      that no single modality expert can represent (e.g., depth-velocity coupling
      when a tool decelerates as it approaches tissue). Zero-initialized projection
      ensures cold start (no effect at initialization).

    Token-level sparse routing (DiffMoE-inspired):
      All experts ALWAYS run on ALL tokens (since each expert processes different
      channels). Sparsity applies only at FUSION — at each spatial position (h, w),
      only top-k=2 expert outputs contribute.

      ActionInputNetwork encodes ALL 9 channels into c_action (shared repr).
      TokenLevelGating produces per-spatial-position gate logits [B*T, 5, H', W'].
      sparse_topk_fusion selects top-k experts per position with re-normalization.
      shared_expert output is added on top (always, regardless of routing).

    Capacity scheduling (3 stages):
      Stage I  (dense):     All experts, softmax fusion — warm up expert branches
      Stage II (annealing): Sparse top-k fusion — learn to route
      Stage III (target):   Sparse top-k fusion — stable routing

    Architecture:
      [B, T, H, W, 9]
        → split 5 modalities
        → 5 × ModalityBranch (patchify + conv1 + conv2 + conv3-residual)  -> [B*T, 256, H/8, W/8] each
        → ActionInputNetwork(full 9ch) → c_action [B*T, 256, H/8, W/8]
        → TokenLevelGating(c_action, τ(t)) → gate_logits [B*T, 5, H/8, W/8]  (token-level!)
        → sparse_topk_fusion(experts, gate_logits, top_k=2) → h_fused [B*T, 256, H/8, W/8]
        → shared_expert(full 9ch) → h_shared [B*T, 256, H/8, W/8]
        → h_fused = h_fused + shared_expert_proj(h_shared)  (zero-init proj, cold start)
        → aux losses (load balance + capacity predictor BCE) → cached
        → post_spatial Conv2d(256->512, s=2)  -> [B*T, 512, H/16, W/16]
        → reshape  -> [B, 512, T, H/16, W/16]
        → temporal_encoder (4 × Conv3d, 4x temporal down)
        → channel_proj (Conv3d -> model_dim)
        → trilinear interp if shape mismatch
        → zero_conv (ControlNet cold-start)
      Output [B, model_dim, T', H', W']

    Training side-effects:
      self._last_gate_weights  [B*T, 5]    (spatial-averaged, for monitoring)
      self._last_aux_loss      scalar       (combined aux loss for trainer)
      self._last_routing_mask  [B*T, 5, H', W']  (for diagnostics)
    """

    def __init__(
        self,
        in_channels: int = 9,
        model_dim: int = 3072,
        num_experts: int = 5,
        t_emb_dim: int = 256,
        temporal_downsample: Tuple[bool, ...] = (True, True, False, False),
        patch_size: int = 2,
        top_k: int = 2,
        capacity_pred_loss_weight: float = 0.01,
        kp_alb_loss_weight: float = 0.01,
        src_loss_weight: float = 0.005,
        use_capacity_predictor: bool = True,
        num_sub_experts: int = 3,
        sub_expert_top_k: int = 1,
        sub_lb_loss_weight: float = 0.005,
        use_shared_expert: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.temporal_downsample = temporal_downsample
        self.patch_size = patch_size
        self.top_k = top_k
        self.capacity_pred_loss_weight = capacity_pred_loss_weight
        self.kp_alb_loss_weight = kp_alb_loss_weight
        self.src_loss_weight = src_loss_weight
        self._sub_lb_loss_weight = sub_lb_loss_weight
        self.use_shared_expert = use_shared_expert

        # ── 5 Modality-Specialized Expert Branches (Hierarchical MoE) ──────
        # Each branch uses HierarchicalModalityExpert: outer specialization by
        # modality channel + inner top-1 routing across 3 sub-experts
        # (Fine / Transport / Skip) per spatial token.
        self.branch_semantic = HierarchicalModalityExpert(
            3, out_channels=256, patch_size=patch_size,
            num_sub_experts=num_sub_experts, sub_top_k=sub_expert_top_k)
        self.branch_depth    = HierarchicalModalityExpert(
            1, out_channels=256, patch_size=patch_size,
            num_sub_experts=num_sub_experts, sub_top_k=sub_expert_top_k)
        self.branch_rotation = HierarchicalModalityExpert(
            1, out_channels=256, patch_size=patch_size,
            num_sub_experts=num_sub_experts, sub_top_k=sub_expert_top_k)
        self.branch_velocity = HierarchicalModalityExpert(
            3, out_channels=256, patch_size=patch_size,
            num_sub_experts=num_sub_experts, sub_top_k=sub_expert_top_k)
        self.branch_accel    = HierarchicalModalityExpert(
            1, out_channels=256, patch_size=patch_size,
            num_sub_experts=num_sub_experts, sub_top_k=sub_expert_top_k)

        # ── Shared Expert (DiffMoE n_shared_experts concept) ──────────────
        # Always-active branch that sees all 9 channels simultaneously.
        # Captures cross-modal interactions that no single modality expert can
        # represent (e.g., depth-velocity coupling, rotation-semantic linkage).
        # mid_channels=128 gives 2× capacity relative to single-modality branches.
        # shared_expert_proj is zero-initialized (ControlNet cold-start): at
        # initialization its output is zero so training begins from the MoE-only
        # baseline and the shared expert contribution is learned progressively.
        if self.use_shared_expert:
            self.shared_expert = ModalityBranch(
                in_channels=in_channels,  # 9 channels: all modalities
                patch_size=patch_size,
                mid_channels=128,
                out_channels=256,
            )
            self.shared_expert_proj = zero_module(nn.Conv2d(256, 256, kernel_size=1))

        # ── Shared Action Input Network (for gating only) ─────────────────
        # Encodes full 9ch into c_action so the gating network can see all
        # modalities holistically.  Experts do NOT receive c_action.
        self.action_input_network = ActionInputNetwork(
            in_channels=in_channels, out_channels=256, patch_size=patch_size,
        )

        # ── Token-Level Gating (replaces batch-level ContentAwareTimestepGating) ─
        # Produces per-spatial-position gate logits [B*T, E, H', W']
        self.token_gating = TokenLevelGating(
            feature_dim=256, t_emb_dim=t_emb_dim, num_experts=num_experts,
        )

        # ── DiffMoE-inspired Capacity Predictor (for efficient inference) ──
        self.capacity_predictor = CapacityPredictor(256, num_experts) if use_capacity_predictor else None

        # ── Post-fusion spatial downsampling ──────────────────────────────
        # [B*T, 256, H/8, W/8] -> [B*T, 512, H/16, W/16]
        self.post_spatial = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            RMS_norm(512, images=False),
            nn.SiLU(),
        )

        # ── Temporal encoding (4 x Conv3d) ────────────────────────────────
        # Same design as SkeletonEncoder Stage 2
        self.temporal_encoder = nn.ModuleList()
        for downsample in temporal_downsample:
            stride = (2, 1, 1) if downsample else (1, 1, 1)
            self.temporal_encoder.append(
                nn.Sequential(
                    nn.Conv3d(512, 512, kernel_size=3, stride=stride, padding=1),
                    RMS_norm(512, images=False),
                    nn.SiLU(),
                )
            )

        # ── Channel projection ────────────────────────────────────────────
        # Same as SkeletonEncoder Stage 3
        self.channel_proj = nn.Sequential(
            nn.Conv3d(512, model_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(model_dim, model_dim, kernel_size=1),
        )

        # ── Zero conv (ControlNet cold-start) ─────────────────────────────
        self.zero_conv = zero_module(
            nn.Conv3d(model_dim, model_dim, kernel_size=1)
        )

        # Cache for training monitoring and aux loss
        self._last_gate_weights: Optional[torch.Tensor] = None
        self._last_gate_weights_spatial: Optional[torch.Tensor] = None
        self._last_aux_loss: Optional[torch.Tensor] = None
        self._last_routing_mask: Optional[torch.Tensor] = None
        self._last_kp_alb_loss: Optional[torch.Tensor] = None
        self._last_src_loss: Optional[torch.Tensor] = None
        self._last_gate_logits: Optional[torch.Tensor] = None
        self._last_inner_gate: Optional[torch.Tensor] = None
        self._last_skip_prob: Optional[torch.Tensor] = None
        self._last_control_features: Optional[torch.Tensor] = None
        # Capacity scheduling stage: 'dense' (Stage I), 'annealing' (Stage II), 'target' (Stage III)
        self._capacity_stage: str = 'dense'
        # Stage II annealing coefficient: 0.0 (all dense) → 1.0 (all sparse)
        # Written by WanTrainer._update_moe_capacity() every training step.
        self._annealing_alpha: float = 0.0

    def forward(
        self,
        skeleton_maps: torch.Tensor,
        target_shape: Tuple[int, int, int],
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            skeleton_maps: [B, T, H, W, 9]  (9-channel skeleton maps)
            target_shape:  (T', H', W')  target latent shape
            timesteps:     [B] flow-matching timesteps in [0, 1]
                           If None, uses uniform weights (1/num_experts each).

        Returns:
            encoded: [B, model_dim, T', H', W']
        """
        B, T, H, W, C = skeleton_maps.shape
        T_target, H_target, W_target = target_shape

        # Nullify gate cache at every forward so stale data is never exposed
        self._last_gate_weights = None
        self._last_gate_weights_spatial = None
        self._last_aux_loss = None
        self._last_routing_mask = None
        self._last_kp_alb_loss = None
        self._last_src_loss = None
        self._last_gate_logits = None
        self._last_inner_gate = None
        self._last_skip_prob = None
        self._last_control_features = None

        # # Guard: NaN / Inf in input
        # if torch.isnan(skeleton_maps).any() or torch.isinf(skeleton_maps).any():
        #     import logging
        #     logging.warning("[ModalityAwareSoftMoEEncoder] NaN/Inf in input, returning zeros")
        #     return torch.zeros(B, self.model_dim, T_target, H_target, W_target,
        #                        dtype=skeleton_maps.dtype, device=skeleton_maps.device)

        # # Guard: all-zero input (avoids RMSNorm numerical issues)
        # if skeleton_maps.abs().max() < 1e-8:
        #     import logging
        #     logging.warning("[ModalityAwareSoftMoEEncoder] All-zero input, returning zeros")
        #     return torch.zeros(B, self.model_dim, T_target, H_target, W_target,
        #                        dtype=skeleton_maps.dtype, device=skeleton_maps.device)
        # Removed early returns for NaN/All-zero inputs here to preserve DDP/ZeRO graph consistency across all ranks.

        # ── Step 1: Split into 5 modality tensors ─────────────────────────
        semantic = skeleton_maps[..., 0:3]   # [B, T, H, W, 3]
        depth    = skeleton_maps[..., 3:4]   # [B, T, H, W, 1]
        rotation = skeleton_maps[..., 4:5]   # [B, T, H, W, 1]
        velocity = skeleton_maps[..., 5:8]   # [B, T, H, W, 3]  (vel_u, vel_v, vel_z)
        accel    = skeleton_maps[..., 8:9]   # [B, T, H, W, 1]

        # ── Nested Checkpoint Helper ──────────────────────────────────────────
        import torch.utils.checkpoint as cp
        def _ckp(module, x):
            # if self.training and x.requires_grad:
            if self.training:
                return cp.checkpoint(module, x, use_reentrant=False)
            return module(x)

        # if self.training:
        #     semantic.requires_grad_(True)
        #     depth.requires_grad_(True)
        #     rotation.requires_grad_(True)
        #     velocity.requires_grad_(True)
        #     accel.requires_grad_(True)

        # ── Step 2: Modality-Specialized Expert Branches ───────────────────
        # Each expert only sees its own modality channels (structural specialization)
        # h_sem = self.branch_semantic(semantic)   # [B*T, 256, H/8, W/8]
        # h_dep = self.branch_depth(depth)         # [B*T, 256, H/8, W/8]
        # h_rot = self.branch_rotation(rotation)   # [B*T, 256, H/8, W/8]
        # h_vel = self.branch_velocity(velocity)   # [B*T, 256, H/8, W/8]
        # h_acc = self.branch_accel(accel)         # [B*T, 256, H/8, W/8]
        # # print(semantic.max(), semantic.min(), h_sem.max(), h_sem.min(), depth.max(), depth.min(), h_dep.max(), h_dep.min(),  )
        h_sem = _ckp(self.branch_semantic, semantic)   # [B*T, 256, H/8, W/8]
        h_dep = _ckp(self.branch_depth, depth)         # [B*T, 256, H/8, W/8]
        h_rot = _ckp(self.branch_rotation, rotation)   # [B*T, 256, H/8, W/8]
        h_vel = _ckp(self.branch_velocity, velocity)   # [B*T, 256, H/8, W/8]
        h_acc = _ckp(self.branch_accel, accel)         # [B*T, 256, H/8, W/8]

        expert_outputs = [h_sem, h_dep, h_rot, h_vel, h_acc]
        inner_gates = [
            getattr(self.branch_semantic, "_last_sub_gate_probs", None),
            getattr(self.branch_depth, "_last_sub_gate_probs", None),
            getattr(self.branch_rotation, "_last_sub_gate_probs", None),
            getattr(self.branch_velocity, "_last_sub_gate_probs", None),
            getattr(self.branch_accel, "_last_sub_gate_probs", None),
        ]
        if all(g is not None for g in inner_gates):
            self._last_inner_gate = torch.stack(inner_gates, dim=0).mean(dim=0)
        skip_probs = [
            getattr(self.branch_semantic, "_last_skip_prob", None),
            getattr(self.branch_depth, "_last_skip_prob", None),
            getattr(self.branch_rotation, "_last_skip_prob", None),
            getattr(self.branch_velocity, "_last_skip_prob", None),
            getattr(self.branch_accel, "_last_skip_prob", None),
        ]
        if all(s is not None for s in skip_probs):
            self._last_skip_prob = torch.stack(skip_probs, dim=0).mean(dim=0)

        # ── Step 3: Token-Level Gating ────────────────────────────────────
        # ActionInputNetwork encodes ALL 9ch → c_action (holistic view for gating)
        # c_action = self.action_input_network(skeleton_maps)  # [B*T, 256, H/8, W/8]
        # if self.training:
        #     skeleton_maps.requires_grad_(True)
        c_action = _ckp(self.action_input_network, skeleton_maps)  # [B*T, 256, H/8, W/8]

        if timesteps is not None:
            t = timesteps.float().to(skeleton_maps.device)
            gate_logits = self.token_gating(c_action, t, T)  # [B*T, 5, H', W']
        else:
            # Uniform logits when timestep is unavailable (e.g. eval without t)
            gate_logits = torch.zeros(
                B * T, self.num_experts, c_action.shape[2], c_action.shape[3],
                dtype=skeleton_maps.dtype, device=skeleton_maps.device,
            )
        self._last_gate_logits = gate_logits

        # ── Step 4: Fusion with capacity scheduling ──────────────────────
        routing_mask = None
        if self._capacity_stage == 'dense' and self.training:
            # Stage I: dense softmax fusion (all experts contribute, no sparse routing)
            stacked = torch.stack(expert_outputs, dim=1)  # [B*T, E, 256, H', W']
            scores = gate_logits.softmax(dim=1).to(device=stacked.device, dtype=stacked.dtype)  # [B*T, E, H', W']
            h_fused = (stacked * scores.unsqueeze(2)).sum(dim=1)  # [B*T, 256, H', W']
            gate_weights_spatial = scores
            # Synthetic routing_mask for CapacityPredictor and KP-ALB pretraining.
            # Selects the same top-k experts that Stage II/III would hard-route to,
            # providing training signal before sparse routing begins.
            # h_fused above is unchanged — this mask is only consumed by aux losses.
            topk_idx = scores.topk(self.top_k, dim=1).indices  # [B*T, top_k, H', W']
            routing_mask = torch.zeros_like(scores).scatter_(1, topk_idx, 1.0)
        elif self._capacity_stage == 'annealing' and self.training:
            # Stage II: linearly interpolate dense → sparse fusion.
            # alpha=0 at stage boundary (all dense), alpha=1 at stage end (all sparse).
            # This avoids the hard loss jump that a discrete switch causes.
            alpha = float(self._annealing_alpha)  # set by WanTrainer._update_moe_capacity()

            # Dense path (always computed for interpolation)
            stacked = torch.stack(expert_outputs, dim=1)  # [B*T, E, 256, H', W']
            scores_dense = gate_logits.softmax(dim=1).to(device=stacked.device, dtype=stacked.dtype)  # [B*T, E, H', W']
            h_dense = (stacked * scores_dense.unsqueeze(2)).sum(dim=1)  # [B*T, 256, H', W']

            # Sparse path (top-k routing)
            h_sparse, routing_mask, scores_sparse = sparse_topk_fusion(
                expert_outputs, gate_logits, self.top_k,
            )

            # Linear blend
            h_fused = (1.0 - alpha) * h_dense + alpha * h_sparse
            # Gate weights for monitoring: blend dense and sparse probabilities
            gate_weights_spatial = (1.0 - alpha) * scores_dense + alpha * scores_sparse
        elif self.training:
            # Stage III: fully sparse top-k fusion
            h_fused, routing_mask, gate_weights_spatial = sparse_topk_fusion(
                expert_outputs, gate_logits, self.top_k,
            )
        else:
            # Inference: use capacity predictor for dynamic routing if available
            if self.capacity_predictor is not None and self.capacity_predictor.expert_threshold.abs().sum() > 0:
                h_fused, routing_mask, gate_weights_spatial = self._inference_dynamic_routing(
                    expert_outputs, c_action, gate_logits,
                )
            else:
                # Fallback: sparse top-k
                h_fused, routing_mask, gate_weights_spatial = sparse_topk_fusion(
                    expert_outputs, gate_logits, self.top_k,
                )

        # ── Step 4b: Shared Expert (always runs, cross-modal) ─────────────
        # shared_expert processes all 9 channels and adds a cross-modal
        # correction to h_fused. shared_expert_proj is zero-initialized
        # (ControlNet cold-start), so training starts from the MoE-only
        # baseline. The contribution grows as the projection learns.
        if self.use_shared_expert:
            h_shared = self.shared_expert(skeleton_maps)                  # [B*T, 256, H/8, W/8]
            h_fused = h_fused + self.shared_expert_proj(h_shared)         # residual add

        post_weight = self.post_spatial[0].weight
        h_fused = h_fused.to(device=post_weight.device, dtype=post_weight.dtype)
        self._last_gate_weights_spatial = gate_weights_spatial

        # ── Auxiliary losses (training only) ──────────────────────────────
        if self.training:
            aux_loss = torch.tensor(0.0, dtype=skeleton_maps.dtype, device=skeleton_maps.device)
            # Clear single-step caches
            self._last_kp_alb_loss = None
            self._last_src_loss = None

            if routing_mask is not None:
                # ── KP-ALB: replaces uniform load balance loss ────────────────
                kp_alb_loss = compute_kinematic_prior_loss(gate_logits, routing_mask, skeleton_maps)
                aux_loss = aux_loss + self.kp_alb_loss_weight * kp_alb_loss
                self._last_kp_alb_loss = kp_alb_loss.detach()

                # ── Capacity predictor BCE + SRC ──────────────────────────────
                if self.capacity_predictor is not None:
                    cap_pred = self.capacity_predictor(c_action.detach())
                    cap_loss = F.binary_cross_entropy_with_logits(
                    #     cap_pred, routing_mask.detach(),
                    # )
                        cap_pred.float(), routing_mask.detach().float(),
                    ).to(cap_pred.dtype)
                    aux_loss = aux_loss + self.capacity_pred_loss_weight * cap_loss

                    # SRC: spatiotemporal routing consistency (needs T > 1)
                    src_loss = compute_spatiotemporal_consistency_loss(cap_pred, skeleton_maps)
                    aux_loss = aux_loss + self.src_loss_weight * src_loss
                    self._last_src_loss = src_loss.detach()

                    # Update EMA thresholds (no grad)
                    self.capacity_predictor.update_threshold(cap_pred, routing_mask)

            # ── Inner sub-expert load-balance losses from all 5 branches ────
            # Each HierarchicalModalityExpert stores _last_sub_lb_loss after forward.
            # Average across branches so the weight is comparable to the outer loss.
            _sub_lb = (
                self.branch_semantic._last_sub_lb_loss +
                self.branch_depth._last_sub_lb_loss +
                self.branch_rotation._last_sub_lb_loss +
                self.branch_velocity._last_sub_lb_loss +
                self.branch_accel._last_sub_lb_loss
            ) / 5.0
            aux_loss = aux_loss + self._sub_lb_loss_weight * _sub_lb

            self._last_aux_loss = aux_loss
            self._last_routing_mask = routing_mask.detach() if routing_mask is not None else None
            # Cache spatial-average gate weights for per-expert monitoring
            self._last_gate_weights = gate_weights_spatial.mean(dim=(2, 3))  # [B*T, E] -> spatial avg

        # ── Step 5: Post-fusion spatial downsampling ──────────────────────
        x = self.post_spatial(h_fused)   # [B*T, 512, H/16, W/16]

        # Reshape to [B, 512, T, H/16, W/16] for 3D convolutions
        _, C_hidden, H_enc, W_enc = x.shape
        x = x.reshape(B, T, C_hidden, H_enc, W_enc).permute(0, 2, 1, 3, 4)

        # ── Step 6: Temporal encoding (3D CNN) ────────────────────────────
        for layer in self.temporal_encoder:
            # x = layer(x)   # [B, 512, T/4, H/16, W/16]
            # if self.training:
            #     x.requires_grad_(True)
            x = _ckp(layer, x)   # [B, 512, T/4, H/16, W/16]
            
        # ── Step 7: Channel projection ────────────────────────────────────
        x = self.channel_proj(x)   # [B, model_dim, T/4, H/16, W/16]

        # ── Step 8: Trilinear interpolation if shape mismatch ─────────────
        if x.shape[2:] != (T_target, H_target, W_target):
            x = F.interpolate(
                x, size=(T_target, H_target, W_target),
                mode='trilinear', align_corners=False,
            )

        # ── Step 9: Zero conv (ControlNet cold-start) ────────────────────
        x = self.zero_conv(x)
        self._last_control_features = x

        # Guard: NaN / Inf in output
        if torch.isnan(x).any() or torch.isinf(x).any():
            import logging
            logging.warning("[ModalityAwareSoftMoEEncoder] NaN/Inf in output, returning zeros")
            return torch.zeros_like(x)

        return x

    def get_last_intermediates(self) -> dict:
        """
        Return detached routing/control signals for teacher-student distillation
        and adaptive execution analysis. Missing values are kept as None so the
        caller can skip unavailable losses.
        """
        return {
            "outer_gate": self._last_gate_weights_spatial,
            "outer_gate_summary": self._last_gate_weights,
            "outer_gate_logits": self._last_gate_logits,
            "routing_mask": self._last_routing_mask,
            "inner_gate": self._last_inner_gate,
            "skip_prob": self._last_skip_prob,
            "control_features": self._last_control_features,
            "aux_loss": self._last_aux_loss,
            "kp_alb_loss": self._last_kp_alb_loss,
            "src_loss": self._last_src_loss,
        }

    def _inference_dynamic_routing(
        self,
        expert_outputs: list,
        c_action: torch.Tensor,
        gate_logits: torch.Tensor,
    ):
        """
        Inference-time dynamic routing using capacity predictor thresholds.

        Uses the learned per-expert thresholds from training to determine
        which experts to activate at each spatial position, avoiding the
        need for the full top-k computation.

        Args:
            expert_outputs: list of 5 tensors, each [B*T, 256, H', W']
            c_action: [B*T, 256, H', W'] feature map (detached for predictor)
            gate_logits: [B*T, E, H', W'] raw gate logits

        Returns:
            h_fused, routing_mask, gate_weights (same as sparse_topk_fusion)
        """
        expert_dtype = expert_outputs[0].dtype
        expert_device = expert_outputs[0].device
        cap_sigmoid = torch.sigmoid(self.capacity_predictor(c_action.detach()))
        # Apply threshold per expert
        threshold = self.capacity_predictor.expert_threshold.view(1, -1, 1, 1).to(cap_sigmoid.device, cap_sigmoid.dtype)
        routing_mask = (cap_sigmoid > threshold).to(device=expert_device, dtype=expert_dtype)

        # Ensure at least 1 expert per spatial position (fallback to argmax)
        no_expert = routing_mask.sum(dim=1, keepdim=True) == 0  # [B*T, 1, H', W']
        if no_expert.any():
            scores = gate_logits.softmax(dim=1).to(device=expert_device, dtype=expert_dtype)
            argmax_idx = scores.argmax(dim=1, keepdim=True)  # [B*T, 1, H', W']
            fallback_mask = torch.zeros_like(routing_mask).scatter_(1, argmax_idx, 1.0)
            routing_mask = torch.where(no_expert.expand_as(routing_mask), fallback_mask, routing_mask)

        # Sparse weighted sum with re-normalization
        scores = gate_logits.softmax(dim=1).to(device=expert_device, dtype=expert_dtype)
        sparse_scores = scores * routing_mask
        sparse_sum = sparse_scores.sum(dim=1, keepdim=True).clamp(min=1e-8)
        gate_weights = sparse_scores / sparse_sum

        stacked = torch.stack(expert_outputs, dim=1)
        h_fused = (stacked * gate_weights.unsqueeze(2)).sum(dim=1)

        return h_fused.to(dtype=expert_dtype), routing_mask, gate_weights


class ActionProjection(nn.Module):
    """
    Zero-initialized projection layer for action feature injection.
    Used in WanAttentionBlockWithAction.
    """

    def __init__(self, dim: int):
        """
        Initialize ActionProjection.

        Args:
            dim: Feature dimension
        """
        super().__init__()
        # self.proj = zero_module(nn.Linear(dim, dim)) ##--->
        self.proj =  nn.Linear(dim, dim) ##--->

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project action features.

        Args:
            x: [B, L, dim] action features

        Returns:
            projected: [B, L, dim] projected features
        """
        return self.proj(x)


# Test function for verification
def test_action_encoder():
    """
    Test ActionEncoder with dummy data.
    """
    print("Testing ActionEncoder...")

    # Create encoder
    encoder = ActionEncoder(action_dim=128, model_dim=3072)

    # Create dummy action features
    B, T = 2, 61  # 2 videos, 61 frames
    action_features = torch.randn(B, T, 128)

    # Target shape (matches VAE latent)
    # For 61 frames video at 30fps: T' = 15 (4x compression)
    # For 480x640 image: H' = 30, W' = 30 (16x compression)
    target_shape = (15, 30, 30)

    # Forward pass
    encoded = encoder(action_features, target_shape)

    print(f"Input shape: {action_features.shape}")
    print(f"Output shape: {encoded.shape}")
    print(f"Expected shape: ({B}, 3072, {target_shape[0]}, {target_shape[1]}, {target_shape[2]})")

    # Verify zero initialization
    print(f"\nZero initialization check:")
    print(f"Output mean: {encoded.mean().item():.6f}")
    print(f"Output std: {encoded.std().item():.6f}")
    print(f"Output max: {encoded.abs().max().item():.6f}")

    if encoded.abs().max().item() < 1e-6:
        print("✓ Zero initialization verified!")
    else:
        print("✗ Warning: Output not zero-initialized")

    # Test gradient flow
    loss = encoded.sum()
    loss.backward()

    print(f"\nGradient flow check:")
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    print(f"Gradients present: {has_grad}")

    if has_grad:
        print("✓ Gradient flow verified!")
    else:
        print("✗ Warning: No gradients")

    print("\nActionEncoder test completed!")


def test_skeleton_encoder():
    """
    Test SkeletonEncoder with dummy skeleton maps.
    """
    print("\nTesting SkeletonEncoder...")

    # Create encoder
    encoder = SkeletonEncoder(in_channels=3, model_dim=3072)

    # Create dummy skeleton maps
    B, T, H, W, C = 2, 61, 512, 512, 3  # 2 videos, 61 frames, 512x512 resolution, 3 channels
    skeleton_maps = torch.randn(B, T, H, W, C)

    # Target shape (matches VAE latent)
    # For 61 frames video at 30fps: T' = 15 (4x compression)
    # For 512x512 image: H' = 32, W' = 32 (16x compression)
    target_shape = (15, 32, 32)

    # Forward pass
    encoded = encoder(skeleton_maps, target_shape)

    print(f"Input shape: {skeleton_maps.shape}")
    print(f"Output shape: {encoded.shape}")
    print(f"Expected shape: ({B}, 3072, {target_shape[0]}, {target_shape[1]}, {target_shape[2]})")

    # Verify zero initialization
    print(f"\nZero initialization check:")
    print(f"Output mean: {encoded.mean().item():.6f}")
    print(f"Output std: {encoded.std().item():.6f}")
    print(f"Output max: {encoded.abs().max().item():.6f}")

    if encoded.abs().max().item() < 1e-6:
        print("✓ Zero initialization verified!")
    else:
        print("✗ Warning: Output not zero-initialized")

    # Test gradient flow
    loss = encoded.sum()
    loss.backward()

    print(f"\nGradient flow check:")
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    print(f"Gradients present: {has_grad}")

    if has_grad:
        print("✓ Gradient flow verified!")
    else:
        print("✗ Warning: No gradients")

    print("\nSkeletonEncoder test completed!")


if __name__ == "__main__":
    test_action_encoder()
    test_skeleton_encoder()

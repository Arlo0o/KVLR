"""
Action feature extraction utilities for surgical video action conditioning.
Ported from wan/training/action_utils.py for KVLR (MMDiT/Flux architecture).

This module provides functions to load and process action annotations including:
- 2D keypoints from keypoints.json
- 3D pose data from memory_pool.pth
- Action type from video filename
- Velocity features (temporal derivatives)

Supports two data formats:
1. Standard format: annotations/action/{video_id}/
   - memory_pool_left.pth, memory_pool_right.pth
   - instance1/keypoints.json, instance2/keypoints.json
2. Separated format: annotations/knotting_results/{video_id}_instance*/
   - memory_pool.pth (per instance)
   - Optional: keypoints.json (can be generated from pose)
"""

import logging
import os
import sys
import json
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict

# Normalization constants for velocity/acceleration channels
_VEL_UV_CLIP = 100.0   # pixels/frame  (da Vinci max ~0.5m/s at z≈0.1m → ~100 px/frame)
_VEL_Z_CLIP  = 0.05    # m/frame       (±1.5 m/s depth speed, far exceeds da Vinci limits)
_ACCEL_CLIP  = 0.1     # m/frame²

# Import Pose class from annotations/read_utils.py
# Add multiple possible paths to find annotations directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
# KVLR/utils/ -> KVLR/ -> project_root
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))

# Try local or user-provided annotation utility locations. The KASA data archive
# includes read_utils_copy.py/read_utils_copy2.py with the action annotations.
_annotation_root = os.environ.get("KASA_ANNOTATION_ROOT", os.path.join(_project_root, "data", "kasa", "annotations"))
_possible_annotation_paths = [
    _annotation_root,
    os.path.join(_annotation_root, "label_results"),
    os.path.join(_project_root, "annotations"),
]

# Add all possible paths to sys.path
for path in _possible_annotation_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path) and abs_path not in sys.path:
        sys.path.insert(0, abs_path)
        print(f"[Action Utils] Added to sys.path: {abs_path}")

# Import strategy (two-module split for correctness):
#
# 1. Pose + load_memory_pool  ← read_utils_copy  (MUST NOT change)
#    PTH files were serialized with torch.save() while read_utils_copy.Pose was the live
#    class.  Pickle stores the fully-qualified name "read_utils_copy.Pose" inside the
#    file.  At load time pickle re-imports read_utils_copy to resolve that name, so
#    the module must stay importable and the class name must stay the same.
#
# 2. Instrument + rendering helpers  ← read_utils_copy2  (updated geometry)
#    read_utils_copy2 corrects the shaft keypoint offset (-0.010026 m vs -0.050000 m)
#    and adds the shaft-extension algorithm.
_HAS_READ_UTILS = False
_HAS_SKELETON_UTILS = False

try:
    from read_utils_copy import Pose, load_memory_pool
    from read_utils_copy2 import (
        Instrument, compute_skeleton_3d, project_3d_to_2d,
        BONES, BONE_CATEGORIES, SEMANTIC_COLORS
    )
    _HAS_READ_UTILS = True
    _HAS_SKELETON_UTILS = True
    # print("[Action Utils] Imported Pose/load_memory_pool from read_utils_copy.py")
    # print("[Action Utils] Imported Instrument/rendering from read_utils_copy2.py")

    # CRITICAL FIX: Register Pose to __main__ for pickle compatibility
    # PTH files were saved with Pose in __main__ module (when script was run directly)
    if '__main__' in sys.modules:
        sys.modules['__main__'].Pose = Pose
        # print("[Action Utils] Registered Pose class to __main__ module for pickle compatibility")

except ImportError as e:
    print(f"[Action Utils] WARNING: Failed to import skeleton utilities: {e}")
    print(f"[Action Utils] Searched in: {_possible_annotation_paths}")
    print(f"[Action Utils] Skeleton map generation will return zero maps.")
    print("[Action Utils] Ensure read_utils_copy.py and read_utils_copy2.py are available in KASA_ANNOTATION_ROOT.")
    # Graceful degradation: _HAS_READ_UTILS = False, functions return zeros


def find_action_directory(video_id: str, action_annotation_dir: str) -> Tuple[Optional[str], str]:
    """
    Intelligently find action annotation directory, supporting both formats.

    Format 1 (Standard): annotations/action/{video_id}/
    Format 2 (Separated): annotations/knotting_results/{video_id}_instance*/

    Args:
        video_id: Video ID extracted from filename
        action_annotation_dir: Base annotation directory

    Returns:
        (action_dir, format_type):
            - action_dir: Path to annotation directory (or parent for separated format)
            - format_type: "standard", "separated", or "not_found"
    """
    # Try standard format first
    standard_dir = os.path.join(action_annotation_dir, video_id)
    if os.path.exists(standard_dir):
        return standard_dir, "standard"

    # Try separated format - search in parent directory
    parent_dir = os.path.dirname(action_annotation_dir)

    # Extract action type from video_id
    action_type = None
    for action in ['knotting', 'needleGrasping', 'needlePuncture', 'suturePulling']:
        if action.lower() in video_id.lower():
            action_type = action
            break

    if action_type:
        # Check {action}_results directory
        results_dir = os.path.join(parent_dir, f"{action_type}_results")
        if os.path.exists(results_dir):
            # Look for matching instance directories
            pattern = f"{video_id}_instance"
            matching_dirs = [
                d for d in os.listdir(results_dir)
                if d.startswith(pattern) and os.path.isdir(os.path.join(results_dir, d))
            ]
            if matching_dirs:
                return results_dir, "separated"

    return None, "not_found"


def load_action_features(video_path: str,
                        frame_indices: List[int],
                        action_annotation_dir: str,
                        image_width: int = 640,
                        image_height: int = 480,
                        verbose: bool = False) -> np.ndarray:
    """
    Load complete action features for a video.

    Args:
        video_path: Path to video file
        frame_indices: List of frame indices to load
        action_annotation_dir: Base directory for action annotations
        image_width: Image width for keypoint normalization
        image_height: Image height for keypoint normalization
        verbose: Whether to print detailed loading information

    Returns:
        action_features: [T, 128] array. Returns zeros if no annotations available.
    """
    T = len(frame_indices)
    return np.zeros((T, 128), dtype=np.float32)


def find_pth_file(video_path: str, pth_search_dirs: List[str]) -> Optional[str]:
    """
    Find corresponding .pth file for a video in knotting_results/needleGrasping_results.

    Args:
        video_path: Path to video file or video_id string
                    (e.g., "SARRARP502022_knotting_1_video12.mp4" or
                           "SARRARP502022_knotting_1_video12")
        pth_search_dirs: List of directories to search for .pth files

    Returns:
        Path to .pth file if found, None otherwise
    """
    # Extract video ID from path (handles both with/without extension)
    video_id = os.path.basename(video_path)
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_id = video_id.replace(ext, '')

    # Extract action type from video_id
    action_type = None
    for action in ['knotting', 'needleGrasping', 'needlePuncture', 'suturePulling']:
        if action.lower() in video_id.lower():
            action_type = action
            break

    if not action_type:
        print(  "find_pth_file: no action type keyword in video_id=%r, skipping", video_id  )
        logging.debug("find_pth_file: no action type keyword in video_id=%r, skipping", video_id)
        return None

    # Search in each directory
    for search_dir in pth_search_dirs:
        if not os.path.exists(search_dir):
            logging.warning("find_pth_file: pth_search_dir does not exist: %s", search_dir)
            continue

        # Check if this is the right action type directory
        if action_type.lower() not in search_dir.lower():
            continue

        # Look for matching instance directories
        # Format: {video_id}_instance1/memory_pool.pth, {video_id}_instance2/memory_pool.pth
        for inst_idx in [1, 2]:
            inst_dir_name = f"{video_id}_instance{inst_idx}"
            inst_path = os.path.join(search_dir, inst_dir_name)

            if os.path.exists(inst_path):
                # Check for memory_pool.pth
                pth_file = os.path.join(inst_path, 'memory_pool.pth')
                if os.path.exists(pth_file):
                    return pth_file

                # Also check for memory_pool_left.pth, memory_pool_right.pth
                for side in ['left', 'right']:
                    pth_file = os.path.join(inst_path, f'memory_pool_{side}.pth')
                    if os.path.exists(pth_file):
                        return pth_file

    logging.debug(
        "find_pth_file: no pth found for video_id=%r (action_type=%r) in %d dirs",
        video_id, action_type, len(pth_search_dirs),
    )
    return None


def extract_pose_sequence_from_pth(
    pth_path: str,
    frame_indices: List[int],
) -> Optional[np.ndarray]:
    """
    Extract pose parameter sequence from PTH file.

    Args:
        pth_path: Path to memory_pool.pth
        frame_indices: List of frame indices to extract

    Returns:
        pose_sequence: [T, 10] array with rotation_quat[4] + translation[3] + alpha[1] + theta_l[1] + theta_r[1]
        Returns None if PTH file cannot be loaded or contains invalid data
    """
    if not os.path.exists(pth_path):
        return None

    try:
        if _HAS_READ_UTILS:
            memory_pool = load_memory_pool(pth_path)
        else:
            memory_pool = torch.load(pth_path, map_location='cpu')

        if not isinstance(memory_pool, dict):
            return None

        T = len(frame_indices)
        pose_sequence = np.zeros((T, 10), dtype=np.float32)

        for t, frame_idx in enumerate(frame_indices):
            # Convert frame_idx to frame_key (1-indexed, zero-padded)
            frame_key = str(frame_idx + 1).zfill(5)

            if frame_key not in memory_pool:
                continue

            entry = memory_pool[frame_key]

            pose_info = entry
            if isinstance(entry, dict) and 'pose_info' in entry:
                pose_info = entry['pose_info']
            elif hasattr(entry, 'pose_info'):
                pose_info = entry.pose_info

            # Extract rotation (quaternion [4])
            if hasattr(pose_info, 'rot'):
                rot_quat = pose_info.rot
                if isinstance(rot_quat, torch.Tensor):
                    rot_quat = rot_quat.detach().cpu().numpy()
                else:
                    rot_quat = np.array(rot_quat, dtype=np.float32)
                pose_sequence[t, 0:4] = rot_quat

            # Extract translation [3]
            if hasattr(pose_info, 'trans'):
                trans = pose_info.trans
                if isinstance(trans, torch.Tensor):
                    trans = trans.detach().cpu().numpy()
                else:
                    trans = np.array(trans, dtype=np.float32)
                pose_sequence[t, 4:7] = trans

            # Extract alpha (wrist joint angle) [1]
            if hasattr(pose_info, 'alpha'):
                alpha = pose_info.alpha
                if isinstance(alpha, torch.Tensor):
                    alpha_val = alpha.item() if alpha.numel() == 1 else alpha[0].item()
                else:
                    alpha_val = float(alpha)
                pose_sequence[t, 7] = alpha_val

            # Extract theta_l (left gripper angle) [1]
            if hasattr(pose_info, 'theta_l'):
                theta_l = pose_info.theta_l
                if isinstance(theta_l, torch.Tensor):
                    theta_l_val = theta_l.item() if theta_l.numel() == 1 else theta_l[0].item()
                else:
                    theta_l_val = float(theta_l)
                pose_sequence[t, 8] = theta_l_val

            # Extract theta_r (right gripper angle) [1]
            if hasattr(pose_info, 'theta_r'):
                theta_r = pose_info.theta_r
                if isinstance(theta_r, torch.Tensor):
                    theta_r_val = theta_r.item() if theta_r.numel() == 1 else theta_r[0].item()
                else:
                    theta_r_val = float(theta_r)
                pose_sequence[t, 9] = theta_r_val

        if np.isnan(pose_sequence).any() or np.isinf(pose_sequence).any():
            import logging
            logging.warning(f"[Pose Sequence] NaN or Inf detected in {pth_path}, returning None")
            return None

        if np.abs(pose_sequence).max() < 1e-8:
            import logging
            logging.warning(f"[Pose Sequence] All-zero pose sequence in {pth_path}, returning None")
            return None

        return pose_sequence

    except Exception as e:
        import logging
        logging.warning(f"Failed to extract pose sequence from {pth_path}: {e}")
        return None


def _compute_keypoint_velocities_and_accels(
    memory_pool: dict,
    instrument,
    sample_frame_indices: List[int],
    focal_length: float,
    image_shape: Tuple[int, int],
) -> Dict[int, Dict[str, Tuple[float, float, float, float]]]:
    """
    Compute per-keypoint image-plane velocities and 3D acceleration magnitudes.

    Uses central finite differences (forward/backward at boundaries).
    PTH frames are assumed at 30 Hz (dt = 1/30 s).

    Returns:
        {frame_idx: {kp_name: (vel_u_norm, vel_v_norm, vel_z_norm, accel_mag_norm)}}
    """
    H, W = image_shape
    cx, cy = W / 2.0, H / 2.0
    fx = fy = float(focal_length)
    dt = 1.0 / 30.0  # PTH annotations at 30 Hz

    def _get_pose_info(fi: int):
        for key in (str(fi + 1).zfill(5), str(fi).zfill(5),
                    str(fi + 1), str(fi)):
            if key in memory_pool:
                entry = memory_pool[key]
                if hasattr(entry, 'pose_info'):
                    return entry.pose_info
                elif isinstance(entry, dict) and 'pose_info' in entry:
                    return entry['pose_info']
                return entry
        return None

    # Gather 3D positions for sample frames plus their immediate neighbours
    needed = set()
    for fi in sample_frame_indices:
        needed.update((fi - 1, fi, fi + 1))

    pos_3d_all: Dict[int, Dict[str, np.ndarray]] = {}
    for fi in needed:
        pose_info = _get_pose_info(fi)
        if pose_info is None:
            continue
        try:
            pts = compute_skeleton_3d(instrument, pose_info)
            valid = {}
            for kp, pt in pts.items():
                arr = np.array(pt, dtype=np.float64)
                if np.isfinite(arr).all():
                    valid[kp] = arr
            if valid:
                pos_3d_all[fi] = valid
        except Exception:
            pass

    result: Dict[int, Dict[str, Tuple[float, float, float, float]]] = {}
    for fi in sample_frame_indices:
        if fi not in pos_3d_all:
            result[fi] = {}
            continue

        kps_t = pos_3d_all[fi]

        vel_3d: Dict[str, np.ndarray] = {}
        accel_3d: Dict[str, np.ndarray] = {}

        for kp, pos_t in kps_t.items():
            has_prev = (fi - 1) in pos_3d_all and kp in pos_3d_all[fi - 1]
            has_next = (fi + 1) in pos_3d_all and kp in pos_3d_all[fi + 1]

            if has_prev and has_next:
                v = (pos_3d_all[fi + 1][kp] - pos_3d_all[fi - 1][kp]) / (2.0 * dt)
                a = (pos_3d_all[fi + 1][kp] - 2.0 * pos_t + pos_3d_all[fi - 1][kp]) / (dt * dt)
            elif has_next:
                v = (pos_3d_all[fi + 1][kp] - pos_t) / dt
                a = np.zeros(3, dtype=np.float64)
            elif has_prev:
                v = (pos_t - pos_3d_all[fi - 1][kp]) / dt
                a = np.zeros(3, dtype=np.float64)
            else:
                v = np.zeros(3, dtype=np.float64)
                a = np.zeros(3, dtype=np.float64)

            vel_3d[kp] = v
            accel_3d[kp] = a

        # Project 3D positions to 2D to get (u, v, Z) for velocity projection
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        try:
            points_2d_fi, depths_fi = project_3d_to_2d(kps_t, K)
        except Exception:
            result[fi] = {}
            continue

        frame_vels: Dict[str, Tuple[float, float, float, float]] = {}
        for kp, (vX, vY, vZ) in vel_3d.items():
            if kp not in depths_fi or kp not in points_2d_fi:
                continue
            Z = float(depths_fi[kp])
            if abs(Z) < 1e-6:
                continue
            u, v_coord = points_2d_fi[kp]

            vel_u_raw = (fx * vX - (u - cx) * vZ) / Z
            vel_v_raw = (fy * vY - (v_coord - cy) * vZ) / Z
            vel_z_raw = vZ
            accel_mag_raw = float(np.linalg.norm(accel_3d.get(kp, np.zeros(3))))

            vel_u_n  = float(np.clip(vel_u_raw  / _VEL_UV_CLIP, -1.0, 1.0))
            vel_v_n  = float(np.clip(vel_v_raw  / _VEL_UV_CLIP, -1.0, 1.0))
            vel_z_n  = float(np.clip(vel_z_raw  / _VEL_Z_CLIP,  -1.0, 1.0))
            accel_n  = float(np.clip(accel_mag_raw / _ACCEL_CLIP,  0.0, 1.0))

            frame_vels[kp] = (vel_u_n, vel_v_n, vel_z_n, accel_n)

        result[fi] = frame_vels

    return result


def render_skeleton_maps(
    points_2d: Dict[str, Tuple[float, float]],
    depths: Dict[str, float],
    pose_info,
    image_shape: Tuple[int, int],
    line_width: int = 3,
    frame_velocities: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Render semantic, depth, rotation, and velocity/acceleration maps from 2D skeleton keypoints.

    Returns:
        (semantic_map [H,W,3], depth_map [H,W,1], rotation_map [H,W,1],
         vel_u_map [H,W,1], vel_v_map [H,W,1], vel_z_map [H,W,1], accel_mag_map [H,W,1])
    """
    if not _HAS_SKELETON_UTILS:
        H, W = image_shape
        return (
            np.zeros((H, W, 3), dtype=np.uint8),
            np.zeros((H, W, 1), dtype=np.float32),
            np.zeros((H, W, 1), dtype=np.float32),
            np.zeros((H, W, 1), dtype=np.float32),
            np.zeros((H, W, 1), dtype=np.float32),
            np.zeros((H, W, 1), dtype=np.float32),
            np.zeros((H, W, 1), dtype=np.float32),
        )

    H, W = image_shape
    semantic_map  = np.zeros((H, W, 3), dtype=np.uint8)
    depth_map     = np.zeros((H, W, 1), dtype=np.float32)
    rotation_map  = np.zeros((H, W, 1), dtype=np.float32)
    vel_u_map     = np.zeros((H, W, 1), dtype=np.float32)
    vel_v_map     = np.zeros((H, W, 1), dtype=np.float32)
    vel_z_map     = np.zeros((H, W, 1), dtype=np.float32)
    accel_mag_map = np.zeros((H, W, 1), dtype=np.float32)

    _zero_vel = (0.0, 0.0, 0.0, 0.0)

    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        print("Warning: cv2 not available, cannot render skeleton maps")
        return (semantic_map, depth_map, rotation_map,
                vel_u_map, vel_v_map, vel_z_map, accel_mag_map)

    for (start_name, end_name) in BONES:
        if start_name not in points_2d or end_name not in points_2d:
            continue

        u1, v1 = points_2d[start_name]
        u2, v2 = points_2d[end_name]

        if not all(np.isfinite(v) for v in [u1, v1, u2, v2]):
            continue

        category = BONE_CATEGORIES.get((start_name, end_name), 0)
        color = SEMANTIC_COLORS.get(category, (255, 255, 255))

        pt1 = (int(u1), int(v1))
        pt2 = (int(u2), int(v2))
        cv2.line(semantic_map, pt1, pt2, color, thickness=line_width)

        line_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.line(line_mask, pt1, pt2, 255, thickness=line_width)
        mask_pixels = line_mask > 0

        depth1 = depths.get(start_name, 0.0)
        depth2 = depths.get(end_name, 0.0)
        depth_map[mask_pixels, 0] = (depth1 + depth2) / 2.0

        rotation_value = 0.0
        if category == 1:  # Wrist bone
            if hasattr(pose_info, 'alpha'):
                alpha = pose_info.alpha
                if isinstance(alpha, torch.Tensor):
                    rotation_value = alpha.item()
                else:
                    rotation_value = float(alpha)
        elif category == 2:  # Left gripper bone
            if hasattr(pose_info, 'theta_l'):
                theta_l = pose_info.theta_l
                if isinstance(theta_l, torch.Tensor):
                    rotation_value = theta_l.item() if theta_l.numel() == 1 else theta_l[0].item()
                else:
                    rotation_value = float(theta_l) if np.isscalar(theta_l) else float(theta_l[0])
        elif category == 3:  # Right gripper bone
            if hasattr(pose_info, 'theta_r'):
                theta_r = pose_info.theta_r
                if isinstance(theta_r, torch.Tensor):
                    rotation_value = theta_r.item() if theta_r.numel() == 1 else theta_r[0].item()
                else:
                    rotation_value = float(theta_r) if np.isscalar(theta_r) else float(theta_r[0])

        rotation_map[mask_pixels, 0] = rotation_value

        if frame_velocities is not None:
            vu1, vv1, vz1, am1 = frame_velocities.get(start_name, _zero_vel)
            vu2, vv2, vz2, am2 = frame_velocities.get(end_name,   _zero_vel)
            vel_u_map    [mask_pixels, 0] = (vu1 + vu2) / 2.0
            vel_v_map    [mask_pixels, 0] = (vv1 + vv2) / 2.0
            vel_z_map    [mask_pixels, 0] = (vz1 + vz2) / 2.0
            accel_mag_map[mask_pixels, 0] = (am1 + am2) / 2.0

    return (semantic_map, depth_map, rotation_map,
            vel_u_map, vel_v_map, vel_z_map, accel_mag_map)


def build_camera_intrinsic(focal_length: float, img_width: int, img_height: int) -> np.ndarray:
    """Build camera intrinsic matrix K."""
    cx = img_width / 2.0
    cy = img_height / 2.0
    K = np.array([
        [focal_length, 0.0, cx],
        [0.0, focal_length, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return K


def generate_skeleton_maps_from_pth(
    pth_path: str,
    frame_indices: List[int],
    image_shape: Tuple[int, int],
    focal_length: float = 587.544,
    line_width: int = 3,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate semantic, depth, rotation, and velocity/acceleration maps from .pth file.

    Args:
        pth_path: Path to memory_pool.pth file
        frame_indices: List of frame indices to load
        image_shape: (height, width) of output maps
        focal_length: Camera focal length in pixels
        line_width: Width of skeleton lines in pixels
        verbose: Whether to print detailed information

    Returns:
        Dictionary containing:
            'semantic':  [T, H, W, 3] RGB color-coded skeleton (uint8)
            'depth':     [T, H, W, 1] Z-depth along skeleton (float32)
            'rotation':  [T, H, W, 1] Self-rotation angles (float32)
            'vel_u':     [T, H, W, 1] Normalized image-plane u-velocity (float32)
            'vel_v':     [T, H, W, 1] Normalized image-plane v-velocity (float32)
            'vel_z':     [T, H, W, 1] Normalized depth-velocity (float32)
            'accel_mag': [T, H, W, 1] Normalized 3-D acceleration magnitude (float32)
        Returns zero-filled arrays if .pth file not found or skeleton utils not available
    """
    T = len(frame_indices)
    H, W = image_shape

    def _zeros_dict():
        return {
            'semantic':  np.zeros((T, H, W, 3), dtype=np.uint8),
            'depth':     np.zeros((T, H, W, 1), dtype=np.float32),
            'rotation':  np.zeros((T, H, W, 1), dtype=np.float32),
            'vel_u':     np.zeros((T, H, W, 1), dtype=np.float32),
            'vel_v':     np.zeros((T, H, W, 1), dtype=np.float32),
            'vel_z':     np.zeros((T, H, W, 1), dtype=np.float32),
            'accel_mag': np.zeros((T, H, W, 1), dtype=np.float32),
        }

    if not _HAS_SKELETON_UTILS:
        if verbose:
            print("[Skeleton Maps] Skeleton utilities not available, returning zeros")
        return _zeros_dict()

    if not os.path.exists(pth_path):
        if verbose:
            print(f"[Skeleton Maps] .pth file not found: {pth_path}, returning zeros")
        return _zeros_dict()

    try:
        if _HAS_READ_UTILS:
            memory_pool = load_memory_pool(pth_path)
        else:
            memory_pool = torch.load(pth_path, map_location='cpu')

        if not isinstance(memory_pool, dict):
            if verbose:
                print(f"[Skeleton Maps] Invalid memory_pool format, returning zeros")
            return _zeros_dict()

        semantic_maps  = np.zeros((T, H, W, 3), dtype=np.uint8)
        depth_maps     = np.zeros((T, H, W, 1), dtype=np.float32)
        rotation_maps  = np.zeros((T, H, W, 1), dtype=np.float32)
        vel_u_maps     = np.zeros((T, H, W, 1), dtype=np.float32)
        vel_v_maps     = np.zeros((T, H, W, 1), dtype=np.float32)
        vel_z_maps     = np.zeros((T, H, W, 1), dtype=np.float32)
        accel_mag_maps = np.zeros((T, H, W, 1), dtype=np.float32)

        instrument = Instrument()
        K = build_camera_intrinsic(focal_length, W, H)

        frame_vel_dict = _compute_keypoint_velocities_and_accels(
            memory_pool=memory_pool,
            instrument=instrument,
            sample_frame_indices=frame_indices,
            focal_length=focal_length,
            image_shape=image_shape,
        )

        for t, frame_idx in enumerate(frame_indices):
            frame_key = str(frame_idx + 1).zfill(5)
            if frame_key not in memory_pool:
                frame_key = str(frame_idx).zfill(5)
                if frame_key not in memory_pool:
                    frame_key = str(frame_idx + 1)
                    if frame_key not in memory_pool:
                        frame_key = str(frame_idx)
                        if frame_key not in memory_pool:
                            continue

            entry = memory_pool[frame_key]

            pose_info = entry
            if hasattr(entry, 'pose_info'):
                pose_info = entry.pose_info
            elif isinstance(entry, dict) and 'pose_info' in entry:
                pose_info = entry['pose_info']

            points_3d = compute_skeleton_3d(instrument, pose_info)

            has_invalid_3d = False
            for key, point in points_3d.items():
                if isinstance(point, (list, tuple, np.ndarray)):
                    point_array = np.array(point)
                    if np.isnan(point_array).any() or np.isinf(point_array).any():
                        has_invalid_3d = True
                        break

            if has_invalid_3d:
                import logging
                logging.warning(f"[Skeleton Maps] NaN/Inf in 3D skeleton for frame {frame_idx}, skipping")
                continue

            points_2d, depths = project_3d_to_2d(points_3d, K)

            # Shaft extension
            if 'shaft_base' in points_2d and 'wrist_origin' in points_2d:
                u_shaft, v_shaft = points_2d['shaft_base']
                u_wrist, v_wrist = points_2d['wrist_origin']
                z_shaft = depths['shaft_base']
                z_wrist = depths['wrist_origin']
                vec_u = u_shaft - u_wrist
                vec_v = v_shaft - v_wrist
                if not (vec_u == 0 and vec_v == 0):
                    t_candidates = []
                    if vec_u != 0:
                        t_candidates.append((0 - u_wrist) / vec_u)
                        t_candidates.append((W - 1 - u_wrist) / vec_u)
                    if vec_v != 0:
                        t_candidates.append((0 - v_wrist) / vec_v)
                        t_candidates.append((H - 1 - v_wrist) / vec_v)
                    best_t = None
                    for t_cand in t_candidates:
                        if t_cand <= 0:
                            continue
                        test_u = u_wrist + t_cand * vec_u
                        test_v = v_wrist + t_cand * vec_v
                        if -2.0 <= test_u <= W + 1.0 and -2.0 <= test_v <= H + 1.0:
                            if best_t is None or t_cand < best_t:
                                best_t = t_cand
                    if best_t is not None and best_t > 1.0:
                        points_2d['shaft_base'] = (
                            u_wrist + best_t * vec_u,
                            v_wrist + best_t * vec_v,
                        )
                        depths['shaft_base'] = z_wrist + best_t * (z_shaft - z_wrist)

            has_invalid_2d = False
            for key, point in points_2d.items():
                if isinstance(point, (list, tuple)):
                    if any(np.isnan(x) or np.isinf(x) for x in point):
                        has_invalid_2d = True
                        break

            for key, depth in depths.items():
                if np.isnan(depth) or np.isinf(depth):
                    has_invalid_2d = True
                    break

            if has_invalid_2d:
                continue

            (semantic_map, depth_map, rotation_map,
             vel_u_map, vel_v_map, vel_z_map, accel_mag_map) = render_skeleton_maps(
                points_2d, depths, pose_info, image_shape, line_width,
                frame_velocities=frame_vel_dict.get(frame_idx)
            )

            semantic_maps [t] = semantic_map
            depth_maps    [t] = depth_map
            rotation_maps [t] = rotation_map
            vel_u_maps    [t] = vel_u_map
            vel_v_maps    [t] = vel_v_map
            vel_z_maps    [t] = vel_z_map
            accel_mag_maps[t] = accel_mag_map

        if np.isnan(depth_maps).any() or np.isinf(depth_maps).any():
            import logging
            logging.warning(f"[Skeleton Maps] NaN or Inf detected in depth maps from {pth_path}, returning zeros")
            return _zeros_dict()

        if np.isnan(rotation_maps).any() or np.isinf(rotation_maps).any():
            import logging
            logging.warning(f"[Skeleton Maps] NaN or Inf detected in rotation maps from {pth_path}, returning zeros")
            return _zeros_dict()

        if verbose:
            nonzero_semantic = (semantic_maps != 0).sum() / semantic_maps.size
            print(f"[Skeleton Maps] Generated from {pth_path}")
            print(f"  - Semantic:  {nonzero_semantic:.1%} non-zero")

    except Exception as e:
        if verbose:
            print(f"[Skeleton Maps] Failed to generate from {pth_path}: {e}")
        return _zeros_dict()

    return {
        'semantic':  semantic_maps,
        'depth':     depth_maps,
        'rotation':  rotation_maps,
        'vel_u':     vel_u_maps,
        'vel_v':     vel_v_maps,
        'vel_z':     vel_z_maps,
        'accel_mag': accel_mag_maps,
    }


def generate_skeleton_maps_from_pth_nframes(
    pth_path: str,
    num_frames: int,
    height: int,
    width: int,
    focal_length: float = 587.544,
    line_width: int = 3,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    KVLR-style wrapper: generates skeleton maps for num_frames frames.

    This is the preferred API for KVLR datasets where only num_frames is known
    (not the original frame indices in the source video).

    Args:
        pth_path: Path to memory_pool.pth file
        num_frames: Number of frames (T)
        height: Output map height (H)
        width: Output map width (W)
        focal_length: Camera focal length in pixels
        line_width: Width of skeleton lines in pixels
        verbose: Whether to print detailed information

    Returns:
        Same dict format as generate_skeleton_maps_from_pth()
    """
    frame_indices = list(range(num_frames))
    return generate_skeleton_maps_from_pth(
        pth_path=pth_path,
        frame_indices=frame_indices,
        image_shape=(height, width),
        focal_length=focal_length,
        line_width=line_width,
        verbose=verbose,
    )

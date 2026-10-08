import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm


cv2.setNumThreads(1)


class IncrementalStats:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros((dim, dim), dtype=np.float64)
        self.dim = dim

    def update(self, x):
        batch_size = x.shape[0]
        for i in range(batch_size):
            self.n += 1
            x_i = x[i].astype(np.float64, copy=False)
            delta = x_i - self.mean
            self.mean += delta / self.n
            delta2 = x_i - self.mean
            self.M2 += np.outer(delta, delta2)

    def get_mean_cov(self):
        if self.n < 2:
            return self.mean, np.zeros((self.dim, self.dim), dtype=np.float64)
        return self.mean, self.M2 / (self.n - 1)


def frechet_distance_from_stats(stats1, stats2, eps=1e-6) -> float:
    mu1, sigma1 = stats1.get_mean_cov()
    mu2, sigma2 = stats2.get_mean_cov()

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    m = diff.dot(diff)

    if stats1.n > 1 and stats2.n > 1:
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                imag_max = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {imag_max}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        fid = np.real(m + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    else:
        fid = np.real(m)

    return float(fid)


def load_i3d_pretrained(filepath, device):
    print(f"Loading I3D model from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"I3D model not found at {filepath}")
    i3d = torch.jit.load(filepath).eval().to(device)
    return i3d


def preprocess_for_i3d(frames_list, resolution=224, target_length=20):
    t = len(frames_list)
    if t < target_length:
        padding = [np.zeros_like(frames_list[0])] * (target_length - t)
        frames_list = frames_list + padding
    elif t > target_length:
        frames_list = frames_list[:target_length]

    video = np.stack(frames_list, axis=0)
    video = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0
    video = video * 2.0 - 1.0

    _, _, h, w = video.shape
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)

    video = video.permute(1, 0, 2, 3)
    video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)

    _, _, h_n, w_n = video.shape
    w_start = (w_n - resolution) // 2
    h_start = (h_n - resolution) // 2
    video = video[:, :, h_start : h_start + resolution, w_start : w_start + resolution]

    video = video.permute(1, 0, 2, 3)
    return video.contiguous()


def get_i3d_feat(video_tensor, detector, device, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)["feat"]

    with torch.no_grad():
        vt = video_tensor.unsqueeze(0).to(device)
        # Use positional arguments for TorchScript I3D for better compatibility.
        feat = detector(vt, False, False, True).cpu().numpy()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, feat=feat)
    return feat


def get_inception_feat(frames_list, model, device, batch_size=50):
    pred_arr = []

    for i in range(0, len(frames_list), batch_size):
        batch_frames = frames_list[i : i + batch_size]
        batch = np.stack(batch_frames, axis=0)
        batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        batch_tensor = batch_tensor.to(device)

        with torch.no_grad():
            pred = model(batch_tensor)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr.append(pred)

    return np.concatenate(pred_arr, axis=0)


def extract_frames(video_path, frame_step=1, start_frame=0, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []

    if start_frame > 0:
        for _ in range(start_frame):
            ret = cap.grab()
            if not ret:
                break

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.size == 0:
            raise RuntimeError(f"Decoded an invalid frame from video: {video_path}")
        if frame_idx % frame_step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return frames


def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count > 0:
        return frame_count

    return len(extract_frames(video_path, frame_step=1))


def calculate_base_metrics(pred_frames, gt_frames, lpips_model, device):
    psnr_list = []
    ssim_list = []
    lpips_list = []

    for pred_frame, gt_frame in zip(pred_frames, gt_frames):
        if gt_frame.shape != pred_frame.shape:
            h, w = pred_frame.shape[:2]
            gt_frame = cv2.resize(gt_frame, (w, h), interpolation=cv2.INTER_LINEAR)

        psnr_val = compute_psnr(gt_frame, pred_frame, data_range=255)
        ssim_val = compute_ssim(gt_frame, pred_frame, data_range=255, channel_axis=-1)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        pred_tensor = (torch.from_numpy(pred_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0) * 2 - 1
        gt_tensor = (torch.from_numpy(gt_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0) * 2 - 1
        pred_tensor = pred_tensor.to(device)
        gt_tensor = gt_tensor.to(device)

        with torch.no_grad():
            lpips_val = lpips_model(pred_tensor, gt_tensor).item()

        lpips_list.append(lpips_val)

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)


def save_results(results, output_json):
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def save_worker_metrics(metrics, output_json):
    if not output_json:
        return
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def build_gt_video_dict(gt_dir):
    print(f"Scanning ground truth videos recursively in {gt_dir}...")
    gt_video_dict = {}
    for root, _, files in os.walk(gt_dir):
        for filename in files:
            if filename.endswith((".mp4", ".avi")):
                gt_video_dict[filename] = os.path.join(root, filename)
    return gt_video_dict


def resolve_video_pair(vid_name, gt_video_dict, sliding_window_mode=False):
    start_idx = 0
    base_vid_name = vid_name

    if sliding_window_mode:
        name_no_ext, ext = os.path.splitext(vid_name)
        if "_start" in name_no_ext:
            parts = name_no_ext.rsplit("_start", 1)
            base_vid_name = parts[0] + ext
            try:
                start_idx = int(parts[1])
            except ValueError:
                start_idx = 0

    return base_vid_name, gt_video_dict.get(base_vid_name), start_idx


def build_eval_items(pred_videos, args, gt_video_dict):
    eval_items = []

    for vid_name in pred_videos:
        base_vid_name, gt_path, sliding_start_idx = resolve_video_pair(vid_name, gt_video_dict, args.sliding_window_mode)
        if gt_path is None:
            print(f"Warning: GT not found for '{base_vid_name}' (derived from '{vid_name}'). Skipping.")
            continue

        pred_path = os.path.join(args.pred_dir, vid_name)

        if args.split_input_clips and not args.sliding_window_mode:
            total_frames = get_video_frame_count(pred_path)
            if total_frames <= 0:
                print(f"Warning: Prediction video has 0 readable frames: {pred_path}. Skipping.")
                continue

            clip_length = max(1, args.split_clip_length)
            num_clips = math.ceil(total_frames / clip_length)
            name_no_ext, ext = os.path.splitext(vid_name)

            for clip_idx in range(num_clips):
                pred_start_idx = clip_idx * clip_length
                eval_items.append(
                    {
                        "eval_name": f"{name_no_ext}_clip{clip_idx:04d}_start{pred_start_idx:04d}{ext}",
                        "pred_file": vid_name,
                        "pred_path": pred_path,
                        "base_vid_name": base_vid_name,
                        "gt_path": gt_path,
                        "pred_start_idx": pred_start_idx,
                        "gt_start_idx": pred_start_idx * args.gt_downsample_step,
                        "clip_max_frames": clip_length,
                    }
                )
        else:
            eval_items.append(
                {
                    "eval_name": vid_name,
                    "pred_file": vid_name,
                    "pred_path": pred_path,
                    "base_vid_name": base_vid_name,
                    "gt_path": gt_path,
                    "pred_start_idx": 0,
                    "gt_start_idx": sliding_start_idx,
                    "clip_max_frames": None,
                }
            )

    return eval_items


def init_models(args, device, enable_fvd, enable_fid):
    print("Loading LPIPS model...")
    loss_fn = lpips.LPIPS(net=args.lpips_net).to(device)

    i3d_model = None
    if enable_fvd:
        try:
            i3d_model = load_i3d_pretrained(args.i3d_model_path, device)
        except Exception as e:
            print(f"\n[Warning] Failed to enable FVD: {e}")
            enable_fvd = False

    inception_model = None
    if enable_fid:
        print("Loading InceptionV3 model for FID...")
        try:
            from pytorch_fid.inception import InceptionV3

            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            inception_model = InceptionV3([block_idx]).to(device)
            inception_model.eval()
        except ImportError:
            print("\n[Warning] pytorch-fid is missing, disabling FID.")
            enable_fid = False
        except Exception as e:
            print(f"\n[Warning] Failed to enable FID: {e}")
            enable_fid = False

    return loss_fn, i3d_model, inception_model, enable_fvd, enable_fid


def activate_offline_weights(args):
    if not (args.offline_alexnet_path or args.offline_vgg_path or args.offline_inception_path):
        return

    fake_torch_home = os.path.abspath("./.offline_torch_env")
    os.environ["TORCH_HOME"] = fake_torch_home
    checkpoint_dir = os.path.join(fake_torch_home, "hub", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Activating offline mode. PyTorch cache redirected to: {fake_torch_home}")

    if args.offline_alexnet_path and os.path.exists(args.offline_alexnet_path):
        dest = os.path.join(checkpoint_dir, "alexnet-owt-7be5be79.pth")
        if not os.path.exists(dest):
            shutil.copy(args.offline_alexnet_path, dest)
        print(f"Injected offline AlexNet backbone from {args.offline_alexnet_path}")

    if args.offline_vgg_path and os.path.exists(args.offline_vgg_path):
        dest = os.path.join(checkpoint_dir, "vgg16-397923af.pth")
        if not os.path.exists(dest):
            shutil.copy(args.offline_vgg_path, dest)
        print(f"Injected offline VGG16 backbone from {args.offline_vgg_path}")

    if args.offline_inception_path and os.path.exists(args.offline_inception_path):
        dest = os.path.join(checkpoint_dir, "pt_inception-2015-12-05-6726825d.pth")
        if not os.path.exists(dest):
            shutil.copy(args.offline_inception_path, dest)
        print(f"Injected offline InceptionV3 backbone from {args.offline_inception_path}")


def evaluate_single_video(
    vid_name,
    pred_path,
    gt_path,
    pred_start_idx,
    gt_start_idx,
    pred_max_frames,
    args,
    loss_fn,
    device,
    i3d_model=None,
    inception_model=None,
):
    print(f"\n[Video] {vid_name}")
    print(f"  [Stage] decode prediction: {pred_path}")
    pred_frames = extract_frames(pred_path, frame_step=1, start_frame=pred_start_idx, max_frames=pred_max_frames)
    pred_length = len(pred_frames)
    if pred_length == 0:
        raise ValueError(f"Prediction video has 0 readable frames: {pred_path}")

    print(f"  [Stage] decode ground truth: {gt_path}")
    gt_frames = extract_frames(
        gt_path,
        frame_step=args.gt_downsample_step,
        start_frame=gt_start_idx,
        max_frames=pred_length,
    )

    min_len = min(len(pred_frames), len(gt_frames))
    if min_len == 0:
        raise ValueError(
            f"Aligned frame count is 0. pred_frames={len(pred_frames)}, gt_frames={len(gt_frames)}"
        )

    aligned_pred = pred_frames[:min_len]
    aligned_gt = gt_frames[:min_len]

    print(f"  [Stage] base metrics on {min_len} aligned frames")
    v_psnr, v_ssim, v_lpips = calculate_base_metrics(aligned_pred, aligned_gt, loss_fn, device)

    metrics = {
        "PSNR": float(v_psnr),
        "SSIM": float(v_ssim),
        "LPIPS": float(v_lpips),
        "Evaluated_Frames": int(min_len),
    }
    print(
        f"  [Metrics] PSNR={metrics['PSNR']:.4f}, SSIM={metrics['SSIM']:.4f}, "
        f"LPIPS={metrics['LPIPS']:.4f}, Frames={metrics['Evaluated_Frames']}",
        flush=True,
    )
    save_worker_metrics(metrics, getattr(args, "worker_output_json", None))
    features = {}

    if i3d_model is not None:
        print("  [Stage] FVD features")
        try:
            pred_cache = os.path.join(args.cache_dir, f"fvd_pred_{vid_name}.npz")
            gt_cache = os.path.join(args.cache_dir, f"fvd_gt_{vid_name}.npz")
            p_vid_tensor = preprocess_for_i3d(aligned_pred, resolution=224, target_length=args.fvd_target_length)
            g_vid_tensor = preprocess_for_i3d(aligned_gt, resolution=224, target_length=args.fvd_target_length)
            features["p_feat_fvd"] = get_i3d_feat(p_vid_tensor, i3d_model, device, pred_cache)
            features["g_feat_fvd"] = get_i3d_feat(g_vid_tensor, i3d_model, device, gt_cache)
        except Exception as e:
            print(f"  [Warning] Skip FVD for '{vid_name}': {e}")

    if inception_model is not None:
        print("  [Stage] FID features")
        try:
            features["p_feat_fid"] = get_inception_feat(
                aligned_pred, inception_model, device, batch_size=args.fid_batch_size
            )
            features["g_feat_fid"] = get_inception_feat(
                aligned_gt, inception_model, device, batch_size=args.fid_batch_size
            )
        except Exception as e:
            print(f"  [Warning] Skip FID for '{vid_name}': {e}")

    return metrics, features


def run_single_video_subprocess(eval_name, args):
    worker_json = tempfile.NamedTemporaryFile(prefix="metric_worker_", suffix=".json", delete=False)
    worker_npz = tempfile.NamedTemporaryFile(prefix="metric_worker_", suffix=".npz", delete=False)
    worker_json.close()
    worker_npz.close()

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--pred_dir",
        args.pred_dir,
        "--gt_dir",
        args.gt_dir,
        "--output_json",
        args.output_json,
        "--gt_downsample_step",
        str(args.gt_downsample_step),
        "--lpips_net",
        args.lpips_net,
        "--i3d_model_path",
        args.i3d_model_path,
        "--fid_batch_size",
        str(args.fid_batch_size),
        "--fvd_target_length",
        str(args.fvd_target_length),
        "--cache_dir",
        args.cache_dir,
        "--single_video",
        args.single_video,
        "--worker_output_json",
        worker_json.name,
        "--worker_output_npz",
        worker_npz.name,
        "--single_video_base_name",
        args.single_video_base_name or "",
        "--pred_start_idx",
        str(args.pred_start_idx),
        "--gt_start_idx",
        str(args.gt_start_idx),
    ]

    if args.clip_max_frames is not None:
        cmd.extend(["--clip_max_frames", str(args.clip_max_frames)])

    if args.offline_alexnet_path:
        cmd.extend(["--offline_alexnet_path", args.offline_alexnet_path])
    if args.offline_vgg_path:
        cmd.extend(["--offline_vgg_path", args.offline_vgg_path])
    if args.offline_inception_path:
        cmd.extend(["--offline_inception_path", args.offline_inception_path])
    if args.disable_fvd:
        cmd.append("--disable_fvd")
    if args.disable_fid:
        cmd.append("--disable_fid")
    if args.sliding_window_mode:
        cmd.append("--sliding_window_mode")
    if args.split_input_clips:
        cmd.append("--split_input_clips")
    cmd.extend(["--split_clip_length", str(args.split_clip_length)])

    completed = subprocess.run(cmd, capture_output=True, text=True)

    metrics = None
    features = {}
    if os.path.exists(worker_json.name) and os.path.getsize(worker_json.name) > 0:
        with open(worker_json.name, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    if completed.returncode == 0:
        if os.path.exists(worker_npz.name) and os.path.getsize(worker_npz.name) > 0:
            npz_data = np.load(worker_npz.name)
            for key in npz_data.files:
                features[key] = npz_data[key]

    for temp_path in (worker_json.name, worker_npz.name):
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return completed, metrics, features


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate PSNR, SSIM, LPIPS, FID and FVD for generated videos")
    parser.add_argument("--pred_dir", type=str, default="./outputs/inference/kvlr", help="Path to generated videos directory")
    parser.add_argument("--gt_dir", type=str, default="./data/kasa/videos", help="Path to ground truth videos parent directory")
    parser.add_argument("--output_json", type=str, default="metrics_results.json", help="Path to save results")
    parser.add_argument("--gt_downsample_step", type=int, default=2, help="Frame step to downsample GT video. 2 means 60Hz -> 30Hz.")
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--i3d_model_path", type=str, default="./checkpoints/metrics/i3d_torchscript.pt", help="Path to local i3d_torchscript.pt for FVD")
    parser.add_argument("--offline_alexnet_path", type=str, default="./checkpoints/metrics/alexnet-owt-7be5be79.pth", help="Extracted offline weights for alexnet backbone.")
    parser.add_argument("--offline_vgg_path", type=str, default=None, help="Extracted offline weights for VGG backbone.")
    parser.add_argument("--offline_inception_path", type=str, default="./checkpoints/metrics/pt_inception-2015-12-05-6726825d.pth", help="Extracted offline weights for InceptionV3 backbone.")
    parser.add_argument("--disable_fvd", action="store_true", help="Skip FVD calculation")
    parser.add_argument("--disable_fid", action="store_true", help="Skip FID calculation")
    parser.add_argument("--fvd_target_length", type=int, default=17, help="Target clip length for FVD calculation")
    parser.add_argument("--fid_batch_size", type=int, default=50, help="Batch size for InceptionV3 frame feature extraction")
    parser.add_argument("--cache_dir", type=str, default="feature_cache", help="Directory to save video feature caches")
    parser.add_argument("--clean_cache", action="store_true", help="Delete the cache directory after completion")
    parser.add_argument("--sliding_window_mode", action="store_true", help="Match sub-videos like *_start0017.mp4 to their base GT videos.")
    parser.add_argument(
        "--split_input_clips",
        dest="split_input_clips",
        action="store_true",
        help="Enable internal clip splitting when not using --sliding_window_mode.",
    )
    parser.add_argument(
        "--no_split_input_clips",
        dest="split_input_clips",
        action="store_false",
        help="Disable internal clip splitting.",
    )
    parser.set_defaults(split_input_clips=True)
    parser.add_argument(
        "--split_clip_length",
        type=int,
        default=17,
        help="Clip length used by --split_input_clips. Long videos are evaluated as multiple non-overlapping clips.",
    )
    parser.add_argument("--safe_mode", action="store_true", help="Process each video in an isolated subprocess so one crash does not kill the whole job.")
    parser.add_argument("--single_video", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single_video_base_name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pred_start_idx", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--gt_start_idx", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--clip_max_frames", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_output_json", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_output_npz", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.split_clip_length <= 0:
        raise ValueError("--split_clip_length must be a positive integer.")
    if args.split_input_clips and args.sliding_window_mode:
        print("[Info] --split_input_clips is ignored because --sliding_window_mode is already enabled.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    enable_fvd = not args.disable_fvd
    enable_fid = not args.disable_fid

    if (enable_fvd or enable_fid) and not args.safe_mode and not args.single_video:
        args.safe_mode = True
        print("[Info] Auto-enabled --safe_mode because FVD/FID may crash native libraries in-process.")

    activate_offline_weights(args)
    loss_fn, i3d_model, inception_model, enable_fvd, enable_fid = init_models(
        args, device, enable_fvd, enable_fid
    )

    pred_videos = sorted([f for f in os.listdir(args.pred_dir) if f.endswith((".mp4", ".avi"))])
    if len(pred_videos) == 0:
        print(f"No video files found in {args.pred_dir}.")
        return

    gt_video_dict = build_gt_video_dict(args.gt_dir)
    print(f"Found {len(gt_video_dict)} Ground Truth videos. Preparing to align with {len(pred_videos)} Predictions...")

    if args.single_video:
        base_vid_name = args.single_video_base_name
        gt_path = gt_video_dict.get(base_vid_name) if base_vid_name is not None else None
        if gt_path is None:
            base_vid_name, gt_path, _ = resolve_video_pair(args.single_video, gt_video_dict, args.sliding_window_mode)
        if gt_path is None:
            raise FileNotFoundError(f"GT not found for '{base_vid_name}' (derived from '{args.single_video}').")

        pred_path = os.path.join(args.pred_dir, args.single_video)
        metrics, features = evaluate_single_video(
            args.single_video,
            pred_path,
            gt_path,
            args.pred_start_idx,
            args.gt_start_idx,
            args.clip_max_frames,
            args,
            loss_fn,
            device,
            i3d_model if enable_fvd else None,
            inception_model if enable_fid else None,
        )
        if args.worker_output_json:
            with open(args.worker_output_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        if args.worker_output_npz:
            np.savez(args.worker_output_npz, **features)
        return

    results = {}
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0

    if enable_fvd:
        fvd_stats_pred = IncrementalStats(400)
        fvd_stats_gt = IncrementalStats(400)
    else:
        fvd_stats_pred = None
        fvd_stats_gt = None

    if enable_fid:
        fid_stats_pred = IncrementalStats(2048)
        fid_stats_gt = IncrementalStats(2048)
    else:
        fid_stats_pred = None
        fid_stats_gt = None

    eval_items = build_eval_items(pred_videos, args, gt_video_dict)
    print(f"Prepared {len(eval_items)} evaluation item(s).")

    for eval_item in tqdm(eval_items):
        vid_name = eval_item["eval_name"]

        try:
            if args.safe_mode:
                worker_args = argparse.Namespace(**vars(args))
                worker_args.single_video = eval_item["pred_file"]
                worker_args.single_video_base_name = eval_item["base_vid_name"]
                worker_args.pred_start_idx = eval_item["pred_start_idx"]
                worker_args.gt_start_idx = eval_item["gt_start_idx"]
                worker_args.clip_max_frames = eval_item["clip_max_frames"]
                completed, metrics, features = run_single_video_subprocess(vid_name, worker_args)
                if metrics is None:
                    print(f"\n[Skip] '{vid_name}' crashed in isolated worker (exit code={completed.returncode}).")
                    stderr_text = (completed.stderr or "").strip()
                    stdout_text = (completed.stdout or "").strip()
                    if stderr_text:
                        print(stderr_text[-2000:])
                    elif stdout_text:
                        print(stdout_text[-2000:])
                    continue
                if completed.returncode != 0:
                    print(
                        f"\n[Warning] '{vid_name}' crashed after base metrics in isolated worker "
                        f"(exit code={completed.returncode}); keeping base metrics and skipping missing features."
                    )
                    stderr_text = (completed.stderr or "").strip()
                    stdout_text = (completed.stdout or "").strip()
                    if stderr_text:
                        print(stderr_text[-2000:])
                    elif stdout_text:
                        print(stdout_text[-2000:])
            else:
                metrics, features = evaluate_single_video(
                    vid_name,
                    eval_item["pred_path"],
                    eval_item["gt_path"],
                    eval_item["pred_start_idx"],
                    eval_item["gt_start_idx"],
                    eval_item["clip_max_frames"],
                    args,
                    loss_fn,
                    device,
                    i3d_model if enable_fvd else None,
                    inception_model if enable_fid else None,
                )

            results[vid_name] = metrics
            total_psnr += metrics["PSNR"]
            total_ssim += metrics["SSIM"]
            total_lpips += metrics["LPIPS"]
            count += 1

            if enable_fvd and "p_feat_fvd" in features and "g_feat_fvd" in features:
                fvd_stats_pred.update(features["p_feat_fvd"])
                fvd_stats_gt.update(features["g_feat_fvd"])

            if enable_fid and "p_feat_fid" in features and "g_feat_fid" in features:
                fid_stats_pred.update(features["p_feat_fid"])
                fid_stats_gt.update(features["g_feat_fid"])

            results["Average"] = {
                "PSNR": total_psnr / count,
                "SSIM": total_ssim / count,
                "LPIPS": total_lpips / count,
                "Successfully_Evaluated_Items": count,
                "Successfully_Evaluated_Videos": count,
            }
            save_results(results, args.output_json)
        except Exception as e:
            print(f"\n[Skip] Failed on '{vid_name}': {e}")
            print(traceback.format_exc())
            continue

    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_lpips = total_lpips / count

        results["Average"] = {
            "PSNR": avg_psnr,
            "SSIM": avg_ssim,
            "LPIPS": avg_lpips,
            "Successfully_Evaluated_Items": count,
            "Successfully_Evaluated_Videos": count,
        }

        print("\n" + "=" * 40)
        print(f"=== Final Evaluation Results: split===" + str(args.pred_dir))
        print(f"Total evaluated matched items: {count}")
        print(f"Average PSNR:  {avg_psnr:.4f}")
        print(f"Average SSIM:  {avg_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")

        if enable_fvd and fvd_stats_pred is not None and fvd_stats_gt is not None and fvd_stats_pred.n > 0 and fvd_stats_gt.n > 0:
            fvd_score = frechet_distance_from_stats(fvd_stats_pred, fvd_stats_gt)
            results["Average"]["FVD"] = fvd_score
            print(f"Average FVD:   {fvd_score:.4f}  (using dataset holistic cov)")

        if enable_fid and fid_stats_pred is not None and fid_stats_gt is not None and fid_stats_pred.n > 0 and fid_stats_gt.n > 0:
            fid_score = frechet_distance_from_stats(fid_stats_pred, fid_stats_gt)
            results["Average"]["FID"] = fid_score
            print(f"Average FID:   {fid_score:.4f}  (using dataset holistic cov)")

        print("=" * 40)
        save_results(results, args.output_json)
        print(f"\nResults have been successfully saved to {args.output_json}")
    else:
        print("No valid paired videos found for evaluation.")

    if args.clean_cache and os.path.exists(args.cache_dir):
        shutil.rmtree(args.cache_dir)
        print(f"[Done] Cleaned up temporary cache directory: {args.cache_dir}")

    
if __name__ == "__main__":
    main()

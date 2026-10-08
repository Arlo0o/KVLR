import logging
import os
import sys
import types
import random

import numpy as np
import pandas as pd
import torch
from PIL import ImageFile
from torchvision.datasets.folder import pil_loader

from KVLR.registry import DATASETS

from .read_video import read_video
from .utils import (
    get_transforms_image,
    get_transforms_video,
    is_img,
    is_valid_video_file,
    map_target_fps,
    read_file,
    temporal_random_crop,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_KEYS = ("neg", "path")
K = 10000


class Iloc:
    def __init__(self, data, sharded_folder, sharded_folders, rows_per_shard):
        self.data = data
        self.sharded_folder = sharded_folder
        self.sharded_folders = sharded_folders
        self.rows_per_shard = rows_per_shard

    def __getitem__(self, index):
        return Item(
            index,
            self.data,
            self.sharded_folder,
            self.sharded_folders,
            self.rows_per_shard,
        )


class Item:
    def __init__(self, index, data, sharded_folder, sharded_folders, rows_per_shard):
        self.index = index
        self.data = data
        self.sharded_folder = sharded_folder
        self.sharded_folders = sharded_folders
        self.rows_per_shard = rows_per_shard

    def __getitem__(self, key):
        index = self.index
        if key in self.data.columns:
            return self.data[key].iloc[index]
        else:
            shard_idx = index // self.rows_per_shard
            idx = index % self.rows_per_shard
            shard_parquet = os.path.join(self.sharded_folder, self.sharded_folders[shard_idx])
            try:
                text_parquet = pd.read_parquet(shard_parquet, engine="fastparquet")
                path = text_parquet["path"].iloc[idx]
                assert path == self.data["path"].iloc[index]
            except Exception as e:
                print(f"Error reading {shard_parquet}: {e}")
                raise
            return text_parquet[key].iloc[idx]

    def to_dict(self):
        index = self.index
        ret = {}
        ret.update(self.data.iloc[index].to_dict())
        shard_idx = index // self.rows_per_shard
        idx = index % self.rows_per_shard
        shard_parquet = os.path.join(self.sharded_folder, self.sharded_folders[shard_idx])
        try:
            text_parquet = pd.read_parquet(shard_parquet, engine="fastparquet")
            path = text_parquet["path"].iloc[idx]
            assert path == self.data["path"].iloc[index]
            ret.update(text_parquet.iloc[idx].to_dict())
        except Exception as e:
            print(f"Error reading {shard_parquet}: {e}")
            ret.update({"text": ""})
        return ret


class EfficientParquet:
    def __init__(self, df, sharded_folder):
        self.data = df
        self.total_rows = len(df)
        self.rows_per_shard = (self.total_rows + K - 1) // K
        self.sharded_folder = sharded_folder
        assert os.path.exists(sharded_folder), f"Sharded folder {sharded_folder} does not exist."
        self.sharded_folders = os.listdir(sharded_folder)
        self.sharded_folders = sorted(self.sharded_folders)

    def __len__(self):
        return self.total_rows

    @property
    def iloc(self):
        return Iloc(self.data, self.sharded_folder, self.sharded_folders, self.rows_per_shard)


@DATASETS.register_module("text")
class TextDataset(torch.utils.data.Dataset):
    """
    Dataset for text data
    """

    def __init__(
        self,
        data_path: str = None,
        tokenize_fn: callable = None,
        fps_max: int = 16,
        vmaf: bool = False,
        memory_efficient: bool = False,
        **kwargs,
    ):
        self.data_path = data_path
        self.data = read_file(data_path, memory_efficient=memory_efficient)
        self.memory_efficient = memory_efficient
        self.tokenize_fn = tokenize_fn
        self.vmaf = vmaf

        if fps_max is not None:
            self.fps_max = fps_max
        else:
            self.fps_max = 999999999

    def to_efficient(self):
        if self.memory_efficient:
            addition_data_path = self.data_path.split(".")[0]
            self._data = self.data
            self.data = EfficientParquet(self._data, addition_data_path)

    def getitem(self, index: int) -> dict:
        ret = dict()
        sample = self.data.iloc[index].to_dict()
        sample_fps = sample.get("fps", np.nan)
        new_fps, sampling_interval = map_target_fps(sample_fps, self.fps_max)
        ret.update({"sampling_interval": sampling_interval})

        if "text" in sample:
            ret["text"] = sample.pop("text")
            postfixs = []
            if new_fps != 0 and self.fps_max < 999:
                postfixs.append(f"{new_fps} FPS")
            if self.vmaf and "score_vmafmotion" in sample and not np.isnan(sample["score_vmafmotion"]):
                postfixs.append(f"{int(sample['score_vmafmotion'] + 0.5)} motion score")
            postfix = " " + ", ".join(postfixs) + "." if postfixs else ""
            ret["text"] = ret["text"] + postfix
            if self.tokenize_fn is not None:
                ret.update({k: v.squeeze(0) for k, v in self.tokenize_fn(ret["text"]).items()})

        if "ref" in sample:  # i2v & v2v reference
            ret["ref"] = sample.pop("ref")

        # name of the generated sample
        if "name" in sample:  # sample name (`dataset_idx`)
            ret["name"] = sample.pop("name")
        else:
            ret["index"] = index  # use index for name
        valid_sample = {k: v for k, v in sample.items() if k in VALID_KEYS}
        ret.update(valid_sample)
        return ret

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.data)


@DATASETS.register_module("video_text")
class VideoTextDataset(TextDataset):
    def __init__(
        self,
        transform_name: str = None,
        bucket_class: str = "Bucket",
        rand_sample_interval: int = None,  # random sample_interval value from [1, min(rand_sample_interval, video_allowed_max)]
        validate_videos: bool = False,
        validation_num_probe_frames: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transform_name = transform_name
        self.bucket_class = bucket_class
        self.rand_sample_interval = rand_sample_interval
        self.validate_videos = validate_videos
        self.validation_num_probe_frames = max(1, validation_num_probe_frames)
        self._video_valid_cache = {}
        if self.validate_videos:
            self._filter_invalid_entries()

    def _filter_invalid_entries(self) -> None:
        valid_flags = []
        dropped = 0
        for _, sample in self.data.iterrows():
            path = sample.get("path")
            if not isinstance(path, str):
                valid_flags.append(False)
                dropped += 1
                continue
            if is_img(path):
                valid = os.path.exists(path)
            else:
                valid = self._is_video_valid(path, refresh=True)
            valid_flags.append(valid)
            if not valid:
                dropped += 1
        if dropped > 0:
            print(f"[VideoTextDataset] Removed {dropped} invalid samples before training")
        self.data = self.data[np.array(valid_flags, dtype=bool)].reset_index(drop=True)

    def get_image(self, index: int, height: int, width: int) -> dict:
        sample = self.data.iloc[index]
        path = sample["path"]
        # loading
        image = pil_loader(path)

        # transform
        transform = get_transforms_image(self.transform_name, (height, width))
        image = transform(image)

        # CHW -> CTHW
        video = image.unsqueeze(1)

        return {"video": video}

    def get_video(self, index: int, num_frames: int, height: int, width: int, sampling_interval: int) -> dict:
        sample = self.data.iloc[index]
        path = sample["path"]

        # loading
        vframes, vinfo = read_video(path, backend="av")

        if self.rand_sample_interval is not None:
            # randomly sample from 1 - self.rand_sample_interval
            video_allowed_max = min(len(vframes) // num_frames, self.rand_sample_interval)
            sampling_interval = random.randint(1, video_allowed_max)

        # Sampling video frames
        video = temporal_random_crop(vframes, num_frames, sampling_interval)

        video = video.clone()
        del vframes

        # transform
        transform = get_transforms_video(self.transform_name, (height, width))
        video = transform(video)  # T C H W
        video = video.permute(1, 0, 2, 3)

        ret = {"video": video}

        return ret

    def get_image_or_video(self, index: int, num_frames: int, height: int, width: int, sampling_interval: int) -> dict:
        sample = self.data.iloc[index]
        path = sample["path"]

        if is_img(path):
            return self.get_image(index, height, width)
        if not self._is_video_valid(path):
            raise ValueError(f"Corrupted or unreadable video encountered: {path}")
        return self.get_video(index, num_frames, height, width, sampling_interval)

    def _is_video_valid(self, path: str, refresh: bool = False) -> bool:
        cached = None if refresh else self._video_valid_cache.get(path)
        if cached is not None:
            return cached
        valid = is_valid_video_file(path, num_probe_frames=self.validation_num_probe_frames)
        self._video_valid_cache[path] = valid
        if not valid:
            print(f"[VideoTextDataset] Skipping invalid video: {path}")
        return valid

    def getitem(self, index: str) -> dict:
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]
        ret = dict()
        ret.update(super().getitem(index))
        try:
            ret.update(self.get_image_or_video(index, num_frames, height, width, ret["sampling_interval"]))
        except Exception as e:
            path = self.data.iloc[index]["path"]
            print(f"video {path}: {e}")
            return None
        return ret

    def __getitem__(self, index):
        return self.getitem(index)


@DATASETS.register_module("cached_video_text")
class CachedVideoTextDataset(VideoTextDataset):
    def __init__(
        self,
        transform_name: str = None,
        bucket_class: str = "Bucket",
        rand_sample_interval: int = None,  # random sample_interval value from [1, min(rand_sample_interval, video_allowed_max)]
        cached_video: bool = False,
        cached_text: bool = False,
        return_latents_path: bool = False,
        load_original_video: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transform_name = transform_name
        self.bucket_class = bucket_class
        self.rand_sample_interval = rand_sample_interval
        self.cached_video = cached_video
        self.cached_text = cached_text
        self.return_latents_path = return_latents_path
        self.load_original_video = load_original_video

    def get_latents(self, path):
        try:
            latents = torch.load(path, map_location=torch.device("cpu"))
        except Exception as e:
            print(f"Error loading latents from {path}: {e}")
            return torch.zeros_like(torch.randn(1, 1, 1, 1))
        return latents

    def get_conditioning_latents(self, index: int) -> dict:
        sample = self.data.iloc[index]
        latents_path = sample["latents_path"]
        text_t5_path = sample["text_t5_path"]
        text_clip_path = sample["text_clip_path"]
        res = dict()
        if self.cached_video:
            latents = self.get_latents(latents_path)
            res["video_latents"] = latents
        if self.cached_text:
            text_t5 = self.get_latents(text_t5_path)
            text_clip = self.get_latents(text_clip_path)
            res["text_t5"] = text_t5
            res["text_clip"] = text_clip
        if self.return_latents_path:
            res["latents_path"] = latents_path
            res["text_t5_path"] = text_t5_path
            res["text_clip_path"] = text_clip_path
        return res

    def getitem(self, index: str) -> dict:
        # a hack to pass in the (time, height, width) info from sampler
        real_index, num_frames, height, width = [int(val) for val in index.split("-")]
        ret = dict()
        if self.load_original_video:
            ret.update(super().getitem(index))
        try:
            ret.update(self.get_conditioning_latents(real_index))
        except Exception as e:
            path = self.data.iloc[real_index]["path"]
            print(f"video {path}: {e}")
            return None
        return ret

    def __getitem__(self, index):
        return self.getitem(index)


@DATASETS.register_module("surgical_action_video_text")
class SurgicalActionVideoTextDataset(VideoTextDataset):
    """
    VideoTextDataset extended with action conditioning (skeleton maps) support.

    For each video, tries to find a corresponding memory_pool.pth file and
    generate 9-channel skeleton maps (semantic/depth/rotation/vel_u/vel_v/vel_z/accel_mag).

    Frame alignment for 60fps video + 30fps PTH annotations
    --------------------------------------------------------
    The source surgical videos are 60fps; PTH annotations are sampled at 30fps.
    PTH key '00001' → video frame 0, '00002' → video frame 2, …  (every-other-frame pattern).
    VideoTextDataset.get_video() uses temporal_random_crop() which picks a RANDOM start offset
    in the 60fps video and uses sampling_interval=2 to down-sample to 30fps.  We must therefore
    track the exact 60fps frame indices loaded and convert them to PTH 0-based frame indices
    (pth_idx = video_frame_idx // sampling_interval) so the skeleton maps match the video clip.

    If no pth file is found, the sample is returned without skeleton_maps
    (graceful degradation for mixed datasets).

    Args:
        pth_search_dirs: List of directories to search for memory_pool.pth files
        focal_length: Camera focal length in pixels for 3D→2D projection
        line_width: Skeleton line width in pixels for rendering
        action_dropout: Probability of dropping skeleton maps for a whole sample (batch-level CFG)
        **kwargs: Forwarded to VideoTextDataset
    """

    def __init__(
        self,
        pth_search_dirs: list = None,
        focal_length: float = 587.544,
        line_width: int = 3,
        action_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pth_search_dirs = pth_search_dirs or []
        self.focal_length = focal_length
        self.line_width = line_width
        self.action_dropout = action_dropout

        # Per-call state set by our get_video() override; safe in DataLoader workers
        # (each worker is a separate process with its own copy of the dataset instance).
        self._last_video_frame_indices = None   # np.ndarray of 60fps frame indices
        self._last_video_sampling_interval = 1  # sampling_interval used by temporal_random_crop

        # Pickle compatibility: register Pose to __main__ so torch.load of PTH files works
        try:
            from KVLR.utils.action_utils import _HAS_READ_UTILS
            if _HAS_READ_UTILS:
                from KVLR.utils.action_utils import Pose as _Pose
                if '__main__' not in sys.modules:
                    sys.modules['__main__'] = types.ModuleType('__main__')
                sys.modules['__main__'].Pose = _Pose
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Override get_video() to track the actual 60fps frame indices used
    # by temporal_random_crop so that we can align PTH annotations.
    # ------------------------------------------------------------------
    def get_video(self, index: int, num_frames: int, height: int, width: int, sampling_interval: int) -> dict:
        """Load video frames and record which 60fps frame indices were selected."""
        sample = self.data.iloc[index]
        path = sample["path"]

        vframes, vinfo = read_video(path, backend="av")

        if self.rand_sample_interval is not None:
            video_allowed_max = min(len(vframes) // num_frames, self.rand_sample_interval)
            sampling_interval = random.randint(1, video_allowed_max)

        # return_frame_indices=True gives us the actual 60fps indices chosen
        video, frame_indices = temporal_random_crop(
            vframes, num_frames, sampling_interval, return_frame_indices=True
        )

        # Store for PTH alignment in getitem() (set before any exception can occur)
        self._last_video_frame_indices = frame_indices        # np.ndarray, 60fps space
        self._last_video_sampling_interval = sampling_interval

        video = video.clone()
        del vframes

        transform = get_transforms_video(self.transform_name, (height, width))
        if transform is not None:
            video = transform(video)   # T C H W
        video = video.permute(1, 0, 2, 3)  # → C T H W

        return {"video": video}

    # ------------------------------------------------------------------
    # getitem: load video (via parent → our get_video), then attach
    # skeleton maps using the frame-aligned PTH lookup.
    # ------------------------------------------------------------------
    def getitem(self, index: str | int) -> dict:
        # Reset per-call state so stale indices are never used
        self._last_video_frame_indices = None
        self._last_video_sampling_interval = 1

        if isinstance(index, int):
            # Bypass VideoTextDataset's sampler string-pack hack completely for offline inference.
            # Directly execute TextDataset's logic to fetch pandas row metadata.
            ret = TextDataset.getitem(self, index)
            if ret is None:
                return None
            parts = [str(index)]
            real_index = index
        else:
            # Call parent to get video + text; our get_video() override fires here
            ret = super().getitem(index)
            if ret is None:
                return None

            # Parse real sample index from "real_idx-num_frames-height-width"
            try:
                parts = index.split("-")
                real_index = int(parts[0])
            except (ValueError, IndexError):
                return ret

        try:
            sample = self.data.iloc[real_index]
            # pandas Series supports .get() since pandas ≥ 0.21
            video_path = sample.get("path", "") if hasattr(sample, "get") else sample["path"]

            from KVLR.utils.action_utils import find_pth_file, generate_skeleton_maps_from_pth

            # 1. Locate the PTH file -------------------------------------------------
            pth_path = None
            try:
                pth_col = sample["pth_path"]
                if pth_col and os.path.isfile(str(pth_col)):
                    pth_path = str(pth_col)
            except (KeyError, TypeError):
                pass

            if pth_path is None and video_path:
                pth_path = find_pth_file(video_path, self.pth_search_dirs)
            # print( "########## pth_path", pth_path, video_path, self.pth_search_dirs  )


            if pth_path is None:
                return ret

            # 2. Determine output spatial resolution from loaded video ----------------
            vid = ret.get("video", None)
            if vid is not None:
                # video tensor shape: [C, T, H, W]
                actual_height = vid.shape[2]
                actual_width  = vid.shape[3]
            else:
                actual_height = int(parts[2]) if len(parts) > 2 else 768
                actual_width  = int(parts[3]) if len(parts) > 3 else 768

            # 3. Compute PTH frame indices from tracked 60fps video frame indices ----
            #
            # PTH annotations are at 30fps; source video is 60fps.
            # sampling_interval=2 means every-other 60fps frame was selected.
            # Mapping:  pth_frame_idx (0-based) = video_frame_idx // sampling_interval
            # PTH file key: str(pth_frame_idx + 1).zfill(5)  (1-indexed)
            #
            # Example (sampling_interval=2):
            #   video frame [40, 42, 44, …, 104]  →  pth idx [20, 21, 22, …, 52]
            #   → PTH keys '00021', '00022', …, '00053'
            video_frame_indices  = self._last_video_frame_indices
            sampling_interval    = self._last_video_sampling_interval

            if video_frame_indices is not None and sampling_interval > 0:
                pth_frame_indices = [int(fi // sampling_interval) for fi in video_frame_indices]
            else:
                # Fallback: use sequential indices from 0.
                # This is only reached if get_video() was NOT called (e.g., image sample).
                num_frames = vid.shape[1] if vid is not None else (int(parts[1]) if len(parts) > 1 else 33)
                start_video_frame = 0
                ref_path = sample.get("ref", "")
                if isinstance(ref_path, str) and "frame_" in ref_path:
                    import re
                    match = re.search(r"frame_(\d+)", ref_path)
                    if match:
                        start_video_frame = int(match.group(1))
                
                interval = 2 # 60fps video -> 30fps pth
                start_pth_frame = start_video_frame // interval
                pth_frame_indices = [start_pth_frame + i for i in range(num_frames)]

            # 4. Generate aligned skeleton maps -------------------------------------
            # NOTE: Do NOT use ThreadPoolExecutor here.
            # DataLoader workers are forked processes on Linux. Spawning a thread
            # inside a forked process that then calls cv2 (which uses OpenMP/TBB
            # thread pools) causes a deadlock: the child inherits the parent's
            # mutex state but not its threads, so cv2's internal lock is never
            # released. The ThreadPoolExecutor.submit().result(timeout=60) would
            # then always time out, stalling every sample for 60 s.
            # Direct call is safe — if it hangs, the DataLoader worker watchdog
            # will kill and restart the worker process automatically.
            skeleton_maps = generate_skeleton_maps_from_pth(
                pth_path=pth_path,
                frame_indices=pth_frame_indices,
                image_shape=(actual_height, actual_width),
                focal_length=self.focal_length,
                line_width=self.line_width,
            )
 

            # 5. NaN/Inf guard (learned from Wan codebase) --------------------------
            has_nan_inf = False
            for k, v in skeleton_maps.items():
                if np.isnan(v).any() or np.isinf(v).any():
                    has_nan_inf = True
                    break

            if has_nan_inf:
                raise RuntimeError(f"NaN/Inf in skeleton_maps for {pth_path}")

            # 6. Convert to float32 tensors -----------------------------------------
            skeleton_maps_tensor = {}
            for k, v in skeleton_maps.items():
                skeleton_maps_tensor[k] = torch.from_numpy(v.astype(np.float32))
            ret["skeleton_maps"] = skeleton_maps_tensor

        except Exception as e:
            # Graceful degradation: return sample without skeleton_maps.
            # Log so we can identify problematic PTH files (mirrors Wan codebase behavior).
            print( "skeleton_maps skipped for index %d: %s", index, e   )
            logging.warning(pth_path, "skeleton_maps skipped for index %d: %s", index, e)

        return ret

    def __getitem__(self, index):
        return self.getitem(index)


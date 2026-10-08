#!/usr/bin/env python3
"""Convert KASA caption annotations into the CSV format used by KVLR."""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import pandas as pd


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def get_aspect_ratio(width, height):
    gcd = math.gcd(width, height)
    return f"{width // gcd}:{height // gcd}"


def resolve_action_video_dir(video_root, action, video_filename=None):
    candidates = [
        os.path.join(video_root, action),
        os.path.join(video_root, action, action),
    ]
    if video_filename is not None:
        for candidate in candidates:
            if os.path.isfile(os.path.join(candidate, video_filename)):
                return candidate
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def build_video_paths(action, video_filename, video_root, frames_root):
    video_path = os.path.join(resolve_action_video_dir(video_root, action, video_filename), video_filename)
    video_stem = Path(video_filename).stem
    first_frame = os.path.join(frames_root, action, video_stem, "frame_000000.png")
    return video_path, first_frame


def verify_directory_structure(video_root):
    actions = ["knotting", "needleGrasping", "needlePuncture"]
    missing = [
        os.path.join(video_root, action)
        for action in actions
        if not os.path.isdir(os.path.join(video_root, action))
        and not os.path.isdir(os.path.join(video_root, action, action))
    ]
    if missing:
        raise FileNotFoundError("Missing video directories:\n" + "\n".join(missing))


def create_KVLR_dataset(
    target_resolution="256px",
    video_root="./data/kasa/videos",
    frames_root="./data/kasa/frames",
    annotations_root="./data/kasa/annotations",
    output_csv=None,
):
    resolution_map = {
        "256px": 256,
        "512px": 512,
        "768px": 768,
        "1024px": 1024,
    }
    if target_resolution not in resolution_map:
        raise ValueError(f"Unsupported resolution: {target_resolution}")

    target_size = resolution_map[target_resolution]
    verify_directory_structure(video_root)

    annotation_files = {
        "knotting": os.path.join(annotations_root, "knotting_captions_all.json"),
        "needleGrasping": os.path.join(annotations_root, "needleGrasping_captions_all.json"),
        "needlePuncture": os.path.join(annotations_root, "needlePuncture_captions_all.json"),
    }

    records = []
    for action, annotation_file in annotation_files.items():
        if not os.path.exists(annotation_file):
            print(f"Skipping missing annotation file: {annotation_file}")
            continue

        with open(annotation_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        for video_filename, annotation in annotations.items():
            if annotation.get("error") is not None:
                continue

            video_path, first_frame = build_video_paths(action, video_filename, video_root, frames_root)
            if not os.path.exists(video_path) or not os.path.exists(first_frame):
                continue

            video_info = get_video_info(video_path)
            if video_info is None:
                continue

            records.append(
                {
                    "path": video_path,
                    "text": annotation.get("one_line", ""),
                    "num_frames": video_info["num_frames"],
                    "height": target_size,
                    "width": target_size,
                    "aspect_ratio": get_aspect_ratio(video_info["width"], video_info["height"]),
                    "resolution": target_resolution,
                    "fps": video_info["fps"],
                    "ref": first_frame,
                }
            )

    if not records:
        raise RuntimeError("No valid KASA records were found.")

    output_csv = output_csv or os.path.join(annotations_root, f"surgical_dataset_KVLR_{target_resolution}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)
    print(f"Wrote {len(records)} records to {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Create a KVLR training CSV from KASA annotations.")
    parser.add_argument("--resolution", default="256px", choices=["256px", "512px", "768px", "1024px"])
    parser.add_argument("--video-root", default=os.environ.get("KASA_VIDEO_ROOT", "./data/kasa/videos"))
    parser.add_argument("--frames-root", default=os.environ.get("KASA_FRAMES_ROOT", "./data/kasa/frames"))
    parser.add_argument("--annotations-root", default=os.environ.get("KASA_ANNOTATION_ROOT", "./data/kasa/annotations"))
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    create_KVLR_dataset(
        target_resolution=args.resolution,
        video_root=args.video_root,
        frames_root=args.frames_root,
        annotations_root=args.annotations_root,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()

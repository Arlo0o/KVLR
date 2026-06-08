#!/usr/bin/env python3
"""Sample an inference-task CSV from a KASA training CSV."""

import argparse
import csv
import os

import pandas as pd


def main():
    default_annotations = os.environ.get("KASA_ANNOTATION_ROOT", "./data/kasa/annotations")

    parser = argparse.ArgumentParser(description="Create an inference-task CSV from a training CSV.")
    parser.add_argument("-i", "--input", default=os.path.join(default_annotations, "surgical_rarp.csv"))
    parser.add_argument("-o", "--output", default=os.path.join(default_annotations, "inference_tasks_val_set.csv"))
    parser.add_argument("-r", "--ratio", type=float, default=0.15)
    parser.add_argument("-s", "--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {args.input}")

    sampled_df = df.sample(frac=args.ratio, random_state=args.seed)
    expected_cols = ["task_name", "video_id", "frame_num", "image_path", "prompt"]

    if "prompt" not in sampled_df.columns and "text" in sampled_df.columns:
        sampled_df["prompt"] = sampled_df["text"]
    if "frame_num" not in sampled_df.columns and "num_frames" in sampled_df.columns:
        sampled_df["frame_num"] = sampled_df["num_frames"]

    video_col = "path" if "path" in sampled_df.columns else ("video_path" if "video_path" in sampled_df.columns else None)
    if video_col:
        if "video_id" not in sampled_df.columns:
            sampled_df["video_id"] = sampled_df[video_col].apply(lambda x: os.path.splitext(os.path.basename(str(x)))[0])
        if "task_name" not in sampled_df.columns:
            sampled_df["task_name"] = sampled_df[video_col].apply(
                lambda x: os.path.basename(os.path.dirname(os.path.normpath(str(x))))
            )
        if "image_path" not in sampled_df.columns:
            frames_root = os.environ.get("KASA_FRAMES_ROOT", "./data/kasa/frames")
            sampled_df["image_path"] = sampled_df.apply(
                lambda row: os.path.join(frames_root, row["task_name"], row["video_id"], "frame_000000.png"),
                axis=1,
            )

    for col in expected_cols:
        if col not in sampled_df.columns:
            sampled_df[col] = "" if col != "video_id" else [f"sample_{i:06d}" for i in range(len(sampled_df))]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sampled_df[expected_cols].to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    print(f"Wrote {len(sampled_df)} inference tasks to {args.output}")


if __name__ == "__main__":
    main()

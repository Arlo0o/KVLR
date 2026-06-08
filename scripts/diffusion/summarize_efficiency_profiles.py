#!/usr/bin/env python3
"""Summarize KVLR efficiency profile JSON files."""

import argparse
import json
import os


def load_profile(path):
    with open(path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    profile["path"] = path
    return profile


def main():
    parser = argparse.ArgumentParser(description="Summarize KVLR teacher/student efficiency profiles.")
    parser.add_argument("profiles", nargs="+", help="Profile JSON files.")
    parser.add_argument("--output", default=None, help="Optional JSON summary path.")
    args = parser.parse_args()

    profiles = [load_profile(path) for path in args.profiles]
    teacher = next((p for p in profiles if "teacher" in p.get("name", "").lower()), None)
    summary = {"profiles": profiles}

    if teacher and teacher.get("elapsed_seconds"):
        baseline = teacher["elapsed_seconds"]
        for profile in profiles:
            elapsed = profile.get("elapsed_seconds")
            if elapsed:
                profile["speedup_vs_teacher"] = baseline / elapsed

    print(json.dumps(summary, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

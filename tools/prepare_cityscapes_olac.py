#!/usr/bin/env python3
# Copyright (c) 2026 Wenbo Wei.
# Licensed under the MIT License (see LICENSE).
# PEMOLA: Occlusion-Aware Panoptic Segmentation with Joint Position Embedding
# and Occlusion-Level Attention (ICME 2026).
"""Build the Cityscapes-OLAC layout from an official Cityscapes copy.

Cityscapes itself may not be redistributed, so this repo ships only the new
occlusion-level annotations (datasets/cityscapes_olac/occlusion_label_*.json).
This script slices a locally downloaded Cityscapes into per-occlusion-level
subsets (low / mid / high) matching those labels.

Prerequisites:
    1. Download leftImg8bit and gtFine from https://www.cityscapes-dataset.com/
       into CITYSCAPES_ROOT.
    2. Generate the panoptic annotations with cityscapesscripts:
           CITYSCAPES_DATASET=$CITYSCAPES_ROOT python -m \
               cityscapesscripts.preparation.createPanopticImgs
       (this creates gtFine/cityscapes_panoptic_{train,val}{,.json})

Usage:
    python tools/prepare_cityscapes_olac.py \
        --cityscapes_root datasets/data/cityscapes \
        --labels_dir      datasets/cityscapes_olac \
        --output          datasets/data/cityscapes_olac

By default files are symlinked; pass --copy to materialise real copies.
"""

import argparse
import json
import os
import shutil


LEVELS = ["low", "mid", "high"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Slice official Cityscapes into Cityscapes-OLAC occlusion-level subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cityscapes_root", default="datasets/data/cityscapes",
                        help="official Cityscapes root (leftImg8bit/, gtFine/ incl. cityscapes_panoptic_*)")
    parser.add_argument("--labels_dir", default="datasets/cityscapes_olac",
                        help="folder holding occlusion_label_{train,val}.json (shipped in this repo)")
    parser.add_argument("--output", default="datasets/data/cityscapes_olac",
                        help="output root for the sliced dataset")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--copy", action="store_true",
                        help="copy files instead of symlinking")
    return parser.parse_args()


def place(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def slice_split(split, root, labels_dir, out, copy):
    label_path = os.path.join(labels_dir, f"occlusion_label_{split}.json")
    with open(label_path) as f:
        labels = json.load(f)

    pan_json_path = os.path.join(root, "gtFine", f"cityscapes_panoptic_{split}.json")
    if not os.path.isfile(pan_json_path):
        raise FileNotFoundError(
            f"{pan_json_path} not found — run cityscapesscripts createPanopticImgs first (see module docstring)")
    with open(pan_json_path) as f:
        pan = json.load(f)
    pan_images = {im["id"]: im for im in pan["images"]}
    pan_annos = {an["image_id"]: an for an in pan["annotations"]}

    missing = sorted(set(labels) - set(pan_images))
    if missing:
        raise KeyError(f"{len(missing)} labelled images absent from {pan_json_path}, e.g. {missing[:3]}")

    for level in LEVELS:
        stems = sorted(k for k, v in labels.items() if v == level)
        sub = f"{split}_{level}"

        for stem in stems:
            city = stem.split("_")[0]
            img = f"{stem}_leftImg8bit.png"
            place(os.path.join(root, "leftImg8bit", split, city, img),
                  os.path.join(out, "leftImg8bit", sub, city, img), copy)

            src_gt_dir = os.path.join(root, "gtFine", split, city)
            for name in os.listdir(src_gt_dir):
                if name.startswith(stem + "_gtFine"):
                    place(os.path.join(src_gt_dir, name),
                          os.path.join(out, "gtFine", sub, city, name), copy)

            pan_png = pan_annos[stem]["file_name"]
            place(os.path.join(root, "gtFine", f"cityscapes_panoptic_{split}", pan_png),
                  os.path.join(out, "gtFine", f"cityscapes_panoptic_{sub}", pan_png), copy)

        sliced = {
            "images": [pan_images[s] for s in stems],
            "annotations": [pan_annos[s] for s in stems],
            "categories": pan["categories"],
        }
        out_json = os.path.join(out, "gtFine", f"cityscapes_panoptic_{sub}.json")
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(sliced, f)
        print(f"{sub}: {len(stems)} images")

    shutil.copy2(label_path, os.path.join(out, "gtFine", os.path.basename(label_path)))


def main():
    args = parse_args()
    for split in args.splits:
        slice_split(split, args.cityscapes_root, args.labels_dir, args.output, args.copy)
    print(f"Done. Cityscapes-OLAC written to {args.output}")


if __name__ == "__main__":
    main()

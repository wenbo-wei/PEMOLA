#!/usr/bin/env python3
# Copyright (c) 2026 Wenbo Wei.
# Licensed under the MIT License (see LICENSE).
# PEMOLA: Occlusion-Aware Panoptic Segmentation with Joint Position Embedding
# and Occlusion-Level Attention (ICME 2026).
"""Build the Cityscapes-OLAC layout from an official Cityscapes copy.

Cityscapes itself may not be redistributed. Download the new occlusion-level
annotations from the Cityscapes-OLAC GitHub Release into datasets/cityscapes_olac/
as occlusion_label_{train,val}.json before running this script (see README).
This script includes the complete train/val splits and per-occlusion-level
subsets (low / mid / high) matching those labels.

Prerequisites:
    1. Download leftImg8bit and gtFine from https://www.cityscapes-dataset.com/
       into CITYSCAPES_ROOT.
    2. Generate semantic training IDs and panoptic annotations:
           CITYSCAPES_DATASET=$CITYSCAPES_ROOT python -m \
               cityscapesscripts.preparation.createTrainIdLabelImgs
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
import filecmp
import json
import os
import shutil
import tempfile


LEVELS = ["low", "mid", "high"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build complete and per-level Cityscapes-OLAC splits from official Cityscapes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cityscapes_root", default="datasets/data/cityscapes",
                        help="official Cityscapes root (leftImg8bit/, gtFine/ incl. cityscapes_panoptic_*)")
    parser.add_argument("--labels_dir", default="datasets/cityscapes_olac",
                        help="folder holding occlusion_label_{train,val}.json downloaded from the GitHub Release")
    parser.add_argument("--output", default="datasets/data/cityscapes_olac",
                        help="output root for the complete and per-level dataset")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--copy", action="store_true",
                        help="copy files instead of symlinking")
    return parser.parse_args()


def place(src, dst, copy):
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        if not os.path.isfile(dst) or (
            not os.path.samefile(src, dst) and not filecmp.cmp(src, dst, shallow=False)
        ):
            raise FileExistsError(f"Refusing to replace different existing data: {dst}")
        if not (copy and os.path.islink(dst)):
            return
    if copy:
        # Also materialise a matching symlink left by a previous default run.
        # Never remove its target or leave a half-written destination file.
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(dst), delete=False) as f:
            temporary = f.name
        try:
            shutil.copy2(src, temporary)
            os.replace(temporary, dst)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
    else:
        os.symlink(os.path.abspath(src), dst)


def place_tree(src, dst, copy):
    if not os.path.isdir(src):
        raise FileNotFoundError(src)
    for directory, _, filenames in os.walk(src):
        relative = os.path.relpath(directory, src)
        for filename in sorted(filenames):
            place(os.path.join(directory, filename), os.path.join(dst, relative, filename), copy)


def write_json(data, path, copy):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.lexists(path):
        with open(path) as f:
            if json.load(f) == data:
                if copy and os.path.islink(path):
                    place(os.path.realpath(path), path, copy=True)
                return
        raise FileExistsError(f"Refusing to replace different existing data: {path}")
    with open(path, "x") as f:
        json.dump(data, f)


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

    if set(labels) != set(pan_images) or set(pan_images) != set(pan_annos):
        raise ValueError(f"Image, panoptic annotation and occlusion label IDs must match for {split}")
    if set(labels.values()) - set(LEVELS):
        raise ValueError(f"Unknown occlusion levels in {label_path}")

    # Validate required training inputs before creating this split.
    for stem in pan_images:
        city = stem.split("_")[0]
        required = (
            os.path.join(root, "leftImg8bit", split, city, f"{stem}_leftImg8bit.png"),
            os.path.join(root, "gtFine", split, city, f"{stem}_gtFine_labelTrainIds.png"),
            os.path.join(root, "gtFine", f"cityscapes_panoptic_{split}", pan_annos[stem]["file_name"]),
        )
        for path in required:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} not found (see preparation steps in module docstring)")

    # Include complete splits alongside the occlusion-level subsets so all
    # training and evaluation inputs can be read from cityscapes_olac alone.
    for relative in (
        os.path.join("leftImg8bit", split),
        os.path.join("gtFine", split),
        os.path.join("gtFine", f"cityscapes_panoptic_{split}"),
    ):
        place_tree(os.path.join(root, relative), os.path.join(out, relative), copy)
    place(pan_json_path, os.path.join(out, "gtFine", os.path.basename(pan_json_path)), copy)
    place(label_path, os.path.join(out, "gtFine", os.path.basename(label_path)), copy)
    print(f"{split}: {len(pan_images)} images (complete)")

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
        write_json(sliced, out_json, copy)
        print(f"{sub}: {len(stems)} images")


def main():
    args = parse_args()
    source = os.path.realpath(args.cityscapes_root)
    output = os.path.realpath(args.output)
    if os.path.commonpath([source, output]) in (source, output):
        raise ValueError("Source and output must be separate, non-nested directories")
    # Reject linked directories in an existing destination, which could send
    # writes outside the output or modify the source dataset through a link.
    if os.path.isdir(output):
        for directory, subdirectories, _ in os.walk(output):
            for name in subdirectories:
                path = os.path.join(directory, name)
                if os.path.islink(path):
                    raise ValueError(f"Output contains a linked directory: {path}")
    for split in args.splits:
        slice_split(split, source, args.labels_dir, output, args.copy)
    print(f"Done. Cityscapes-OLAC written to {args.output}")


if __name__ == "__main__":
    main()

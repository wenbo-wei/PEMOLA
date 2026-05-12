#!/usr/bin/env python3
# Copyright (c) 2026 Wenbo Wei.
# Licensed under the MIT License (see LICENSE).
# PEMOLA: Occlusion-Aware Panoptic Segmentation with Joint Position Embedding
# and Occlusion-Level Attention (ICME 2026).

import os
import json
import shutil
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Slice a custom subset from a COCO-format dataset using the images in IMAGE_DIR. "
                    "Outputs go to sibling folders next to IMAGE_DIR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_dir", nargs="?",
                        default="/home/wenbo/data/datasets/custom/custom",
                        help="folder of selected custom images")
    parser.add_argument("--src_root", default="/home/wenbo/data/datasets/coco",
                        help="source COCO dataset root (contains annotations/ and panoptic_<split>/ folders)")
    parser.add_argument("--split", choices=["train2017", "val2017"], default="train2017",
                        help="which split of the source dataset to slice from")
    return parser.parse_args()


def empty_coco_like(src):
    return {
        "info": src.get("info", {}),
        "licenses": src.get("licenses", []),
        "images": [],
        "annotations": [],
        "categories": src["categories"],
    }


def main():
    args = parse_args()

    image_dir = os.path.abspath(args.image_dir)
    out_root = os.path.dirname(image_dir)
    suffix = os.path.basename(image_dir)  # e.g. "custom" -> outputs become panoptic_custom etc.

    src_ann_dir = os.path.join(args.src_root, "annotations")
    instances_path = os.path.join(src_ann_dir, f"instances_{args.split}.json")
    panoptic_path = os.path.join(src_ann_dir, f"panoptic_{args.split}.json")

    for path in (image_dir, instances_path, panoptic_path):
        if not os.path.exists(path):
            raise SystemExit(f"missing required path: {path}")

    mask_dirs = [(f"{kind}_{args.split}", f"{kind}_{suffix}")
                 for kind in ("panoptic", "panoptic_stuff", "panoptic_semseg")]

    out_ann_dir = os.path.join(out_root, "annotations")
    os.makedirs(out_ann_dir, exist_ok=True)
    for _, dst_sub in mask_dirs:
        os.makedirs(os.path.join(out_root, dst_sub), exist_ok=True)

    print(f"source : {args.src_root}")
    print(f"images : {image_dir}")
    print(f"output : {out_root}")

    with open(instances_path) as f:
        instances_ann = json.load(f)
    with open(panoptic_path) as f:
        panoptic_ann = json.load(f)

    fname_to_img = {img["file_name"]: img for img in instances_ann["images"]}
    inst_id_to_anns = {}
    for ann in instances_ann["annotations"]:
        inst_id_to_anns.setdefault(ann["image_id"], []).append(ann)
    pan_id_to_ann = {ann["image_id"]: ann for ann in panoptic_ann["annotations"]}

    new_instances = empty_coco_like(instances_ann)
    new_panoptic = empty_coco_like(panoptic_ann)

    image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png")))
    if not image_files:
        raise SystemExit(f"no .jpg/.png images found in {image_dir}")

    skipped, copied = [], 0
    for im_name in tqdm(image_files, desc="building custom subset"):
        img_info = fname_to_img.get(im_name)
        if img_info is None:
            skipped.append(im_name)
            continue

        image_id = img_info["id"]
        base = os.path.splitext(im_name)[0]
        for src_sub, dst_sub in mask_dirs:
            src = os.path.join(args.src_root, src_sub, base + ".png")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out_root, dst_sub))
                copied += 1

        new_instances["images"].append(img_info)
        new_panoptic["images"].append(img_info)
        if image_id in inst_id_to_anns:
            new_instances["annotations"].extend(inst_id_to_anns[image_id])
        if image_id in pan_id_to_ann:
            new_panoptic["annotations"].append(pan_id_to_ann[image_id])

    with open(os.path.join(out_ann_dir, "instances.json"), "w") as f:
        json.dump(new_instances, f)
    with open(os.path.join(out_ann_dir, "panoptic.json"), "w") as f:
        json.dump(new_panoptic, f)

    print(f"\ndone. wrote {len(new_instances['images'])} images, "
          f"{len(new_instances['annotations'])} instance anns, "
          f"{len(new_panoptic['annotations'])} panoptic anns to {out_ann_dir}")
    print(f"copied {copied} mask files into {out_root}")
    if skipped:
        print(f"skipped {len(skipped)} images not found in instances_{args.split}.json (showing up to 5):")
        for s in skipped[:5]:
            print(f"  - {s}")


if __name__ == "__main__":
    main()

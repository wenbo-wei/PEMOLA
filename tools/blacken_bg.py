#!/usr/bin/env python3
# Copyright (c) 2026 Wenbo Wei.
# Licensed under the MIT License (see LICENSE).
# PEMOLA: Occlusion-Aware Panoptic Segmentation with Joint Position Embedding
# and Occlusion-Level Attention (ICME 2026).

import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from pycocotools.coco import COCO


JPEG_QUALITY = 95


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with black background for coco/cityscapes datasets.")
    parser.add_argument("--dataset", type=str, choices=["coco", "cityscapes"], default="coco",
                        help="dataset type: coco or cityscapes")
    parser.add_argument("--data_path", type=str, default="datasets/data/coco/train2017",
                        help="path to dataset root (images)")
    parser.add_argument("--ann_path", type=str, default="datasets/data/coco/annotations/instances_train2017.json",
                        help="path to annotation (json for coco, gtFine for cityscapes)")
    parser.add_argument("--output", type=str, default="output/black_bg",
                        help="path to output folder")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="number of parallel worker processes")
    parser.add_argument("--chunksize", type=int, default=8,
                        help="number of tasks dispatched to each worker at a time")
    return parser.parse_args()


def save_black_bg(im, mask, out_path):
    output_im = im * mask[..., None]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if out_path.lower().endswith((".jpg", ".jpeg")) else []
    cv2.imwrite(out_path, output_im, params)


def _process_cityscapes_one(task):
    im_path, ann_file, out_path = task
    im = cv2.imread(im_path)
    ann = cv2.imread(ann_file, cv2.IMREAD_UNCHANGED)
    if im is None or ann is None:
        return
    mask = ann >= 1000
    save_black_bg(im, mask, out_path)


_COCO = None
_COCO_DATA_PATH = None
_COCO_OUTPUT = None


def _coco_init(ann_path, data_path, output):
    global _COCO, _COCO_DATA_PATH, _COCO_OUTPUT
    cv2.setNumThreads(0)
    _COCO = COCO(ann_path)
    _COCO_DATA_PATH = data_path
    _COCO_OUTPUT = output


def _process_coco_one(im_id):
    im_ann = _COCO.loadImgs(im_id)[0]
    im_path = os.path.join(_COCO_DATA_PATH, im_ann['file_name'])
    im = cv2.imread(im_path)
    if im is None:
        return
    h, w, _ = im.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in _COCO.loadAnns(_COCO.getAnnIds(imgIds=im_id)):
        if ann.get('iscrowd', 0) == 0:
            mask |= _COCO.annToMask(ann)
    out_name = os.path.splitext(im_ann['file_name'])[0] + ".jpg"
    out_path = os.path.join(_COCO_OUTPUT, out_name)
    save_black_bg(im, mask, out_path)


def process_dataset(dataset, data_path, ann_path, output, workers, chunksize):
    if dataset == "cityscapes":
        tasks = []
        for city in os.listdir(data_path):
            city_dir = os.path.join(data_path, city)
            if not os.path.isdir(city_dir):
                continue
            for file_name in os.listdir(city_dir):
                if not file_name.endswith("_leftImg8bit.png"):
                    continue
                im_path = os.path.join(city_dir, file_name)
                ann_file = os.path.join(ann_path, city, file_name.replace("_leftImg8bit.png", "_gtFine_instanceIds.png"))
                out_name = file_name.replace("_leftImg8bit.png", "_leftImg8bit.jpg")
                out_path = os.path.join(output, city, out_name)
                tasks.append((im_path, ann_file, out_path))

        with Pool(processes=workers, initializer=cv2.setNumThreads, initargs=(0,)) as pool:
            for _ in tqdm(pool.imap_unordered(_process_cityscapes_one, tasks, chunksize=chunksize), total=len(tasks)):
                pass

    elif dataset == "coco":
        with open(ann_path) as f:
            im_ids = [img['id'] for img in json.load(f)['images']]
        with Pool(processes=workers, initializer=_coco_init,
                  initargs=(ann_path, data_path, output)) as pool:
            for _ in tqdm(pool.imap_unordered(_process_coco_one, im_ids, chunksize=chunksize), total=len(im_ids)):
                pass
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def main():
    args = parse_args()
    process_dataset(args.dataset, args.data_path, args.ann_path, args.output, args.workers, args.chunksize)


if __name__ == "__main__":
    main()

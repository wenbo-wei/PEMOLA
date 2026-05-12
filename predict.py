# Copyright (c) 2026 Wenbo Wei.
# Licensed under the MIT License (see LICENSE).
# PEMOLA: Occlusion-Aware Panoptic Segmentation with Joint Position Embedding
# and Occlusion-Level Attention (ICME 2026).
"""
PEMOLA panoptic segmentation prediction script.

Run via scripts/predict_pemola_olac.sh, or invoke directly — all dataset
assets are CLI args, no per-dataset state is hard-coded in this file.
"""

import argparse
import json
import os
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, detection_utils as utils
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import ColorMode, Visualizer

from mask2former import add_maskformer2_config, add_pemola_config


class PemolaPredictor:
    def __init__(self, config_file, weights, metadata_name,
                 cam_dir=None, occlusion_json=None, enable_pemola=True):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_pemola_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
        cfg.MODEL.PEMOLA.PE_MODULATION = enable_pemola
        cfg.freeze()

        self.cfg = cfg
        self.metadata = MetadataCatalog.get(metadata_name)
        self.input_format = cfg.INPUT.FORMAT
        self.augmentations = T.AugmentationList(utils.build_augmentation(cfg, is_train=False))

        self.model = build_model(cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)

        if enable_pemola:
            if not (cam_dir and occlusion_json):
                raise ValueError(
                    "--cam-dir and --occlusion-json are required when PEMOLA modulation is enabled."
                )
            self.cam_dir = cam_dir
            with open(occlusion_json) as f:
                self.occlusion_ann = json.load(f)
            self.occlusion_labels = {name: i for i, name in enumerate(cfg.MODEL.PEMOLA.OCCLUSION_LEVELS)}
        else:
            self.cam_dir = None
            self.occlusion_ann = None
            self.occlusion_labels = None

    def _build_input(self, im_bgr, img_name):
        # Detectron2 inputs follow the format of utils.build_augmentation /
        # DatasetMapper: image is HxWxC in cfg.INPUT.FORMAT (RGB or BGR), augmented
        # then transposed to CxHxW float tensor. For PEMOLA we also feed an
        # occlusion CAM, resized to original image size and run through the same
        # augmentation so it stays spatially aligned with the image.
        image = im_bgr if self.input_format == "BGR" else im_bgr[:, :, ::-1]
        height, width = image.shape[:2]
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        aug_image = aug_input.image
        inputs = {
            "image": torch.as_tensor(np.ascontiguousarray(aug_image.transpose(2, 0, 1))),
            "height": height,
            "width": width,
        }
        if self.occlusion_ann is not None:
            cam = torch.load(
                os.path.join(self.cam_dir, f"{img_name}.pt"), weights_only=False
            ).numpy()
            cam_resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
            cam_aug = transforms.apply_image(cam_resized)
            inputs["occlusion_cam"] = torch.as_tensor(np.ascontiguousarray(cam_aug))
            inputs["occlusion_label"] = self.occlusion_labels[self.occlusion_ann[img_name]]
        return inputs

    def predict(self, image_path, out_dir, out_format="pdf"):
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        im = cv2.imread(image_path)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        inputs = self._build_input(im, img_name)
        with torch.no_grad():
            outputs = self.model([inputs])[0]

        vis = Visualizer(im[:, :, ::-1], self.metadata, scale=1.2,
                         instance_mode=ColorMode.IMAGE_BW)
        panoptic_seg, segments_info = outputs["panoptic_seg"]
        rendered = vis.draw_panoptic_seg(panoptic_seg.to("cpu"), segments_info).get_image()

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{img_name}.{out_format}")
        if out_format == "pdf":
            Image.fromarray(rendered).save(out_path, "PDF")
        else:
            cv2.imwrite(out_path, rendered[:, :, ::-1])
        return out_path


def _iter_inputs(path):
    if os.path.isdir(path):
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            yield from sorted(glob(os.path.join(path, f"*{ext}")))
    else:
        yield path


def parse_args():
    parser = argparse.ArgumentParser(description="PEMOLA panoptic prediction")
    parser.add_argument("--config", required=True, help="path to detectron2 yaml config")
    parser.add_argument("--weights", required=True, help="path to model checkpoint (.pth/.pkl)")
    parser.add_argument("--input", required=True, help="image file or directory of images")
    parser.add_argument("--output-dir", default="output/predictions",
                        help="where to write rendered predictions (default: %(default)s)")
    parser.add_argument("--metadata", default="coco_2017_val_panoptic",
                        help="detectron2 MetadataCatalog name used for visualisation "
                             "(default: %(default)s)")
    parser.add_argument("--cam-dir",
                        help="directory containing per-image *.pt CAM tensors "
                             "(required unless --no-pemola)")
    parser.add_argument("--occlusion-json",
                        help="JSON mapping image name -> occlusion level "
                             "(required unless --no-pemola)")
    parser.add_argument("--no-pemola", action="store_true",
                        help="disable PEMOLA modulation to run the baseline")
    parser.add_argument("--format", choices=("pdf", "png", "jpg"), default="pdf",
                        help="output image format (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = PemolaPredictor(
        config_file=args.config,
        weights=args.weights,
        metadata_name=args.metadata,
        cam_dir=args.cam_dir,
        occlusion_json=args.occlusion_json,
        enable_pemola=not args.no_pemola,
    )
    for image_path in _iter_inputs(args.input):
        out_path = predictor.predict(image_path, args.output_dir, args.format)
        print(f"[ok] {image_path} -> {out_path}")


if __name__ == "__main__":
    main()

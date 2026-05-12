# Copyright (c) Facebook, Inc. and its affiliates.
import os
import cv2
import copy
import json
import logging
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils


__all__ = ["PanopticTestMapper"]


class PanopticTestMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
            self,
            cfg,
            is_train: bool,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
    ):
        """
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """

        # fmt: off
        self.cfg = cfg
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[PanopticTestMapper] Augmentations used in {mode}: {augmentations}")

        if self.cfg.MODEL.PEMOLA.PE_MODULATION:
            self.dataset_root = os.getenv("DETECTRON2_DATASETS", "datasets")
            self.occlusion_maps = {name: i for i, name in enumerate(self.cfg.MODEL.PEMOLA.OCCLUSION_LEVELS)}

            if self.cfg.INPUT.DATASET_MAPPER_NAME == "coco_olac_panoptic_lsj":
                self.cam_dir = os.path.join(self.dataset_root, "coco_olac_cam/cam_pt_val")
                occlusion_label_json = os.path.join(self.dataset_root, "coco_olac/val/occlusion_label_val.json")
            elif self.cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
                self.cam_dir = os.path.join(self.dataset_root, "cityscapes_cam/cam_pt_val")
                occlusion_label_json = os.path.join(self.dataset_root, "cityscapes/gtFine/occlusion_label_val.json")
            else:
                ValueError(f"Unsupported cfg.INPUT.DATASET_MAPPER_NAME: {self.cfg.INPUT.DATASET_MAPPER_NAME}.")

            with open(occlusion_label_json, 'r') as olj:
                self.occlusion_label_ann = json.load(olj)

    @classmethod
    def from_config(cls, cfg, is_train: bool = False):
        augs = utils.build_augmentation(cfg, is_train)

        ret = {
            "cfg": cfg,
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.cfg.MODEL.PEMOLA.PE_MODULATION:
            if self.cfg.INPUT.DATASET_MAPPER_NAME == "coco_olac_panoptic_lsj":
                basename = os.path.basename(dataset_dict['file_name'])
                name, _ = os.path.splitext(basename)
            elif self.cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
                name = dataset_dict['image_id']
            else:
                ValueError(f"Unsupported cfg.INPUT.DATASET_MAPPER_NAME: {self.cfg.INPUT.DATASET_MAPPER_NAME}.")

            dataset_dict["occlusion_label"] = self.occlusion_maps[self.occlusion_label_ann[name]]

            cam = torch.load(os.path.join(self.cam_dir, f"{name}.pt"), weights_only=False).numpy()
            cam_resized = cv2.resize(cam, (dataset_dict['width'], dataset_dict['height']),
                                     interpolation=cv2.INTER_LINEAR)
            cam_transformed = transforms.apply_image(cam_resized)
            dataset_dict["occlusion_cam"] = torch.as_tensor(np.ascontiguousarray(cam_transformed))

        return dataset_dict

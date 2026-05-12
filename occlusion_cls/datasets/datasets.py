import os
import cv2
import json
import numpy as np
from PIL import Image

import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 transform = None,
                 return_name = False):
        super().__init__()
        assert os.path.isdir(data_path), f"{data_path} is not a valid directory."
        self.data_path = data_path
        self.images = sorted(os.listdir(self.data_path))
        self.transform = transform
        self.return_name = return_name

    def __getitem__(self, idx):
        im_name = self.images[idx]
        im_path = os.path.join(self.data_path, im_name)

        with Image.open(im_path) as im:
            im = im.convert("RGB")

        if self.transform is not None:
            im = self.transform(im)

        if self.return_name:
            return im, im_name

        return im

    def __len__(self):
        return len(self.images)


class CocoOlacDataset(BaseDataset):
    def __init__(self,
                 data_path,
                 label_map,
                 mode: str = "train",
                 black_bg: bool = False,
                 transform = None):
        assert label_map, "label_map is required"
        super().__init__(data_path=data_path, transform=transform)
        self.label_map = label_map

        assert mode in ["train", "val", "test"], "mode needs to be train/val/test"
        sub = f"{mode}_blackbg" if black_bg else mode
        im_dir = os.path.join(self.data_path, sub)
        self.images = sorted(os.path.join(im_dir, n) for n in os.listdir(im_dir))

        ocl_ann_path = {
            "train": os.path.join(self.data_path, "occlusion_label_train.json"),
            "val": os.path.join(self.data_path, "occlusion_label_val.json"),
            "test": os.path.join(self.data_path, "occlusion_label_test.json"),
        }[mode]

        with open(ocl_ann_path, "r") as oap:
            self.ocl_ann = json.load(oap)

    def __getitem__(self, idx):
        im_path = self.images[idx]
        im_name = os.path.basename(im_path)
        im_name_main = im_name.split(".")[0]

        with Image.open(im_path) as ip:
            im = ip.convert("RGB")

        if self.transform is not None:
            im = self.transform(im)

        ocl_level = self.ocl_ann[im_name_main]
        label = self.label_map[ocl_level]

        return im, label


class CityscapesDataset(BaseDataset):
    def __init__(self,
                 data_path,
                 label_map,
                 mode: str = "train",
                 black_bg: bool = False,
                 transform=None):
        assert label_map, "label_map is required"
        super().__init__(data_path=data_path, transform=transform)
        self.label_map = label_map

        assert mode in ["train", "val"], "mode needs to be train/val"

        sub = f"{mode}_blackbg" if black_bg else mode
        self.im_dir = os.path.join(self.data_path, "leftImg8bit", sub)

        images = []
        for city in os.listdir(self.im_dir):
            city_dir = os.path.join(self.im_dir, city)
            if not os.path.isdir(city_dir):
                continue
            for im_name in os.listdir(city_dir):
                if im_name.endswith("_leftImg8bit.png"):
                    images.append(os.path.join(city_dir, im_name))
        self.images = sorted(images)

        ann_dir = os.path.join(self.data_path, "gtFine")
        ocl_ann_path = {
            "train": os.path.join(ann_dir, "occlusion_label_train.json"),
            "val": os.path.join(ann_dir, "occlusion_label_val.json"),
        }[mode]

        with open(ocl_ann_path, "r") as oap:
            self.ocl_ann = json.load(oap)

    def __getitem__(self, idx):
        im_path = self.images[idx]
        im_name = os.path.basename(im_path)

        with Image.open(im_path) as ip:
            im = ip.convert("RGB")

        if self.transform is not None:
            im = self.transform(im)

        _im_name = im_name.rsplit('_', 1)[0]
        ocl_level = self.ocl_ann[_im_name]
        label = self.label_map[ocl_level]

        return im, label


class CAMDataset(BaseDataset):
    def __init__(self,
                 data_path,
                 data_name,
                 occlusion_ann_path,
                 visual_size,
                 label_map,
                 transform,
                 return_name = True):
        assert data_name in ("coco", "cityscapes"), f"Unsupported data_name: {data_name}"
        assert occlusion_ann_path, "occlusion_ann_path is required"
        assert visual_size, "visual_size is required"
        assert label_map, "label_map is required"
        assert transform is not None, "transform is required (CAM pipeline needs tensor input)"
        super().__init__(data_path, transform=transform, return_name=return_name)
        self.label_map = label_map

        self.data_name = data_name
        self.data_path_blackbg = f"{self.data_path}_blackbg"
        assert os.path.isdir(self.data_path_blackbg), \
            f"Required blackbg directory does not exist: {self.data_path_blackbg}"

        images = []
        images_blackbg = []
        if self.data_name == 'coco':
            for im_name in os.listdir(self.data_path):
                images.append(os.path.join(self.data_path, im_name))
                images_blackbg.append(os.path.join(self.data_path_blackbg, im_name))

        elif self.data_name == 'cityscapes':
            for city in os.listdir(self.data_path):
                city_dir = os.path.join(self.data_path, city)
                city_black_dir = os.path.join(self.data_path_blackbg, city)
                if not (os.path.isdir(city_dir) and os.path.isdir(city_black_dir)):
                    continue
                for im_name in os.listdir(city_dir):
                    if im_name.endswith("_leftImg8bit.png"):
                        images.append(os.path.join(city_dir, im_name))
                        images_blackbg.append(os.path.join(city_black_dir, im_name))

        # Paired sort so self.images[i] and self.images_blackbg[i] always correspond
        # to the same source image, regardless of how the two parent dirs sort.
        if images:
            pairs = sorted(zip(images, images_blackbg))
            self.images = [p[0] for p in pairs]
            self.images_blackbg = [p[1] for p in pairs]
        else:
            self.images = []
            self.images_blackbg = []

        with open(occlusion_ann_path, "r") as oap:
            self.occlusion_ann = json.load(oap)

        self.visual_size = visual_size

    def visualization_preprocess(self, image):
        image = np.array(image)
        image = cv2.resize(image, (self.visual_size, self.visual_size))
        image = np.float32(image) / 255
        return image

    def __getitem__(self, idx):
        im_path = self.images[idx]
        im_blackbg_path = self.images_blackbg[idx]
        im_name = os.path.basename(im_path)

        with Image.open(im_path) as ip:
            im = ip.convert("RGB")

        with Image.open(im_blackbg_path) as ibp:
            im_blackbg = ibp.convert("RGB")

        im_visual = self.visualization_preprocess(im)

        if self.transform is not None:
            im_tensor = self.transform(im_blackbg)

        if self.data_name == 'coco':
            im_name_main = im_name.split(".")[0]
        elif self.data_name == 'cityscapes':
            im_name_main = im_name.rsplit('_', 1)[0]
        else:
            raise ValueError(f"Unsupported dataset: {self.data_name}")

        ocl_level = self.occlusion_ann[im_name_main]
        label = self.label_map[ocl_level]

        if self.return_name:
            return im_tensor, im_visual, label, im_name_main

        return im_tensor, im_visual, label

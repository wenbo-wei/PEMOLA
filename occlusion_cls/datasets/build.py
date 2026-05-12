# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.distributed as dist
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .datasets import CocoOlacDataset
from .datasets import CityscapesDataset

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except ImportError:
    from timm.data.transforms import _pil_interp


def setup_mixup(config):
    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    return mixup_fn


def _make_label_map(config):
    label_map = {name: i for i, name in enumerate(config.OCCLUSION.LEVELS)}
    assert len(label_map) == config.MODEL.NUM_CLASSES, (
        f"OCCLUSION.LEVELS has {len(label_map)} entries but MODEL.NUM_CLASSES={config.MODEL.NUM_CLASSES}; "
        f"these must agree."
    )
    return label_map


def _build_eval_loader(config, dataset, name):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=config.TEST.SHUFFLE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )
    if dist.get_rank() == 0:
        print(f"successfully build {name} dataset")
    return dataloader


def build_train_loader(config):
    transform = build_transform(config, is_train=True)
    label_map = _make_label_map(config)

    if config.DATA.DATASET == 'coco_olac_cls':
        dataset = CocoOlacDataset(config.DATA.DATA_PATH, label_map=label_map, black_bg=True, transform=transform)
        name = f"{config.DATA.DATASET}_train_blackbg"
        assert len(dataset) > 0, f"{name} dataset is empty"
        if dist.get_rank() == 0:
            print(f"successfully build {name} dataset")

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True
        )

    else:
        raise ValueError(f"Unsupported dataset: {config.DATA.DATASET}")

    return dataloader


def build_test_loader(config):
    test_loader = {}
    transform = build_transform(config)
    label_map = _make_label_map(config)

    if config.DATA.DATASET == 'coco_olac_cls':
        dataset_cls = CocoOlacDataset
        dataset_configs = [
            ("val", False),  # mode, black_bg
            ("test", False),
            ("val", True),
            ("test", True),
        ]
    elif config.DATA.DATASET == 'cityscapes':
        dataset_cls = CityscapesDataset
        dataset_configs = [
            ("train", False),
            ("val", False),
            ("train", True),
            ("val", True),
        ]
    else:
        raise ValueError(f"Unsupported dataset: {config.DATA.DATASET}")

    for mode, black_bg in dataset_configs:
        dataset = dataset_cls(data_path=config.DATA.DATA_PATH, label_map=label_map,
                              mode=mode, black_bg=black_bg, transform=transform)
        suffix = "_blackbg" if black_bg else ""
        name = f"{config.DATA.DATASET}_{mode}{suffix}"
        assert len(dataset) > 0, f"{name} dataset is empty"
        test_loader[name] = _build_eval_loader(config, dataset, name)

    return test_loader


def build_transform(config, is_train=False):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)))
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION)))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

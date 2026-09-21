# Copyright (c) 2026 Wenbo Wei.
# Licensed under the MIT License (see LICENSE).
"""Register complete and occlusion-level Cityscapes-OLAC splits."""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.data.datasets.cityscapes_panoptic import load_cityscapes_panoptic


def register_all_cityscapes_olac_panoptic(root):
    # Keep the 19-class ordering and category mappings used by Detectron2's
    # standard Cityscapes panoptic loader and evaluators.
    metadata = {
        "thing_classes": [category["name"] for category in CITYSCAPES_CATEGORIES],
        "thing_colors": [category["color"] for category in CITYSCAPES_CATEGORIES],
        "stuff_classes": [category["name"] for category in CITYSCAPES_CATEGORIES],
        "stuff_colors": [category["color"] for category in CITYSCAPES_CATEGORIES],
        "thing_dataset_id_to_contiguous_id": {
            category["id"]: category["trainId"]
            for category in CITYSCAPES_CATEGORIES if category["isthing"]
        },
        "stuff_dataset_id_to_contiguous_id": {
            category["id"]: category["trainId"]
            for category in CITYSCAPES_CATEGORIES if not category["isthing"]
        },
    }
    dataset_root = os.path.join(root, "cityscapes_olac")
    splits = [
        split + suffix
        for split in ("train", "val")
        for suffix in ("", "_low", "_mid", "_high")
    ]
    for split in splits:
        name = f"cityscapes_olac_panoptic_{split}"
        image_root = os.path.join(dataset_root, "leftImg8bit", split)
        panoptic_root = os.path.join(dataset_root, "gtFine", f"cityscapes_panoptic_{split}")
        panoptic_json = panoptic_root + ".json"
        DatasetCatalog.register(
            name,
            lambda images=image_root, masks=panoptic_root, annotations=panoptic_json:
                load_cityscapes_panoptic(images, masks, annotations, metadata),
        )
        MetadataCatalog.get(name).set(
            image_root=image_root,
            panoptic_root=panoptic_root,
            panoptic_json=panoptic_json,
            gt_dir=os.path.join(dataset_root, "gtFine", split),
            evaluator_type="cityscapes_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **metadata,
        )


register_all_cityscapes_olac_panoptic(os.getenv("DETECTRON2_DATASETS", "datasets"))

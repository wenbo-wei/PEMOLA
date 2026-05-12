# Copyright (c) 2026 Wenbo Wei.
# Licensed under the MIT License (see LICENSE).
# PEMOLA: Occlusion-Aware Panoptic Segmentation with Joint Position Embedding
# and Occlusion-Level Attention (ICME 2026).

import os
import json
import argparse
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from occlusion_cls.config import get_config
from occlusion_cls.models import build_model
from occlusion_cls.logger import create_logger

from occlusion_cls.datasets import BaseDataset, build_transform


def parse_args():
    parser = argparse.ArgumentParser('Swin Transformer prediction script', add_help=False)
    parser.add_argument('--cfg', type=str,
                        default='configs/occlusion_cls/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml',
                        help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('--batch_size', type=int,
                        default=32,
                        help="batch size for single GPU")
    parser.add_argument('--data_path', type=str,
                        default='datasets/data/coco_olac/val/val',
                        help='path to dataset')
    parser.add_argument('--dataset', type=str,
                        default='coco',
                        help='dataset name')
    parser.add_argument("--occlusion_ann",
                        default="datasets/data/coco_olac/train/occlusion_label_train.json",
                        help="Path to occlusion annotation json file")
    parser.add_argument('--resume', type=str, required=True,
                        help='resume from checkpoint')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)
    return args, config, unparsed


def build_dataloader(config):
    transform = build_transform(config)
    dataset = BaseDataset(config.DATA.DATA_PATH, transform, return_name=True)
    assert len(dataset) > 0

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
    )

    if dist.get_rank() == 0:
        logger.info(f"Successfully built dataset, total samples {len(dataset)}")

    return dataloader


@torch.no_grad()
def predict(config, model, data_loader):
    # Label-completion: if a sample already has a ground-truth occlusion label in
    # OCCLUSION.ANN, use it; otherwise fall back to model prediction. Output is a
    # merged label map, not pure predictions.
    label_dict = {}
    with open(config.OCCLUSION.ANN, 'r') as oa:
        occlusion_ann = json.load(oa)

    logger.info("Start predicting")

    for images, names in tqdm(data_loader, desc=f"Rank {dist.get_rank()} predicting",
                              disable=(dist.get_rank() != 0)):
        images = images.cuda(non_blocking=True)

        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            outputs = model(images)
            preds = outputs.argmax(dim=-1)

        for name, pred in zip(names, preds):
            _name = name.split(".")[0]
            if _name in occlusion_ann:
                label_dict[_name] = occlusion_ann[_name]
            else:
                label_dict[_name] = config.OCCLUSION.LEVELS[int(pred)]

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, label_dict)

    if dist.get_rank() == 0:
        merged = {}
        for d in gathered:
            merged.update(d)

        save_path = os.path.join(config.OUTPUT, "prediction.json")
        with open(save_path, "w") as f:
            json.dump(merged, f, indent=2, sort_keys=True)
        logger.info(f"Prediction saved to {save_path}")
        return merged

    return label_dict


def main(config):
    # build dataset loader
    dataloader = build_dataloader(config)

    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")

    model = build_model(config)
    model.cuda()
    model.eval()

    logger.info(f"Loading from {config.MODEL.RESUME}")
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=False)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    if msg.missing_keys:
        logger.warning(f"Missing keys when loading checkpoint: {msg.missing_keys}")
    if msg.unexpected_keys:
        logger.warning(f"Unexpected keys when loading checkpoint: {msg.unexpected_keys}")

    preds = predict(config, model, dataloader)
    logger.info(f"Prediction finished, total samples {len(preds)}")


if __name__ == '__main__':
    args, config, unparsed = parse_args()

    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.cuda.set_device(config.LOCAL_RANK)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier(device_ids=[torch.cuda.current_device()])

    cudnn.benchmark = True
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(),
                           name=f"{config.MODEL.NAME}")

    if unparsed:
        logger.warning(f"Unparsed (ignored) CLI args: {unparsed}")

    main(config)

    dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.destroy_process_group()

# Copyright (c) 2026 Wenbo Wei.
# Licensed under the MIT License (see LICENSE).
# PEMOLA: Occlusion-Aware Panoptic Segmentation with Joint Position Embedding
# and Occlusion-Level Attention (ICME 2026).
#
# Uses pytorch-grad-cam (Jacob Gildenblat, MIT) as an external dependency:
# https://github.com/jacobgil/pytorch-grad-cam (installed via `pip install grad-cam`).

import os
import cv2
import argparse
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from occlusion_cls.config import get_config
from occlusion_cls.models import build_model
from occlusion_cls.datasets import build_transform, CAMDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",
                        default="configs/occlusion_cls/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml",
                        help="path to config file", )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--batch_size', type=int,
                        default=8,
                        help="batch size for single GPU")
    parser.add_argument("--data_path",
                        default="datasets/data/coco_olac/val/val",
                        help="Input image dir")
    parser.add_argument('--dataset', type=str,
                        default='coco',
                        help='dataset name')
    parser.add_argument("--occlusion_ann",
                        default="datasets/data/coco_olac/val/occlusion_label_val.json",
                        help="Path to occlusion annotation json file")
    parser.add_argument("--resume",
                        default="output/swin_large_patch4_window12_384_22kto1k_finetune/ckpt_epoch_24.pth",
                        help="Resume from checkpoint")
    parser.add_argument("--output", default="output")
    parser.add_argument('--tag', default="coco", help='tag of experiment')

    return parser.parse_args()


def load_checkpoint(config):
    model = build_model(config)
    checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def reshape_transform(tensor):
    B, N, C = tensor.shape
    spatial_size = int(N ** 0.5)
    result = tensor.reshape(B, spatial_size, spatial_size, C)
    result = result.transpose(2, 3).transpose(1, 2)  # (B, C, H, W)
    return result


if __name__ == "__main__":
    args = parse_args()
    config = get_config(args)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    cudnn.benchmark = True

    model = load_checkpoint(config).cuda()

    if config.MODEL.TYPE != 'swin':
        raise NotImplementedError(
            f"CAM script currently only supports swin (got {config.MODEL.TYPE}). "
            f"target_layers and reshape_transform need to be adapted for other architectures."
        )
    target_layers = [model.layers[-1].blocks[-1].norm2]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    transform = build_transform(config)

    label_map = {name: i for i, name in enumerate(config.OCCLUSION.LEVELS)}
    dataset = CAMDataset(config.DATA.DATA_PATH,
                         data_name=config.DATA.DATASET,
                         occlusion_ann_path=config.OCCLUSION.ANN,
                         visual_size=config.DATA.IMG_SIZE,
                         label_map=label_map,
                         transform=transform)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.DATA.BATCH_SIZE, sampler=sampler,
        num_workers=14, pin_memory=True,
    )

    output_cam_image_dir = os.path.join(config.OUTPUT, "cam_image")
    output_cam_pt_dir = os.path.join(config.OUTPUT, "cam_pt")
    if rank == 0:
        os.makedirs(output_cam_image_dir, exist_ok=True)
        os.makedirs(output_cam_pt_dir, exist_ok=True)
    dist.barrier()

    if rank == 0:
        print(f"Start running on {len(dataset)} images ({len(dataloader)} batches/rank) on {world_size} GPU(s)")
    for im_tensors, im_visuals, labels, im_names in tqdm(dataloader, disable=(rank != 0)):
        im_tensors = im_tensors.cuda(non_blocking=True)
        targets = [ClassifierOutputTarget(label) for label in labels]
        grayscale_cams = cam(input_tensor=im_tensors, targets=targets, eigen_smooth=True, aug_smooth=True,)
        for im_visual, grayscale_cam, im_name in zip(im_visuals, grayscale_cams, im_names):
            cam_image = show_cam_on_image(im_visual.numpy(), grayscale_cam, use_rgb=True)
            cv2.imwrite(os.path.join(output_cam_image_dir, f"{im_name}.jpg"), cam_image[:, :, ::-1])
            torch.save(torch.from_numpy(grayscale_cam), os.path.join(output_cam_pt_dir, f"{im_name}.pt"))

    dist.barrier()
    dist.destroy_process_group()

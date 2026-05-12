# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Modified for PEMOLA: Occlusion-Aware Panoptic Segmentation with Joint Position
# Embedding and Occlusion-Level Attention (ICME 2026) by Wenbo Wei, 2026.
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from timm.utils import accuracy, AverageMeter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from occlusion_cls.config import get_config
from occlusion_cls.models import build_model
from occlusion_cls.lr_scheduler import build_scheduler
from occlusion_cls.optimizer import build_optimizer
from occlusion_cls.logger import create_logger
from occlusion_cls.utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

from occlusion_cls.datasets import setup_mixup
from occlusion_cls.datasets import build_train_loader
from occlusion_cls.datasets import build_test_loader


def parse_args():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str,
                        default='configs/occlusion_cls/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml',
                        metavar="FILE", help='path to config file', )
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
                        default='datasets/data/coco_olac_cls',
                        help='path to dataset')
    parser.add_argument('--dataset', type=str,
                        default='coco_olac_cls',
                        help='dataset name')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache_mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='swin only: path to a pretrained .pth (e.g. imagenet22k weights)')
    parser.add_argument('--imagenet_pretrained', action='store_true',
                        help='ResNet only: load torchvision ImageNet1K pretrained weights')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use_checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer in config if provided. Supported: adamw, sgd.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config, unparsed


def main(config):
    if config.EVAL_MODE and not (config.MODEL.RESUME or config.MODEL.PRETRAINED or config.MODEL.IMAGENET_PRETRAINED):
        raise ValueError("EVAL_MODE requires MODEL.RESUME, MODEL.PRETRAINED, or MODEL.IMAGENET_PRETRAINED to be set")

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"number of params: {n_parameters}")
    if config.MODEL.TYPE == 'swin':
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9:.2f}")
    elif config.MODEL.TYPE == 'resnet':
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
            flops = FlopCountAnalysis(model, dummy).total()
        model.train()
        logger.info(f"number of GFLOPs (fvcore): {flops / 1e9:.2f}")

    model.cuda()
    model_without_ddp = model

    # build train loader
    if not config.EVAL_MODE:
        train_loader = build_train_loader(config)
        optimizer = build_optimizer(config, model)
        loss_scaler = NativeScalerWithGradNormCount()
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
        if config.AUG.MIXUP > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif config.MODEL.LABEL_SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
        else:
            criterion = torch.nn.CrossEntropyLoss()

    # build test loader
    test_loaders = build_test_loader(config)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)

    if config.TRAIN.AUTO_RESUME and not config.EVAL_MODE:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    max_accuracy_dict = {}

    if config.MODEL.RESUME:
        max_accuracy_dict = load_checkpoint(
            config,
            model_without_ddp,
            optimizer if not config.EVAL_MODE else None,
            lr_scheduler if not config.EVAL_MODE else None,
            loss_scaler if not config.EVAL_MODE else None,
            logger,
        )
        validate_all(config, test_loaders, model, max_accuracy_dict)
        if config.EVAL_MODE:
            return

    if (config.MODEL.PRETRAINED or config.MODEL.IMAGENET_PRETRAINED) and (not config.MODEL.RESUME):
        # swin: PRETRAINED is a path → use load_pretrained for relative_position_bias etc.
        # resnet: weights are already loaded inside build_model (torchvision or custom path).
        if config.MODEL.TYPE == 'swin' and config.MODEL.PRETRAINED:
            load_pretrained(config, model_without_ddp, logger)
        validate_all(config, test_loaders, model, max_accuracy_dict)
        if config.EVAL_MODE:
            return

    mixup_fn = setup_mixup(config)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, train_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy_dict, optimizer, lr_scheduler, loss_scaler, logger)

        validate_all(config, test_loaders, model, max_accuracy_dict)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, dataloader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(dataloader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(dataloader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss = criterion(outputs, targets)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=True)
        optimizer.zero_grad()
        lr_scheduler.step_update(epoch * num_steps + idx)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def validate_all(config, dataloaders, model, max_accuracy_dict):
    for name, dataloader in dataloaders.items():
        acc1, acc2, loss = validate(config, dataloader, model, name)
        if name not in max_accuracy_dict:
            max_accuracy_dict[name] = 0.0
        max_accuracy_dict[name] = max(max_accuracy_dict[name], acc1)

        logger.info(f"Accuracy on the {len(dataloader.dataset)} {name} images: {acc1:.1f}%")
        logger.info(f'Max accuracy of {name}: {max_accuracy_dict[name]:.2f}%')


@torch.no_grad()
def validate(config, dataloader, model, name):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc2_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(dataloader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc2 = accuracy(output, target, topk=(1, 2))

        acc1 = reduce_tensor(acc1)
        acc2 = reduce_tensor(acc2)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc2_meter.update(acc2.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'{name}: [{idx}/{len(dataloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@2 {acc2_meter.val:.3f} ({acc2_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@2 {acc2_meter.avg:.3f}')
    return acc1_meter.avg, acc2_meter.avg, loss_meter.avg


if __name__ == '__main__':
    args, config, unparsed = parse_args()

    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    # linear scale the learning rate according to total batch size, may not be optimal
    total_bs = config.DATA.BATCH_SIZE * dist.get_world_size()
    scale = total_bs / 512.0
    orig_base_lr = config.TRAIN.BASE_LR
    linear_scaled_lr = orig_base_lr * scale
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * scale
    linear_scaled_min_lr = config.TRAIN.MIN_LR * scale

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    logger.info(f"LR linear-scaled by {scale:.4f} (total_bs={total_bs}/512): "
                f"BASE_LR {orig_base_lr:g} -> {linear_scaled_lr:g}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    if unparsed:
        logger.warning(f"Unparsed (ignored) CLI args: {unparsed}")

    main(config)

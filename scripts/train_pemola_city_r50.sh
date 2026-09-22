#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export DETECTRON2_DATASETS=datasets/data
python ./train_net.py \
--config-file configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml \
--num-gpus 3 \
MODEL.PEMOLA.PE_MODULATION True \
SOLVER.IMS_PER_BATCH 24

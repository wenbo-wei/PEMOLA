cd "$(dirname "$0")/.." || exit 1
export DETECTRON2_DATASETS=datasets/data
# Smoke-test version: MODEL.WEIGHTS "" skips loading a trained checkpoint, so the
# PEMOLA decoder runs with random init (SAM3 ViT still loads from SAM3.CHECKPOINT
# in the yaml). PQ/mIoU will be near random — this only verifies the eval pipeline.
# After training finishes, swap "" back to output/pemola_sam3/model_final.pth.
python ./train_net.py \
--config-file configs/coco_olac/panoptic-segmentation/sam3/pemola_SAM3_bs16_50ep.yaml \
--num-gpus 2 \
--eval-only

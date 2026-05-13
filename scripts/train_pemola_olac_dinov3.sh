cd "$(dirname "$0")/.." || exit 1
export DETECTRON2_DATASETS=datasets/data
python ./train_net.py \
--config-file configs/coco_olac/panoptic-segmentation/dinov3/pemola_DINOv3_bs16_50ep.yaml \
--num-gpus 2

cd "$(dirname "$0")/.." || exit 1
export DETECTRON2_DATASETS=datasets/data
python ./train_net.py \
--config-file configs/coco_olac/panoptic-segmentation/pemola_R50_bs16_50ep.yaml \
--num-gpus 2
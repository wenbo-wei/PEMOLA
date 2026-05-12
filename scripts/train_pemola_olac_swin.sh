cd "$(dirname "$0")/.." || exit 1
export DETECTRON2_DATASETS=datasets/data
python ./train_net.py \
--config-file configs/coco_olac/panoptic-segmentation/swin/pemola_swin_large_IN21k_384_bs16_100ep.yaml \
--num-gpus 2

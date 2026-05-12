cd "$(dirname "$0")/.." || exit 1
export DETECTRON2_DATASETS=datasets/data
python ./train_net.py \
--config-file configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml \
--num-gpus 2 \
--eval-only \
MODEL.WEIGHTS output/base_city_r50/model_final.pth

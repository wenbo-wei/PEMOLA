cd "$(dirname "$0")/.." || exit 1
export DETECTRON2_DATASETS=datasets/data
python ./predict.py \
--config configs/coco_olac/panoptic-segmentation/swin/pemola_swin_large_IN21k_384_bs16_100ep.yaml \
--weights output/pemola_swin/model_final.pth \
--input datasets/data/coco_olac/val/val \
--output-dir output/predictions_swin \
--metadata coco_2017_val_panoptic \
--cam-dir datasets/data/coco_olac_cam/cam_pt_val \
--occlusion-json datasets/data/coco_olac/val/occlusion_label_val.json \
--format pdf

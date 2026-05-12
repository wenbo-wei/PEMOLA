cd "$(dirname "$0")/.." || exit 1
export DETECTRON2_DATASETS=datasets/data
python ./predict.py \
--config configs/coco_olac/panoptic-segmentation/pemola_R50_bs16_50ep.yaml \
--weights output/model_final_pemola.pth \
--input datasets/data/coco_olac/val/val \
--output-dir output/predictions \
--metadata coco_2017_val_panoptic \
--cam-dir datasets/data/coco_olac_cam/cam_pt_val \
--occlusion-json datasets/data/coco_olac/val/occlusion_label_val.json \
--format pdf

cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=12
torchrun \
--nproc_per_node=2 \
--master_port=12345 \
occ_cls_predict.py \
--cfg configs/occlusion_cls/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml \
--opts \
MODEL.NUM_CLASSES 3 \
--resume output/swin_large_patch4_window12_384_22kto1k_finetune/ckpt_epoch_24.pth \
--data_path datasets/data/coco/train2017_blackbg \
--occlusion_ann datasets/data/coco_olac/train/occlusion_label_train.json \
--batch_size 32 \
--output output \
--tag prediction
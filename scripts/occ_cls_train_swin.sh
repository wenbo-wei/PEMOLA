cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=12
torchrun \
--nproc_per_node=2 \
--master_port=12345 \
occ_cls_train.py \
--cfg configs/occlusion_cls/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml \
--opts \
TRAIN.EPOCHS 30 \
TRAIN.BASE_LR 2e-5 \
TRAIN.WARMUP_LR 2e-8 \
TRAIN.MIN_LR 2e-7 \
MODEL.NUM_CLASSES 3 \
--dataset coco_olac_cls \
--data_path datasets/data/coco_olac_cls \
--batch_size 16 \
--output output \
--pretrained occlusion_cls/pretrained/swin_large_patch4_window12_384_22k.pth \
--tag test
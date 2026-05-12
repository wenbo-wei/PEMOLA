cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=12
torchrun \
--nproc_per_node=2 \
--master_port=12345 \
occ_cls_train.py \
--cfg configs/occlusion_cls/resnet/resnet50.yaml \
--opts \
TRAIN.EPOCHS 30 \
TRAIN.BASE_LR 1e-3 \
TRAIN.WARMUP_LR 1e-6 \
TRAIN.MIN_LR 1e-5 \
MODEL.NUM_CLASSES 3 \
--dataset coco_olac_cls \
--data_path datasets/data/coco_olac_cls \
--batch_size 64 \
--output output \
--imagenet_pretrained \
--tag test
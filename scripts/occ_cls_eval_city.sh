cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=12
torchrun \
--nproc_per_node=2 \
--master_port=12345 \
occ_cls_train.py \
--cfg configs/occlusion_cls/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml \
--opts \
MODEL.NUM_CLASSES 3 \
--dataset cityscapes \
--data_path datasets/data/cityscapes \
--resume output/swin_large_patch4_window12_384_22kto1k_finetune/test/ckpt_epoch_1.pth \
--batch_size 64 \
--output output \
--eval \
--tag test
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun \
--nproc_per_node=2 \
--master_port=12345 \
occ_cls_draw_cam.py \
--cfg configs/occlusion_cls/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml \
--opts \
MODEL.NUM_CLASSES 3 \
--dataset coco \
--data_path datasets/data/coco_olac/train/train \
--occlusion_ann datasets/data/coco_olac/train/occlusion_label_train.json \
--resume output/swin_large_patch4_window12_384_22kto1k_finetune/ep29.pth \
--batch_size 8 \
--output output \
--tag test
#!/usr/bin/env bash
# Copyright (c) 2026 Wenbo Wei.
# Licensed under the MIT License (see LICENSE).
#
# Add DINOv3 backbone support to an existing pemola conda env.
# Run AFTER install_env.sh has set up the base env.
# Re-runnable; safe to re-execute individual blocks.
set -euo pipefail

ENV_NAME=pemola
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DINOV3_CKPT_DEST="$PROJECT_ROOT/mask2former/pretrained/dinov3_vitl16.safetensors"

# ---- 0. activate pemola env -----------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- 1. extra pip deps (HuggingFace transformers + safetensors) -----------
# Unlike SAM3, DINOv3 has no pinned-dep conflicts with pemola, so a normal
# pip install is safe. transformers brings tokenizers as a dep.
pip install transformers safetensors

# ---- 2. download DINOv3 ViT-L/16 checkpoint (1.2 GB) ---------------------
# Tries HuggingFace first; if that fails (e.g., network or gated repo), the
# user can manually drop the safetensors at $DINOV3_CKPT_DEST and re-run.
mkdir -p "$(dirname "$DINOV3_CKPT_DEST")"
if [ ! -f "$DINOV3_CKPT_DEST" ]; then
    python - <<PY
from huggingface_hub import hf_hub_download
import shutil
src = hf_hub_download(
    repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
    filename="model.safetensors",
)
shutil.copy(src, r"$DINOV3_CKPT_DEST")
print(f"  [ckpt] copied to $DINOV3_CKPT_DEST")
PY
else
    echo "[install_dinov3] checkpoint already at $DINOV3_CKPT_DEST, skipping download"
fi

# ---- 3. smoke test --------------------------------------------------------
python - <<'PY'
import torch
from mask2former.config import add_maskformer2_config, add_pemola_config
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_backbone
import mask2former

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
add_pemola_config(cfg)
cfg.merge_from_file("configs/coco_olac/panoptic-segmentation/dinov3/pemola_DINOv3_bs16_50ep.yaml")
cfg.MODEL.BACKBONE.NAME = "D2DINOv3"
cfg.freeze()

bb = build_backbone(cfg).cuda().eval()
x = torch.randn(1, 3, 512, 512).cuda()
with torch.no_grad():
    out = bb(x)
print("  [smoke] D2DINOv3 forward OK; outputs:")
for k, v in out.items():
    print(f"          {k}: {tuple(v.shape)}")
PY

echo
echo "✓ DINOv3 backbone ready. Try:  bash scripts/train_pemola_olac_dinov3.sh"

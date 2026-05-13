#!/usr/bin/env bash
# Copyright (c) 2026 Wenbo Wei.
# Licensed under the MIT License (see LICENSE).
#
# Add SAM3 backbone support to an existing pemola conda env.
# Run AFTER install_env.sh has set up the base env.
# Re-runnable; safe to re-execute individual blocks.
set -euo pipefail

ENV_NAME=pemola
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SAM3_CLONE_DIR="${SAM3_CLONE_DIR:-/tmp/sam3_clone}"
SAM3_CKPT_DEST="$PROJECT_ROOT/mask2former/pretrained/sam3.pt"

# ---- 0. activate pemola env -----------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- 1. extra pip deps (sam3 needs einops; pinned-conflict deps in sam3's
#         requires.txt are intentionally skipped via --no-deps below) ------
pip install einops

# ---- 2. clone facebookresearch/sam3 (skip if already cloned) --------------
if [ ! -d "$SAM3_CLONE_DIR" ]; then
    git clone https://github.com/facebookresearch/sam3.git "$SAM3_CLONE_DIR"
else
    echo "[install_sam3] reusing existing clone at $SAM3_CLONE_DIR"
fi

# ---- 3. install sam3 as a regular (copy) pip package, ignoring its pinned
#         deps so it doesn't downgrade pemola's numpy / iopath / ftfy ------
pip uninstall -y sam3 || true
pip install --no-deps "$SAM3_CLONE_DIR"

# ---- 4. patch the installed copy to silence two upstream deprecation
#         warnings (pkg_resources, timm.models.layers) -- editing site-
#         packages keeps the user's clone clean ---------------------------
SAM3_SITE="$(python -c 'import sam3, os; print(os.path.dirname(sam3.__file__))')"

# 4a. timm.models.layers -> timm.layers (one bare import in this file)
python - <<PY
import pathlib, re
p = pathlib.Path(r"$SAM3_SITE/model/video_tracking_multiplex.py")
src = p.read_text()
old = "from timm.models.layers import trunc_normal_"
new = (
    "try:\n"
    "    from timm.layers import trunc_normal_\n"
    "except ModuleNotFoundError:\n"
    "    from timm.models.layers import trunc_normal_"
)
if old in src:
    p.write_text(src.replace(old, new, 1))
    print(f"  [patch] {p.name}: timm.layers fallback installed")
else:
    print(f"  [patch] {p.name}: already patched, skipping")
PY

# 4b. pkg_resources -> importlib.resources (3 call sites + import)
python - <<PY
import pathlib
p = pathlib.Path(r"$SAM3_SITE/model_builder.py")
src = p.read_text()
if "import pkg_resources" in src:
    src = src.replace(
        "import pkg_resources",
        "from importlib.resources import files as _pkg_files",
        1,
    )
    src = src.replace(
        'pkg_resources.resource_filename(\n'
        '            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"\n'
        '        )',
        'str(_pkg_files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))',
    )
    p.write_text(src)
    print(f"  [patch] {p.name}: pkg_resources -> importlib.resources")
else:
    print(f"  [patch] {p.name}: already patched, skipping")
PY

# ---- 5. download SAM3 image-model checkpoint (sam3.pt, ~3.3 GB) ---------
mkdir -p "$(dirname "$SAM3_CKPT_DEST")"
if [ ! -f "$SAM3_CKPT_DEST" ]; then
    python - <<PY
from huggingface_hub import hf_hub_download
import shutil
src = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt")
shutil.copy(src, r"$SAM3_CKPT_DEST")
print(f"  [ckpt] copied to $SAM3_CKPT_DEST")
PY
else
    echo "[install_sam3] checkpoint already at $SAM3_CKPT_DEST, skipping download"
fi

# ---- 6. smoke test --------------------------------------------------------
python - <<'PY'
import torch
import sam3
from mask2former.config import add_maskformer2_config, add_pemola_config
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_backbone
import mask2former

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
add_pemola_config(cfg)
cfg.merge_from_file("configs/coco_olac/panoptic-segmentation/sam3/pemola_SAM3_bs16_50ep.yaml")
cfg.MODEL.SAM3.USE_ACT_CHECKPOINT = False
cfg.MODEL.BACKBONE.NAME = "D2SAM3"
cfg.freeze()

bb = build_backbone(cfg).cuda().eval()
x = torch.randn(1, 3, 512, 512).cuda()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = bb(x)
print("  [smoke] D2SAM3 forward OK; outputs:")
for k, v in out.items():
    print(f"          {k}: {tuple(v.shape)}")
PY

echo
echo "✓ SAM3 backbone ready. Try:  bash scripts/train_pemola_olac_sam3.sh"

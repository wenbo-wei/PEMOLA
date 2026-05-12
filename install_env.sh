#!/usr/bin/env bash
# Copyright (c) 2026 Wenbo Wei.
# Licensed under the MIT License (see LICENSE).
# PEMOLA: Occlusion-Aware Panoptic Segmentation with Joint Position Embedding
# and Occlusion-Level Attention (ICME 2026).
#
# Reproducible install for the `pemola` conda env (RTX 4090, CUDA 13).
# Re-runnable; safe to re-execute individual blocks.
set -euo pipefail

ENV_NAME=pemola
PY_VER=3.12
CUDA_TOOLKIT_VER=13.0
ARCH=8.9   # compute capability for RTX 4090

# ---- 1. base conda env -----------------------------------------------------
conda create -y -n "$ENV_NAME" python="$PY_VER"
# from here on, run inside the env:
# conda activate "$ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- 2. toolchain (gcc/g++ 14 — system gcc 15 is too new for nvcc) --------
conda install -y -c conda-forge gxx_linux-64=14 gcc_linux-64=14

# ---- 3. full CUDA toolkit (minimal cuda-cudart-dev lacks cusparse.h) -------
conda install -y -c nvidia cuda-toolkit="$CUDA_TOOLKIT_VER"

# ---- 4. activate hook: CUDA_HOME + arch list ------------------------------
HOOK_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$HOOK_DIR"
cat > "$HOOK_DIR/cuda_env.sh" <<EOF
export CUDA_HOME=\$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="$ARCH"
EOF
# re-source so the rest of this script sees the vars
conda deactivate && conda activate "$ENV_NAME"

# ---- 5. PyTorch 2.11 cu130 (Tsinghua mirror; --index-url for official ----
# CUDA wheels) ------------------------------------------------------------
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.11.0 torchvision torchaudio

# ---- 6. detectron2 0.6 from source ---------------------------------------
pip install --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# ---- 7. project python deps ----------------------------------------------
pip install -r requirements.txt

# ---- 8. MSDeformAttn CUDA op (sources tracked in repo, already patched ---
# for new PyTorch API: .data<T>() -> .data_ptr<T>(), .type().is_cuda() ----
# -> .is_cuda(), unused q_col vars removed) ------------------------------
( cd mask2former/modeling/pixel_decoder/ops && bash make.sh )

# ---- 9. smoke test -------------------------------------------------------
python - <<'PY'
import torch, detectron2
from mask2former.modeling.pixel_decoder.ops.functions import MSDeformAttnFunction
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "device:", torch.cuda.get_device_name(0))
print("detectron2:", detectron2.__version__)
print("MSDeformAttn import OK")
PY

echo
echo "✓ pemola env ready. Activate with:  conda activate $ENV_NAME"

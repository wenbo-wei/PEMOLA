# Occlusion-Aware Panoptic Segmentation with Joint Position Embedding and Occlusion-Level Attention (ICME 2026)

[Wenbo Wei](https://scholar.google.com/citations?user=45fCx08AAAAJ&hl=en), [Jun Wang](https://scholar.google.com/citations?user=b_jEBHEAAAAJ&hl=en), [Shan Raza](https://scholar.google.com/citations?user=XfBoSP4AAAAJ&hl=en), [Abhir Bhalerao](https://scholar.google.com/citations?user=yPKOcgQAAAAJ&hl=en)

<p align="center">
  <img src="assets/architecture.png" width="900" alt="PEMOLA architecture">
</p>

## Highlights

- **Plug-and-play.** Drop into any transformer-based panoptic segmenter (Mask2Former, Mask DINO) without architectural changes.
- **Occlusion-aware.** Uses both spatial (Grad-CAM attention) and channel-wise (label embedding) occlusion cues.
- **Lightweight.** Adds only a small classifier inference + a single modulation step at the position embedding.
- **Generalises across datasets.** Validated on COCO-OLAC and the newly annotated Cityscapes-OLAC.

## Results

### Panoptic Segmentation on COCO-OLAC (ResNet-50)
| Model | $\text{PQ}$ | $\text{PQ}^{\text{Th}}$ | $\text{PQ}^{\text{St}}$ | $\text{AP}_{\text{pan}}^{\text{Th}}$ | $\text{mIoU}_{\text{pan}}$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| Mask2Former† | 40.7 | 44.5 | 35.0 | 30.0 | 54.2 |
| + PEMOLA | 41.5 | 45.2 | 35.9 | 30.4 | 54.8 |
| Mask DINO† | 44.0 | 48.5 | 37.3 | 33.5 | 53.4 |
| + PEMOLA | 44.8 | 49.4 | 37.8 | 34.2 | 55.3 |

### Panoptic Segmentation on Cityscapes-OLAC (ResNet-50)
| Model | $\text{PQ}$ | $\text{PQ}^{\text{Th}}$ | $\text{PQ}^{\text{St}}$ | $\text{AP}_{\text{pan}}^{\text{Th}}$ | $\text{mIoU}_{\text{pan}}$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| Mask2Former† | 61.5 | 54.0 | 66.9 | 35.2 | 76.1 |
| + PEMOLA | 62.3 | 55.4 | 67.2 | 38.5 | 77.4 |

† denotes retrained model.

## Installation

A reproducible install script for the `pemola` conda env (tested on RTX 4090, CUDA 13) is provided:

```bash
bash install_env.sh
```

The script sets up Python 3.12, gcc/g++ 14, CUDA toolkit 13.0, PyTorch 2.11+cu130, detectron2 0.6 (from source), the MSDeformAttn CUDA op, and the Python deps in `requirements.txt`. See the script for individual blocks if you want to run them piecewise.

## Data Preparation

PEMOLA is evaluated on **COCO-OLAC** ([Wei et al., 2025](https://github.com/wenbo-wei/COCO-OLAC)) and **Cityscapes-OLAC** (introduced in this work).

Place the datasets under `datasets/data/`:

```
datasets/data/
├── coco_olac/
│   ├── train/, val/
│   ├── annotations/        # panoptic + instance jsons
│   └── occlusion_label_{train,val}.json
└── cityscapes_olac/
    └── ...
```

For the **occlusion classifier**, training images are pre-processed by blackening non-object regions:

```bash
python tools/blacken_bg.py --dataset coco --data_path datasets/data/coco/train2017 \
    --ann_path datasets/data/coco/annotations/instances_train2017.json \
    --output_path datasets/data/coco_olac_cls/train
```

## Training

### 1. Occlusion classifier (Swin-L)
```bash
bash scripts/occ_cls_train_swin.sh
```

### 2. PEMOLA panoptic segmentation
```bash
# ResNet-50 backbone
bash scripts/train_pemola_olac_r50.sh

# Swin-L backbone
bash scripts/train_pemola_olac_swin.sh
```

Both scripts use `train_net.py` with the corresponding YAML in `configs/coco_olac/panoptic-segmentation/`.

## Evaluation

```bash
bash scripts/eval_pemola_olac_r50.sh    # R-50
bash scripts/eval_pemola_olac_swin.sh   # Swin-L
```

By default the eval scripts expect the trained checkpoint at `output/pemola/model_final.pth`; override with `MODEL.WEIGHTS <path>`.

## Acknowledgements

PEMOLA is built on top of the following open-source projects:

- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Meta, MIT) — backbone panoptic framework. Modified portions retain the original Meta copyright headers.
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer) (Microsoft, MIT) — Swin-L backbone and occlusion classifier. Modified portions retain the original Microsoft copyright headers.
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) (SenseTime, Apache 2.0) — MSDeformAttn CUDA op. Modified portions retain the original SenseTime copyright headers.
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) (Jacob Gildenblat, MIT) — used as an external dependency for extracting occlusion-level attention via Grad-CAM in `occ_cls_draw_cam.py`. No source code is copied; the library is imported through `pip install grad-cam`.

We thank the authors of these works for releasing their code.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

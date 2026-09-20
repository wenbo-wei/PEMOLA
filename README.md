<h1 align="center">Occlusion-Aware Panoptic Segmentation with Joint Position Embedding and Occlusion-Level Attention (ICME 2026)</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=45fCx08AAAAJ&hl=en">Wenbo Wei</a>,
  <a href="https://scholar.google.com/citations?user=b_jEBHEAAAAJ&hl=en">Jun Wang</a>,
  <a href="https://scholar.google.com/citations?user=yPKOcgQAAAAJ&hl=en">Shan Raza</a>,
  <a href="https://scholar.google.com/citations?user=XfBoSP4AAAAJ&hl=en">Abhir Bhalerao</a>
</p>

<p align="center">
  <a href="#citation"><img src="https://img.shields.io/badge/Paper-ICME%202026-b31b1b.svg" alt="Paper"></a>
  <a href="#pretrained-weights"><img src="https://img.shields.io/badge/Hugging%20Face-Weights-yellow.svg" alt="Hugging Face weights"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.11%2Bcu130-ee4c2c.svg" alt="PyTorch 2.11"></a>
  <a href="#acknowledgements"><img src="https://img.shields.io/badge/built%20on-Mask2Former%20%2F%20Mask%20DINO-4a90e2.svg" alt="Framework"></a>
</p>

<p align="center"><img src="assets/architecture.png" width="900" alt="PEMOLA architecture"></p>

## Table of Contents

- [News](#news)
- [Highlights](#highlights)
- [Model Zoo &amp; Results](#model-zoo--results)
- [Pretrained Weights](#pretrained-weights)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference &amp; Visualization](#inference--visualization)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## News

- [2026-09-20] [Model weights are available on Hugging Face](#pretrained-weights): PEMOLA + Mask2Former R50 on COCO-OLAC and Cityscapes-OLAC, Mask DINO R50 with and without PEMOLA on COCO-OLAC, and the Swin-L 384 occlusion classifier.
- [2026-03] Paper accepted to **ICME 2026**.

## Highlights

- **Plug-and-play.** Drops into any transformer-based panoptic segmenter (Mask2Former, Mask DINO) without architectural changes to the backbone or mask decoder.
- **Occlusion-aware queries.** Combines spatial (Grad-CAM) and channel-wise (label embedding) occlusion cues directly at the joint position embedding.
- **Lightweight.** A single classifier forward pass + one modulation step per query at inference, <1% added FLOPs.
- **Backbone-agnostic.** Validated end-to-end on ResNet-50. Drop-in configs for ResNet-101 and Swin-T/S/B/L are included for easy extension.
- **Generalises across datasets.** Improves PQ on both **COCO-OLAC** and the newly annotated **Cityscapes-OLAC**, with the largest gains on the heavily-occluded subset.

## Model Zoo &amp; Results

`†` denotes our re-trained baseline under identical training schedule for a fair comparison.
Segmentation results below use the *full* OLAC validation set; occlusion-subset results are reported in the paper.

### Panoptic Segmentation on COCO-OLAC

| Backbone | Method | $\text{PQ}$ | $\text{PQ}^{\text{Th}}$ | $\text{PQ}^{\text{St}}$ | $\text{AP}_{\text{pan}}^{\text{Th}}$ | $\text{mIoU}_{\text{pan}}$ | Config / Weights |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-50 | Mask2Former † | 40.7 | 44.5 | 35.0 | 30.0 | 54.2 | [config](configs/coco_olac/panoptic-segmentation/mask2former_COCO-OLAC_R50_bs16_50ep.yaml) |
| ResNet-50 | **+ PEMOLA** | **41.5** | **45.2** | **35.9** | **30.4** | **54.8** | [config](https://huggingface.co/weiwb/PEMOLA-Mask2Former-R50-COCO-OLAC/resolve/main/config.yaml) / [weights](https://huggingface.co/weiwb/PEMOLA-Mask2Former-R50-COCO-OLAC/resolve/main/model_final.pth) |
| ResNet-50 | Mask DINO † | 44.0 | 48.5 | 37.3 | 33.5 | 53.4 | [config](https://huggingface.co/weiwb/MaskDINO-R50-COCO-OLAC/resolve/main/config.yaml) / [weights](https://huggingface.co/weiwb/MaskDINO-R50-COCO-OLAC/resolve/main/model_final.pth) |
| ResNet-50 | **+ PEMOLA** | **44.8** | **49.4** | **37.8** | **34.2** | **55.3** | [config](https://huggingface.co/weiwb/PEMOLA-MaskDINO-R50-COCO-OLAC/resolve/main/config.yaml) / [weights](https://huggingface.co/weiwb/PEMOLA-MaskDINO-R50-COCO-OLAC/resolve/main/model_final.pth) |

### Panoptic Segmentation on Cityscapes-OLAC

| Backbone | Method | $\text{PQ}$ | $\text{PQ}^{\text{Th}}$ | $\text{PQ}^{\text{St}}$ | $\text{AP}_{\text{pan}}^{\text{Th}}$ | $\text{mIoU}_{\text{pan}}$ | Config / Weights |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-50 | Mask2Former † | 61.5 | 54.0 | 66.9 | 35.2 | 76.1 | [training preset](configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml) |
| ResNet-50 | **+ PEMOLA** | **62.3** | **55.4** | **67.2** | **38.5** | **77.4** | [config](https://huggingface.co/weiwb/PEMOLA-Mask2Former-R50-Cityscapes-OLAC/resolve/main/config.yaml) / [weights](https://huggingface.co/weiwb/PEMOLA-Mask2Former-R50-Cityscapes-OLAC/resolve/main/model_final.pth) |

The Cityscapes checkpoints use a 60,000-iteration schedule with a global batch size of 24; use their bundled configurations.

### Occlusion Classifier (auxiliary)

Top-1 accuracies (%) reported in the paper on the COCO-OLAC test split with background-blackened inputs, for three-way (low / mid / high) occlusion classification.

<table>
<thead>
<tr>
  <th align="left">Backbone</th>
  <th align="center">Pretraining</th>
  <th align="center">Input</th>
  <th align="center">Top-1 Acc (%)</th>
  <th align="center">Config / Weights</th>
</tr>
</thead>
<tbody>
<tr><td align="left">ResNet-50</td>       <td align="center">ImageNet-1K</td>       <td align="center">224</td> <td align="center">70.3</td>     <td align="center"><a href="configs/occlusion_cls/resnet/resnet50.yaml">config</a></td></tr>
<tr><td align="left">ResNet-101</td>      <td align="center">ImageNet-1K</td>       <td align="center">224</td> <td align="center">70.8</td>     <td align="center"><a href="configs/occlusion_cls/resnet/resnet101.yaml">config</a></td></tr>
<tr><td align="left">Swin-T</td>          <td align="center">ImageNet-22K</td>      <td align="center">224</td> <td align="center">71.6</td>     <td align="center"><a href="configs/occlusion_cls/swin/swin_tiny_patch4_window7_224_22k.yaml">config</a></td></tr>
<tr><td align="left">Swin-S</td>          <td align="center">ImageNet-22K</td>      <td align="center">224</td> <td align="center">71.4</td>     <td align="center"><a href="configs/occlusion_cls/swin/swin_small_patch4_window7_224_22k.yaml">config</a></td></tr>
<tr><td align="left">Swin-B</td>          <td align="center">ImageNet-22K</td>      <td align="center">224</td> <td align="center">71.7</td>     <td align="center"><a href="configs/occlusion_cls/swin/swin_base_patch4_window7_224_22k.yaml">config</a></td></tr>
<tr><td align="left">Swin-B</td>          <td align="center">ImageNet-22K</td> <td align="center">384</td> <td align="center">75.3</td>     <td align="center"><a href="configs/occlusion_cls/swin/swin_base_patch4_window12_384_22kto1k_finetune.yaml">config</a></td></tr>
<tr><td align="left">Swin-L</td>          <td align="center">ImageNet-22K</td>      <td align="center">224</td> <td align="center">73.0</td>     <td align="center"><a href="configs/occlusion_cls/swin/swin_large_patch4_window7_224_22k.yaml">config</a></td></tr>
<tr><td align="left"><b>Swin-L</b></td>    <td align="center">ImageNet-22K</td> <td align="center">384</td> <td align="center"><b>75.3</b></td> <td align="center"><a href="https://huggingface.co/weiwb/PEMOLA-Occlusion-Swin-L-384/resolve/main/config.yaml">config</a> / <a href="https://huggingface.co/weiwb/PEMOLA-Occlusion-Swin-L-384/resolve/main/ep29.pth">weights</a></td></tr>
</tbody>
</table>

## Pretrained Weights

Download checkpoints and their matching configurations from Hugging Face:

| Model | Model card | Config / Weights |
| :--- | :---: | :---: |
| PEMOLA + Mask2Former R50 — COCO-OLAC | [Hugging Face](https://huggingface.co/weiwb/PEMOLA-Mask2Former-R50-COCO-OLAC) | [config](https://huggingface.co/weiwb/PEMOLA-Mask2Former-R50-COCO-OLAC/resolve/main/config.yaml) / [weights](https://huggingface.co/weiwb/PEMOLA-Mask2Former-R50-COCO-OLAC/resolve/main/model_final.pth) |
| PEMOLA + Mask2Former R50 — Cityscapes-OLAC | [Hugging Face](https://huggingface.co/weiwb/PEMOLA-Mask2Former-R50-Cityscapes-OLAC) | [config](https://huggingface.co/weiwb/PEMOLA-Mask2Former-R50-Cityscapes-OLAC/resolve/main/config.yaml) / [weights](https://huggingface.co/weiwb/PEMOLA-Mask2Former-R50-Cityscapes-OLAC/resolve/main/model_final.pth) |
| Mask DINO R50 — COCO-OLAC | [Hugging Face](https://huggingface.co/weiwb/MaskDINO-R50-COCO-OLAC) | [config](https://huggingface.co/weiwb/MaskDINO-R50-COCO-OLAC/resolve/main/config.yaml) / [weights](https://huggingface.co/weiwb/MaskDINO-R50-COCO-OLAC/resolve/main/model_final.pth) |
| PEMOLA + Mask DINO R50 — COCO-OLAC | [Hugging Face](https://huggingface.co/weiwb/PEMOLA-MaskDINO-R50-COCO-OLAC) | [config](https://huggingface.co/weiwb/PEMOLA-MaskDINO-R50-COCO-OLAC/resolve/main/config.yaml) / [weights](https://huggingface.co/weiwb/PEMOLA-MaskDINO-R50-COCO-OLAC/resolve/main/model_final.pth) |
| Occlusion classifier — Swin-L 384 | [Hugging Face](https://huggingface.co/weiwb/PEMOLA-Occlusion-Swin-L-384) | [config](https://huggingface.co/weiwb/PEMOLA-Occlusion-Swin-L-384/resolve/main/config.yaml) / [weights](https://huggingface.co/weiwb/PEMOLA-Occlusion-Swin-L-384/resolve/main/ep29.pth) |

From the PEMOLA repository root, download the files with:

```bash
python -m pip install huggingface_hub

hf download weiwb/PEMOLA-Mask2Former-R50-COCO-OLAC \
    --local-dir checkpoints/pemola-coco-olac
hf download weiwb/PEMOLA-Mask2Former-R50-Cityscapes-OLAC \
    --local-dir checkpoints/pemola-cityscapes-olac
hf download weiwb/MaskDINO-R50-COCO-OLAC \
    --local-dir checkpoints/maskdino-coco-olac
hf download weiwb/PEMOLA-MaskDINO-R50-COCO-OLAC \
    --local-dir checkpoints/pemola-maskdino-coco-olac
hf download weiwb/PEMOLA-Occlusion-Swin-L-384 \
    --local-dir checkpoints/pemola-occlusion-swin-l
```

Use the configuration shipped with each checkpoint. Mask DINO weights use the dedicated [Mask DINO implementation](https://github.com/wenbo-wei/MaskDINO). The segmentation commands below use this repository's Mask2Former implementation.

**Required inputs:** PEMOLA segmentation uses an image, a per-image CAM tensor, and a `low` / `mid` / `high` occlusion label. The auxiliary classifier supports label prediction and CAM preparation; the current CAM workflow also requires background-blackened images. The [classifier model card](https://huggingface.co/weiwb/PEMOLA-Occlusion-Swin-L-384) provides the preparation commands. See [Inference &amp; Visualization](#inference--visualization) for segmentation usage.

## Installation

Set up the `pemola` conda environment with PyTorch, Detectron2, and the MSDeformAttn operator:

```bash
bash install_env.sh
```

Before running the installer, replace `panopticapi` in `requirements.txt` with `git+https://github.com/cocodataset/panopticapi.git`. Set the installer's `ARCH` to your GPU architecture (default: `8.9` for RTX 4090).

## Data Preparation

PEMOLA is evaluated on **COCO-OLAC** ([Wei *et al.*, 2025](https://github.com/wenbo-wei/COCO-OLAC)) and **Cityscapes-OLAC** (introduced in this work).
Place the datasets under `datasets/data/`:

```
datasets/data/
├── coco_olac/
│   ├── train/, val/                      # RGB images
│   ├── annotations/                      # panoptic + instance JSONs
│   └── occlusion_label_{train,val}.json  # per-instance occlusion level
├── cityscapes/                           # official Cityscapes (leftImg8bit/, gtFine/)
├── cityscapes_olac/                      # built by tools/prepare_cityscapes_olac.py
│   ├── leftImg8bit/{train,val}_{low,mid,high}/
│   └── gtFine/                           # per-level gtFine, panoptic JSONs, occlusion labels
└── coco_olac_cls/
    ├── train/, val/, test/               # flat image directories
    ├── train_blackbg/, val_blackbg/, test_blackbg/
    └── occlusion_label_{train,val,test}.json
```

### Cityscapes-OLAC

The occlusion-level annotations introduced in this work are shipped in this repository under
[`datasets/cityscapes_olac/`](datasets/cityscapes_olac) — `occlusion_label_{train,val}.json`, one
`low / mid / high` level per image (2975 train / 500 val), following the same labelling protocol as COCO-OLAC.
Cityscapes itself may not be redistributed, so build the per-level subsets locally:

```bash
# 1. Download leftImg8bit + gtFine from https://www.cityscapes-dataset.com/
#    into datasets/data/cityscapes/
# 2. Generate the panoptic format:
CITYSCAPES_DATASET=datasets/data/cityscapes \
    python -m cityscapesscripts.preparation.createPanopticImgs
# 3. Slice into occlusion-level subsets (symlinks by default, --copy for real copies):
python tools/prepare_cityscapes_olac.py
```

For the **occlusion classifier**, training images are pre-processed by blackening non-object regions
(this restricts the classifier's receptive field to the instance and removes scene-context bias):

```bash
python tools/blacken_bg.py \
    --dataset    coco \
    --data_path  datasets/data/coco/train2017 \
    --ann_path   datasets/data/coco/annotations/instances_train2017.json \
    --output_path datasets/data/coco_olac_cls/train
```

Set `DETECTRON2_DATASETS=datasets/data` (already exported by every script under `scripts/`) so that detectron2 resolves dataset paths correctly.

## Training

The training scripts under `scripts/` wrap `train_net.py` (Mask2Former) or `occ_cls_train.py` (classifier) and read the corresponding YAML in `configs/`.

### 1. Occlusion classifier

```bash
bash scripts/occ_cls_train_swin.sh    # Swin-L,  used in main results
bash scripts/occ_cls_train_res.sh     # ResNet baselines (50 / 101)
```

### 2. PEMOLA panoptic segmentation on COCO-OLAC

```bash
bash scripts/train_pemola_olac_r50.sh      # ResNet-50
bash scripts/train_pemola_olac_swin.sh     # Swin-L  (IN-22K, 384)
```

> Scripts in `scripts/` ship with `--num-gpus 2` for typical local development. The paper numbers are reproduced with `--num-gpus 3` on **3× A100 (40 GB)**. Adjust the flag (and `SOLVER.IMS_PER_BATCH` in the YAML) to match your hardware.

### 3. PEMOLA panoptic segmentation on Cityscapes-OLAC

Use the configs under `configs/cityscapes/panoptic-segmentation/` with the same `train_net.py` interface, for example:

```bash
export DETECTRON2_DATASETS=datasets/data
python train_net.py \
    --config-file configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml \
    --num-gpus 3
```

## Evaluation

```bash
bash scripts/eval_pemola_olac_r50.sh       # ResNet-50
bash scripts/eval_pemola_olac_swin.sh      # Swin-L
bash scripts/eval_base_city_r50.sh         # Cityscapes-OLAC baseline
```

The wrapper scripts use their configured `output/` checkpoint paths. To evaluate the downloaded COCO-OLAC checkpoint after preparing the dataset, CAMs, and occlusion labels, invoke the entry point directly:

```bash
export DETECTRON2_DATASETS=datasets/data
python train_net.py \
    --config-file checkpoints/pemola-coco-olac/config.yaml \
    --num-gpus 1 \
    --eval-only \
    MODEL.WEIGHTS checkpoints/pemola-coco-olac/model_final.pth
```

## Inference &amp; Visualization

**Per-image Mask2Former panoptic prediction** (writes a colourised PNG):

```bash
python predict.py \
    --config checkpoints/pemola-coco-olac/config.yaml \
    --weights checkpoints/pemola-coco-olac/model_final.pth \
    --input /path/to/image.jpg \
    --output-dir output/pemola-predictions \
    --cam-dir /path/to/cam_pt \
    --occlusion-json /path/to/occlusion_labels.json \
    --format png
```

For `image.jpg`, supply `cam_pt/image.pt` and a JSON entry such as `{"image": "mid"}`. `--input` accepts a single image or a directory. For Cityscapes, use the downloaded `pemola-cityscapes-olac` configuration and weights; the [model card](https://huggingface.co/weiwb/PEMOLA-Mask2Former-R50-Cityscapes-OLAC) explains how to align the `_leftImg8bit` filename suffix with CAM and label keys.

**Occlusion labels and Grad-CAM inputs:** The [released Swin-L 384 model card](https://huggingface.co/weiwb/PEMOLA-Occlusion-Swin-L-384) provides commands that load `checkpoints/pemola-occlusion-swin-l/ep29.pth` for label prediction and CAM generation. Use an empty label JSON for pure predictions, since the prediction script preserves any supplied labels. CAM generation requires matching original and background-blackened image directories.

## Citation

If you find PEMOLA useful for your research, please consider citing:

```bibtex
@inproceedings{wei2026pemola,
  title     = {Occlusion-Aware Panoptic Segmentation with Joint Position Embedding and Occlusion-Level Attention},
  author    = {Wei, Wenbo and Wang, Jun and Raza, Shan and Bhalerao, Abhir},
  booktitle = {IEEE International Conference on Multimedia and Expo (ICME)},
  year      = {2026}
}
```

For the **COCO-OLAC** benchmark used in our experiments, please also cite:

```bibtex
@article{wei2025cocoolac,
  title   = {COCO-OLAC: An Occlusion-Level Annotated COCO Benchmark for Panoptic Segmentation},
  author  = {Wei, Wenbo and others},
  journal = {arXiv preprint},
  year    = {2025}
}
```

## Acknowledgements

PEMOLA is built on top of the following excellent open-source projects.
Modified portions retain the original copyright headers, in accordance with each project's licence.

- [**Mask2Former**](https://github.com/facebookresearch/Mask2Former) (Meta, MIT) — base panoptic framework.
- [**Mask DINO**](https://github.com/IDEA-Research/MaskDINO) (IDEA, Apache 2.0) — alternative panoptic baseline used in our Mask DINO experiments.
- [**Swin Transformer**](https://github.com/microsoft/Swin-Transformer) (Microsoft, MIT) — Swin-L backbone and occlusion classifier.
- [**Deformable DETR**](https://github.com/fundamentalvision/Deformable-DETR) (SenseTime, Apache 2.0) — MSDeformAttn CUDA op.
- [**pytorch-grad-cam**](https://github.com/jacobgil/pytorch-grad-cam) (Jacob Gildenblat, MIT) — used as an external dependency for extracting occlusion-level attention via Grad-CAM in `occ_cls_draw_cam.py`. No source code is copied. The library is imported through `pip install grad-cam`.

We thank the authors of these works for releasing their code.

## License

This project is released under the **MIT License** — see [LICENSE](LICENSE) for the full text.
Code adapted from third-party projects retains its original licence and copyright notice.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

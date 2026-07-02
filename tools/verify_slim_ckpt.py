"""Sanity-check the slimmed occlusion-classifier checkpoints.

For each (yaml, ckpt) pair:
  1. Build the model from the yaml config.
  2. Load the slimmed checkpoint and check for missing / unexpected keys.
  3. Forward a dummy tensor and confirm output shape == [B, NUM_CLASSES].
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from occlusion_cls.config import get_config  # noqa: E402
from occlusion_cls.models import build_model  # noqa: E402


PAIRS = [
    ("configs/occlusion_cls/resnet/resnet50.yaml",
     "output/occ_classifier/resnet50/ep28.pth"),
    ("configs/occlusion_cls/resnet/resnet101.yaml",
     "output/occ_classifier/resnet101/ep28.pth"),
    ("configs/occlusion_cls/swin/swin_tiny_patch4_window7_224_22k.yaml",
     "output/occ_classifier/swin_tiny_patch4_window7_224_22k/ep19.pth"),
    ("configs/occlusion_cls/swin/swin_small_patch4_window7_224_22k.yaml",
     "output/occ_classifier/swin_small_patch4_window7_224_22k/ep19.pth"),
    ("configs/occlusion_cls/swin/swin_base_patch4_window7_224_22k.yaml",
     "output/occ_classifier/swin_base_patch4_window7_224_22k/ep17.pth"),
    ("configs/occlusion_cls/swin/swin_base_patch4_window12_384_22kto1k_finetune.yaml",
     "output/occ_classifier/swin_base_patch4_window12_384_22kto1k_finetune/ep19.pth"),
    ("configs/occlusion_cls/swin/swin_large_patch4_window7_224_22k.yaml",
     "output/occ_classifier/swin_large_patch4_window7_224_22k/ep20.pth"),
    ("configs/occlusion_cls/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml",
     "output/occ_classifier/swin_large_patch4_window12_384_22kto1k_finetune/ep29.pth"),
]


def fake_args(cfg_path: str):
    return SimpleNamespace(cfg=cfg_path, opts=None)


def verify(cfg_path: str, ckpt_path: str) -> bool:
    cfg_abs = PROJECT_ROOT / cfg_path
    ckpt_abs = PROJECT_ROOT / ckpt_path
    if not cfg_abs.exists():
        print(f"  [FAIL] missing yaml: {cfg_path}")
        return False
    if not ckpt_abs.exists():
        print(f"  [FAIL] missing ckpt: {ckpt_path}")
        return False

    config = get_config(fake_args(str(cfg_abs)))
    model = build_model(config)
    model.eval()

    ckpt = torch.load(str(ckpt_abs), map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        print(f"  [FAIL] no 'model' key in ckpt")
        return False

    msg = model.load_state_dict(ckpt["model"], strict=False)
    if msg.missing_keys:
        print(f"  [FAIL] missing keys: {msg.missing_keys[:5]}...")
        return False
    if msg.unexpected_keys:
        print(f"  [FAIL] unexpected keys: {msg.unexpected_keys[:5]}...")
        return False

    with torch.no_grad():
        x = torch.randn(2, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
        out = model(x)

    expected = (2, config.MODEL.NUM_CLASSES)
    if tuple(out.shape) != expected:
        print(f"  [FAIL] wrong output shape: got {tuple(out.shape)}, expected {expected}")
        return False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [ OK ] strict load, fwd shape={tuple(out.shape)}, params={n_params:,}")
    return True


def main():
    passes = 0
    for cfg_path, ckpt_path in PAIRS:
        name = Path(ckpt_path).parent.name
        print(f"\n--- {name} ---")
        print(f"  cfg : {cfg_path}")
        print(f"  ckpt: {ckpt_path}")
        if verify(cfg_path, ckpt_path):
            passes += 1

    print(f"\n{passes}/{len(PAIRS)} checkpoints verified.")
    sys.exit(0 if passes == len(PAIRS) else 1)


if __name__ == "__main__":
    main()

# Copyright (c) Wenbo Wei.
"""
DINOv3 ViT image encoder as a detectron2 backbone for Mask2Former / PEMOLA.

Wraps HuggingFace `transformers.DINOv3ViTModel` (ViT-L/16 pretrained on LVD-1689M
by default), strips its [CLS] + register tokens to recover a dense spatial grid,
then applies a SimpleFPN-style neck to produce 4 multi-scale feature maps
("res2"/"res3"/"res4"/"res5") at strides 4/8/16/32 — the input format expected
by MSDeformAttnPixelDecoder.

Unlike SAM3, DINOv3 supports variable input resolutions natively (RoPE +
interpolated abs pos), so no internal resize is required. Likewise it has no
hardcoded bf16 fused-op constraint, so we don't need an explicit autocast wrap
inside the forward path — outer AMP handles dtypes.
"""

import logging
import math
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)


class _SimpleFPNHead(nn.Module):
    """One branch of a ViTDet-style Simple Feature Pyramid.

    Takes (B, dim, H/p, W/p) from the ViT trunk and rescales spatial by the
    given scale_factor (4.0 = 4x up, 0.5 = 2x down), then projects to d_model
    with 1x1 -> 3x3 convs.
    """

    def __init__(self, in_dim: int, d_model: int, scale: float):
        super().__init__()
        layers = nn.Sequential()
        if scale == 4.0:
            layers.add_module(
                "dconv_2x2_0", nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2)
            )
            layers.add_module("gelu", nn.GELU())
            layers.add_module(
                "dconv_2x2_1",
                nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
            )
            out_dim = in_dim // 4
        elif scale == 2.0:
            layers.add_module(
                "dconv_2x2", nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2)
            )
            out_dim = in_dim // 2
        elif scale == 1.0:
            out_dim = in_dim
        elif scale == 0.5:
            layers.add_module("maxpool_2x2", nn.MaxPool2d(kernel_size=2, stride=2))
            out_dim = in_dim
        else:
            raise NotImplementedError(f"scale_factor={scale} not supported")
        layers.add_module("conv_1x1", nn.Conv2d(out_dim, d_model, kernel_size=1, bias=True))
        layers.add_module(
            "conv_3x3", nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=True)
        )
        self.layers = layers

    def forward(self, x):
        return self.layers(x)


@BACKBONE_REGISTRY.register()
class D2DINOv3(Backbone):
    """DINOv3 ViT + SimpleFPN packaged as a detectron2 Backbone."""

    def __init__(self, cfg, input_shape):
        super().__init__()
        cfg_d = cfg.MODEL.DINOV3

        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        vit_cfg = DINOv3ViTConfig(
            hidden_size=cfg_d.EMBED_DIM,
            num_hidden_layers=cfg_d.DEPTH,
            num_attention_heads=cfg_d.NUM_HEADS,
            intermediate_size=cfg_d.MLP_DIM,
            patch_size=cfg_d.PATCH_SIZE,
            image_size=cfg_d.IMG_SIZE,
            num_register_tokens=cfg_d.NUM_REGISTER_TOKENS,
        )
        self.vit = DINOv3ViTModel(vit_cfg)
        self._patch_size = cfg_d.PATCH_SIZE
        self._n_special = 1 + cfg_d.NUM_REGISTER_TOKENS  # cls + register tokens to skip

        # SimpleFPN: 4 parallel branches producing strides 4/8/16/32 from the
        # single-scale ViT output (which sits at stride=patch_size).
        self.sfp = nn.ModuleList(
            [_SimpleFPNHead(cfg_d.EMBED_DIM, cfg_d.OUT_CHANNELS, s) for s in (4.0, 2.0, 1.0, 0.5)]
        )

        if cfg_d.CHECKPOINT:
            self._load_dinov3_checkpoint(cfg_d.CHECKPOINT)

        if cfg_d.FREEZE_BACKBONE:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

        self._out_features = ("res2", "res3", "res4", "res5")
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {n: cfg_d.OUT_CHANNELS for n in self._out_features}
        self._size_divisibility = cfg_d.SIZE_DIVISIBILITY
        self._freeze_backbone = cfg_d.FREEZE_BACKBONE

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility

    def output_shape(self):
        return {
            n: ShapeSpec(
                channels=self._out_feature_channels[n], stride=self._out_feature_strides[n]
            )
            for n in self._out_features
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_backbone:
            self.vit.eval()
        return self

    def _load_dinov3_checkpoint(self, ckpt_path: str) -> None:
        ckpt_path = os.path.expanduser(ckpt_path)
        # safetensors-only path; we don't accept legacy pickle ckpts here
        from safetensors.torch import load_file as _load_st

        with PathManager.open(ckpt_path, "rb"):
            pass  # ensure file exists / is readable through PathManager
        sd_raw = _load_st(ckpt_path)

        # HF DINOv3ViTModel wraps the transformer trunk under a `model.` attribute,
        # but the published .safetensors files store it without that prefix.
        # Remap: `layer.*` and other trunk keys get `model.` prefixed; embeddings
        # and the final norm keep their bare names.
        sd = {}
        for k, v in sd_raw.items():
            if k.startswith("layer."):
                sd["model." + k] = v
            else:
                sd[k] = v

        # Load into the wrapped HF model. The SFP convs are randomly initialized
        # (no DINOv3 pretrained SFP), so they show up as missing — that's fine.
        # Use vit submodule directly so missing/unexpected stats only cover the ViT.
        m, u = self.vit.load_state_dict(sd, strict=False)
        logger.info(
            f"[D2DINOv3] Loaded {ckpt_path}: ViT(missing={len(m)}, unexpected={len(u)})"
        )
        if len(sd) == 0:
            logger.warning("[D2DINOv3] No weights matched — check checkpoint format.")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        if H % self._patch_size or W % self._patch_size:
            raise RuntimeError(
                f"D2DINOv3 input H={H}, W={W} not divisible by patch_size={self._patch_size}; "
                f"check SIZE_DIVISIBILITY in the yaml."
            )

        # No grad needed through frozen trunk — saves backbone activation memory.
        if self._freeze_backbone and torch.is_grad_enabled():
            with torch.no_grad():
                hidden = self.vit(x).last_hidden_state  # (B, 1+R+N, C)
        else:
            hidden = self.vit(x).last_hidden_state

        # Drop cls + register tokens, reshape patch tokens back to a 2D grid.
        patch_tokens = hidden[:, self._n_special :, :]  # (B, N, C)
        Hp, Wp = H // self._patch_size, W // self._patch_size
        x_last = patch_tokens.transpose(1, 2).reshape(B, -1, Hp, Wp)  # (B, C, Hp, Wp)

        # SFP -> 4 outputs at native sub-strides; bilinear-snap each to PEMOLA's
        # expected stride (since DINOv3 patch=16 already aligns with 4/8/16/32
        # at H=multiple of 32, the snap is usually a no-op).
        target_strides = (4, 8, 16, 32)
        out: Dict[str, torch.Tensor] = {}
        for i, head in enumerate(self.sfp):
            f = head(x_last)
            th = max(1, math.ceil(H / target_strides[i]))
            tw = max(1, math.ceil(W / target_strides[i]))
            if f.shape[-2:] != (th, tw):
                f = F.interpolate(f, size=(th, tw), mode="bilinear", align_corners=False)
            out[self._out_features[i]] = f
        return out

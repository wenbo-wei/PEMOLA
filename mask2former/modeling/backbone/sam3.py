# Copyright (c) Wenbo Wei.
"""
SAM3 image encoder as a detectron2 backbone for Mask2Former / PEMOLA.

Wraps the SAM3 ViT trunk (sam3.model.vitdet.ViT) plus its SimpleFPN-style
convs (from sam3.model.necks.Sam3DualViTDetNeck) and exposes a 4-level
output ("res2"/"res3"/"res4"/"res5") at strides 4/8/16/32 — the input format
expected by MSDeformAttnPixelDecoder.

The native SAM3 patch size is 14, so SFP outputs sit at strides 3.5/7/14/28.
We bilinear-resize each level to the exact target stride so downstream
Mask2Former code sees clean integer strides.
"""

import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)


@BACKBONE_REGISTRY.register()
class D2SAM3(Backbone):
    """SAM3 ViT + SimpleFPN packaged as a detectron2 Backbone."""

    def __init__(self, cfg, input_shape):
        super().__init__()
        cfg_sam3 = cfg.MODEL.SAM3

        from sam3.model.necks import Sam3DualViTDetNeck
        from sam3.model.position_encoding import PositionEmbeddingSine
        from sam3.model.vitdet import ViT

        self.vit = ViT(
            img_size=cfg_sam3.IMG_SIZE,
            pretrain_img_size=cfg_sam3.PRETRAIN_IMG_SIZE,
            patch_size=cfg_sam3.PATCH_SIZE,
            embed_dim=cfg_sam3.EMBED_DIM,
            depth=cfg_sam3.DEPTH,
            num_heads=cfg_sam3.NUM_HEADS,
            mlp_ratio=cfg_sam3.MLP_RATIO,
            norm_layer="LayerNorm",
            drop_path_rate=cfg_sam3.DROP_PATH_RATE,
            qkv_bias=True,
            use_abs_pos=True,
            tile_abs_pos=True,
            global_att_blocks=tuple(cfg_sam3.GLOBAL_ATT_BLOCKS),
            rel_pos_blocks=(),
            use_rope=True,
            use_interp_rope=True,
            window_size=cfg_sam3.WINDOW_SIZE,
            pretrain_use_cls_token=True,
            retain_cls_token=False,
            ln_pre=True,
            ln_post=False,
            return_interm_layers=False,
            bias_patch_embed=False,
            use_act_checkpoint=cfg_sam3.USE_ACT_CHECKPOINT,
        )

        # PositionEmbeddingSine is required by the neck constructor but its
        # output is unused here — Mask2Former adds its own positional encoding.
        dummy_pe = PositionEmbeddingSine(
            num_pos_feats=cfg_sam3.OUT_CHANNELS // 2,
            normalize=True,
        )
        self.neck = Sam3DualViTDetNeck(
            trunk=self.vit,
            position_encoding=dummy_pe,
            d_model=cfg_sam3.OUT_CHANNELS,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            add_sam2_neck=False,
        )

        if cfg_sam3.CHECKPOINT:
            self._load_sam3_checkpoint(cfg_sam3.CHECKPOINT)

        if cfg_sam3.FREEZE_BACKBONE:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()
        if cfg_sam3.FREEZE_NECK:
            for p in self.neck.convs.parameters():
                p.requires_grad = False

        self._out_features = ("res2", "res3", "res4", "res5")
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {
            name: cfg_sam3.OUT_CHANNELS for name in self._out_features
        }
        self._size_divisibility = cfg_sam3.SIZE_DIVISIBILITY
        self._freeze_backbone = cfg_sam3.FREEZE_BACKBONE
        # SAM3 RoPE is precomputed for img_size/patch_size grid, so the ViT
        # must run at a fixed resolution. We resize the input to this size and
        # snap SFP outputs back to PEMOLA's strides on the original size.
        self._vit_input_size = cfg_sam3.IMG_SIZE

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_backbone:
            self.vit.eval()
        return self

    def _load_sam3_checkpoint(self, ckpt_path: str) -> None:
        ckpt_path = os.path.expanduser(ckpt_path)
        with PathManager.open(ckpt_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        trunk_prefixes = (
            "detector.backbone.vision_backbone.trunk.",
            "backbone.vision_backbone.trunk.",
            "sam3_model.backbone.vision_backbone.trunk.",
            # legacy / alternative naming
            "detector.backbone.visual.trunk.",
            "backbone.visual.trunk.",
        )
        conv_prefixes = (
            "detector.backbone.vision_backbone.convs.",
            "backbone.vision_backbone.convs.",
            "sam3_model.backbone.vision_backbone.convs.",
            "detector.backbone.visual.convs.",
            "backbone.visual.convs.",
        )
        vit_sd, neck_sd = {}, {}
        for k, v in ckpt.items():
            matched = False
            for p in trunk_prefixes:
                if k.startswith(p):
                    sub = k[len(p):]
                    # freqs_cis is a deterministic buffer regenerated at init
                    # from input_size/theta. Skip to let our computed one stand.
                    if sub.endswith(".freqs_cis") or sub.endswith(
                        ".freqs_cis_real"
                    ) or sub.endswith(".freqs_cis_imag"):
                        matched = True
                        break
                    vit_sd[sub] = v
                    matched = True
                    break
            if matched:
                continue
            for p in conv_prefixes:
                if k.startswith(p):
                    neck_sd[k[len(p):]] = v
                    break

        m1, u1 = self.vit.load_state_dict(vit_sd, strict=False)
        m2, u2 = self.neck.convs.load_state_dict(neck_sd, strict=False)
        logger.info(
            f"[D2SAM3] Loaded {ckpt_path}: ViT(missing={len(m1)}, unexpected={len(u1)}) "
            f"SFP(missing={len(m2)}, unexpected={len(u2)})"
        )
        if len(vit_sd) == 0:
            logger.warning(
                "[D2SAM3] No ViT weights matched in checkpoint — check key prefixes."
            )

    def forward(self, x: torch.Tensor):
        H, W = x.shape[-2:]
        S = self._vit_input_size
        if (H, W) != (S, S):
            x_in = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
        else:
            x_in = x

        # SAM3 ViT relies on two fused bf16 ops that have hard constraints:
        #   - perflib.fused.addmm_act: requires grad to be disabled
        #   - addmm_act always returns bf16, while the surrounding nn.Linear (fc2)
        #     keeps fp32 weights — only matches when AMP autocast is active.
        # We satisfy both: run ViT under explicit bf16 autocast (independent of
        # outer AMP, so eval works too) + no_grad when trunk is frozen.
        use_no_grad = self._freeze_backbone and torch.is_grad_enabled()
        amp_ctx = torch.amp.autocast(
            device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda
        )
        if use_no_grad:
            with torch.no_grad(), amp_ctx:
                feats = self.vit(x_in)
        else:
            with amp_ctx:
                feats = self.vit(x_in)
        x_last = feats[-1]
        if x_last.ndim != 4:
            raise RuntimeError(
                f"D2SAM3 expects 4D ViT output (B,C,H,W); got shape {tuple(x_last.shape)}"
            )

        target_strides = (4, 8, 16, 32)
        out = {}
        for i, conv_seq in enumerate(self.neck.convs):
            f = conv_seq(x_last)
            th = max(1, math.ceil(H / target_strides[i]))
            tw = max(1, math.ceil(W / target_strides[i]))
            if f.shape[-2:] != (th, tw):
                f = F.interpolate(
                    f, size=(th, tw), mode="bilinear", align_corners=False
                )
            out[self._out_features[i]] = f
        return out

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch.nn as nn

from .swin_transformer import SwinTransformer


def build_model(config):
    model_type = config.MODEL.TYPE

    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=nn.LayerNorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    elif model_type == 'resnet':
        import torchvision.models as models

        name = str(config.MODEL.NAME).lower()
        num_classes = int(config.MODEL.NUM_CLASSES)
        drop_rate = float(getattr(config.MODEL, "DROP_RATE", 0.0))
        use_imagenet = bool(config.MODEL.IMAGENET_PRETRAINED)

        if name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if use_imagenet else None
            model = models.resnet50(weights=weights)
        elif name == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if use_imagenet else None
            model = models.resnet101(weights=weights)
        else:
            raise NotImplementedError(f"Unknown ResNet: {name}")

        in_feat = model.fc.in_features
        if drop_rate and drop_rate > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=drop_rate),
                nn.Linear(in_feat, num_classes)
            )
        else:
            model.fc = nn.Linear(in_feat, num_classes)

    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model

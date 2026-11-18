import math  # noqa: F401
import torch
import torch.nn as nn

import sys

from .AsymST import AsymST
from .vit_encoder import UNIViTBranch


class UNIv1ViTBranch(UNIViTBranch):
    """Original UNI (v1) ViT-L/16 branch (MahmoodLab/UNI, embed_dim 1024, 14x14 patch grid, no register tokens),
    as in the AsymST paper. Same interface as the UNI2-h branch (`UNIViTBranch`), which is left untouched."""
    UNI2H_EMBED_DIM = 1024
    UNI2H_DEPTH = 24
    UNI2H_NUM_HEADS = 16
    UNI2H_PATCH_SIZE = 16
    UNI2H_NUM_REG_TOKENS = 0

    def _build_backbone(self, drop_path_rate: float):
        import timm
        self.vit = timm.create_model(
            "vit_large_patch16_224", img_size=self.img_size, patch_size=self.UNI2H_PATCH_SIZE,
            init_values=1e-5, num_classes=0, dynamic_img_size=True, drop_path_rate=drop_path_rate,
        )

    def _load_uni_weights(self, weights_path: str):
        if not __import__("os").path.isfile(weights_path):
            raise FileNotFoundError(f"UNI (v1) weights not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        msg = self.vit.load_state_dict(state_dict, strict=True)   # standard ViT-L/16 pos_embed (cls + 196), loads as is
        print(f"UNI v1 weights loaded: {msg}")


class AsymSTV1(AsymST):
    """AsymST with the paper's original UNI (v1) ViT-L/16 (1024-d) next to DenseNet-121 (1024-d): both branches share one
    dimension, so the benchmark port's extra `cnn_proj` is an Identity and the model is the paper's architecture.
    Added beside `AsymST` (UNI2-h), which is unchanged."""

    def __init__(self, *args, **kwargs):
        _asym_mod = sys.modules[AsymST.__module__]
        original = _asym_mod.UNIViTBranch
        _asym_mod.UNIViTBranch = UNIv1ViTBranch   # AsymST.__init__ builds its ViT branch from this module global
        try:
            super().__init__(*args, **kwargs)
        finally:
            _asym_mod.UNIViTBranch = original


# the model loader filters config keys by __init__'s signature -> expose AsymST's own
import inspect
AsymSTV1.__init__.__signature__ = inspect.signature(AsymST.__init__)

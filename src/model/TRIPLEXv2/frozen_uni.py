import math
import os

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# UNI2-h (MahmoodLab/UNI2-h) is a timm `vit_giant_patch14_224` with these settings; they were verified to load the
# checkpoint with zero missing/unexpected keys.
_EMBED_DIM, _DEPTH, _HEADS, _PATCH, _MLP_RATIO, _REG_TOKENS = 1536, 24, 24, 14, 2.66667 * 2, 8


def _interpolate_pos_embed(pos_embed, target_grid):
    src_grid = int(math.sqrt(pos_embed.shape[1]))
    if src_grid * src_grid != pos_embed.shape[1]:
        raise ValueError("Position-embedding length is not a square number.")
    dim = pos_embed.shape[2]
    x = pos_embed.reshape(1, src_grid, src_grid, dim).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(target_grid, target_grid), mode="bicubic", align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, dim)


class FrozenUNI:
    """Frozen UNI2-h feature extractor, deliberately NOT an `nn.Module`.

    UNI2-h is ~680M parameters; as a registered submodule every checkpoint would carry ~2.7 GB of weights that never
    change (and `load_state_dict` would demand them). Keeping it outside the module tree keeps checkpoints at the size
    of the trainable head only. The weights are read from `weights_path` (default: `MahmoodLab/UNI2-h` from the
    HuggingFace cache) in `__init__` and moved to the input's device on first use. Everything runs under `no_grad` in
    eval mode, so train-time image augmentation is preserved while the encoder never changes.
    """

    def __init__(self, weights_path=None, img_size=224):
        import timm
        from timm.layers import SwiGLUPacked

        if weights_path is None:
            weights_path = hf_hub_download("MahmoodLab/UNI2-h", "pytorch_model.bin")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"UNI2-h weights not found: {weights_path}")

        self.vit = timm.create_model(
            "vit_giant_patch14_224", img_size=img_size, patch_size=_PATCH, depth=_DEPTH, num_heads=_HEADS,
            init_values=1e-5, embed_dim=_EMBED_DIM, mlp_ratio=_MLP_RATIO, num_classes=0, no_embed_class=True,
            mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=_REG_TOKENS, dynamic_img_size=True,
            drop_path_rate=0.0,
        )
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        for k in [k for k in state_dict if k.startswith("head.") or k.startswith("fc_norm.")]:
            del state_dict[k]
        grid = img_size // _PATCH
        if "pos_embed" in state_dict and state_dict["pos_embed"].shape[1] != grid ** 2:
            # no_embed_class=True: pos_embed covers only the patch tokens
            state_dict["pos_embed"] = _interpolate_pos_embed(state_dict["pos_embed"], grid)
        self.vit.load_state_dict(state_dict, strict=False)
        self.vit.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, img):
        """Returns (patch tokens (B, 256, 1536), CLS embedding (B, 1536)), both after the final norm."""
        if next(self.vit.parameters()).device != img.device:
            self.vit.to(img.device)
        x = self.vit.patch_embed(img)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        # token order with no_embed_class: [cls, register tokens..., patch tokens...]
        return x[:, 1 + _REG_TOKENS:], x[:, 0]

    def __call__(self, img):
        return self.encode(img)[0]

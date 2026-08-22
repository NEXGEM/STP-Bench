import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNIViTBranch(nn.Module):
    """UNI2-h ViT-giant/14 branch.

    The reference AsymST implementation targets the original UNI (ViT-L/16,
    embed_dim=1024). This port targets MahmoodLab/UNI2-h instead (the
    checkpoint actually available for this integration) -- a different timm
    architecture: `vit_giant_patch14_224`, embed_dim=1536, patch_size=14,
    SwiGLU MLP, SiLU activation, and 8 register tokens (`no_embed_class=True`,
    so register/cls tokens are excluded from position embeddings). These
    kwargs are UNI2-h's published config and were verified to load the
    checkpoint with zero missing/unexpected keys.
    """

    UNI2H_EMBED_DIM = 1536
    UNI2H_DEPTH = 24
    UNI2H_NUM_HEADS = 24
    UNI2H_PATCH_SIZE = 14
    UNI2H_MLP_RATIO = 2.66667 * 2
    UNI2H_NUM_REG_TOKENS = 8

    def __init__(
        self,
        uni_weights_path: str,
        img_size: int = 224,
        freeze_strategy: str = "partial",
        unfreeze_last_n: int = 4,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = self.UNI2H_EMBED_DIM
        self.img_size = img_size
        self.patch_size = self.UNI2H_PATCH_SIZE
        self.grid_size = img_size // self.patch_size
        self.num_patches = self.grid_size ** 2
        self.num_reg_tokens = self.UNI2H_NUM_REG_TOKENS
        self.freeze_strategy = freeze_strategy
        self.unfreeze_last_n = unfreeze_last_n

        self._build_backbone(drop_path_rate)
        self._load_uni_weights(uni_weights_path)
        self._apply_freeze_strategy()

    def _build_backbone(self, drop_path_rate: float):
        try:
            import timm
            from timm.layers import SwiGLUPacked
        except ImportError as exc:
            raise ImportError("timm>=0.9.0 with SwiGLUPacked: pip install -U timm") from exc

        self.vit = timm.create_model(
            "vit_giant_patch14_224",
            img_size=self.img_size,
            patch_size=self.UNI2H_PATCH_SIZE,
            depth=self.UNI2H_DEPTH,
            num_heads=self.UNI2H_NUM_HEADS,
            init_values=1e-5,
            embed_dim=self.UNI2H_EMBED_DIM,
            mlp_ratio=self.UNI2H_MLP_RATIO,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=self.num_reg_tokens,
            dynamic_img_size=True,
            drop_path_rate=drop_path_rate,
        )

    def _load_uni_weights(self, weights_path: str):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"UNI2-h weights not found: {weights_path}")

        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location="cpu")

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

        keys_to_remove = [
            k for k in state_dict.keys() if k.startswith("head.") or k.startswith("fc_norm.")
        ]
        for key in keys_to_remove:
            del state_dict[key]

        if "pos_embed" in state_dict:
            pos_embed = state_dict["pos_embed"]
            # no_embed_class=True: pos_embed covers only patch tokens (no
            # slot for cls/register tokens), unlike original UNI(v1).
            if pos_embed.shape[1] != self.num_patches:
                state_dict["pos_embed"] = self._interpolate_pos_embed(pos_embed, self.grid_size)

        self.vit.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _interpolate_pos_embed(pos_embed: torch.Tensor, target_grid: int) -> torch.Tensor:
        num_patches = pos_embed.shape[1]
        src_grid = int(math.sqrt(num_patches))
        if src_grid * src_grid != num_patches:
            raise ValueError("Position-embedding length is not a square number.")

        dim = pos_embed.shape[2]
        patch_embed = pos_embed.reshape(1, src_grid, src_grid, dim).permute(0, 3, 1, 2)
        patch_embed = F.interpolate(
            patch_embed,
            size=(target_grid, target_grid),
            mode="bicubic",
            align_corners=False,
        )
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, dim)
        return patch_embed

    def _apply_freeze_strategy(self):
        if self.freeze_strategy == "full_finetune":
            return

        if self.freeze_strategy not in {"full_freeze", "partial"}:
            raise ValueError(f"Unknown freeze_strategy: {self.freeze_strategy}")

        for param in self.vit.parameters():
            param.requires_grad = False

        if self.freeze_strategy == "partial":
            n_blocks = len(self.vit.blocks)
            start = max(0, n_blocks - self.unfreeze_last_n)
            for idx in range(start, n_blocks):
                for param in self.vit.blocks[idx].parameters():
                    param.requires_grad = True

            if hasattr(self.vit, "norm"):
                for param in self.vit.norm.parameters():
                    param.requires_grad = True
            if hasattr(self.vit, "fc_norm"):
                for param in self.vit.fc_norm.parameters():
                    param.requires_grad = True

    def _forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        return x

    def forward(self, x: torch.Tensor):
        n_blocks = len(self.vit.blocks)
        if self.freeze_strategy == "partial":
            freeze_end = max(0, n_blocks - self.unfreeze_last_n)
        elif self.freeze_strategy == "full_freeze":
            freeze_end = n_blocks
        else:
            freeze_end = 0

        if self.freeze_strategy in {"partial", "full_freeze"}:
            with torch.no_grad():
                x = self._forward_embed(x)
                for blk in self.vit.blocks[:freeze_end]:
                    x = blk(x)
        else:
            x = self._forward_embed(x)

        for blk in self.vit.blocks[freeze_end:]:
            x = blk(x)

        x = self.vit.norm(x)
        vit_cls = x[:, 0]
        # no_embed_class token order is [cls, reg_tokens..., patch_tokens...]
        # (see timm.models.vision_transformer.VisionTransformer._pos_embed) --
        # skip the register tokens so "tokens" stays purely spatial/patch-level
        # for cross-attention with the CNN's spatial feature map.
        vit_tokens = x[:, 1 + self.num_reg_tokens:]
        return vit_tokens, vit_cls

    def get_param_groups(self, base_lr: float, lr_mult_backbone: float = 0.01):
        backbone_params = [p for p in self.vit.parameters() if p.requires_grad]
        if not backbone_params:
            return []
        return [{"params": backbone_params, "lr": base_lr * lr_mult_backbone, "name": "uni_backbone"}]

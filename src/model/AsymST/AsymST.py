import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_encoder import DenseNet121Branch
from .cross_attn import AsymmetricCrossAttentionFusion, AttentivePool
from .vit_encoder import UNIViTBranch


def pearson_corr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-gene Pearson correlation loss: mean(1 - r_j) across genes."""
    B = pred.size(0)
    if B < 3:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_mean = pred.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    cov = (pred_centered * target_centered).sum(dim=0)
    pred_std = (pred_centered ** 2).sum(dim=0).clamp(min=eps).sqrt()
    target_std = (target_centered ** 2).sum(dim=0).clamp(min=eps).sqrt()

    r = cov / (pred_std * target_std + eps)
    r = r.clamp(-1.0, 1.0)
    return (1.0 - r).mean()


def _safe(loss: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(loss).all():
        return loss
    return torch.zeros((), device=loss.device, dtype=loss.dtype)


class ShallowGenePredictionHead(nn.Module):
    """Shallow auxiliary gene-prediction head."""

    def __init__(self, in_dim: int, n_genes: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, n_genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.dropout(x)
        return self.fc(x)


class DeepGenePredictionHead(nn.Module):
    """Deep fused-branch gene-prediction head with a nonlinear bottleneck."""

    def __init__(
        self,
        in_dim: int,
        n_genes: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AsymST(nn.Module):
    """Asymmetric dual-pathway (DenseNet-121 + UNI2-h ViT) fusion model.

    Port of https://github.com/181sxl/AsymST (models/model_v2.py +
    criterion_v2.py), adapted to STP-Bench's model contract:
    `forward(img, label=None, **kwargs)` returns `{"loss", "logits"}` when
    `label` is given (train/val) and `{"logits"}` otherwise (test/predict).
    `logits` is the fused-branch prediction (`pred` in the reference code);
    the CNN/ViT auxiliary predictions are only used internally for the
    multi-objective training loss, matching the reference's
    `AsymSTCriterion`.

    Deviation from the reference: the reference assumes the CNN and ViT
    branches happen to share one output dimension (both 1024 with the
    original UNI ViT-L/16) and hard-fails otherwise. This integration uses
    UNI2-h (embed_dim=1536) instead of UNI(v1), so a linear `cnn_proj`
    layer up-projects DenseNet-121's 1024-dim features into the ViT's
    1536-dim space before fusion -- not present in the original paper.

    Inference-time spatial k-NN Gaussian smoothing (the reference's
    `eval_smooth.py`) is intentionally not ported: it post-processes raw
    predictions pooled across a fold's full test set, which has no
    equivalent hook in STP-Bench's per-batch train/evaluate loop.
    """

    def __init__(
        self,
        num_genes: int,
        uni_weights_path: str,
        cnn_pretrained: bool = True,
        cnn_train_backbone: bool = True,
        vit_freeze_strategy: str = "partial",
        vit_unfreeze_last_n: int = 4,
        vit_drop_path: float = 0.0,
        fusion_heads: int = 16,
        fusion_depth: int = 1,
        dropout: float = 0.1,
        fuse_head_hidden_dim: int = 1024,
        w_fuse: float = 1.0,
        w_cnn: float = 0.3,
        w_vit: float = 0.3,
        w_pcc: float = 0.5,
        max_batch_size: int = 32,
    ):
        super().__init__()

        self.w_fuse = float(w_fuse)
        self.w_cnn = float(w_cnn)
        self.w_vit = float(w_vit)
        self.w_pcc = float(w_pcc)
        self.max_batch_size = max_batch_size

        # Encoders
        self.cnn_branch = DenseNet121Branch(
            pretrained=cnn_pretrained,
            train_backbone=cnn_train_backbone,
        )
        self.vit_branch = UNIViTBranch(
            uni_weights_path=uni_weights_path,
            img_size=224,
            freeze_strategy=vit_freeze_strategy,
            unfreeze_last_n=vit_unfreeze_last_n,
            drop_path_rate=vit_drop_path,
        )

        cnn_out_dim = self.cnn_branch.out_dim
        vit_out_dim = self.vit_branch.embed_dim
        # See class docstring: reference code required cnn_out_dim ==
        # vit_out_dim outright. Project CNN features into the ViT's space
        # instead, so both branches operate in one common dimension for
        # fusion regardless of which ViT backbone is plugged in.
        self.cnn_proj = (
            nn.Identity() if cnn_out_dim == vit_out_dim else nn.Linear(cnn_out_dim, vit_out_dim)
        )

        self.branch_dim = vit_out_dim
        self.branch_latent_dim = self.branch_dim * 2

        # Branch pooling
        self.cnn_pool = AttentivePool(
            dim=self.branch_dim,
            num_heads=fusion_heads,
            dropout=dropout,
            learnable_query=True,
        )
        self.vit_pool = AttentivePool(
            dim=self.branch_dim,
            num_heads=fusion_heads,
            dropout=dropout,
            learnable_query=True,
        )

        # Cross-attention fusion
        self.fusion = AsymmetricCrossAttentionFusion(
            cnn_dim=self.branch_dim,
            vit_dim=self.branch_dim,
            num_heads=fusion_heads,
            depth=fusion_depth,
            dropout=dropout,
        )

        # Latent norms
        self.cnn_latent_norm = nn.LayerNorm(self.branch_latent_dim)
        self.vit_latent_norm = nn.LayerNorm(self.branch_latent_dim)
        self.fuse_latent_norm = nn.LayerNorm(self.fusion.out_dim)

        # Prediction heads
        self.pred_head_fuse = DeepGenePredictionHead(
            in_dim=self.fusion.out_dim,
            n_genes=num_genes,
            hidden_dim=fuse_head_hidden_dim,
            dropout=dropout,
        )
        self.pred_head_cnn = ShallowGenePredictionHead(
            self.branch_latent_dim, num_genes, dropout
        )
        self.pred_head_vit = ShallowGenePredictionHead(
            self.branch_latent_dim, num_genes, dropout
        )

    def encode(self, x: torch.Tensor) -> dict:
        if x.ndim != 4:
            raise ValueError(f"Input tensor must be 4D, got {x.ndim}D")

        cnn_tokens, cnn_gap = self.cnn_branch(x)
        cnn_tokens = self.cnn_proj(cnn_tokens)
        cnn_gap = self.cnn_proj(cnn_gap)
        vit_tokens, vit_cls = self.vit_branch(x)

        cnn_pool = self.cnn_pool(cnn_tokens)
        vit_pool = self.vit_pool(vit_tokens)

        fuse_out = self.fusion(
            cnn_tokens=cnn_tokens,
            vit_tokens=vit_tokens,
        )

        z_cnn = self.cnn_latent_norm(torch.cat([cnn_pool, cnn_gap], dim=-1))
        z_vit = self.vit_latent_norm(torch.cat([vit_pool, vit_cls], dim=-1))
        z_fuse = self.fuse_latent_norm(fuse_out["z"])

        return {"z": z_fuse, "z_cnn": z_cnn, "z_vit": z_vit}

    def _predict_all(self, img: torch.Tensor):
        enc = self.encode(img)
        pred_fuse = self.pred_head_fuse(enc["z"])
        pred_cnn = self.pred_head_cnn(enc["z_cnn"])
        pred_vit = self.pred_head_vit(enc["z_vit"])
        return pred_fuse, pred_cnn, pred_vit

    def _compute_loss(self, pred_fuse, pred_cnn, pred_vit, label):
        target = label.float()

        loss_fuse_mse = _safe(F.mse_loss(pred_fuse, target))
        loss_fuse_pcc = _safe(pearson_corr_loss(pred_fuse, target))
        loss_fuse = loss_fuse_mse + self.w_pcc * loss_fuse_pcc

        loss_cnn = _safe(F.mse_loss(pred_cnn, target))
        loss_vit = _safe(F.mse_loss(pred_vit, target))

        total = self.w_fuse * loss_fuse + self.w_cnn * loss_cnn + self.w_vit * loss_vit
        return _safe(total)

    def forward(self, img: torch.Tensor, label=None, **kwargs) -> dict:
        # Train batches are always small (dataloader-controlled), so a
        # batch bigger than max_batch_size only happens during whole-slide
        # val/test/predict passes -- chunk the forward pass there to bound
        # peak memory (DenseNet121 + UNI2-h on a full slide's worth of
        # spots at once otherwise OOMs), then compute the loss once over
        # the concatenated predictions so it matches an unchunked pass.
        if img.shape[0] > self.max_batch_size and not self.training:
            chunks = img.split(self.max_batch_size, dim=0)
            with torch.no_grad():
                preds = [self._predict_all(c) for c in chunks]
            pred_fuse = torch.cat([p[0] for p in preds], dim=0)
            if label is None:
                return {"logits": pred_fuse}
            pred_cnn = torch.cat([p[1] for p in preds], dim=0)
            pred_vit = torch.cat([p[2] for p in preds], dim=0)
        else:
            pred_fuse, pred_cnn, pred_vit = self._predict_all(img)
            if label is None:
                return {"logits": pred_fuse}

        total = self._compute_loss(pred_fuse, pred_cnn, pred_vit, label)
        return {"loss": total, "logits": pred_fuse}

    def get_param_groups(self, base_lr: float, vit_lr_mult: float = 0.01):
        groups = []
        vit_groups = self.vit_branch.get_param_groups(
            base_lr, lr_mult_backbone=vit_lr_mult
        )
        groups.extend(vit_groups)

        vit_param_ids = {
            id(param) for group in vit_groups for param in group["params"]
        }
        default_params = [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in vit_param_ids
        ]
        if default_params:
            groups.append({"params": default_params, "lr": base_lr, "name": "default"})
        return groups

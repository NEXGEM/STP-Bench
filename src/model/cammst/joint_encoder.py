# Vendored from https://github.com/Kyyle2114/CAMMST (models/joint_encoder.py); only the intra-package imports
# were made relative. NOTE: the upstream repository ships no LICENSE file.
from __future__ import annotations

from typing import Dict, Optional, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .common import Block, FeedForward, MultiHeadSelfAttention, make_norm_layer

__all__ = ["CrossModalJointEncoder"]


class SharedBlockWithPathNorms(nn.Module):
    """Transformer block with shared attn/ffn and per-path norms."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ffn_activation: Literal["gelu", "swiglu"] = "swiglu",
        norm_type: Literal["rms", "layer"] = "rms",
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            activation=ffn_activation,
            dropout=ffn_dropout,
        )
        self.norm1_image = make_norm_layer(norm_type, dim, norm_eps)
        self.norm2_image = make_norm_layer(norm_type, dim, norm_eps)
        self.norm1_gene = make_norm_layer(norm_type, dim, norm_eps)
        self.norm2_gene = make_norm_layer(norm_type, dim, norm_eps)

    def forward_patch(self, tokens: torch.Tensor, coords: Optional[torch.Tensor]) -> torch.Tensor:
        x = tokens + self.attn(self.norm1_image(tokens), coords=coords)
        x = x + self.ffn(self.norm2_image(x))
        return x

    def forward_gene(self, tokens: torch.Tensor, coords: Optional[torch.Tensor]) -> torch.Tensor:
        x = tokens + self.attn(self.norm1_gene(tokens), coords=coords)
        x = x + self.ffn(self.norm2_gene(x))
        return x


class ProjectionHead(nn.Module):
    """MLP projection from embedding dimension (d) to contrastive dimension (c_dim)."""

    def __init__(
        self,
        embed_dim: int,
        contrastive_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            embed_dim: Input embedding dimension (d, typically aligned with image encoder/UNI dim).
            contrastive_dim: Projection dim for contrastive head.
            dropout: Dropout probability inside the MLP.
        """
        super().__init__()
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1.")
        if contrastive_dim < 1:
            raise ValueError("contrastive_dim must be >= 1.")

        self.projection = nn.Linear(embed_dim, contrastive_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(contrastive_dim, contrastive_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(contrastive_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings of shape (B, V, d).

        Returns:
            Tensor of shape (B, V, c_dim) after projection and normalization.
        """
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, V, d), got {x.shape}")

        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)

        return x


class CrossModalJointEncoder(nn.Module):
    """CAV-MAE style joint encoder with separate modality paths and contrastive learning."""

    def __init__(
        self,
        embed_dim: int,
        joint_depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_activation: str = "swiglu",
        norm_type: str = "rms",
        contrastive_dim: Optional[int] = None,
        decoder_depth: int = 2,
    ) -> None:
        """
        Args:
            embed_dim: Token embedding dimension (d).
            joint_depth: Number of Transformer blocks for the fusion path.
            num_heads: Attention heads.
            mlp_ratio: FFN expansion ratio for all blocks.
            attn_dropout: Attention dropout for all blocks.
            proj_dropout: Projection/FFN dropout for all blocks.
            ffn_activation: FFN activation function, {"gelu", "swiglu"}.
            norm_type: Normalization type, {"rms", "layer"}.
            contrastive_dim: Projection dim for contrastive head (defaults to embed_dim).
            decoder_depth: Number of decoder transformer blocks (must be >= 1).
            Decoder uses same num_heads, mlp_ratio, attn_dropout, proj_dropout as joint blocks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.contrastive_dim = contrastive_dim if contrastive_dim is not None else embed_dim

        self.fuse_linear = nn.Linear(embed_dim * 2, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.joint_blocks = nn.ModuleList(
            [
                SharedBlockWithPathNorms(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=attn_dropout,
                    proj_dropout=proj_dropout,
                    ffn_dropout=proj_dropout,
                    ffn_activation=ffn_activation,
                    norm_type=norm_type,
                    norm_eps=1e-6,
                )
                for _ in range(joint_depth)
            ]
        )
        self.proj_patch = ProjectionHead(embed_dim, self.contrastive_dim)
        self.proj_gene = ProjectionHead(embed_dim, self.contrastive_dim)

        if decoder_depth < 1:
            raise ValueError(f"decoder_depth must be >= 1, got {decoder_depth}")
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=attn_dropout,
                    proj_dropout=proj_dropout,
                    ffn_dropout=proj_dropout,
                    ffn_activation=ffn_activation,
                    norm_type=norm_type,
                    norm_eps=1e-6,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = make_norm_layer("rms", embed_dim, 1e-6)

        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        patch_emb: torch.Tensor,  # (B, N, d)
        gene_emb: torch.Tensor,  # (B, V, d) aligned to vis_idx
        mask: torch.Tensor,  # (B, N) bool, True=masked
        vis_idx: torch.Tensor,  # (B, V) long; visible token indices
        coords: torch.Tensor,  # (B, N, 2) spatial coords for ALiBi
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            patch_emb: Patch embeddings (B, N, d).
            gene_emb: Gene embeddings for visible tokens (B, V, d) ordered by vis_idx.
            mask: Boolean mask (B, N), True=masked positions.
            vis_idx: Visible token indices from sampler (B, V).
            coords: Spatial coordinates (B, N, 2) for ALiBi bias.

        Returns:
            Dict with:
              - fused_tokens: fused sequence (B, N, d)
              - decoded_tokens: tokens after decoder blocks (B, N, d)
              - contrastive_patch: contrastive proj for patch (B, V, c_dim)
              - contrastive_gene: contrastive proj for gene (B, V, c_dim)
        """
        if patch_emb.dim() != 3:
            raise ValueError(f"patch_emb must have shape (B, N, d), got {patch_emb.shape}")
        if gene_emb.dim() != 3:
            raise ValueError(f"gene_emb must have shape (B, V, d), got {gene_emb.shape}")
        if mask.dim() != 2:
            raise ValueError(f"mask must have shape (B, N), got {mask.shape}")
        if vis_idx.dim() != 2:
            raise ValueError(f"vis_idx must have shape (B, V), got {vis_idx.shape}")

        bsz, seq_len, dim = patch_emb.shape
        if dim != self.embed_dim:
            raise ValueError(f"patch_emb embed_dim mismatch: expected {self.embed_dim}, got {dim}")
        if mask.shape != (bsz, seq_len):
            raise ValueError(f"mask shape mismatch: expected ({bsz}, {seq_len}), got {mask.shape}")
        if coords.shape != (bsz, seq_len, 2):
            raise ValueError(f"coords shape mismatch: expected ({bsz}, {seq_len}, 2), got {coords.shape}")
        if vis_idx.shape[0] != bsz:
            raise ValueError(f"vis_idx batch size mismatch: expected {bsz}, got {vis_idx.shape[0]}")
        if vis_idx.shape[1] != gene_emb.shape[1]:
            raise ValueError(f"vis_idx and gene_emb visible length mismatch: {vis_idx.shape[1]} vs {gene_emb.shape[1]}")
        if vis_idx.dtype != torch.long:
            raise ValueError(f"vis_idx must be torch.long, got {vis_idx.dtype}")
        visible_count = (~mask).sum(dim=1)
        if not torch.equal(visible_count, torch.full_like(visible_count, gene_emb.shape[1])):
            raise ValueError("visible count from mask must match gene_emb/vis_idx length")

        # refine patch tokens over all N using shared blocks (patch norms)
        patch_tokens = patch_emb
        for blk in self.joint_blocks:
            patch_tokens = blk.forward_patch(patch_tokens, coords=coords)

        # gather visible tokens and coords
        gather_vis = vis_idx.unsqueeze(-1).expand(-1, -1, dim)
        patch_vis = patch_tokens.gather(dim=1, index=gather_vis)  # (B, V, d)
        coords_vis = coords.gather(dim=1, index=vis_idx.unsqueeze(-1).expand(-1, -1, 2))  # (B, V, 2)

        # refine gene tokens 
        gene_refined = gene_emb
        for blk in self.joint_blocks:
            gene_refined = blk.forward_gene(gene_refined, coords=coords_vis)

        # fuse gene and patch tokens
        fused_vis = torch.cat([gene_refined, patch_vis], dim=-1)
        fused_vis = self.fuse_linear(fused_vis)  # (B, V, d)

        # get contrastive features
        c_patch = F.normalize(self.proj_patch(patch_vis), dim=-1)
        c_gene = F.normalize(self.proj_gene(gene_refined), dim=-1)

        # make decoder input
        fused_full = self.fuse_linear(
            torch.cat(
                [
                    self.mask_token.expand(bsz, seq_len, dim),  # gene slot masked
                    patch_tokens,
                ],
                dim=-1,
            )
        ).clone()
        for b in range(bsz):
            fused_full[b, vis_idx[b]] = fused_vis[b]

        decoded = fused_full
        for blk in self.decoder_blocks:
            decoded = blk(decoded, coords=coords)
        decoded = self.decoder_norm(decoded)

        return {
            "fused_tokens": fused_full,  # (B, N, d) masked slots contain mask_token+patch fused baseline
            "decoded_tokens": decoded,  # (B, N, d) after decoder blocks
            "contrastive_patch": c_patch,  # (B, V, c_dim)
            "contrastive_gene": c_gene,  # (B, V, c_dim)
        }

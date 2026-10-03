# Vendored from https://github.com/Kyyle2114/CAMMST (models/gene_encoder.py); only the intra-package imports
# were made relative. NOTE: the upstream repository ships no LICENSE file.
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

__all__ = ["GeneEncoder"]


class GeneEncoder(nn.Module):
    """MLP projection from gene dimension (G: number of genes) to embedding dimension (d)."""

    def __init__(
        self,
        num_genes: int,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            num_genes: Input gene dimension (G, number of genes).
            embed_dim: Output embedding dimension (d, typically aligned with image encoder/UNI dim).
            hidden_dim: Hidden dimension for the MLP. Defaults to 2 * embed_dim.
            dropout: Dropout probability inside the MLP.
        """
        super().__init__()
        if num_genes < 1:
            raise ValueError("num_genes must be >= 1.")
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1.")
        hidden = hidden_dim if hidden_dim is not None else embed_dim * 2

        self.proj = nn.Sequential(
            nn.Linear(num_genes, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, genes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            genes: Gene profiles of shape (B, V, G) where V is the number of visible
                tokens/regions for which gene measurements are available, and G is
                the number of genes.

        Returns:
            Tensor of shape (B, V, d) after projection and normalization.
        """
        if genes.dim() != 3:
            raise ValueError(f"genes must have shape (B, V, G), got {genes.shape}")
        out = self.proj(genes)
        out = self.norm(out)
        return out

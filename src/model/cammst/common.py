# Vendored from https://github.com/Kyyle2114/CAMMST (models/common.py); only the intra-package imports
# were made relative. NOTE: the upstream repository ships no LICENSE file.
from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    "MultiHeadSelfAttention",
    "FeedForward",
    "Block",
    "RMSNorm",
    "make_norm_layer",
]


class RMSNorm(nn.Module):
    """Root mean square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """
        Args:
            dim: Embedding dimension.
            eps: Numerical stability term.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


def make_norm_layer(norm_type: Literal["rms", "layer"], dim: int, eps: float) -> nn.Module:
    """Factory for normalization layers."""
    if norm_type == "rms":
        return RMSNorm(dim, eps=eps)
    if norm_type == "layer":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"norm_type must be 'rms' or 'layer', got '{norm_type}'")


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional ALiBi positional bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """
        Args:
            dim: Embedding dimension.
            num_heads: Number of attention heads.
            attn_dropout: Dropout applied to attention probabilities.
            proj_dropout: Dropout applied after the output projection.
            bias: Whether to add bias terms to linear projections.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.register_buffer("alibi_slopes", self._build_alibi_slopes(num_heads), persistent=False)

    @staticmethod
    def _build_alibi_slopes(num_heads: int) -> torch.Tensor:
        """Create head-specific slopes for ALiBi-style distance penalties."""
        return torch.tensor([1.0 / (2.0 ** (h + 1)) for h in range(num_heads)], dtype=torch.float32).view(
            1, num_heads, 1, 1
        )

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D).
            coords: Optional coordinates tensor of shape (B, N, 2). When provided,
                ALiBi-style spatial bias is subtracted from attention logits based on
                pairwise Euclidean distances and head-specific slopes.

        Returns:
            Tensor of shape (B, N, D) after attention.
        """
        bsz, seq_len, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, Hd)

        if coords is None:
            logits = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        else:
            logits = q @ k.transpose(-2, -1)  # (B, H, N, N) without scaling

        # 2D ALiBi positional encoding
        if coords is not None:
            if coords.dim() != 3 or coords.shape[0] != bsz or coords.shape[1] != seq_len or coords.shape[2] != 2:
                raise ValueError("coords must have shape (B, N, 2)")
            # pairwise Euclidean distance matrix per batch: (B, N, N)
            diff = coords.unsqueeze(2) - coords.unsqueeze(1)
            dist = torch.sqrt(torch.clamp((diff**2).sum(dim=-1), min=1e-12))
            # apply head-specific slopes
            slopes = self.alibi_slopes.to(dtype=logits.dtype, device=logits.device)  # (1, H, 1, 1)
            logits = logits - slopes * dist.unsqueeze(1)

        attn = logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, H, N, Hd)
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Module):
    """Two-layer feedforward network with GELU or SwiGLU."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: Literal["gelu", "swiglu"] = "gelu",
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            dim: Input and output embedding dimension.
            hidden_dim: Hidden dimension size.
            activation: Activation type, one of {"gelu", "swiglu"}.
            dropout: Dropout probability applied after activations and projection.
        """
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, dim)
        elif activation == "swiglu":
            self.fc1 = nn.Linear(dim, hidden_dim * 2)
            self.fc2 = nn.Linear(hidden_dim, dim)
        else:
            raise ValueError("activation must be 'gelu' or 'swiglu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feedforward transformation."""
        if self.activation == "gelu":
            x = self.fc1(x)
            x = F.gelu(x)
        else:
            x_proj = self.fc1(x)
            x1, x2 = x_proj.chunk(2, dim=-1)
            x = F.silu(x1) * x2
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Pre-norm Transformer block (MHSA + FFN)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ffn_activation: Literal["gelu", "swiglu"] = "gelu",
        norm_type: Literal["rms", "layer"] = "rms",
        norm_eps: float = 1e-6,
    ) -> None:
        """
        Args:
            dim: Embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for the feedforward hidden dimension.
            attn_dropout: Dropout applied to attention probabilities.
            proj_dropout: Dropout after attention output projection.
            ffn_dropout: Dropout inside the feedforward network.
            ffn_activation: Activation type for FFN, {"gelu", "swiglu"}.
            norm_type: Normalization type, {"rms", "layer"}.
            norm_eps: Epsilon used in normalization layers.
        """
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.norm1 = make_norm_layer(norm_type=norm_type, dim=dim, eps=norm_eps)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = make_norm_layer(norm_type=norm_type, dim=dim, eps=norm_eps)
        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            activation=ffn_activation,
            dropout=ffn_dropout,
        )

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D).
            coords: Optional coordinates tensor (B, N, 2). When provided, spatial ALiBi
                bias is applied inside attention.

        Returns:
            Tensor of shape (B, N, D) after one Transformer block.
        """
        x = x + self.attn(self.norm1(x), coords=coords)
        x = x + self.ffn(self.norm2(x))
        return x
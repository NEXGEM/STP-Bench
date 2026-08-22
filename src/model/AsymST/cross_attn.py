import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(self, dim: int, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer in which ViT tokens query CNN tokens."""

    def __init__(self, dim: int, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_drop = nn.Dropout(dropout)

    def forward(self, query_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(query_tokens)
        kv = self.kv_norm(kv_tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return query_tokens + self.out_drop(out)


class SelfAttentionFFNBlock(nn.Module):
    """Self-attention followed by a feed-forward network."""

    def __init__(self, dim: int, num_heads: int = 16, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.attn_drop(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class AttentivePool(nn.Module):
    """Single-query attentive pooling."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        dropout: float = 0.1,
        learnable_query: bool = True,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.learnable_query = learnable_query
        if learnable_query:
            self.query = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.query, std=0.02)
        else:
            self.register_parameter("query", None)

        self.query_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_drop = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout, mlp_ratio=mlp_ratio)

    def forward(self, tokens: torch.Tensor, query: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = tokens.size(0)

        if query is None:
            if self.query is None:
                raise ValueError("A query must be provided when learnable_query is disabled.")
            query = self.query.expand(batch_size, -1, -1)
        elif query.ndim == 2:
            query = query.unsqueeze(1)

        q = self.query_norm(query)
        kv = self.token_norm(tokens)
        pooled, _ = self.attn(q, kv, kv, need_weights=False)
        pooled = query + self.out_drop(pooled)
        pooled = pooled + self.ffn(self.ffn_norm(pooled))
        return pooled.squeeze(1)


class AsymmetricFusionLayer(nn.Module):
    """Fusion layer that updates only the ViT tokens."""

    def __init__(self, dim: int, num_heads: int = 16, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(dim, num_heads=num_heads, dropout=dropout)
        self.vit_refine = SelfAttentionFFNBlock(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, vit_tokens: torch.Tensor, cnn_tokens: torch.Tensor) -> torch.Tensor:
        vit_tokens = self.cross_attn(vit_tokens, cnn_tokens)
        vit_tokens = self.vit_refine(vit_tokens)
        return vit_tokens


class AsymmetricCrossAttentionFusion(nn.Module):
    """
    Asymmetric cross-attention fusion.

    ViT tokens retrieve local information from CNN tokens, while CNN tokens are not
    overwritten by a reciprocal update. The fused embedding concatenates an updated
    ViT summary and a ViT-guided CNN summary.
    """

    def __init__(
        self,
        cnn_dim: int = 1024,
        vit_dim: int = 1024,
        num_heads: int = 16,
        depth: int = 1,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if cnn_dim != vit_dim:
            raise ValueError("CNN and ViT token dimensions must match.")

        self.dim = cnn_dim
        self.depth = depth

        self.cnn_token_norm = nn.LayerNorm(self.dim)
        self.vit_token_norm = nn.LayerNorm(self.dim)

        self.layers = nn.ModuleList([
            AsymmetricFusionLayer(
                dim=self.dim,
                num_heads=num_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        self.vit_fused_pool = AttentivePool(
            dim=self.dim,
            num_heads=num_heads,
            dropout=dropout,
            learnable_query=True,
        )
        self.cnn_guided_pool = AttentivePool(
            dim=self.dim,
            num_heads=num_heads,
            dropout=dropout,
            learnable_query=False,
        )

        self.out_dim = self.dim * 2

    def forward(self, cnn_tokens: torch.Tensor, vit_tokens: torch.Tensor) -> dict:
        cnn_tokens = self.cnn_token_norm(cnn_tokens)
        vit_tokens = self.vit_token_norm(vit_tokens)

        vit_tokens_fused = vit_tokens
        for layer in self.layers:
            vit_tokens_fused = layer(vit_tokens_fused, cnn_tokens)

        vit_fused_summary = self.vit_fused_pool(vit_tokens_fused)
        cnn_guided_summary = self.cnn_guided_pool(cnn_tokens, query=vit_fused_summary)

        z = torch.cat([vit_fused_summary, cnn_guided_summary], dim=-1)

        return {
            "z": z,
            "vit_tokens_fused": vit_tokens_fused,
            "vit_fused_summary": vit_fused_summary,
            "cnn_guided_summary": cnn_guided_summary,
        }

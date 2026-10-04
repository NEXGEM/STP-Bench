# MMDiT denoiser of BioFlow (https://github.com/hrterry/BioFLow, bioflow/model/vpredictor.py), reproduced here so that
# the model folder is self-contained. Building blocks are the reference's, unchanged; only the thin wrapper at the
# end differs (no device handling, no unused `coords`).
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def _to_device(module, *tensors):
    device = next(module.parameters()).device
    return [t.to(device) for t in tensors]


# -----------------------------
# Timestep Embedder
# -----------------------------
class TimestepEmbedder(nn.Module):
    """
    Sinusoidal γ(t) + 2-layer MLP -> hidden_dim
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim: int, max_period: int = 10000):
        """
        t: [B] float/long
        returns: [B, dim]
        """
        if t.dtype != torch.float32 and t.dtype != torch.float64:
            t = t.float()
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        t = t.view(-1)  # [B]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb  # [B, hidden]


# -----------------------------
# MLP
# -----------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = approx_gelu()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


# -----------------------------
# Gated Decomposed Attention (GDA) with t in QKV
# -----------------------------
class DecomposedAttention(nn.Module):
    """
    Self-attention on unified sequence; weights four sub-blocks of attention matrix via four learnable gates:
    img↔img / img↔gene / gene↔img / gene↔gene.

    Time embedding γ(t) is concatenated to each token representation before QKV.
    """
    def __init__(self, d_model, d_time, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        in_dim = d_model + d_time
        self.qkv = nn.Linear(in_dim, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        self.g_img2img   = nn.Parameter(torch.tensor(1.0))
        self.g_img2gene  = nn.Parameter(torch.tensor(1.0))
        self.g_gene2img  = nn.Parameter(torch.tensor(1.0))
        self.g_gene2gene = nn.Parameter(torch.tensor(1.0))

    def forward(self, z, t_emb, num_img_tokens: int):
        """
        z: [B, N, d_model],  N = M + C
        t_emb: [B, d_time]
        num_img_tokens: M
        returns: [B, N, d_model]
        """
        B, N, C = z.shape
        M = num_img_tokens
        d_time = t_emb.shape[-1]

        t_expanded = t_emb[:, None, :].expand(B, N, d_time)
        h = torch.cat([z, t_expanded], dim=-1)  # [B, N, d_model + d_time]

        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, N, D]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        g_ii = torch.sigmoid(self.g_img2img)
        g_ig = torch.sigmoid(self.g_img2gene)
        g_gi = torch.sigmoid(self.g_gene2img)
        g_gg = torch.sigmoid(self.g_gene2gene)

        top_left     = attn[:, :, :M, :M]      * g_ii
        top_right    = attn[:, :, :M, M:]      * g_ig
        bottom_left  = attn[:, :, M:, :M]      * g_gi
        bottom_right = attn[:, :, M:, M:]      * g_gg
        attn = torch.cat([
            torch.cat([top_left, top_right], dim=-1),
            torch.cat([bottom_left, bottom_right], dim=-1)
        ], dim=-2)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------------
# MMDiT Block (with ReZero residual scaling)
# -----------------------------
class MMDiTBlock(nn.Module):
    def __init__(self, d_model, d_time, num_heads, mlp_ratio=4.0, attn_drop=0., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = DecomposedAttention(
            d_model=d_model,
            d_time=d_time,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = Mlp(d_model, int(d_model * mlp_ratio), drop=drop)

        # ReZero residual scaling
        self.alpha_attn = nn.Parameter(torch.tensor(1e-3))
        self.alpha_mlp  = nn.Parameter(torch.tensor(1e-3))

    def forward(self, z, t_emb, num_img_tokens):
        z = z + self.alpha_attn * self.attn(self.norm1(z), t_emb, num_img_tokens)
        z = z + self.alpha_mlp  * self.mlp(self.norm2(z))
        return z


# -----------------------------
# MMDiT Transformer (unified sequence, spot->C outputs)
# -----------------------------
class MMDiTTransformer(nn.Module):
    """
    Input:
      img_tokens:  [B, M, d_img_in]
      gene_tokens: [B, C, d_gene_in] or gene_indices: [B, C]
      t:           [B]
    Output:
      v_pred:      [B, M, out_dim]  (out_dim = n_genes = C for this task)
    """
    def __init__(
        self,
        d_img_in: int,
        d_gene_in: int = None,
        n_genes: int = None,
        gene_input_is_indices: bool = False,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        time_hidden_dim: int = None,
        out_dim: int = None,   # 默认输出维度设置为 n_genes
    ):
        super().__init__()
        assert (gene_input_is_indices and n_genes is not None) or (not gene_input_is_indices and d_gene_in is not None), \
            "gene_input_is_indices=True requires n_genes; otherwise requires d_gene_in"

        self.n_genes = n_genes
        self.gene_input_is_indices = gene_input_is_indices
        self.d_model = d_model
        self.out_dim = out_dim if out_dim is not None else n_genes
        assert self.out_dim is not None, "out_dim/n_genes cannot be None"

        # Project to d_model
        self.img_proj = nn.Linear(d_img_in, d_model)
        if gene_input_is_indices:
            embed_dim = d_gene_in if d_gene_in is not None else d_model
            self.gene_embed = nn.Embedding(n_genes, embed_dim)
            self.gene_proj = nn.Linear(embed_dim, d_model) if embed_dim != d_model else nn.Identity()
        else:
            self.gene_proj = nn.Linear(d_gene_in, d_model)

        # time embedding
        self.time_hidden_dim = time_hidden_dim if time_hidden_dim is not None else d_model
        self.time_embedder = TimestepEmbedder(self.time_hidden_dim)

        # transformer blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(
                d_model=d_model,
                d_time=self.time_hidden_dim,
                num_heads=n_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_dropout,
                drop=dropout,
            ) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.output_proj_spot = nn.Linear(d_model, self.out_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, img_tokens, gene_tokens=None, gene_indices=None, t=None):
        """
        Returns: v_pred [B, M, C]
        """
        assert t is not None, "Time step t must be provided"
        if self.gene_input_is_indices:
            assert gene_indices is not None
        else:
            assert gene_tokens is not None

        B, M, d_img_in = img_tokens.shape

        img_z = self.img_proj(img_tokens)  # [B, M, d_model]
        if self.gene_input_is_indices:
            gene_emb = self.gene_embed(gene_indices)  # [B, C, emb]
            gene_z = self.gene_proj(gene_emb)         # [B, C, d_model]
            C = gene_indices.shape[1]
        else:
            gene_z = self.gene_proj(gene_tokens)      # [B, C, d_model]
            C = gene_tokens.shape[1]

        z = torch.cat([img_z, gene_z], dim=1)         # [B, M+C, d_model]
        t_emb = self.time_embedder(t)                 # [B, d_time]

        for blk in self.blocks:
            z = blk(z, t_emb, num_img_tokens=M)

        z = self.final_norm(z)
        spot_hidden = z[:, :M, :]                     # [B, M, d_model]
        v_pred = self.output_proj_spot(spot_hidden)   # [B, M, C]
        return v_pred


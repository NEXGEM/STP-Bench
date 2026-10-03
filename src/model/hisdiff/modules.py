# Copied from this repository's Stem (src/model/stem/stem.py) so that HisDiff stays self-contained; the DiT
# building blocks are the same as upstream HisDiff's (https://github.com/wenwenmin/HisDiff, model.py).
import math

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class GeneJointEmbedding(nn.Module):
    def __init__(self, input_size, hidden_dim):
        """
        input_size: number of genes in input
        hidden_dim: num hidden dimension
        """
        super().__init__()
        self.gene_name_ebd = nn.Parameter(torch.empty((input_size, hidden_dim)),requires_grad=True) # trainable look-up table to encode gene name
        torch.nn.init.kaiming_uniform_(self.gene_name_ebd, a=math.sqrt(5))
        self.gene_count_ebd = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),            
        )
        torch.nn.init.xavier_uniform_(self.gene_count_ebd[0].weight)
        torch.nn.init.xavier_uniform_(self.gene_count_ebd[2].weight)

        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        x: (N, NumGene) tensor of inputs
        """
        gene_count_ebd = self.gene_count_ebd(x.unsqueeze(2))      # (N, NumGene, hidden_dim)
        gene_name_ebd = self.gene_name_ebd                        # (NumGene, hidden_dim)
        gene_joint_ebd = torch.add(gene_count_ebd, gene_name_ebd) # (N, NumGene, hidden_dim)
        return gene_joint_ebd                                     # (N, NumGene, hidden_dim)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, 
                       hidden_features=mlp_hidden_dim, 
                       act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, 2, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = torch.permute(self.linear(x), (0, 2, 1))  # (N, NumGene, hidden_size) -> (N, 2, NumGene)
        return x


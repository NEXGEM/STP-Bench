# Vendored from https://github.com/mcpathology/nh2st (models/module.py), class bodies unchanged except where
# marked "STP-Bench". NOTE: the upstream repository ships no LICENSE file.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, emb_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if 'x_kv' in kwargs.keys():
            kwargs['x_kv'] = self.norm(kwargs['x_kv'])

        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, emb_dim, heads=4, dropout=0., attn_bias=False):
        super().__init__()

        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'

        dim_head = emb_dim // heads
        project_out = not (heads == 1)

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_kv = nn.Linear(emb_dim, emb_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_q, x_kv, return_attn=False):
        # qkv = self.to_qkv(x) # b x n x d*3

        # qkv = rearrange(qkv, 'b n (h d a) -> b n a h d', h = self.heads, a=3)
        # out = flash_attn_qkvpacked_func(qkv, self.drop_p, softmax_scale=None, causal=False)
        # out = rearrange(out, 'b n h d -> b n (h d)')

        # return self.to_out(out)

        q = self.to_q(x_q)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(x_kv).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # if mask is not None:
        #     fill_value = torch.finfo(torch.float16).min
        #     ind_mask = mask.shape[-1]
        #     qk[:, :, -ind_mask:, -ind_mask:] = qk[:, :, -ind_mask:, -ind_mask:].masked_fill(mask == 0, fill_value)

        attn_weights = self.attend(qk)  # b h n n
        if return_attn:
            attn_weights_averaged = attn_weights.mean(dim=1)

        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if return_attn:
            return self.to_out(out), attn_weights_averaged[:, 0]
        else:
            return self.to_out(out)


class CrossEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., attn_bias=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadCrossAttention(emb_dim, heads=heads, dropout=dropout, attn_bias=attn_bias)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x_q, x_kv, return_attn=False):
        for attn, ff in self.layers:
            if return_attn:
                attn_out, attn_weights = attn(x_q, x_kv=x_kv, return_attn=return_attn)
                x_q += attn_out  # residual connection after attention
                x_q = ff(x_q) + x_q  # residual connection after feed forward net
            else:
                x_q = attn(x_q, x_kv=x_kv) + x_q
                x_q = ff(x_q) + x_q  # residual connection after feed forward net

        if return_attn:
            return x_q, attn_weights
        else:
            return x_q


class TWOFusionEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        super().__init__()

        self.fusion_layer = CrossEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, x_t=None,x_g=None):

        if x_g.dim() == 1:
            x_g = x_g.unsqueeze(0)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        if x_g.dim() == 2:
            x_g = x_g.unsqueeze(1)
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        # Target token
        x_g = self.fusion_layer(x_g, x_t)
        x_g = self.norm(x_g.squeeze(1))  # STP-Bench: was .squeeze(), which also removed a batch dim of size 1

        return x_g


class HGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HGNNLayer, self).__init__()
        self.conv = pyg_nn.HypergraphConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        return x


class HGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HGNN, self).__init__()
        self.layer1 = HGNNLayer(in_channels, hidden_channels)
        self.layer2 = HGNNLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = torch.mean(x, dim=0, keepdim=True)
        return x


class EXPLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EXPLayer, self).__init__()
        self.conv = pyg_nn.HypergraphConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = x.float()
        x = self.conv(x, edge_index)
        x = F.relu(x)
        return x


class EXPNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EXPNN, self).__init__()
        self.layer1 = EXPLayer(in_channels, hidden_channels)
        self.layer2 = EXPLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = torch.mean(x, dim=0, keepdim=True)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=250):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        #self.relu = nn.ReLU()

    def forward(self, x):
        #x = self.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

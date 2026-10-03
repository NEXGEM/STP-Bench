"""MISO (Owkin): local-attention MIL over the spots of a slide, trained with a cosine loss.

Port of https://github.com/owkin/miso_code (`miso/models/local_attention.py`,
`miso/models/layers/attention.py`, `miso/engine/losses.py`, `miso/engine/training.py`), licensed
CC BY-NC-SA 4.0 (Copyright (c) Owkin Inc.).

Each spot is a token. A token attends only to its K spatial neighbors' keys/values (local self-attention,
https://arxiv.org/abs/2205.06672); the stack is
    embedding -> [LayerNorm -> local attention (+x) -> LayerNorm -> linear (+x)] x depth -> linear head.

This model is genuinely slide-level (tokens attend to other spots of the slide), so inference batches are
NOT chunked: splitting a slide would silently change which neighbors a spot can see. One slide is one
training step, as in the reference.

Differences from the reference, on purpose: padding/masking is not ported (a slide is never padded, batch
size is 1), the layer names are kept so reference weights load directly, and `device` handling is left to the
trainer.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalSelfAttention(nn.Module):
    def __init__(self, in_features, out_features, d_model, num_heads=8, dropout=0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(in_features, d_model * 3, bias=True)
        self.to_out = nn.Linear(d_model, out_features, bias=False)
        self.dropout_func = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, neighbors: torch.LongTensor) -> torch.Tensor:
        """x: (B, N, IN), neighbors: (B, N, K) indices into the N tokens of the same item -> (B, N, OUT)."""
        b, n, _ = x.shape
        h, hd = self.num_heads, self.head_dim
        q, k, v = (t.view(b, n, h, hd).permute(0, 2, 1, 3) for t in self.to_qkv(x).chunk(3, dim=-1))  # (B,H,N,hd)

        # gather the K neighbors' keys/values of every token (flat index = (batch, head, token))
        kn = neighbors.shape[-1]
        off = (torch.arange(h, device=x.device)[None, :, None, None] * n
               + torch.arange(b, device=x.device)[:, None, None, None] * n * h)
        idx = (neighbors.unsqueeze(1).expand(b, h, n, kn) + off).reshape(-1)
        k = k.reshape(b * h * n, hd)[idx].view(b, h, n, kn, hd)
        v = v.reshape(b * h * n, hd)[idx].view(b, h, n, kn, hd)

        attn = torch.softmax(torch.einsum("bhnd,bhnkd->bhnk", q, k) * self.scale, dim=-1)
        attn = self.dropout_func(attn)
        out = torch.einsum("bhnk,bhnkd->bhnd", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, h * hd)
        return self.to_out(out)


class TilesMLP(nn.Module):
    def __init__(self, in_features, out_features=1, hidden: Optional[List[int]] = None, bias=True,
                 activation: nn.Module = nn.Sigmoid(), dropout: Optional[nn.Module] = None):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        if hidden is not None:
            for hdim in hidden:
                self.hidden_layers.append(nn.Linear(in_features, hdim, bias=bias))
                self.hidden_layers.append(activation)
                if dropout:
                    self.hidden_layers.append(dropout)
                in_features = hdim
        self.hidden_layers.append(nn.Linear(in_features, out_features, bias=bias))

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x


class Transformer(nn.Module):
    def __init__(self, in_features, depth, d_model, num_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(in_features),
                LocalSelfAttention(in_features, out_features=in_features, d_model=d_model,
                                   num_heads=num_heads, dropout=dropout),
                nn.LayerNorm(in_features),
                TilesMLP(in_features=in_features, out_features=in_features, hidden=feed_forward_hidden,
                         activation=nn.ReLU(), dropout=nn.Dropout(dropout)),
            ]))

    def forward(self, x, neighbors):
        for norm1, attention_layer, norm2, feed_forward in self.layers:
            x1 = attention_layer(norm1(x), neighbors) + x
            x = feed_forward(norm2(x1)) + x1
        return x


class LocalAttentionMIL(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim: Optional[int] = None, depth=1, d_model=128,
                 num_heads=4, feed_forward_hidden: Optional[List[int]] = None, dropout=0.0):
        super().__init__()
        if emb_dim is not None:
            self.emb_layer = nn.Linear(input_dim, emb_dim, bias=True)
            self.emb_dim = emb_dim
            self.enable_embedding = True
        else:
            self.emb_dim = input_dim
            self.enable_embedding = False
        self.transformer = Transformer(self.emb_dim, depth, d_model, num_heads, feed_forward_hidden, dropout)
        self.mlp = nn.Linear(self.emb_dim, output_dim)

    def forward(self, x, neighbors):
        """x: (B, N, F), neighbors: (B, N, K) -> (B, N, OUT). A 2-D x is treated as one item."""
        if x.ndim == 2:
            x, neighbors = x[None], neighbors[None]
        if self.enable_embedding:
            x = self.emb_layer(x)
        return self.mlp(self.transformer(x, neighbors))


class MISO(nn.Module):
    def __init__(self,
                 num_genes=200,
                 img_embedding_dim=1536,
                 emb_dim=2048,
                 depth=1,
                 d_model=1024,
                 num_heads=64,
                 feed_forward_hidden: Optional[List[int]] = None,
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        self.net = LocalAttentionMIL(input_dim=img_embedding_dim, output_dim=num_genes, emb_dim=emb_dim,
                                     depth=depth, d_model=d_model, num_heads=num_heads,
                                     feed_forward_hidden=feed_forward_hidden, dropout=dropout)

    @staticmethod
    def cosine_loss(pred, label, eps=1e-8):
        """Reference CosineSimilarityLoss: 1 - cosine over the spot axis, averaged over genes. Computed in
        float32: squared expression summed over thousands of spots overflows float16."""
        with torch.autocast(device_type=pred.device.type, enabled=False):
            return (1 - F.cosine_similarity(pred.float(), label.float(), dim=0, eps=eps)).mean()

    def forward(self, img_emb, neighbors, label=None, **kwargs):
        """img_emb: (N, D) spots of one slide, neighbors: (N, K), label: (N, G) -> logits (N, G)."""
        phase = kwargs.get('phase', 'train')
        pred = self.net(img_emb, neighbors)[0]
        if phase == 'train':
            result = {'logits': pred}
            if label is not None:
                result['loss'] = self.cosine_loss(pred, label)
            return result
        return {'logits': torch.relu(pred)}  # the reference applies a ReLU at prediction time

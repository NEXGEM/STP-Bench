"""HisDiff: hierarchical-context conditional diffusion for spatial gene expression.

Re-implementation of https://github.com/wenwenmin/HisDiff (`model.py::HisDiff`, `train.py`, `sample.py`).
The denoiser is a DiT over gene tokens (as in Stem); what differs is the condition: the spot embedding
(local) queries the embeddings of its 9 nearest spots (neighbor) and of its 49 nearest spots (global)
through two cross-attention layers, and their sum is mapped to the DiT conditioning vector.

The diffusion process (`diffusion/`) and the DiT building blocks (`modules.py`) are copies of this repository's Stem
ones (HisDiff's own `diffusion/` is byte-for-byte Stem's apart from line endings); every model folder is
self-contained, so nothing is imported from `model.stem`.

Differences from the reference, on purpose:
  * the cross-attention width follows `label_size` (the patch-encoder embedding width, 1536 for
    `uni_v2`) instead of the hard-coded 1024 of UNI v1;
  * only the `HisDiff` network is ported (the reference's `Unet` variant is unused by `train.py`);
  * validation/test sampling clamps outputs at 0 like the repo's Stem (expression is non-negative);
  * the optimizer, learning-rate schedule and epochs come from the benchmark's training loop.
"""
import torch
import torch.nn as nn

from .diffusion import create_diffusion
from .modules import DiTBlock, FinalLayer, GeneJointEmbedding, TimestepEmbedder


class HisDiffModel(nn.Module):
    def __init__(self,
                 input_size=200,
                 hidden_size=384,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 label_size=1536,
                 learn_sigma=True,
                 cross_attn_heads=8,
                 ):
        super().__init__()
        assert label_size % cross_attn_heads == 0, \
            f"label_size ({label_size}) must be divisible by cross_attn_heads ({cross_attn_heads})"
        self.learn_sigma = learn_sigma
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gene_joint_embed = GeneJointEmbedding(input_size, hidden_size)
        self.time_embed = TimestepEmbedder(hidden_size)
        # reference: crossAttention_ln (local -> neighbor spots) and crossAttention_gn (local -> global spots)
        self.cross_attn_neighbor = nn.MultiheadAttention(label_size, num_heads=cross_attn_heads, batch_first=True)
        self.cross_attn_global = nn.MultiheadAttention(label_size, num_heads=cross_attn_heads, batch_first=True)
        self.img_ebd = nn.Sequential(
            nn.Linear(label_size, label_size, bias=True),
            nn.ReLU(),
            nn.Linear(label_size, hidden_size, bias=True),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        # the reference initializes img_ebd[2] (twice) and leaves img_ebd[0] at the xavier init above
        nn.init.normal_(self.img_ebd[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, local_ebd, neighbor_ebd, global_ebd):
        """x: (N, 1, G) noised expression, t: (N,), local_ebd: (N, D), neighbor_ebd: (N, Kn, D),
        global_ebd: (N, Kg, D)  ->  (N, 2, G) (mean and variance channels)."""
        if x.dim() == 3:
            x = x.squeeze(1)
        query = local_ebd.unsqueeze(1)                                           # (N, 1, D)
        y1, _ = self.cross_attn_neighbor(query, neighbor_ebd, neighbor_ebd)
        y2, _ = self.cross_attn_global(query, global_ebd, global_ebd)
        y = self.img_ebd(y1.squeeze(1) + y2.squeeze(1))                          # (N, hidden)
        x = self.gene_joint_embed(x)                                             # (N, G, hidden)
        c = self.time_embed(t) + y
        for block in self.blocks:
            x = block(x, c)
        return self.final_layer(x, c)                                            # (N, 2, G)


class HisDiff(nn.Module):
    def __init__(self,
                 num_genes=200,
                 hidden_size=384,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 label_size=1536,
                 learn_sigma=True,
                 max_batch_size=1024,
                 **kwargs):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.num_genes = num_genes
        self.diffusion = create_diffusion(timestep_respacing="")
        self.model = HisDiffModel(input_size=num_genes, hidden_size=hidden_size, depth=depth,
                                  num_heads=num_heads, mlp_ratio=mlp_ratio, label_size=label_size,
                                  learn_sigma=learn_sigma)

    @staticmethod
    def _context(img_emb, neighbor_emb, global_emb, neighbor_idx, global_idx, lo=None, hi=None):
        """Condition tensors for spots [lo:hi]. Training batches carry the gathered context; whole-slide
        batches carry kNN indices into the slide's own embeddings (gathering here avoids shipping
        Kg x D floats per spot). Indices always point into the full slide, so cutting a chunk keeps every
        query's own neighbors."""
        sl = slice(lo, hi)
        local = img_emb[sl]
        nb = neighbor_emb[sl] if neighbor_emb is not None else img_emb[neighbor_idx[sl]]
        gl = global_emb[sl] if global_emb is not None else img_emb[global_idx[sl]]
        return dict(local_ebd=local, neighbor_ebd=nb, global_ebd=gl)

    def forward(self, img_emb, label=None, neighbor_emb=None, global_emb=None,
                neighbor_idx=None, global_idx=None, **kwargs):
        phase = kwargs.get('phase', 'train')

        if phase == 'train':
            x = label.unsqueeze(1)                                               # (N, 1, G)
            model_kwargs = self._context(img_emb, neighbor_emb, global_emb, neighbor_idx, global_idx)
            t = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=x.device)
            loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
            return {'loss': loss_dict["loss"].mean()}

        if phase not in ('val', 'test'):
            raise ValueError("Invalid phase. Must be 'train', 'val', or 'test'.")

        n = img_emb.size(0)
        samples = []
        for i in range(0, n, self.max_batch_size):
            j = min(i + self.max_batch_size, n)
            chunk = self._context(img_emb, neighbor_emb, global_emb, neighbor_idx, global_idx, i, j)
            z = torch.randn(j - i, 1, self.num_genes, device=img_emb.device)
            samples.append(self.diffusion.p_sample_loop(
                self.model.forward, z.shape, z, clip_denoised=False, model_kwargs=chunk,
                progress=True, device=img_emb.device))
        return {'logits': torch.clamp(torch.cat(samples, dim=0).squeeze(1), min=0)}

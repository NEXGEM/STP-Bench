import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree

from .denoiser import MMDiTTransformer


def zinb_prior(shape, total_count=1.0, logits=0.1, zi_logits=0.0, device=None):
    """Zero-inflated negative binomial counts, plain torch.

    Same distribution as `scvi.distributions.ZeroInflatedNegativeBinomial(total_count, logits, zi_logits)`
    (NB with mean `total_count * exp(logits)`, forced to 0 with probability `sigmoid(zi_logits)`), without the scvi
    dependency. Non-negative by construction.
    """
    nb = torch.distributions.NegativeBinomial(
        total_count=torch.tensor(float(total_count), device=device), logits=torch.tensor(float(logits), device=device)
    ).sample(shape)
    gate = torch.distributions.Bernoulli(logits=torch.tensor(float(zi_logits), device=device)).sample(shape)
    return nb * (1.0 - gate)


class PatchSampler:
    """Training-time local chunk sampler (BioFlow `data/sampling_utils.py` + `distribution_utils.py`).

    Picks a random spot, then its `k` nearest spots, with `k = max(min_samples, int(N * ratio))` and `ratio` drawn from
    `distribution` ('uniform', 'beta_a_b', 'constant_r', 'cosine', 'sqrt', 'square').
    """

    def __init__(self, distribution='uniform', min_samples=2):
        self.min_samples = min_samples
        if distribution == 'uniform':
            self.draw = np.random.rand
        elif distribution.startswith('constant_'):
            ratio = float(distribution.split('_')[1])
            self.draw = lambda: ratio
        elif distribution.startswith('beta_'):
            a, b = [float(x) for x in distribution.split('_')[1:]]
            self.draw = lambda: np.random.beta(a, b)
        elif distribution == 'cosine':
            self.draw = lambda: 1 - math.cos(np.random.rand() * math.pi * 0.5)
        elif distribution == 'sqrt':
            self.draw = lambda: math.sqrt(np.random.rand())
        elif distribution == 'square':
            self.draw = lambda: np.random.rand() ** 2
        else:
            raise ValueError(f"Unknown patch distribution: {distribution}")

    def __call__(self, coords):
        n = len(coords)
        k = min(n, max(self.min_samples, int(n * self.draw())))
        if k == n:
            return np.arange(n)
        center = coords[np.random.randint(0, n)]
        return KDTree(coords).query(center, k=k)[1]


class BioFlow(nn.Module):
    """BioFlow (https://github.com/hrterry/BioFLow): support-preserving flow matching from H&E embeddings to expression.

    A slide-level model: an MMDiT denoiser jointly attends over all spots of a slide and all genes, so inference runs
    the whole slide at once (batch 1, never chunked). Training samples local spot chunks (`PatchSampler`).

    mode='paper' (default) follows the published equations:
        x_t = (1 - t) x0 + t x1,      target  v* = x1 - x0,
        v_hat = -x_t / delta + softplus(u),             (the same v_hat in training and sampling)
        Euler:  x_{t+h} = x_t + h v_hat,  h = 1 / n_sample_steps  (paper: h = 0.05, delta = 0.1).
    With x0 >= 0 and h <= delta every Euler state, hence the output, is >= 0 without any clamp:
        x_{t+h} = (1 - h/delta) x_t + h softplus(u) >= 0.   `delta >= 1 / n_sample_steps` is therefore enforced.

    mode='released_code' reproduces what the released repository does instead (it contradicts the paper): target
    (x1 - x0) / (1 - t), v_hat = -x_t / (1 - t) + softplus(u) in training, and sampling with the raw network output u
    followed by clamp(min=0) after every step. Pick `prior_sampler: gaussian` and `n_sample_steps: 10` to match its
    defaults. It is kept only for comparison.

    The prior lives in log1p space like the labels (the benchmark's labels are log1p-normalized).
    """

    def __init__(self, num_genes=200, feature_dim=1536, hidden_dim=128, n_layers=4, n_heads=4, dropout=0.2,
                 attn_dropout=0.2, prior_sampler='zinb', zinb_total_count=1.0, zinb_logits=0.1, zinb_zi_logits=0.0,
                 delta=0.1, n_sample_steps=20, mode='paper', clamp_output=False, patch_distribution='uniform',
                 sample_times=5):
        super().__init__()
        if mode not in ('paper', 'released_code'):
            raise ValueError(f"mode must be 'paper' or 'released_code', got {mode!r}")
        if prior_sampler not in ('zinb', 'gaussian', 'zero'):
            raise ValueError(f"prior_sampler must be zinb | gaussian | zero, got {prior_sampler!r}")
        if mode == 'paper':
            if prior_sampler == 'gaussian':
                raise ValueError("the paper-faithful mode needs a non-negative prior (zinb or zero), not gaussian")
            if n_sample_steps < 1 or 1.0 / n_sample_steps > delta + 1e-9:
                raise ValueError(
                    f"h = 1/n_sample_steps = {1.0 / max(n_sample_steps, 1):.4g} must be <= delta = {delta}: "
                    "the support-preserving update is only non-negative for h <= delta")
            if clamp_output:
                raise ValueError("clamp_output is an ablation of the released-code sampler, not of the paper mode")
        self.mode = mode
        self.num_genes = num_genes
        self.delta = delta
        self.n_sample_steps = n_sample_steps
        self.clamp_output = clamp_output
        self.prior_sampler = prior_sampler
        self.zinb_args = (zinb_total_count, zinb_logits, zinb_zi_logits)
        self.sample_times = sample_times
        self.patch_sampler = PatchSampler(patch_distribution)

        self.image_transform = nn.Linear(feature_dim, feature_dim)
        self.backbone = MMDiTTransformer(
            d_img_in=feature_dim, d_gene_in=None, n_genes=num_genes, gene_input_is_indices=True,
            d_model=hidden_dim, n_layers=n_layers, n_heads=n_heads, mlp_ratio=4.0, dropout=dropout,
            attn_dropout=attn_dropout, time_hidden_dim=hidden_dim, out_dim=num_genes,
        )

    # ------------------------------------------------------------------ flow
    def sample_prior(self, shape, device):
        if self.prior_sampler == 'zinb':
            return torch.log1p(zinb_prior(shape, *self.zinb_args, device=device))
        if self.prior_sampler == 'zero':
            return torch.zeros(shape, device=device)
        return torch.randn(shape, device=device)

    def raw_output(self, x_t, img_features, t):
        """Unconstrained network output u_theta, (B, M, C). `x_t` only supplies the shape (the reference ignores it)."""
        B, M, C = x_t.shape
        gene_indices = torch.arange(C, device=x_t.device).unsqueeze(0).expand(B, -1)
        return self.backbone(img_tokens=self.image_transform(img_features), gene_indices=gene_indices, t=t)

    def velocity(self, x_t, img_features, t):
        """v_hat = -x_t / delta + softplus(u) (paper), or -x_t / (1 - t) + softplus(u) (released_code)."""
        phi = F.softplus(self.raw_output(x_t, img_features, t))
        if self.mode == 'paper':
            return -x_t / self.delta + phi
        return -x_t / (1.0 - t[:, None, None] + 1e-8) + phi

    def target(self, x1, x0, t):
        if self.mode == 'paper':
            return x1 - x0
        return (x1 - x0) / (1.0 - t[:, None, None] + 1e-8)

    def _chunk_loss(self, img_features, label):
        x0 = self.sample_prior(label.shape, label.device)
        t = torch.rand(label.shape[0], device=label.device)
        x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * label
        v_hat = self.velocity(x_t, img_features, t)
        return F.mse_loss(v_hat, self.target(label, x0, t))

    # ------------------------------------------------------------------ train / sample
    def forward(self, img_features, coords=None, label=None, **kwargs):
        phase = kwargs.get('phase', 'test')
        if img_features.dim() == 2:
            img_features = img_features.unsqueeze(0)
        if phase != 'train':
            return {'logits': self.sample(img_features).squeeze(0)}

        if label is None:
            raise ValueError("BioFlow training needs the label")
        if label.dim() == 2:
            label = label.unsqueeze(0)
        if coords is None:
            raise ValueError("BioFlow training samples local spot chunks from the coordinates")
        coords = coords.reshape(-1, coords.shape[-1]).detach().float().cpu().numpy()

        # one slide per step; `sample_times` local chunks, as the reference's `sample_times` chunks per epoch
        loss = 0.0
        for _ in range(self.sample_times):
            idx = torch.as_tensor(self.patch_sampler(coords), device=label.device, dtype=torch.long)
            loss = loss + self._chunk_loss(img_features[:, idx], label[:, idx])
        return {'loss': loss / self.sample_times}

    @torch.no_grad()
    def sample(self, img_features):
        """Euler integration t = 0, h, ..., 1 - h with h = 1 / n_sample_steps; returns x_1 (B, M, C)."""
        assert img_features.shape[0] == 1, "BioFlow inference runs one whole slide at a time"
        x = self.sample_prior((1, img_features.shape[1], self.num_genes), img_features.device)
        h = 1.0 / self.n_sample_steps
        for s in range(self.n_sample_steps):
            t = torch.full((1,), s * h, device=x.device)
            if self.mode == 'paper':
                x = x + h * self.velocity(x, img_features, t)
            else:  # released code: raw network output, clamp after every step
                x = x + h * self.raw_output(x, img_features, t)
                x = torch.clamp(x, min=0.0)
            if self.clamp_output:
                x = torch.clamp(x, min=0.0)
        return x

# Vendored from https://github.com/Kyyle2114/CAMMST (models/loss.py); only the intra-package imports
# were made relative. NOTE: the upstream repository ships no LICENSE file.
"""Loss functions for CAMMST model."""

from __future__ import annotations

import torch
from torch.nn import functional as F

__all__ = [
    "soft_cross_entropy",
    "compute_pcc_loss",
    "compute_bio_salience_sampling_loss",
    "bio_salience_regression",
    "bio_salience_scale_aware_ranking",
]


def soft_cross_entropy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Custom cross-entropy for soft targets.

    Args:
        preds: Predicted logits (B*V, B*V).
        targets: Soft target probabilities (B*V, B*V).
        reduction: Reduction mode, "none" or "mean".

    Returns:
        Loss value (scalar if reduction="mean", tensor if reduction="none").
    """
    log_softmax = F.log_softmax(preds, dim=-1)
    loss = (-targets * log_softmax).sum(dim=-1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"reduction must be 'none' or 'mean', got {reduction}")


def compute_pcc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute gene-wise PCC (Pearson Correlation Coefficient) loss using vectorized operations.

    Each gene's correlation is computed using spatial samples (spots) as observations,
    enabling proper statistical correlation analysis of spatial gene expression patterns.

    Args:
        pred: Predicted gene expressions (M, G) - masked spots x genes.
        target: Target gene expressions (M, G) - masked spots x genes.

    Returns:
        PCC loss value (scalar) - negative average correlation across genes.
    """
    # 1. compute gene-wise means (dim=0: average across masked spots)
    # result shape: (1, G)
    pred_mean = pred.mean(dim=0, keepdim=True)  # (1, G)
    target_mean = target.mean(dim=0, keepdim=True)  # (1, G)

    # 2. center the data (broadcasting applied automatically)
    # (M, G) - (1, G) -> (M, G)
    pred_centered = pred - pred_mean  # (M, G)
    target_centered = target - target_mean  # (M, G)

    # 3. compute gene-wise covariance
    # multiply (M, G) tensors and average across spot dim -> (G,) vector
    covariance = (pred_centered * target_centered).mean(dim=0)  # (G,)

    # 4. compute gene-wise standard deviations -> (G,) vector
    pred_std = torch.sqrt((pred_centered**2).mean(dim=0) + 1e-8)  # (G,)
    target_std = torch.sqrt((target_centered**2).mean(dim=0) + 1e-8)  # (G,)

    # 5. compute gene-wise correlations (element-wise division) -> (G,)
    correlation = covariance / (pred_std * target_std)  # (G,)

    # 6. average correlation across all genes
    # PCC loss: negative correlation (we want to maximize correlation)
    pcc_loss = -correlation.mean()

    return pcc_loss


def compute_bio_salience_sampling_loss(
    logits: torch.Tensor,  # (B, N)
    p_x: torch.Tensor,  # (B, N)
    bio_salience_score: torch.Tensor,  # (B, N)
    method: str = "scale_aware_ranking",
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Compute Bio-Salience Guided Sampling Loss.

    Args:
        logits: Token logits before softmax (B, N).
        p_x: Token probabilities (B, N).
        bio_salience_score: Ground truth deviation scores (B, N).
        method: Loss method ("regression", "scale_aware_ranking").
        beta: Beta exponent for ranking loss weighting.

    Returns:
        Bio-Salience sampling loss value (scalar).
    """
    batch_size, num_spots = p_x.shape
    device = p_x.device

    # min-max normalization (0~1)
    G_min = bio_salience_score.min(dim=1, keepdim=True).values
    G_max = bio_salience_score.max(dim=1, keepdim=True).values
    G_normalized = (bio_salience_score - G_min) / (G_max - G_min + 1e-8)  # (B, N)

    if method == "regression":
        return bio_salience_regression(logits, G_normalized)

    elif method == "scale_aware_ranking":
        return bio_salience_scale_aware_ranking(logits, G_normalized, beta)

    else:
        raise ValueError(f"Unknown bio_salience_method: {method}")


def bio_salience_regression(
    logits: torch.Tensor,  # (B, N)
    G_normalized: torch.Tensor,  # (B, N) in [0, 1]
) -> torch.Tensor:
    """
    Direct MSE Regression of G_normalized using logits.
    Logits are mapped to [0, 1] via sigmoid for scale match with G_normalized.

    Args:
        logits: Token logits before softmax (B, N).
        G_normalized: Normalized deviation scores (B, N) in [0, 1].

    Returns:
        MSE regression loss (scalar).
    """
    logits_scaled = torch.sigmoid(logits)
    return F.mse_loss(logits_scaled, G_normalized)


def bio_salience_scale_aware_ranking(
    logits: torch.Tensor,  # (B, N)
    G_normalized: torch.Tensor,  # (B, N) in [0, 1]
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Scale-Aware Ranking Loss.
    Pair-wise ranking loss weighted by deviation difference.

    where:
    - S_i: logit for spot i (before softmax)
    - w_ij = |G_i - G_j|^beta

    Args:
        logits: Token logits before softmax (B, N).
        G_normalized: Normalized deviation scores (B, N) in [0, 1].
        beta: Beta exponent for weighting.

    Returns:
        Scale-aware ranking loss (scalar).
    """
    batch_size, num_spots = logits.shape
    device = logits.device

    loss_total = 0.0

    for b in range(batch_size):
        logits_b = logits[b]  # (N,)
        G_b = G_normalized[b]  # (N,)

        # vectorized pair-wise computation
        # logits_diff[i, j] = logits_i - logits_j
        logits_diff = logits_b.unsqueeze(1) - logits_b.unsqueeze(0)  # (N, N)

        # G_diff[i, j] = G_i - G_j
        G_diff = G_b.unsqueeze(1) - G_b.unsqueeze(0)  # (N, N)

        # weight: w_ij = |G_i - G_j|^beta
        weights = torch.abs(G_diff) ** beta  # (N, N)

        # sign: +1 if G_i > G_j, -1 otherwise
        sign = torch.sign(G_diff)  # (N, N)

        # log-sigmoid loss: log(1 + exp(-logits_diff * sign))
        # for numerical stability, use F.softplus
        # softplus(x) = log(1 + exp(x))
        # log(1 + exp(-logits_diff * sign)) = softplus(-logits_diff * sign)
        ranking_loss = F.softplus(-logits_diff * sign)  # (N, N)

        # weighted loss
        weighted_loss = weights * ranking_loss  # (N, N)

        # use all pairs (upper triangle only to avoid duplicates)
        triu_indices = torch.triu_indices(
            num_spots, num_spots, offset=1, device=device
        )
        loss_b = weighted_loss[triu_indices[0], triu_indices[1]].mean()

        loss_total += loss_b

    return loss_total / batch_size

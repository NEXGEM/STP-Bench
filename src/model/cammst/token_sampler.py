# Vendored from https://github.com/Kyyle2114/CAMMST (models/token_sampler.py); only the intra-package imports
# were made relative. NOTE: the upstream repository ships no LICENSE file.
from __future__ import annotations

import torch
from torch import nn
from typing import Optional

from .common import Block

__all__ = ["AdaptiveTokenSampler", "RandomTokenSampler"]


class AdaptiveTokenSampler(nn.Module):
    """Sampler network that selects visible tokens via learned probabilities."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        visible_ratio: float = 0.1,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_activation: str = "swiglu",
        norm_type: str = "layer",
        region_based: bool = False,
        num_regions: Optional[int] = None,
    ) -> None:
        """
        Args:
            embed_dim: Patch embedding dimension (d).
            num_heads: Heads for the probability block.
            visible_ratio: Fraction of tokens to keep visible (0 < r <= 1).
            mlp_ratio: FFN expansion ratio inside the probability Block.
            attn_dropout: Attention dropout inside the probability Block.
            proj_dropout: Projection dropout inside the probability Block.
            ffn_activation: FFN activation function, {"gelu", "swiglu"}.
            norm_type: Normalization type, {"rms", "layer"}.
            region_based: If True, pick centers then expand with local neighbors for visible tokens.
            num_regions: Number of centers/regions when region_based is True (required).
        """
        super().__init__()
        if not (0.0 < visible_ratio <= 1.0):
            raise ValueError("visible_ratio must be in (0, 1].")
        if region_based and (num_regions is None or num_regions < 1):
            raise ValueError("num_regions must be provided and >= 1 when region_based is True.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.visible_ratio = visible_ratio
        self.region_based = region_based
        self.num_regions = num_regions

        self.prob_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            ffn_activation=ffn_activation,
            norm_type=norm_type,
        )
        self.prob_linear = nn.Linear(embed_dim, 1)
        self.prob_flatten = torch.nn.Flatten(start_dim=1)
        self.softmax = nn.Softmax(dim=-1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        patch_emb: torch.Tensor,
        coords: torch.Tensor,
        visible_ratio: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            patch_emb: Tensor of shape (B, N, d) with patch embeddings.
            coords: Tensor of shape (B, N, 2) spatial coordinates; used for ALiBi
                spatial bias and for region-based sampling.
            visible_ratio: Optional fraction of tokens to keep visible (0 < r <= 1).
                If None, uses self.visible_ratio.

        Returns:
            logits: Token logits before softmax, shape (B, N).
            p_x: Token probabilities, shape (B, N).
            vis_idx: Visible token indices, shape (B, V).
            mask: Boolean mask (B, N), True for masked tokens, False for visible.
            centers: Centers selected for region-based sampling, shape (num_centers,).
        """
        if patch_emb.dim() != 3:
            raise ValueError(f"patch_emb must have shape (B, N, d), got {patch_emb.shape}")
        bsz, seq_len, dim = patch_emb.shape
        if dim != self.embed_dim:
            raise ValueError(f"patch_emb embed_dim mismatch: expected {self.embed_dim}, got {dim}")

        if coords.shape != (bsz, seq_len, 2):
            raise ValueError(f"coords must have shape (B, N, 2), got {coords.shape}")

        # apply Block with ALiBi using coords, then linear + flatten
        logits = self.prob_block(patch_emb, coords=coords)  # (B, N, d)
        logits = self.prob_linear(logits)  # (B, N, 1)
        logits = self.prob_flatten(logits)  # (B, N)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        p_x = self.softmax(logits)  # (B, N)

        # use provided visible_ratio or default to self.visible_ratio
        ratio = visible_ratio if visible_ratio is not None else self.visible_ratio
        visible_tokens = max(1, int(round(seq_len * ratio)))
        visible_tokens = min(visible_tokens, seq_len)

        if self.region_based:
            # region-based: sample centers based on p_x, then expand with k-NN in coordinate space
            dist = torch.cdist(coords, coords)  # (B, N, N)
            num_centers = min(self.num_regions, seq_len)
            tokens_per_region = max(1, visible_tokens // num_centers)
            vis_idx_list = []
            
            for b in range(bsz):
                # select centers based on probabilities
                if self.training:
                    centers = torch.multinomial(p_x[b], num_samples=num_centers, replacement=False)
                else:
                    # for inference, select top num_centers tokens with highest probabilities
                    centers = torch.topk(p_x[b], k=num_centers, largest=True).indices
                
                # calculate distance from each token to its nearest selected center
                # dist[b, :, centers] -> (N, num_centers): distances to selected centers
                # .min(dim=1) -> (N,): distance to the closest center for each token
                d_to_nearest_center = dist[b, :, centers].min(dim=1).values 

                # expand regions (K-NN)
                candidate_mask = torch.zeros(seq_len, device=patch_emb.device, dtype=torch.bool)
                for c in centers:
                    k = min(tokens_per_region, seq_len)
                    knn = torch.topk(dist[b, c], k=k, largest=False).indices
                    candidate_mask[knn] = True

                candidate_idx = torch.nonzero(candidate_mask).squeeze(1)

                # adjust to match exactly `visible_tokens` using distance
                if candidate_idx.numel() > visible_tokens:
                    # case 1: too many tokens selected (regions overlap heavily or broad coverage)
                    # prune tokens that are farthest from any center.
                    # select `visible_tokens` with the smallest `d_to_nearest_center`.
                    cand_dists = d_to_nearest_center[candidate_idx]
                    top = torch.topk(cand_dists, k=visible_tokens, largest=False).indices
                    vis_idx_b = candidate_idx[top]

                elif candidate_idx.numel() < visible_tokens:
                    # case 2: too few tokens selected (regions overlap too much)
                    # fill remaining slots with unselected tokens closest to any center.
                    vis_idx_b = candidate_idx
                    remaining_count = visible_tokens - vis_idx_b.numel()
                    
                    # clone distances to ensure safety
                    temp_dists = d_to_nearest_center.clone()
                    
                    # mask already selected tokens by setting their distance to infinity
                    # prevents duplicate selection.
                    temp_dists[vis_idx_b] = float('inf')
                    
                    # calculate how many valid candidates remain (those not set to inf)
                    valid_count = (temp_dists < float('inf')).sum()
                    
                    # determine actual number to sample (handle edge case where remaining < needed)
                    actual_k = min(remaining_count, valid_count)
                    
                    if actual_k > 0:
                        # select closest remaining tokens
                        additional = torch.topk(temp_dists, k=actual_k, largest=False).indices
                        vis_idx_b = torch.cat([vis_idx_b, additional], dim=0)
                else:
                    # exact match
                    vis_idx_b = candidate_idx
                
                vis_idx_list.append(vis_idx_b)
            vis_idx = torch.stack(vis_idx_list, dim=0)
        else:
            # scattered sampling
            if self.training:
                vis_idx = torch.multinomial(p_x, num_samples=visible_tokens, replacement=False)  # (B, V)
                centers = None
            else:
                # for inference, select top visible_tokens tokens with highest probabilities
                vis_idx = torch.topk(p_x, k=visible_tokens, largest=True).indices
                centers = None
            
        # remove duplicate indices to ensure shape consistency
        vis_idx = torch.unique(vis_idx).unsqueeze(0)    # bsz is always 1 in this case

        mask = torch.ones((bsz, seq_len), device=patch_emb.device, dtype=torch.bool)
        mask.scatter_(dim=-1, index=vis_idx, value=False)  # False for visible, True for masked

        return logits, p_x, vis_idx, mask, centers


class RandomTokenSampler(nn.Module):
    """Random sampler that selects visible tokens randomly without learning."""
    def __init__(
        self,
        embed_dim: int,
        visible_ratio: float = 0.1,
        region_based: bool = False,
        num_regions: Optional[int] = None,
    ) -> None:
        """
        Args:
            embed_dim: Patch embedding dimension (d).
            visible_ratio: Fraction of tokens to keep visible (0 < r <= 1).
            region_based: If True, pick centers then expand with local neighbors for visible tokens.
            num_regions: Number of centers/regions when region_based is True (required).
        """
        super().__init__()
        if not (0.0 < visible_ratio <= 1.0):
            raise ValueError("visible_ratio must be in (0, 1].")
        if region_based and (num_regions is None or num_regions < 1):
            raise ValueError("num_regions must be provided and >= 1 when region_based is True.")
        self.embed_dim = embed_dim
        self.visible_ratio = visible_ratio
        self.region_based = region_based
        self.num_regions = num_regions

    def forward(
        self,
        patch_emb: torch.Tensor,
        coords: torch.Tensor,
        visible_ratio: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            patch_emb: Tensor of shape (B, N, d) with patch embeddings.
            coords: Tensor of shape (B, N, 2) spatial coordinates; used for
                region-based sampling when region_based is True.
            visible_ratio: Optional fraction of tokens to keep visible (0 < r <= 1).
                If None, uses self.visible_ratio.

        Returns:
            logits: Token logits before softmax, shape (B, N). Uniform for random sampling.
            p_x: Token probabilities, shape (B, N). Uniform distribution for random sampling.
            vis_idx: Visible token indices, shape (B, V).
            mask: Boolean mask (B, N), True for masked tokens, False for visible.
            centers: Centers selected for region-based sampling, shape (num_centers,).
        """
        if patch_emb.dim() != 3:
            raise ValueError(f"patch_emb must have shape (B, N, d), got {patch_emb.shape}")
        bsz, seq_len, dim = patch_emb.shape
        if dim != self.embed_dim:
            raise ValueError(f"patch_emb embed_dim mismatch: expected {self.embed_dim}, got {dim}")

        if coords.shape != (bsz, seq_len, 2):
            raise ValueError(f"coords must have shape (B, N, 2), got {coords.shape}")

        device = patch_emb.device

        # create uniform probability distribution
        # for uniform distribution, logits are all zeros (log(1/N) + constant)
        logits = torch.zeros((bsz, seq_len), device=device)  # (B, N)
        p_x = torch.ones((bsz, seq_len), device=device) / seq_len  # (B, N)

        # use provided visible_ratio or default to self.visible_ratio
        ratio = visible_ratio if visible_ratio is not None else self.visible_ratio
        visible_tokens = max(1, int(round(seq_len * ratio)))
        visible_tokens = min(visible_tokens, seq_len)

        if self.region_based:
            # region-based: randomly sample centers, then expand with k-NN in coordinate space
            dist = torch.cdist(coords, coords)  # (B, N, N)
            num_centers = min(self.num_regions, seq_len)
            tokens_per_region = max(1, visible_tokens // num_centers)
            vis_idx_list = []

            for b in range(bsz):
                # randomly select centers (uniform random selection)
                centers = torch.randperm(seq_len, device=device)[:num_centers]

                # calculate distance from each token to its nearest selected center
                # dist[b, :, centers] -> (N, num_centers): distances to selected centers
                # .min(dim=1) -> (N,): distance to the closest center for each token
                d_to_nearest_center = dist[b, :, centers].min(dim=1).values

                # expand regions (K-NN)
                candidate_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
                for c in centers:
                    k = min(tokens_per_region, seq_len)
                    knn = torch.topk(dist[b, c], k=k, largest=False).indices
                    candidate_mask[knn] = True

                candidate_idx = torch.nonzero(candidate_mask).squeeze(1)

                # adjust to match exactly `visible_tokens` using distance
                if candidate_idx.numel() > visible_tokens:
                    # case 1: too many tokens selected (regions overlap heavily or broad coverage)
                    # prune tokens that are farthest from any center.
                    # select `visible_tokens` with the smallest `d_to_nearest_center`.
                    cand_dists = d_to_nearest_center[candidate_idx]
                    top = torch.topk(cand_dists, k=visible_tokens, largest=False).indices
                    vis_idx_b = candidate_idx[top]

                elif candidate_idx.numel() < visible_tokens:
                    # case 2: too few tokens selected (regions overlap too much)
                    # fill remaining slots with unselected tokens closest to any center.
                    vis_idx_b = candidate_idx
                    remaining_count = visible_tokens - vis_idx_b.numel()

                    # clone distances to ensure safety
                    temp_dists = d_to_nearest_center.clone()

                    # mask already selected tokens by setting their distance to infinity
                    # prevents duplicate selection.
                    temp_dists[vis_idx_b] = float('inf')

                    # calculate how many valid candidates remain (those not set to inf)
                    valid_count = (temp_dists < float('inf')).sum()

                    # determine actual number to sample (handle edge case where remaining < needed)
                    actual_k = min(remaining_count, valid_count)

                    if actual_k > 0:
                        # select closest remaining tokens
                        additional = torch.topk(temp_dists, k=actual_k, largest=False).indices
                        vis_idx_b = torch.cat([vis_idx_b, additional], dim=0)
                else:
                    # exact match
                    vis_idx_b = candidate_idx

                vis_idx_list.append(vis_idx_b)
            vis_idx = torch.stack(vis_idx_list, dim=0)
        else:
            # scattered sampling: randomly select tokens
            vis_idx = torch.stack([
                torch.randperm(seq_len, device=device)[:visible_tokens]
                for _ in range(bsz)
            ], dim=0)  # (B, V)
            centers = None

        # remove duplicate indices to ensure shape consistency
        vis_idx = torch.unique(vis_idx).unsqueeze(0)    # bsz is always 1 in this case

        mask = torch.ones((bsz, seq_len), device=device, dtype=torch.bool)
        mask.scatter_(dim=-1, index=vis_idx, value=False)  # False for visible, True for masked

        return logits, p_x, vis_idx, mask, centers

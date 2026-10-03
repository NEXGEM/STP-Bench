# Vendored from https://github.com/Kyyle2114/CAMMST (models/cammst.py); only the intra-package imports
# were made relative. NOTE: the upstream repository ships no LICENSE file.
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .common import make_norm_layer
from .gene_encoder import GeneEncoder
from .token_sampler import AdaptiveTokenSampler, RandomTokenSampler
from .joint_encoder import CrossModalJointEncoder
from .loss import (
    soft_cross_entropy,
    compute_pcc_loss,
    compute_bio_salience_sampling_loss,
)

__all__ = ["CAMMST"]


class CAMMST(nn.Module):
    """Contrastive & Adaptive Multi-modal Masked Autoencoder for Spatial Transcriptomics."""

    def __init__(
        self,
        input_dim: int = 1536,  # input feature dimension (1536 for UNI2-h)
        embed_dim: int = 768,
        num_genes: int = 250,
        visible_ratio: float = 0.1,
        joint_depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_activation: str = "swiglu",  # "gelu" or "swiglu"
        contrastive_dim: Optional[int] = None,
        decoder_depth: int = 2,
        norm_type: str = "rms",  # "rms" or "layer"
        region_based_sampling: bool = False,
        num_regions: Optional[int] = None,
        sampler_type: str = "adaptive",  # "adaptive" or "random"
    ) -> None:
        """
        Args:
            input_dim: Input feature dimension (1536 for UNI2-h).
            embed_dim: Token embedding dimension.
            num_genes: Number of genes in expression profiles.
            visible_ratio: Fraction of tokens to keep visible (0 < r <= 1).
            joint_depth: Number of Transformer blocks for the fusion path.
            num_heads: Attention heads.
            mlp_ratio: FFN expansion ratio for all blocks.
            attn_dropout: Attention dropout for all blocks.
            proj_dropout: Projection/FFN dropout for all blocks.
            ffn_activation: FFN activation function, {"gelu", "swiglu"}.
            contrastive_dim: Projection dim for contrastive head (defaults to embed_dim).
            decoder_depth: Number of decoder transformer blocks (must be >= 1).
            norm_type: Normalization type, {"rms", "layer"}.
            region_based_sampling: If True, pick centers then expand with local neighbors.
            num_regions: Number of centers/regions when region_based is True.
            sampler_type: Token sampler type, {"adaptive", "random"}.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_genes = num_genes
        self.input_dim = input_dim
        self.visible_ratio = visible_ratio
        self.norm_type = norm_type
        self.sampler_type = sampler_type

        # UNI feature projection (UNI dim -> embed_dim)
        self.uni_proj = nn.Linear(input_dim, embed_dim)

        # Gene Encoder: projects gene profiles to embedding space
        self.gene_encoder = GeneEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
        )

        # Token Sampler: selects visible tokens (adaptive or random)
        if sampler_type == "adaptive":
            self.token_sampler = AdaptiveTokenSampler(
                embed_dim=embed_dim,
                num_heads=num_heads,
                visible_ratio=visible_ratio,
                ffn_activation=ffn_activation,
                norm_type=norm_type,
                region_based=region_based_sampling,
                num_regions=num_regions,
            )
        elif sampler_type == "random":
            self.token_sampler = RandomTokenSampler(
                embed_dim=embed_dim,
                visible_ratio=visible_ratio,
                region_based=region_based_sampling,
                num_regions=num_regions,
            )
        else:
            raise ValueError(f"sampler_type must be 'adaptive' or 'random', got {sampler_type}")

        # Joint Encoder: fuses patch and gene representations
        self.joint_encoder = CrossModalJointEncoder(
            embed_dim=embed_dim,
            joint_depth=joint_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            ffn_activation=ffn_activation,
            norm_type=norm_type,
            contrastive_dim=contrastive_dim,
            decoder_depth=decoder_depth,
        )

        # gene reconstruction (prediction) head
        self.recon_norm = make_norm_layer(norm_type, embed_dim, 1e-6)
        self.recon_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(embed_dim, num_genes),
        )

        # initialize parameters
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize model parameters."""
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
        features: torch.Tensor,
        gt_expressions: torch.Tensor,
        coords: torch.Tensor,
        visible_ratio: Optional[float] = None,
        bio_salience_score: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of CAMMST model.

        Args:
            features: Pre-computed UNI patch embeddings of shape (B, N, d).
            gt_expressions: Ground truth gene expression profiles of shape (B, N, G).
            coords: Spatial coordinates of shape (B, N, 2).
            visible_ratio: Optional fraction of tokens to keep visible (0 < r <= 1).
                If None, uses the model's default visible_ratio.
            bio_salience_score: Optional bio-salience deviation scores of shape (B, N).
                Used as target for Bio-Salience Guided Sampling Loss.

        Returns:
            Dictionary containing predictions and intermediate representations:
            - pred_expressions: Predicted gene expressions for masked spots (B*M, G) - flattened
            - gt_masked: Ground truth gene expressions for masked spots (B*M, G) - flattened
            - p_x: Token probabilities (B, N)
            - vis_idx: Visible token indices (B, V)
            - mask: Boolean mask (B, N)
            - bio_salience_score: Pass-through of input bio_salience_score (B, N)
            - ... (joint encoder outputs)
        """
        if features.dim() != 3:
            raise ValueError(f"features must have shape (B, N, d), got {features.shape}")
        if gt_expressions.dim() != 3:
            raise ValueError(f"gt_expressions must have shape (B, N, G), got {gt_expressions.shape}")
        if coords.dim() != 3 or coords.shape[2] != 2:
            raise ValueError(f"coords must have shape (B, N, 2), got {coords.shape}")
        if visible_ratio is not None and not (0.0 < visible_ratio <= 1.0):
            raise ValueError(f"visible_ratio must be in (0, 1], got {visible_ratio}")

        batch_size, num_spots, uni_embed_dim = features.shape
        _, _, num_genes = gt_expressions.shape

        if num_genes != self.num_genes:
            raise ValueError(f"gene count mismatch: {num_genes} vs {self.num_genes}")

        # project UNI features to model embedding dimension
        patch_emb = self.uni_proj(features)  # (B, N, input_dim) -> (B, N, embed_dim)

        # step 1: adaptive token sampling - select informative visible tokens
        logits, p_x, vis_idx, mask, centers = self.token_sampler(patch_emb, coords, visible_ratio)

        # step 2: gene encoding for visible tokens
        # gather gene expressions for selected visible spots
        vis_idx_expanded = vis_idx.unsqueeze(-1).expand(-1, -1, num_genes)
        visible_expressions = gt_expressions.gather(dim=1, index=vis_idx_expanded)  # (B, V, G)
        gene_emb = self.gene_encoder(visible_expressions)  # (B, V, embed_dim)

        # step 3: joint encoding - fuse image and gene representations
        joint_output = self.joint_encoder(
            patch_emb=patch_emb,
            gene_emb=gene_emb,
            mask=mask,
            vis_idx=vis_idx,
            coords=coords,
        )

        # step 4: gene reconstruction prediction
        prediction_tokens = joint_output["decoded_tokens"]

        # predict only for masked (unseen) positions
        # boolean indexing flattens (B, N, D) to (B*M, D) where M is number of masked spots
        masked_tokens = prediction_tokens[mask]  # extract tokens for masked spots
        masked_tokens = self.recon_norm(masked_tokens)
        pred_expressions = self.recon_head(masked_tokens)  # (B*M, num_genes)

        # gather ground truth for masked positions
        gt_masked = gt_expressions[mask]  # (B*M, G)

        return {
            # predictions
            "pred_expressions": pred_expressions,  # (B*M, G)
            "gt_masked": gt_masked,  # (B*M, G)

            # sampling outputs
            "logits": logits,  # (B, N)
            "p_x": p_x,  # (B, N)
            "vis_idx": vis_idx,  # (B, V)
            "mask": mask,  # (B, N)
            "centers": centers,  # (B, num_centers) or None

            # bio-salience score (pass-through)
            "bio_salience_score": bio_salience_score,  # (B, N) or None

            # joint encoder outputs
            **joint_output,
        }

    @torch.inference_mode()
    def infer_with_all_mask(
        self,
        features: torch.Tensor,
        gt_expressions: torch.Tensor,
        coords: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference method with all spots masked (no adaptive sampling).

        This method is used during validation phase to predict gene expressions
        for all spots without using any visible gene information. All spots are
        treated as masked tokens.

        Args:
            features: Pre-computed UNI patch embeddings of shape (B, N, d).
            gt_expressions: Ground truth gene expression profiles of shape (B, N, G).
                Used only for gathering ground truth for evaluation.
            coords: Spatial coordinates of shape (B, N, 2).
            visible_ratio: Optional fraction of tokens to keep visible (0 < r <= 1).
                Ignored in this method since all tokens are masked.

        Returns:
            Dictionary containing predictions and intermediate representations:
            - pred_expressions: Predicted gene expressions for all spots (B*N, G)
            - gt_masked: Ground truth gene expressions for all spots (B*N, G)
            - p_x: Token probabilities (B, N) - all zeros since no sampling
            - vis_idx: Visible token indices (B, 0) - empty since all masked
            - mask: Boolean mask (B, N) - all True
            - ... (joint encoder outputs)
        """
        if features.dim() != 3:
            raise ValueError(f"features must have shape (B, N, d), got {features.shape}")
        if gt_expressions.dim() != 3:
            raise ValueError(f"gt_expressions must have shape (B, N, G), got {gt_expressions.shape}")
        if coords.dim() != 3 or coords.shape[2] != 2:
            raise ValueError(f"coords must have shape (B, N, 2), got {coords.shape}")

        batch_size, num_spots, uni_embed_dim = features.shape
        _, _, num_genes = gt_expressions.shape

        if num_genes != self.num_genes:
            raise ValueError(f"gene count mismatch: {num_genes} vs {self.num_genes}")

        # project UNI features to model embedding dimension
        patch_emb = self.uni_proj(features)  # (B, N, input_dim) -> (B, N, embed_dim)

        # all spots are masked: no adaptive sampling
        # create empty visible indices and mask all spots
        device = patch_emb.device
        logits = torch.zeros((batch_size, num_spots), device=device)  # (B, N) - dummy logits
        vis_idx = torch.empty((batch_size, 0), dtype=torch.long, device=device)  # (B, 0)
        mask = torch.ones((batch_size, num_spots), dtype=torch.bool, device=device)  # (B, N) - all True
        p_x = torch.zeros((batch_size, num_spots), device=device)  # (B, N) - dummy probabilities

        # no gene encoding since all spots are masked
        # create empty gene embeddings
        gene_emb = torch.empty((batch_size, 0, self.embed_dim), device=device)  # (B, 0, d)

        # joint encoding - fuse image and gene representations
        # all gene slots will use mask_token in joint_encoder
        joint_output = self.joint_encoder(
            patch_emb=patch_emb,
            gene_emb=gene_emb,
            mask=mask,
            vis_idx=vis_idx,
            coords=coords,
        )

        # gene reconstruction prediction
        prediction_tokens = joint_output["decoded_tokens"]

        # predict for all positions (all are masked)
        # boolean indexing flattens (B, N, D) to (B*N, D)
        prediction_tokens = prediction_tokens[mask]
        prediction_tokens = self.recon_norm(prediction_tokens)
        pred_expressions = self.recon_head(prediction_tokens)  # (B*N, num_genes)

        # ground truth for all positions
        gt_masked = gt_expressions[mask]  # (B*N, G)

        return {
            # predictions
            "pred_expressions": pred_expressions,  # (B*N, G)
            "gt_masked": gt_masked,  # (B*N, G)

            # sampling outputs (dummy values since no sampling)
            "logits": logits,  # (B, N) - all zeros
            "p_x": p_x,  # (B, N) - all zeros
            "vis_idx": vis_idx,  # (B, 0) - empty
            "mask": mask,  # (B, N) - all True

            # joint encoder outputs
            **joint_output,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        recon_weight: float = 1.0,
        sample_weight: float = 1e-4,
        contrast_weight: float = 0.01,
        contrast_temp: float = 1.0,
        pcc_weight: float = 0.5,
        contrastive_type: str = "soft",
        bio_salience_method: str = "scale_aware_ranking",
        bio_salience_beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.

        Args:
            outputs: Output dictionary from forward pass.
            recon_weight: Weight for reconstruction loss.
            sample_weight: Weight for sampling loss.
            contrast_weight: Weight for contrastive loss.
            contrast_temp: Temperature for contrastive loss.
            pcc_weight: Weight for PCC loss (gene-wise spatial correlation).
            contrastive_type: Type of contrastive loss, "hard" (InfoNCE) or "soft" (BLEEP-style).
            bio_salience_method: Bio-salience loss method ("regression", "scale_aware_ranking").
            bio_salience_beta: Beta exponent for scale-aware ranking weighting.

        Returns:
            Dictionary of loss components and total loss.
        """
        pred_expressions = outputs["pred_expressions"]  # (M, G)
        gt_masked = outputs["gt_masked"]  # (M, G)
        logits = outputs["logits"]  # (B, N)
        p_x = outputs["p_x"]  # (B, N)
        contrastive_patch = outputs.get("contrastive_patch")  # (B, V, c_dim)
        contrastive_gene = outputs.get("contrastive_gene")  # (B, V, c_dim)

        # --- Reconstruction Loss ---
        # reduction='none' -> (M, G)
        mse_elements = F.mse_loss(pred_expressions, gt_masked, reduction='none')

        # (M,): Per-spot average error
        per_spot_mse = mse_elements.mean(dim=-1)

        # scalar: overall average error (reconstruction loss)
        recon_loss = per_spot_mse.mean()

        # --- Gene-wise PCC Loss ---
        # already flattened: (M, G)
        num_genes = pred_expressions.shape[-1]

        pred_flat = pred_expressions.reshape(-1, num_genes) # (M, G) - M masked spots
        gt_flat = gt_masked.reshape(-1, num_genes)          # (M, G) - M masked spots

        pcc_loss = compute_pcc_loss(pred_flat, gt_flat)

        # --- Sampling Loss (Bio-Salience) ---
        sample_loss = torch.tensor(0.0, device=pred_expressions.device)
        if sample_weight > 0.0:
            bio_salience_score = outputs.get("bio_salience_score")
            if bio_salience_score is None:
                raise ValueError("bio_salience_score required but not provided in outputs")
            sample_loss = compute_bio_salience_sampling_loss(
                logits=logits,
                p_x=p_x,
                bio_salience_score=bio_salience_score,
                method=bio_salience_method,
                beta=bio_salience_beta,
            )

        # --- Contrastive loss: align image and gene representations ---
        contrast_loss = torch.tensor(0.0, device=pred_expressions.device)
        if contrastive_patch is not None and contrastive_gene is not None:
            # representations are already L2-normalized from joint_encoder
            patch_norm = contrastive_patch  # (B, V, contrast_dim)
            gene_norm = contrastive_gene    # (B, V, contrast_dim)

            batch_size, num_visible, contrast_dim = patch_norm.shape

            # only compute contrastive loss if there are visible tokens
            if num_visible > 0:
                # reshape for matrix multiplication: (B*V, contrast_dim)
                patch_flat = patch_norm.view(-1, contrast_dim)
                gene_flat = gene_norm.view(-1, contrast_dim)

                # similarity matrix: (B*V, B*V)
                logits = torch.mm(patch_flat, gene_flat.t()) / contrast_temp

                if contrastive_type == "hard":
                    # hard contrastive loss (InfoNCE): maximize similarity for corresponding patch-gene pairs
                    # positive pairs: same spatial location (patch[i] ↔ gene[i])
                    # negative pairs: different spatial locations
                    labels = torch.arange(batch_size * num_visible, device=logits.device)
                    contrast_loss = F.cross_entropy(logits, labels)
                elif contrastive_type == "soft":
                    # soft contrastive loss (BLEEP style): similarity-based soft targets
                    # compute internal similarities
                    patch_similarity = torch.mm(patch_flat, patch_flat.t())  # (B*V, B*V)
                    gene_similarity = torch.mm(gene_flat, gene_flat.t())    # (B*V, B*V)

                    # soft target: average of patch and gene similarities
                    targets = F.softmax(
                        ((patch_similarity + gene_similarity) / 2) / contrast_temp, dim=-1
                    )

                    # bidirectional loss
                    patch_to_gene_loss = soft_cross_entropy(logits, targets, reduction='none')
                    gene_to_patch_loss = soft_cross_entropy(logits.t(), targets.t(), reduction='none')
                    contrast_loss = (patch_to_gene_loss + gene_to_patch_loss).mean() / 2.0
                else:
                    raise ValueError(f"contrastive_type must be 'hard' or 'soft', got {contrastive_type}")

        # total loss: L = L_mse + pcc_weight * L_pcc + sample_weight * L_sample + contrast_weight * L_contrast
        total_loss = (
            recon_weight * recon_loss +
            pcc_weight * pcc_loss +
            sample_weight * sample_loss +
            contrast_weight * contrast_loss
        )

        return {
            "total_loss": total_loss,
            "recon_loss": recon_weight * recon_loss,
            "pcc_loss": pcc_weight * pcc_loss,
            "sample_loss": sample_weight * sample_loss,
            "contrast_loss": contrast_weight * contrast_loss,
        }

    @torch.inference_mode()
    def slide_inference(
        self,
        features: torch.Tensor,
        gt_expressions: torch.Tensor,
        coords: torch.Tensor,
        visible_ratio: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference method that predicts gene expressions for all spots in a slide.

        Args:
            features: Pre-computed UNI patch embeddings of shape (B, N, d).
            gt_expressions: Ground truth gene expression profiles of shape (B, N, G).
            coords: Spatial coordinates of shape (B, N, 2).
            visible_ratio: Fraction of tokens to keep visible (0 <= r <= 1).
                If 0, all spots are masked (no visible spots).
                If > 0, uses adaptive token sampling.

        Returns:
            Dictionary containing predictions for all spots:
            - pred_expressions: Predicted gene expressions for all spots (B*N, G)
            - pred_expressions_with_gt: Predictions with GT for visible spots, predictions for masked (B*N, G)
            - gt_all: Ground truth gene expressions for all spots (B*N, G)
            - p_x: Token probabilities (B, N)
            - vis_idx: Visible token indices (B, V)
            - mask: Boolean mask (B, N) - True for masked spots
            - ... (joint encoder outputs)
        """
        if features.dim() != 3:
            raise ValueError(f"features must have shape (B, N, d), got {features.shape}")
        if gt_expressions.dim() != 3:
            raise ValueError(f"gt_expressions must have shape (B, N, G), got {gt_expressions.shape}")
        if coords.dim() != 3 or coords.shape[2] != 2:
            raise ValueError(f"coords must have shape (B, N, 2), got {coords.shape}")
        if not (0.0 <= visible_ratio <= 1.0):
            raise ValueError(f"visible_ratio must be in [0, 1], got {visible_ratio}")

        batch_size, num_spots, uni_embed_dim = features.shape
        _, _, num_genes = gt_expressions.shape

        if num_genes != self.num_genes:
            raise ValueError(f"gene count mismatch: {num_genes} vs {self.num_genes}")

        # project UNI features to model embedding dimension
        patch_emb = self.uni_proj(features)  # (B, N, input_dim) -> (B, N, embed_dim)

        # step 1: token sampling based on visible_ratio
        if visible_ratio == 0.0:
            # all spots are masked - no visible spots
            device = patch_emb.device
            logits = torch.zeros((batch_size, num_spots), device=device)  # (B, N) - dummy logits
            vis_idx = torch.empty((batch_size, 0), dtype=torch.long, device=device)  # (B, 0)
            mask = torch.ones((batch_size, num_spots), dtype=torch.bool, device=device)  # (B, N) - all True
            p_x = torch.zeros((batch_size, num_spots), device=device)  # (B, N) - dummy probabilities
            centers = None
        else:
            # adaptive token sampling
            logits, p_x, vis_idx, mask, centers = self.token_sampler(patch_emb, coords, visible_ratio)

        # step 2: prepare gene embeddings for visible spots (if any)
        if len(vis_idx[0]) > 0:  # there are visible spots
            vis_idx_expanded = vis_idx.unsqueeze(-1).expand(-1, -1, num_genes)
            visible_expressions = gt_expressions.gather(dim=1, index=vis_idx_expanded)  # (B, V, G)
            gene_emb = self.gene_encoder(visible_expressions)  # (B, V, embed_dim)
        else:
            # no visible spots
            gene_emb = torch.empty((batch_size, 0, self.embed_dim), device=features.device)  # (B, 0, d)

        # step 3: joint encoding - fuse image and gene representations
        joint_output = self.joint_encoder(
            patch_emb=patch_emb,
            gene_emb=gene_emb,
            mask=mask,
            vis_idx=vis_idx,
            coords=coords,
        )

        # step 4: gene reconstruction prediction for all spots
        prediction_tokens = joint_output["decoded_tokens"]

        # predict for all positions
        prediction_tokens = self.recon_norm(prediction_tokens)
        pred_expressions = self.recon_head(prediction_tokens)  # (B, N, num_genes)
        pred_expressions_flat = pred_expressions.view(batch_size * num_spots, self.num_genes)  # (B*N, G)

        # create pred_expressions_with_gt: visible spots use GT, masked spots use predictions
        pred_expressions_with_gt = pred_expressions.clone()  # (B, N, G)
        if len(vis_idx[0]) > 0:  # there are visible spots
            # for each batch element
            for b in range(batch_size):
                vis_indices = vis_idx[b]  # (V,)
                # put GT values at visible positions
                pred_expressions_with_gt[b, vis_indices] = gt_expressions[b, vis_indices]
        pred_expressions_with_gt_flat = pred_expressions_with_gt.view(batch_size * num_spots, self.num_genes)  # (B*N, G)

        # flatten ground truth
        gt_all = gt_expressions.view(batch_size * num_spots, self.num_genes)  # (B*N, G)
        
        pred_expressions_masked = pred_expressions_flat[mask.squeeze(0)]
        gt_masked = gt_all[mask.squeeze(0)]
        
        return {
            # predictions
            "pred_expressions": pred_expressions_flat,  # (B*N, G) - all predictions
            "pred_expressions_with_gt": pred_expressions_with_gt_flat,  # (B*N, G) - GT for visible, pred for masked
            "gt_all": gt_all,  # (B*N, G)
            
            # predictions for masked spots
            "pred_expressions_masked": pred_expressions_masked,  # (B*M, G) - predictions for masked spots
            "gt_masked": gt_masked,  # (B*M, G) - ground truth for masked spots

            # sampling outputs
            "logits": logits,  # (B, N)
            "p_x": p_x,  # (B, N)
            "vis_idx": vis_idx,  # (B, V)
            "mask": mask,  # (B, N) - True for masked spots
            "centers": centers,  # (B, num_centers) - Centers selected for region-based sampling

            # joint encoder outputs
            **joint_output,
        }

    @torch.inference_mode()
    def get_vis_indices(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        visible_ratio: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract visible token indices using the adaptive token sampler.

        This method only performs sampling without any gene encoding or prediction,
        useful for visualization of selected regions.

        Args:
            features: Pre-computed UNI patch embeddings of shape (B, N, d).
            coords: Spatial coordinates of shape (B, N, 2).
            visible_ratio: Optional fraction of tokens to keep visible (0 < r <= 1).
                If None, uses the model's default visible_ratio.

        Returns:
            p_x: Token probabilities, shape (B, N).
            vis_idx: Visible token indices, shape (B, V).
            mask: Boolean mask (B, N), True for masked tokens, False for visible.
            centers: Centers selected for region-based sampling, shape (num_centers,).
        """
        if features.dim() != 3:
            raise ValueError(f"features must have shape (B, N, d), got {features.shape}")
        if coords.dim() != 3 or coords.shape[2] != 2:
            raise ValueError(f"coords must have shape (B, N, 2), got {coords.shape}")

        # project UNI features to model embedding dimension
        patch_emb = self.uni_proj(features)  # (B, N, input_dim) -> (B, N, embed_dim)

        # perform adaptive token sampling
        _, p_x, vis_idx, mask, centers = self.token_sampler(patch_emb, coords, visible_ratio)

        return p_x, vis_idx, mask, centers
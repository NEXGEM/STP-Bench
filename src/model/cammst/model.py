import torch
import torch.nn as nn
import torch.nn.functional as F

from .cammst import CAMMST


class CAMMSTModel(nn.Module):
    """STP-Bench wrapper around CAMMST (Kim et al., MICCAI 2026, https://github.com/Kyyle2114/CAMMST).

    CAMMST is a masked autoencoder over the spots of a slide: a subset of spots reveals its gene
    expression (`visible_ratio`) and the model reconstructs the rest from the UNI2-h embeddings,
    the spot coordinates (ALiBi) and the visible expression. It is genuinely slide-level (every spot
    attends to every other spot), so inference is NOT chunked: splitting a slide would change what
    a spot can attend to.

    Training follows the reference (visible spots revealed, loss on the masked ones). Evaluation
    (val / test / predict) is image-only: it runs `infer_with_all_mask`, i.e. every spot masked,
    which is the reference's `visible_ratio=0` setting and the only setting that is comparable with
    the other benchmark models. The label is never passed to the network at evaluation time -- a
    zero tensor stands in for it -- so held-out expression cannot leak into a prediction.
    """

    def __init__(self, num_genes, input_dim=1536, embed_dim=768, visible_ratio=0.1, joint_depth=3,
                 num_heads=8, mlp_ratio=2.0, attn_dropout=0.1, proj_dropout=0.1, ffn_activation='swiglu',
                 contrastive_dim=384, decoder_depth=2, norm_type='rms', region_based_sampling=True,
                 num_regions=3, sampler_type='adaptive', recon_weight=1.0, pcc_weight=0.5,
                 sample_weight=1e-3, contrast_weight=0.05, contrast_temp=1.0, contrastive_type='soft',
                 bio_salience_method='scale_aware_ranking', bio_salience_beta=1.5, non_negative_output=True):
        super().__init__()
        self.non_negative_output = non_negative_output
        self.model = CAMMST(
            input_dim=input_dim, embed_dim=embed_dim, num_genes=num_genes, visible_ratio=visible_ratio,
            joint_depth=joint_depth, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_dropout=attn_dropout,
            proj_dropout=proj_dropout, ffn_activation=ffn_activation, contrastive_dim=contrastive_dim,
            decoder_depth=decoder_depth, norm_type=norm_type, region_based_sampling=region_based_sampling,
            num_regions=num_regions, sampler_type=sampler_type,
        )
        self.num_genes = num_genes
        self.loss_kwargs = dict(
            recon_weight=recon_weight, pcc_weight=pcc_weight, sample_weight=sample_weight,
            contrast_weight=contrast_weight, contrast_temp=contrast_temp, contrastive_type=contrastive_type,
            bio_salience_method=bio_salience_method, bio_salience_beta=bio_salience_beta,
        )

    def _nonneg(self, pred):
        return F.softplus(pred) if self.non_negative_output else pred

    def forward(self, img_emb, coord, label=None, bio_salience_score=None, **kwargs):
        phase = kwargs.get('phase', 'train')
        features = img_emb.unsqueeze(0)  # (1, N, D): one slide per step
        coords = coord.float().unsqueeze(0)  # (1, N, 2)

        if phase == 'train':
            outputs = self.model(
                features, label.unsqueeze(0), coords,
                bio_salience_score=None if bio_salience_score is None else bio_salience_score.unsqueeze(0),
            )
            outputs['pred_expressions'] = self._nonneg(outputs['pred_expressions'])  # the losses see the softplus output
            losses = self.model.compute_loss(outputs, **self.loss_kwargs)
            # logits are the masked spots only (M, G), so they are not comparable with the full label
            return {'loss': losses['total_loss'], 'logits': outputs['pred_expressions']}

        # image-only inference; the label (if any) is used for the loss below, never as network input
        zeros = torch.zeros(1, features.shape[1], self.num_genes, device=features.device, dtype=features.dtype)
        outputs = self.model.infer_with_all_mask(features, zeros, coords)
        pred = self._nonneg(outputs['pred_expressions'].clone())  # (N, G); leaves inference mode for downstream metrics
        if label is None:
            return {'logits': pred}
        losses = self.model.compute_loss(
            {**outputs, 'pred_expressions': pred, 'gt_masked': label},
            recon_weight=self.loss_kwargs['recon_weight'], pcc_weight=self.loss_kwargs['pcc_weight'],
            sample_weight=0.0, contrast_weight=0.0,
        )
        return {'loss': losses['total_loss'], 'logits': pred}

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frozen_uni import FrozenUNI
from .module import HGNN, EXPNN, TWOFusionEncoder, Decoder


class NH2ST(nn.Module):
    """NH2ST (Qu et al., MICCAI 2025, https://github.com/mcpathology/nh2st) with a frozen UNI2-h image encoder.

    The reference encodes the target patch (and its 9 nearest spots' patches) with a ResNet18 trained
    end-to-end and flattens the 7x7x512 feature map. Here the target patch goes through frozen UNI2-h
    *online* (so train-time flip/rotate augmentation is kept) and its CLS embedding (1536-d, the benchmark's
    `uni_v2` embedding) replaces the flattened map: `fc: 1536 -> emb_dim`. The 9 neighbors' image features are
    the cached `uni_v2` embeddings of those spots (encoding 9 more patches per sample online is not feasible).

    The prediction is `decoder(fc(UNI(image)))` -- the neighbor hypergraph branch, the expression encoders, the
    cross-attention and both contrastive losses only shape the training (they never feed the prediction, as in the
    reference). At evaluation only the image is used, so held-out expression cannot leak.

    The unused heads of the reference (`fc_target`, `fc_neighbor`, `fc_nuclei`, `fc_global`) are not ported.
    """

    def __init__(self, num_genes=250, emb_dim=512, depth1=1, num_heads1=8, mlp_ratio1=1.0, dropout1=0.5,
                 temperature1=0.05, temperature2=0.05, loss_ratio1=1.0, loss_ratio2=0.5, hyperedge_k=3,
                 max_batch_size=256, uni_weights_path=None, non_negative_output=True):
        super().__init__()
        self.non_negative_output = non_negative_output
        self.emb_dim = emb_dim
        self.num_genes = num_genes
        self.ratio1, self.ratio2 = loss_ratio1, loss_ratio2
        self.temperature1, self.temperature2 = temperature1, temperature2
        self.hyperedge_k = hyperedge_k
        self.max_batch_size = max_batch_size

        self.uni = FrozenUNI(uni_weights_path)  # frozen, outside the module tree (not in checkpoints)
        self.fc = nn.Linear(1536, emb_dim)
        self.decoder = Decoder(input_dim=emb_dim, output_dim=num_genes)

        self.exp_encoder = nn.Sequential(
            nn.Linear(num_genes, emb_dim),
            nn.Linear(emb_dim, emb_dim)
        )
        self.neighbor_encoder = HGNN(1536, 1024, emb_dim)
        self.neighbor_exp_encoder = EXPNN(emb_dim, 1024, emb_dim)
        self.cross_encoder = TWOFusionEncoder(emb_dim, depth1, num_heads1, int(emb_dim * mlp_ratio1), dropout1)

    # ------------------------------------------------------------------ training-only branch
    @staticmethod
    def contrastive_loss(features1, features2, temperature, negative_weight=0.1):
        if features1.dim() == 1:
            features1 = features1.unsqueeze(0)
        if features2.dim() == 1:
            features2 = features2.unsqueeze(0)
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        similarity_matrix = torch.mm(features1, features2.t()) / temperature
        batch_size = features1.size(0)
        mask = torch.eye(batch_size, device=features1.device)
        similarity_matrix = similarity_matrix * mask + similarity_matrix * (1 - mask) * negative_weight
        labels = torch.arange(batch_size, device=features1.device)
        return F.cross_entropy(similarity_matrix, labels)

    def build_hyperedges(self, nbr_emb):
        """Hypergraph of every sample's K neighbors, as one disjoint graph over the whole batch.

        Per sample (as `NGHist2ST.build_hypergraph`): node i owns a hyperedge made of its `hyperedge_k` most
        cosine-similar *other* nodes plus itself. Returns hyperedge_index over B*K nodes / B*K hyperedges.
        """
        B, K, _ = nbr_emb.shape
        feats = F.normalize(nbr_emb.float(), dim=-1)
        sim = torch.bmm(feats, feats.transpose(1, 2))
        sim = sim.masked_fill(torch.eye(K, dtype=torch.bool, device=sim.device), -1)
        topk = sim.topk(min(self.hyperedge_k, K - 1), dim=-1).indices  # (B, K, k)
        ids = torch.arange(K, device=sim.device).view(1, K, 1)
        nodes = torch.cat([topk, ids.expand(B, K, 1)], dim=-1)  # (B, K, k+1)
        edges = ids.expand(B, K, nodes.shape[-1])
        offset = (torch.arange(B, device=sim.device) * K).view(B, 1, 1)
        return torch.stack([(nodes + offset).reshape(-1), (edges + offset).reshape(-1)])

    @staticmethod
    def _hgnn_mean(net, x_nodes, hyperedge_index, B, K):
        # HGNN / EXPNN forward without its per-graph mean: the batch is one disjoint graph, mean per sample
        h = net.layer2(net.layer1(x_nodes, hyperedge_index), hyperedge_index)
        return h.view(B, K, -1).mean(1)

    def _train_step(self, img, label, nbr_emb, nbr_exp):
        if label is None or nbr_emb is None or nbr_exp is None:
            raise ValueError("NH2ST training needs label, nbr_emb and nbr_exp")
        B, K, _ = nbr_emb.shape

        patch_fusion = self.fc(self.uni.encode(img)[1])
        patch_exp = self.exp_encoder(label)
        pred_exp = self._decode(patch_fusion)

        hyperedge_index = self.build_hyperedges(nbr_emb)
        neighbors = self._hgnn_mean(self.neighbor_encoder, nbr_emb.reshape(B * K, -1).float(), hyperedge_index, B, K)
        neighbor_exps = self._hgnn_mean(
            self.neighbor_exp_encoder, self.exp_encoder(nbr_exp.reshape(B * K, -1)), hyperedge_index, B, K)

        patch_fusion = self.cross_encoder(patch_exp, patch_fusion)
        patch_exp = self.cross_encoder(patch_fusion, patch_exp)
        neighbors = self.cross_encoder(neighbor_exps, neighbors)
        neighbor_exps = self.cross_encoder(neighbors, neighbor_exps)

        loss_patch = self.contrastive_loss(patch_fusion, patch_exp, self.temperature1)
        loss_neighbor = self.contrastive_loss(neighbors, neighbor_exps, self.temperature2)
        loss = self.ratio1 * loss_patch + self.ratio2 * loss_neighbor + F.mse_loss(pred_exp, label)
        return {'loss': loss, 'logits': pred_exp}

    # ------------------------------------------------------------------ inference: image only
    def _decode(self, features):
        out = self.decoder(features)
        return F.softplus(out) if self.non_negative_output else out

    def _predict(self, img):
        return self._decode(self.fc(self.uni.encode(img)[1]))

    def forward(self, img, label=None, nbr_emb=None, nbr_exp=None, **kwargs):
        if kwargs.get('phase', 'test') == 'train':
            return self._train_step(img, label, nbr_emb, nbr_exp)

        # spots are independent, so slide-level batches are chunked (frozen UNI2-h on every chunk)
        logits = torch.cat([self._predict(c) for c in img.split(self.max_batch_size, dim=0)], dim=0)
        if label is not None:
            return {'loss': F.mse_loss(logits, label), 'logits': logits}
        return {'logits': logits}

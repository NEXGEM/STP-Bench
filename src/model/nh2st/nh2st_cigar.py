import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .nh2st import NH2ST
from .module import HGNN, EXPNN, TWOFusionEncoder, Decoder


def load_cigar_resnet18(weights_path=None):
    """ResNet18 initialised from the cigar (`tenpercent_resnet18`) weights, without avgpool/fc: (B,3,224,224) -> (B,512,7,7)."""
    weights_path = weights_path or os.path.join('weights', 'cigar', 'tenpercent_resnet18.ckpt')
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"cigar ResNet18 weights not found: {weights_path}")
    model = torchvision.models.resnet18(weights=None)
    state = torch.load(weights_path, map_location='cpu', weights_only=False)['state_dict']
    state = {k.replace('model.', '').replace('resnet.', ''): v for k, v in state.items()}
    own = model.state_dict()
    own.update({k: v for k, v in state.items() if k in own})
    model.load_state_dict(own)
    return nn.Sequential(*list(model.children())[:-2])


class NH2STCigar(NH2ST):
    """NH2ST as in the paper: ResNet18 (cigar weights as init) trained end-to-end on the target patch, its 7x7x512
    feature map flattened -> fc: 25088 -> emb_dim (instead of the benchmark NH2ST's frozen UNI2-h). The neighbors'
    image features are the cached `cigar` embeddings (512-d; the benchmark cannot encode 9 more patches per sample
    online). Added next to NH2ST, which is unchanged."""

    def __init__(self, num_genes=250, emb_dim=512, depth1=1, num_heads1=8, mlp_ratio1=1.0, dropout1=0.5,
                 temperature1=0.05, temperature2=0.05, loss_ratio1=1.0, loss_ratio2=0.5, hyperedge_k=3,
                 max_batch_size=256, cigar_weights_path=None, non_negative_output=True, img_dim=512, feat_map=49, **unused):
        nn.Module.__init__(self)   # NH2ST.__init__ would load UNI2-h
        self.non_negative_output = non_negative_output
        self.emb_dim, self.num_genes = emb_dim, num_genes
        self.ratio1, self.ratio2 = loss_ratio1, loss_ratio2
        self.temperature1, self.temperature2 = temperature1, temperature2
        self.hyperedge_k = hyperedge_k
        self.max_batch_size = max_batch_size

        self.encoder = load_cigar_resnet18(cigar_weights_path)   # trainable
        self.fc = nn.Linear(512 * feat_map, emb_dim)
        self.decoder = Decoder(input_dim=emb_dim, output_dim=num_genes)
        self.exp_encoder = nn.Sequential(nn.Linear(num_genes, emb_dim), nn.Linear(emb_dim, emb_dim))
        self.neighbor_encoder = HGNN(img_dim, 1024, emb_dim)
        self.neighbor_exp_encoder = EXPNN(emb_dim, 1024, emb_dim)
        self.cross_encoder = TWOFusionEncoder(emb_dim, depth1, num_heads1, int(emb_dim * mlp_ratio1), dropout1)

    def _train_step(self, img, label, nbr_emb, nbr_exp):
        if label is None or nbr_emb is None or nbr_exp is None:
            raise ValueError("NH2ST training needs label, nbr_emb and nbr_exp")
        B, K, _ = nbr_emb.shape

        patch_fusion = self.fc(self.encoder(img).flatten(1))
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

    def _predict(self, img):
        return self._decode(self.fc(self.encoder(img).flatten(1)))

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frozen_uni import FrozenUNI
from .module import GlobalEncoder, NeighborEncoder, FusionEncoder

warnings.filterwarnings("ignore", category=FutureWarning)


class TRIPLEXv2(nn.Module):
    """TRIPLEX whose target encoder is a frozen UNI2-h instead of an end-to-end ResNet18.

    Identical to `TRIPLEX.TRIPLEX` (neighbor / global / fusion encoders, supervised + distillation
    losses, non-negative output) except for the target branch: the 224x224 target patch goes through
    frozen UNI2-h *online* (under `no_grad`, so training augmentation -- random flips / 90-degree
    rotations -- is preserved) and its 256 patch tokens are projected `1536 -> emb_dim` by a
    trainable linear layer (the ResNet's 49 tokens x 512 were projected `512 -> emb_dim`).
    The frozen encoder's weights are not part of the state dict (see `FrozenUNI`).
    """

    def __init__(self,
                 num_genes=250,
                 emb_dim=512,
                 depth1=2,
                 depth2=2,
                 depth3=2,
                 num_heads1=8,
                 num_heads2=8,
                 num_heads3=8,
                 mlp_ratio1=2.0,
                 mlp_ratio2=2.0,
                 mlp_ratio3=2.0,
                 dropout1=0.1,
                 dropout2=0.1,
                 dropout3=0.1,
                 kernel_size=3,
                 res_neighbor=(5, 5),
                 pos_layer='APEG',
                 max_batch_size=256,
                 non_negative_output: bool = True,
                 uni_weights_path=None):
        """
        Args are those of `TRIPLEX.TRIPLEX`, plus:
            max_batch_size (int): spots per chunk at inference; 256 (not TRIPLEX's 1024) because every
                chunk now also runs the 680M-parameter UNI2-h.
            uni_weights_path (str): local UNI2-h `pytorch_model.bin`; default downloads/reads
                `MahmoodLab/UNI2-h` from the HuggingFace cache.
        """
        super().__init__()

        if emb_dim > 1536:
            self.global_mapping = nn.Linear(emb_dim, 1536)
            self.neighbor_mapping = nn.Linear(emb_dim, 1536)
            emb_dim = 1536

        self.alpha = 0.3
        self.emb_dim = emb_dim
        self.max_batch_size = max_batch_size
        self.non_negative_output = non_negative_output

        # Target Encoder: frozen UNI2-h (outside the module tree) + trainable projection
        self.uni = FrozenUNI(uni_weights_path)
        self.target_linear = nn.Linear(1536, emb_dim)
        self.fc_target = nn.Linear(emb_dim, num_genes)

        # Neighbor Encoder
        self.neighbor_encoder = NeighborEncoder(emb_dim,
                                                depth3,
                                                num_heads3,
                                                int(emb_dim * mlp_ratio3),
                                                dropout=dropout3,
                                                resolution=res_neighbor)
        self.fc_neighbor = nn.Linear(emb_dim, num_genes)

        # Global Encoder
        self.global_encoder = GlobalEncoder(emb_dim,
                                            depth2,
                                            num_heads2,
                                            int(emb_dim * mlp_ratio2),
                                            dropout2,
                                            kernel_size,
                                            pos_layer)
        self.fc_global = nn.Linear(emb_dim, num_genes)

        # Fusion Layer
        self.fusion_encoder = FusionEncoder(emb_dim,
                                            depth1,
                                            num_heads1,
                                            int(emb_dim * mlp_ratio1),
                                            dropout1)

        self.fc = nn.Linear(emb_dim, num_genes)

    def forward(self,
                img,
                mask,
                neighbor_emb,
                position=None,
                global_emb=None,
                pid=None,
                sid=None,
                **kwargs):

        phase = kwargs.get('phase', 'test')
        if phase == 'train':
            if 'dataset' not in kwargs:
                raise ValueError('Please provide dataset for training phase')
            return self._process_training_batch(img, mask, neighbor_emb, pid, sid, kwargs['dataset'], kwargs['label'])
        return self._process_inference_batch(img, mask, neighbor_emb, position, global_emb, sid)

    def _process_training_batch(self, img, mask, neighbor_emb, pid, sid, dataset, label):
        global_emb, position = self.retrieve_global_emb(pid, dataset)

        if getattr(self, 'global_mapping', None) is not None:
            neighbor_emb = self.neighbor_mapping(neighbor_emb)
            for k in global_emb.keys():
                global_emb[k] = self.global_mapping(global_emb[k])

        target_token = self.encode_target(img)
        neighbor_token = self.neighbor_encoder(neighbor_emb, mask)
        global_token = self.encode_global(global_emb, position, pid, sid)
        fusion_token = self.fusion_encoder(target_token, neighbor_token, global_token, mask=mask)

        return self._get_outputs(fusion_token, target_token, neighbor_token, global_token, label)

    def _process_inference_batch(self, img, mask, neighbor_emb, position, global_emb, sid=None):
        if getattr(self, 'global_mapping', None) is not None:
            neighbor_emb = self.neighbor_mapping(neighbor_emb)
            global_emb = self.global_mapping(global_emb)

        # The global token of every spot depends on the whole slide only, so it is computed once and
        # indexed per chunk (TRIPLEX recomputes it for every chunk; the values are the same).
        global_token = self.encode_global(global_emb, position)  # (N, D)
        index = torch.arange(img.shape[0], device=img.device) if sid is None else sid

        preds = []
        for im, nb, mk, idx in zip(img.split(self.max_batch_size, dim=0),
                                   neighbor_emb.split(self.max_batch_size, dim=0),
                                   mask.split(self.max_batch_size, dim=0),
                                   index.split(self.max_batch_size, dim=0)):
            target_token = self.encode_target(im)
            neighbor_token = self.neighbor_encoder(nb, mk)
            fusion_token = self.fusion_encoder(target_token, neighbor_token, global_token[idx], mask=mk)
            preds.append(self.fc(fusion_token))

        pred = torch.cat(preds, dim=0)
        if self.non_negative_output:
            pred = F.softplus(pred)
        return {'logits': pred}

    def encode_target(self, img):
        return self.target_linear(self.uni(img))  # B x 256 x emb_dim

    def encode_global(self, global_emb, position, pid=None, sid=None):
        if isinstance(global_emb, dict):
            global_token = torch.zeros((sid.shape[0], self.emb_dim)).to(sid.device)
            for _id, x_g in global_emb.items():
                batch_idx = pid == _id
                pos = position[_id]
                g_token = self.global_encoder(x_g, pos).squeeze(0)  # N x D
                global_token[batch_idx] = g_token[sid[batch_idx]]  # B x D
            return global_token
        return self.global_encoder(global_emb, position).squeeze(0)  # N x D

    def _get_outputs(self, fusion_token, target_token, neighbor_token, global_token, label):
        if self.non_negative_output:
            output = F.softplus(self.fc(fusion_token))
            out_target = F.softplus(self.fc_target(target_token.mean(1)))
            out_neighbor = F.softplus(self.fc_neighbor(neighbor_token.mean(1)))
            out_global = F.softplus(self.fc_global(global_token))
        else:
            output = self.fc(fusion_token)
            out_target = self.fc_target(target_token.mean(1))
            out_neighbor = self.fc_neighbor(neighbor_token.mean(1))
            out_global = self.fc_global(global_token)

        preds = (output, out_target, out_neighbor, out_global)
        loss = self.calculate_loss(preds, label)
        return {'loss': loss, 'logits': output}

    def calculate_loss(self, preds, label):
        loss = F.mse_loss(preds[0], label)                       # Supervised loss for Fusion
        for i in range(1, len(preds)):
            loss += F.mse_loss(preds[i], label) * (1 - self.alpha)  # Supervised loss
            loss += F.mse_loss(preds[0], preds[i]) * self.alpha     # Distillation loss
        return loss

    def retrieve_global_emb(self, pid, dataset):
        device = pid.device
        unique_pid = pid.unique()

        global_emb = {}
        pos = {}
        for pid in unique_pid:
            pid = int(pid)
            _id = dataset.int2id[pid]

            global_emb[pid] = dataset.global_embs[_id].clone().to(device).unsqueeze(0)
            pos[pid] = dataset.pos_dict[_id].clone().to(device)

        return global_emb, pos

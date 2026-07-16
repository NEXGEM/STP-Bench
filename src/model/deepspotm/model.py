import os
import sys

import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "DeepSpotM", "src"))
from deepspotm import DeepSpotM


class DeepSpotMModule(nn.Module):
    """Wraps the pretrained DeepSpotM foundation model (ratschlab/DeepSpotM)
    for zero-shot inference, following the OmiCLIP/STPath pattern: no
    training, weights loaded once from a HuggingFace repo or local
    directory in __init__.
    """

    def __init__(self, repo_id_or_path='ratschlab/DeepSpotM', source=None, device='cpu', gene_path=None):
        super(DeepSpotMModule, self).__init__()
        self.model, self.image_processor = DeepSpotM.from_pretrained(
            repo_id_or_path, source=source, device=device,
        )

    def forward(self, img, label=None, **kwargs):
        device = kwargs.get('device', 'cuda')
        dataset = kwargs.get('dataset', None)
        if dataset is None:
            raise ValueError("Please provide dataset in kwargs to resolve target gene indices.")

        gene_idx = self.model.genes_to_indices(dataset.genes).to(img.device)
        expression, _, _ = self.model(img, gene_indices=gene_idx)
        output = expression.to(device)

        if label is not None:
            loss = F.mse_loss(output, label)
            return {'loss': loss, 'logits': output}
        else:
            return {'logits': output}

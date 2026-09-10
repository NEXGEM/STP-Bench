import torch
import torch.nn as nn
import torch.nn.functional as F

# pip-installed package (see requirements/models/DeepSpotM.txt), not vendored --
# `uv pip install -r requirements/models/DeepSpotM.txt` before using this model.
from deepspotm import DeepSpotM


class DeepSpotMModule(nn.Module):
    """Wraps the pretrained DeepSpotM foundation model (ratschlab/DeepSpotM)
    for zero-shot inference, following the OmiCLIP/STPath pattern: no
    training, weights loaded once from a HuggingFace repo or local
    directory in __init__.
    """

    def __init__(self, repo_id_or_path='ratschlab/DeepSpotM', source=None, device='cpu', gene_path=None,
                 max_batch_size=32):
        super(DeepSpotMModule, self).__init__()
        self.model, self.image_processor = DeepSpotM.from_pretrained(
            repo_id_or_path, source=source, device=device,
        )
        self.max_batch_size = max_batch_size

    def forward(self, img, label=None, **kwargs):
        device = kwargs.get('device', 'cuda')
        dataset = kwargs.get('dataset', None)
        if dataset is None:
            raise ValueError("Please provide dataset in kwargs to resolve target gene indices.")

        gene_idx = self.model.genes_to_indices(dataset.genes).to(img.device)

        # Eval/predict batches are slide-level (every patch of a slide in one
        # batch), which can be hundreds of tiles — split into chunks so the
        # giant Midnight ViT backbone doesn't OOM on a single forward pass.
        if img.shape[0] > self.max_batch_size:
            chunks = img.split(self.max_batch_size, dim=0)
            expression = torch.cat(
                [self.model(chunk, gene_indices=gene_idx)[0] for chunk in chunks], dim=0,
            )
        else:
            expression, _, _ = self.model(img, gene_indices=gene_idx)
        output = expression.to(device)

        if label is not None:
            loss = F.mse_loss(output, label)
            return {'loss': loss, 'logits': output}
        else:
            return {'logits': output}

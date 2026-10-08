import torch

from .default import DefaultAdapter


class GeneRAGAdapter(DefaultAdapter):
    """Default batch contract for GeneRAG's two stages.

    Stage 1 (train / val): the linear decoder is trained on the anchor genes and validated on the
    anchor genes only, so the checkpoint is chosen by the decoder's own quality. No retrieval runs.
    Stage 2 (test / predict): the decoder is frozen and the retrieval against the reference bank
    reconstructs the full gene panel.

    The bank (training spots' expression + embeddings) is identical for every slide of a run, so it is
    built once by the dataset and installed on the model on the first test/predict step rather than
    shipped in every batch.
    """

    _STAGE_LOADERS = {'test': 'test_dataloaders', 'predict': 'predict_dataloaders'}

    def val_num_outputs(self, config):
        return config.MODEL.num_anchor_genes

    def _eval_dataset(self, module):
        stage = module.trainer.state.stage
        attr = self._STAGE_LOADERS.get(getattr(stage, 'value', str(stage)), 'val_dataloaders')
        return getattr(module.trainer, attr).dataset

    def get_label(self, module, batch, outputs, stage):
        label = batch["label"]
        if stage == 'val':
            # validation labels come from the full-panel eval dataset; score the anchor genes only
            idx = torch.as_tensor(self._eval_dataset(module).anchor_idx, dtype=torch.long, device=label.device)
            label = label.index_select(-1, idx)
        return label

    def forward(self, module, batch, phase):
        if phase not in ("train", "val") and not module.model.has_bank:
            module.model.set_bank(*self._eval_dataset(module).get_bank())
        return super().forward(module, batch, phase)

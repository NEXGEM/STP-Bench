import os

import h5py


class ModelAdapter:
    """Adapter interface for model-specific LightningModule behavior."""

    squeeze_specs = ()
    flatten_specs = ()

    def squeeze_batch(self, batch):
        for key, ndim in self.squeeze_specs:
            if key in batch and hasattr(batch[key], "shape") and len(batch[key].shape) == ndim:
                batch[key] = batch[key].squeeze(0)
        for key, ndim in self.flatten_specs:
            if key in batch and hasattr(batch[key], "shape") and len(batch[key].shape) == ndim:
                batch[key] = batch[key].view(-1)
        return batch

    def prepare_batch(self, module, batch, stage):
        batch = self.squeeze_batch(batch)
        return batch

    def forward(self, module, batch, phase):
        if phase == "train":
            return module.model(**batch, phase=phase)
        if module.config.DATA.get('gene_output_indices') is not None:
            # External evaluation against a dataset that only measures a
            # subset of the training gene panel: the label here is already
            # narrowed to that subset, but most models' forward() computes
            # its own internal loss against the model's full fixed-width
            # output, which would crash on the width mismatch before
            # BaseModule ever gets a chance to slice it (see
            # BaseModule._slice_gene_outputs). Withhold label so the model
            # just returns its raw (unsliced) logits; BaseModule computes
            # loss/metrics itself after slicing.
            batch = {k: v for k, v in batch.items() if k != "label"}
        return module.model(**batch, phase=phase, device=module.device.type)

    def get_label(self, module, batch, outputs, stage):
        return batch["label"]

    def get_predict_dataset(self, module):
        return module._trainer.predict_dataloaders.dataset

    def prepare_predict_batch(self, module, batch, dataset):
        return self.squeeze_batch(batch)

    def inference_prediction_context(self, module):
        dataset = module._trainer.predict_dataloaders.dataset
        return dataset.name, dataset.genes, self._read_predict_coords(dataset)

    @staticmethod
    def _read_predict_coords(dataset):
        """Read spatial coords for the current predict-time sample from the
        same patch h5 the dataset itself reads (see STDataset.load_img),
        so predictions can carry obsm['spatial'] without a second WSI pass.
        Returns None (not an error) if the patch h5 has no 'coords'
        dataset — save_predictions treats that as "no spatial info"."""
        path = os.path.join(dataset.img_dir, f"{dataset.name}.h5")
        if not os.path.isfile(path):
            path = os.path.join(dataset.img_dir, f"{dataset.name}_patches.h5")
        if not os.path.isfile(path):
            return None
        with h5py.File(path, 'r') as f:
            if 'coords' not in f:
                return None
            return f['coords'][:]

    def evaluation_prediction_context(self, module, batch_idx):
        dataset = module._trainer.test_dataloaders.dataset
        name = dataset.int2id[batch_idx]
        genes = dataset.genes
        id2dir = getattr(dataset, "id2dir", None)
        return name, genes, id2dir

    def after_optimizer_step(self, module):
        ema = getattr(module.model, "ema", None)
        if ema is not None:
            ema.update()

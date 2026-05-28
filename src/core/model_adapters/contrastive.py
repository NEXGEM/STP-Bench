from .base import ModelAdapter


class ContrastiveAdapter(ModelAdapter):
    """Adapter for models that need dataset context during test/predict."""

    squeeze_specs = (
        ("img", 5),
        ("label", 3),
        ("img_emb", 3),
        ("coord", 3),
        ("pred", 3),
        ("adj", 3),
        ("oris", 3),
        ("sfs", 3),
    )

    def prepare_batch(self, module, batch, stage):
        batch = self.squeeze_batch(batch)
        if stage == "test":
            batch["dataset"] = module._trainer.test_dataloaders.dataset
        elif stage == "val":
            dl = module._trainer.val_dataloaders
            if isinstance(dl, list):
                dl = dl[0]
            batch["dataset"] = dl.dataset
        return batch

    def prepare_predict_batch(self, module, batch, dataset):
        batch = self.squeeze_batch(batch)
        batch["dataset"] = dataset
        return batch

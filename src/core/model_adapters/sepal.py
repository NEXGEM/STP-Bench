from .base import ModelAdapter


class SepalAdapter(ModelAdapter):
    """Adapter for Sepal dataloaders and label handling."""

    squeeze_specs = (
        ("label", 3),
        ("pred", 3),
        ("adj", 3),
        ("oris", 3),
        ("sfs", 3),
    )

    def get_label(self, module, batch, outputs, stage):
        return outputs["label"]

    def get_predict_dataset(self, module):
        return module._trainer.predict_dataloaders

    def inference_prediction_context(self, module):
        dataloader = module._trainer.predict_dataloaders
        return dataloader.name, getattr(dataloader, "genes", None)

    def evaluation_prediction_context(self, module, batch_idx):
        dataloader = module._trainer.test_dataloaders
        return dataloader.int2id[batch_idx], dataloader.genes, None

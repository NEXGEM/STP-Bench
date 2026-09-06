from .base import ModelAdapter


class TriplexAdapter(ModelAdapter):
    """Adapter for TRIPLEX-style models requiring dataset context."""

    squeeze_specs = (
        ("img", 5),
        ("img2", 5),
        ("img3", 5),
        ("label", 3),
        ("mask", 3),
        ("neighbor_emb", 4),
        ("sub_spot_emb", 4),
        ("img_emb", 3),
        ("spot_emb", 3),
        ("position", 3),
    )
    flatten_specs = (
        ("pid", 2),
        ("sid", 2),
    )

    def prepare_batch(self, module, batch, stage):
        batch = self.squeeze_batch(batch)
        if stage == "train":
            batch["dataset"] = module._trainer.train_dataloader.dataset
        return batch

    def prepare_predict_batch(self, module, batch, dataset):
        batch = self.squeeze_batch(batch)
        batch["position"] = dataset.position.clone().to(batch["img"].device)
        batch["global_emb"] = dataset.global_emb.clone().to(batch["img"].device).unsqueeze(0)
        return batch

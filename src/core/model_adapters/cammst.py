from .default import DefaultAdapter


class CammstAdapter(DefaultAdapter):
    """Default batch contract; every batch is one slide, so the DataLoader batch dim (1) of `img_emb`,
    `coord`, `label` and `bio_salience_score` is squeezed (train and test alike)."""

    squeeze_specs = DefaultAdapter.squeeze_specs + (
        ("coord", 3),
        ("bio_salience_score", 2),
    )

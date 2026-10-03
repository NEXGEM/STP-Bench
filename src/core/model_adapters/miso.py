from .default import DefaultAdapter


class MisoAdapter(DefaultAdapter):
    """Default batch contract; every batch is one slide, so the DataLoader batch dim (1) of `img_emb`,
    `label` and `neighbors` is squeezed (train and test alike)."""

    squeeze_specs = DefaultAdapter.squeeze_specs + (
        ("neighbors", 3),
    )

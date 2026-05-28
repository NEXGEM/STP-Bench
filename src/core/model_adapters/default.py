from .base import ModelAdapter


class DefaultAdapter(ModelAdapter):
    """Default adapter for models following the standard **batch contract."""

    squeeze_specs = (
        ("img", 5),
        ("label", 3),
        ("img_emb", 3),
    )

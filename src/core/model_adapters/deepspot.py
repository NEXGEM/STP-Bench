from .base import ModelAdapter


class DeepSpotAdapter(ModelAdapter):
    """Adapter for DeepSpot's spot/sub_spot/neighbor embedding batch contract."""

    squeeze_specs = (
        ("label", 3),
        ("spot_emb", 3),
        ("sub_spot_emb", 4),
        ("neighbor_emb", 4),
    )

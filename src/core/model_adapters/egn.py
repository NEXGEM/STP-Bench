from .base import ModelAdapter


class EGNAdapter(ModelAdapter):
    """Adapter for EGN/EGGN: squeezes the DataLoader batch dim for slide-level ei/ej/yj."""

    squeeze_specs = (
        ("img", 5),
        ("label", 3),
        ("ei", 4),
        ("ej", 4),
        ("yj", 4),
    )

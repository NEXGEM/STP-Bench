from .base import ModelAdapter


class StemAdapter(ModelAdapter):
    """Adapter for Stem's explicit EMA update hook."""

    squeeze_specs = (
        ("img_emb", 3),
        ("label", 3),
    )

    def after_optimizer_step(self, module):
        module.model.update_ema()

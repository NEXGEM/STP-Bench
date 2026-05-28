from .base import ModelAdapter


class StemAdapter(ModelAdapter):
    """Adapter for Stem's explicit EMA update hook."""

    def after_optimizer_step(self, module):
        module.model.update_ema()


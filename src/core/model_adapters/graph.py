from .base import ModelAdapter


class GraphAdapter(ModelAdapter):
    """Adapter for graph models that consume the full batch object."""

    squeeze_specs = (
        ("ei", 4),
        ("ej", 4),
        ("yj", 4),
    )

    def forward(self, module, batch, phase):
        return module.model(batch, phase=phase)

    def get_label(self, module, batch, outputs, stage):
        return batch["window"].y

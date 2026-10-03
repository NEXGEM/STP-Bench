from .default import DefaultAdapter


class HisDiffAdapter(DefaultAdapter):
    """Default batch contract; whole-slide test batches also carry the kNN index tensors, whose
    DataLoader batch dim (1) must be squeezed like `img_emb` and `label`."""

    squeeze_specs = DefaultAdapter.squeeze_specs + (
        ("neighbor_idx", 3),
        ("global_idx", 3),
    )

"""Checkpoint loading utilities for DeepSpotM.

These helpers centralize the patterns the training repo previously
duplicated in workflows/common.py: filtering vocab-specific keys before
load, asserting no unexplained state-dict drift, and building a
predict-time model with optional cross-vocab support.
"""

from typing import Iterable, Optional, Sequence, Tuple

import torch

from .model import DeepSpotM


# Keys whose shape is tied to the gene vocabulary. When loading a
# cross-vocab checkpoint over a default-vocab model (or vice versa), drop
# these from the state dict before load — the caller is expected to
# install vocab-specific weights via CrossAttentionGeneDecoder.swap_vocabulary.
VOCAB_SPECIFIC_PREFIXES: Tuple[str, ...] = (
    "gene_decoder.gene_embeddings",
    "gene_decoder._router_bio_emb",
)


def _multi_source_drop_prefixes(
    all_sources: Sequence[str], kept_source: str,
) -> Tuple[str, ...]:
    """Build the state-dict prefixes to drop when collapsing a multi-source
    decoder to one source at load time.

    Drops:
      * ``bio_{kept}`` — its shape is about to change (vocab swap).
      * ``bio_{other}``, ``adapters.{other}.*``, ``routers.{other}.*`` for
        every source other than ``kept_source`` — those parameters no
        longer exist on the collapsed model.
    """
    prefixes = [f"gene_decoder.multi_source.bio_{kept_source}"]
    for other in all_sources:
        if other == kept_source:
            continue
        prefixes.extend([
            f"gene_decoder.multi_source.bio_{other}",
            f"gene_decoder.multi_source.adapters.{other}.",
            f"gene_decoder.routers.{other}.",
        ])
    return tuple(prefixes)


def filter_state_dict(state_dict: dict, drop_prefixes: Iterable[str]) -> dict:
    """Return a copy of ``state_dict`` with any key starting with one of
    ``drop_prefixes`` removed."""
    drop = tuple(drop_prefixes)
    return {k: v for k, v in state_dict.items() if not k.startswith(drop)}


def check_state_dict_load(
    missing: Sequence[str],
    unexpected: Sequence[str],
    *,
    allow_missing_prefixes: Iterable[str] = (),
    allow_unexpected_prefixes: Iterable[str] = (),
) -> None:
    """Raise if load_state_dict produced any unexplained drift.

    ``missing`` are weights the model expects but the checkpoint omitted.
    ``unexpected`` are weights in the checkpoint that the model lacks.
    Anything outside the allow lists indicates that the runtime model and
    the checkpoint disagree — predictions would silently use uninitialised
    weights, which is exactly the case we want to surface.
    """
    allow_missing = tuple(allow_missing_prefixes)
    allow_unexpected = tuple(allow_unexpected_prefixes)

    bad_missing = [k for k in missing if not k.startswith(allow_missing)]
    bad_unexpected = [k for k in unexpected if not k.startswith(allow_unexpected)]

    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "load_state_dict produced unaccounted-for key drift between "
            "checkpoint and runtime model: "
            f"{len(bad_missing)} missing (e.g. {bad_missing[:3]}), "
            f"{len(bad_unexpected)} unexpected (e.g. {bad_unexpected[:3]}). "
            "Usually means the training-time and runtime architectures "
            "disagree (e.g. hparam drift, missing layer)."
        )


def load_for_prediction(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    token_path: Optional[str] = None,
    gene_embeddings_path: Optional[str] = None,
    source: Optional[str] = None,
) -> Tuple[DeepSpotM, callable]:
    """Build a DeepSpotM from a checkpoint for inference.

    Returns ``(model, image_processor)``. Model is moved to ``device``,
    set to eval, and the backbone's transforms switched to center-crop.

    For cross-vocab checkpoints (training-time vocab != target vocab) pass
    both ``token_path`` and ``gene_embeddings_path``. We build the model
    with the default vocab so ``CrossAttentionGeneDecoder``'s init-time
    asserts pass, swap the gene decoder to the target vocab, then load the
    cross-vocab weights — the freshly-swapped vocab-specific layers stay
    intact because their keys were filtered out of the state dict.

    ``source`` is required when the checkpoint's hparams declare a
    multi-source decoder with more than one source (``init_gene_embeddings``
    a list with >1 entry). The checkpoint is collapsed to the named source
    at load time; per-source parameters for the other sources are dropped
    from the state dict before loading. For an already-collapsed checkpoint
    (single-element ``init_gene_embeddings`` list) the source is inferred,
    so passing it is optional.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cross_vocab = bool(token_path and gene_embeddings_path)

    if cross_vocab:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hparams = dict(ckpt.get("hyper_parameters", {}))
        hparams["token_path"] = None  # build with default vocab

        ckpt_sources = hparams.get("init_gene_embeddings")
        is_multi = isinstance(ckpt_sources, (list, tuple))
        if is_multi:
            if source is None:
                if len(ckpt_sources) == 1:
                    source = ckpt_sources[0]
                else:
                    raise ValueError(
                        "Loading a multi-source checkpoint with sources "
                        f"{list(ckpt_sources)} for cross-vocab prediction "
                        "requires the ``source`` argument to select which "
                        "pathway to keep."
                    )
            if source not in ckpt_sources:
                raise ValueError(
                    f"Unknown source {source!r}. Available in checkpoint: "
                    f"{list(ckpt_sources)}"
                )

        model = DeepSpotM(**hparams)
        model.swap_gene_vocabulary(
            gene_embeddings_path, token_path, source=source if is_multi else None,
        )

        if is_multi:
            drop_prefixes = _multi_source_drop_prefixes(ckpt_sources, source)
        else:
            drop_prefixes = VOCAB_SPECIFIC_PREFIXES
        state_dict = filter_state_dict(ckpt["state_dict"], drop_prefixes)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        check_state_dict_load(
            missing, unexpected,
            allow_missing_prefixes=drop_prefixes,
        )
    else:
        model = DeepSpotM.load_from_checkpoint(
            checkpoint_path, map_location="cpu", strict=False,
        )
        # Multi-source predict without a vocab swap: select the active
        # pathway. After STRS-style collapse the saved hparams already
        # name the one remaining source, so ``set_source`` is a no-op
        # consistency check; for an uncollapsed multi checkpoint the
        # caller must pass ``source`` explicitly.
        if model.sources is not None:
            if source is None:
                if len(model.sources) == 1:
                    source = model.sources[0]
                else:
                    raise ValueError(
                        f"Multi-source checkpoint with sources {model.sources} "
                        "requires ``source`` to be set before prediction."
                    )
            model.set_source(source)

    model.to(device).eval()
    model.set_eval_transforms(center_crop=True)
    return model, model.backbone.transforms

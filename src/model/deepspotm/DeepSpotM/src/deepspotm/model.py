from typing import List, Optional

import torch
import torch.nn as nn
import lightning as pl
from peft import get_peft_model, LoraConfig

from .image_encoder import build_encoder
from .modules import (
    StructureExpression,
    CrossAttentionGeneDecoder,
)


class DeepSpotM(pl.LightningModule):
    """
    Predicts spatial gene expression from histology images.

    All architectural choices are exposed as hyperparameters so that every
    component can be toggled on/off for ablation studies.

    --- LoRA save / load contract ---
    When ``use_lora=True``, the backbone is wrapped by PEFT (``PeftModel`` →
    ``LoraModel`` → ``ImageEncoder``). State-dict keys carry the
    ``backbone.base_model.model.`` prefix and the adapters live alongside.

    The contract is to keep LoRA *wrapped* at predict time:

    * ``forward`` / ``predict_genes``: wrapped.
    * ``deepspotm.load_for_prediction``: wrapped.

    Do not call ``model.backbone.merge_and_unload()`` in the
    inference path: doing so silently breaks the load-side round-trip
    because a freshly constructed ``DeepSpotM(use_lora=True)`` expects
    the prefixed keys.
    """

    def __init__(
        self,
        encoder_model: str = "MidnightEncoder",
        # --- Cross-attention gene decoder ---
        use_cross_attention: bool = True,
        gene_embed_dim: int = 256,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 2,
        cross_attn_dropout: float = 0.1,
        init_gene_embeddings = "xavier",          # "xavier" | str key in _GENE_EMB_LOADERS | list[str] for multi-source
        # --- Gene router (hypernetwork for zero-shot) ---
        use_router: bool = True,
        router_hidden_dim: int = None,
        # --- Multi-source modality encoding (only used when init_gene_embeddings is a list) ---
        use_modality_encoding: bool = True,
        # --- LoRA fine-tuning ---
        use_lora: bool = True,
        r: int = 8,
        # --- Custom gene vocabulary ---
        token_path: str = None,
        # --- Self-contained checkpoint loading ---
        # backbone_pretrained=False builds the vision backbone architecture
        # without downloading its upstream weights (they are loaded from the
        # checkpoint instead). bio_dims maps each multi-source name to its
        # bio_dim so the gene decoder can allocate parameters by shape without
        # reading the raw gene-embedding .pt assets. Both are set by
        # from_pretrained; defaults preserve training-time behaviour.
        backbone_pretrained: bool = True,
        bio_dims: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs"])

        # ------ Expression vocabulary ------
        expression_info = StructureExpression(token_path=token_path)
        # Ordered gene names: output column i of forward() corresponds to
        # gene_names[i]. Stored so callers can map the prediction vector back
        # to gene symbols without re-parsing tokens.csv.
        self.gene_names = list(expression_info.gene_names_ordered)
        self.n_genes = len(self.gene_names)
        self._gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}

        # ------ Vision backbone ------
        # PEFT's get_peft_model freezes the underlying base model automatically
        # and marks only the LoRA adapter parameters trainable. When LoRA is
        # disabled we freeze the backbone explicitly here.
        self.backbone = build_encoder(encoder_model, pretrained=backbone_pretrained)
        if use_lora:
            lora_config = LoraConfig(
                r=r,
                lora_alpha=r * 2,
                lora_dropout=0.2,
                target_modules=self.backbone.LORA_TARGET_MODULES,
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        else:
            self.freeze_backbone()

        # ------ Gene prediction head ------
        if use_cross_attention:
            self.gene_decoder = CrossAttentionGeneDecoder(
                n_genes=self.n_genes,
                patch_dim=self.backbone.patch_dim,
                gene_embed_dim=gene_embed_dim,
                num_heads=cross_attn_heads,
                num_layers=cross_attn_layers,
                dropout=cross_attn_dropout,
                init_gene_embeddings=init_gene_embeddings,
                use_router=use_router,
                router_hidden_dim=router_hidden_dim,
                use_modality_encoding=use_modality_encoding,
                bio_dims=bio_dims,
            )
        else:
            self.gene_head = nn.Sequential(
                nn.Linear(self.backbone.hidden_dim, gene_embed_dim),
                nn.GELU(),
                nn.Linear(gene_embed_dim, self.n_genes),
            )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def swap_gene_vocabulary(
        self,
        embeddings_path: str,
        token_path: str,
        use_router: bool = None,
        router_hidden_dim: int = None,
        source: str = None,
    ) -> list:
        """Retarget the gene decoder to a custom vocabulary in-place.

        Loads embeddings from ``embeddings_path``, validates the row count
        matches the ``token_path`` gene list, and rebuilds the vocab-specific
        layers (``gene_embeddings``, ``gene_query_proj``, and either the
        router's ``_router_bio_emb`` + MLPs or the shared output head).
        Returns the new gene name list.

        Args:
            embeddings_path: path to a (n_genes, bio_dim) .pt tensor.
            token_path: token CSV with ``token_type == 'gene'`` rows.
            use_router: optional override for the output head — keep the
                current router/no-router setting if None, ``True`` rebuilds
                the router for the new bio_dim, ``False`` replaces it with a
                shared linear head.
            router_hidden_dim: optional override for the router MLP hidden dim;
                ignored when use_router resolves to False.
            source: required when the model is multi-source. Names the
                source pathway to keep; the model is collapsed to that
                single source and its bio param is swapped to the new
                vocabulary's embeddings. ``hparams['init_gene_embeddings']``
                is updated to ``[source]`` so a future ``load_from_checkpoint``
                rebuilds the same collapsed structure.
        """
        import pandas as pd
        from .modules import CrossAttentionGeneDecoder

        if not isinstance(self.gene_decoder, CrossAttentionGeneDecoder):
            raise RuntimeError(
                "swap_gene_vocabulary requires use_cross_attention=True; this "
                "checkpoint uses the MLP head and has no vocab-specific layers."
            )

        tokens = pd.read_csv(token_path)
        gene_list = tokens.query("token_type == 'gene'").token.values.tolist()
        bio_emb = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        if bio_emb.shape[0] != len(gene_list):
            raise ValueError(
                f"Gene embedding rows ({bio_emb.shape[0]}) must match token "
                f"list length ({len(gene_list)}) in {token_path}."
            )
        self.gene_decoder.swap_vocabulary(
            bio_emb,
            use_router=use_router,
            router_hidden_dim=router_hidden_dim,
            source=source,
        )
        self.n_genes = len(gene_list)
        if use_router is not None:
            self.hparams["use_router"] = use_router
        # Reflect a multi-source collapse in the hyperparameters so a saved
        # checkpoint round-trips correctly: load_from_checkpoint will rebuild
        # MultiSourceGeneEmbedding with just this one source.
        if source is not None and isinstance(
            self.hparams.get("init_gene_embeddings"), (list, tuple)
        ):
            self.hparams["init_gene_embeddings"] = [source]
        return gene_list

    # ------------------------------------------------------------------ #
    #  Multi-source dispatch                                              #
    # ------------------------------------------------------------------ #
    def set_source(self, name: str) -> None:
        """Select the active gene-embedding source for multi-source models.

        Only valid when the model was constructed with ``init_gene_embeddings``
        as a list of source names. Must be called before each forward in
        multi-source mode — the selection is not persisted in ``state_dict``
        so freshly loaded checkpoints start with no active source.
        """
        decoder = getattr(self, "gene_decoder", None)
        multi = getattr(decoder, "multi_source", None) if decoder is not None else None
        if multi is None:
            raise RuntimeError(
                "set_source requires a multi-source decoder. Build the model "
                "with init_gene_embeddings=[<source>, ...]."
            )
        multi.set_active(name)

    @classmethod
    def load_for_inference(cls, checkpoint_path, *, source=None, **kwargs):
        """Inference-optimized loader that skips unused multi-source banks.

        Multi-source ckpts hold one frozen gene-embedding bank and one adapter
        per source (evo2/orthrus/prott5/scgpt/apertus, ~750 MB on disk in
        total for the five default sources, plus their adapter weights). At
        inference only one source is active; building and loading the other
        four is pure waste.

        Pass ``source=<name>`` to:
          1. Override ``init_gene_embeddings`` to ``[source]`` so the model's
             ``MultiSourceGeneEmbedding`` is built with only that source's
             bank and adapter (the registry only opens that one ``.pt`` file).
          2. Load the ckpt's state_dict with ``strict=False`` so the unused
             sources' tensors are silently dropped.
          3. Call :meth:`set_source` so the model is ready for forward.

        For single-source ckpts, pass ``source=None`` (or omit) — this
        behaves identically to :meth:`load_from_checkpoint`. Passing
        ``source`` to a single-source ckpt raises ``RuntimeError`` after
        load (the single ``bio_emb`` buffer doesn't match the per-source
        ``bio_<name>`` layout, so ``strict=False`` would silently lose the
        actual bio weights — we surface that explicitly).
        """
        if source is None:
            return cls.load_from_checkpoint(checkpoint_path, **kwargs)
        kwargs.setdefault("strict", False)
        kwargs["init_gene_embeddings"] = [source]
        model = cls.load_from_checkpoint(checkpoint_path, **kwargs)
        if getattr(model, "sources", None) is None:
            raise RuntimeError(
                f"load_for_inference(source={source!r}) called on a "
                f"single-source ckpt. Use load_from_checkpoint() instead."
            )
        model.set_source(source)
        return model

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        *,
        source: Optional[str] = None,
        device=None,
        revision: Optional[str] = None,
    ):
        """Load a published DeepSpotM model from a Hugging Face repo or a local
        directory.

        Expects the directory/repo to contain ``config.json`` (saved
        hyper-parameters, including ``bio_dims``), ``model.safetensors`` (the
        baked-in weights — vision backbone + LoRA adapters + gene decoder),
        and ``tokens.csv`` (the gene vocabulary).

        The vision backbone is built offline (``backbone_pretrained=False``)
        and its weights come from the checkpoint, so loading needs no network
        access to the upstream backbone repo.

        Returns ``(model, image_processor)``: the model on ``device`` in eval
        mode with center-crop eval transforms, and the backbone's image
        transform callable.
        """
        import json
        import os
        from safetensors.torch import load_file

        if os.path.isdir(repo_id_or_path):
            cfg_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")
            token_path = os.path.join(repo_id_or_path, "tokens.csv")
        else:
            from huggingface_hub import hf_hub_download

            def _dl(fn):
                return hf_hub_download(repo_id_or_path, fn, revision=revision)

            cfg_path = _dl("config.json")
            weights_path = _dl("model.safetensors")
            token_path = _dl("tokens.csv")

        with open(cfg_path) as f:
            hparams = json.load(f)
        hparams.pop("_deepspotm_format_version", None)
        hparams["token_path"] = token_path
        hparams["backbone_pretrained"] = False

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = cls(**hparams)
        state_dict = load_file(weights_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # LoRA buffers/PEFT bookkeeping may be absent from a clean export; only
        # surface drift in the substantive weight namespaces.
        bad_missing = [
            k for k in missing
            if k.startswith(("gene_decoder.", "backbone.")) and "lora" not in k.lower()
        ]
        if bad_missing:
            raise RuntimeError(
                "from_pretrained: checkpoint is missing expected weights "
                f"(e.g. {bad_missing[:5]}). The config and weights disagree."
            )

        if model.sources is not None:
            if source is None:
                if len(model.sources) == 1:
                    source = model.sources[0]
                else:
                    raise ValueError(
                        f"Multi-source model with sources {model.sources} "
                        "requires `source=` to select the active pathway."
                    )
            model.set_source(source)

        model.to(device).eval()
        model.set_eval_transforms(center_crop=True)
        return model, model.backbone.transforms

    @property
    def sources(self) -> Optional[List[str]]:
        """Registered source names if multi-source, else ``None``."""
        decoder = getattr(self, "gene_decoder", None)
        multi = getattr(decoder, "multi_source", None) if decoder is not None else None
        return list(multi.source_names) if multi is not None else None

    @property
    def current_source(self) -> Optional[str]:
        """Currently active source name, or ``None`` if unset / single-source.

        Read-side symmetry with :meth:`set_source`. Returns ``None`` both for
        single-source models (no concept of "current") and for freshly loaded
        multi-source checkpoints before :meth:`set_source` has been called.
        """
        decoder = getattr(self, "gene_decoder", None)
        multi = getattr(decoder, "multi_source", None) if decoder is not None else None
        return multi.active_source if multi is not None else None

    def set_eval_transforms(self, **build_kwargs) -> None:
        """Switch the backbone to evaluation transforms.

        Reaches through PEFT wrappers if LoRA is enabled. Use at predict time
        with e.g. ``center_crop=True``.
        """
        backbone = self.backbone
        # PEFT wraps the backbone twice (PeftModel -> LoraModel -> encoder).
        # __getattr__ delegation usually works, but we be explicit so a future
        # PEFT release that breaks the chain won't silently regress to no-op.
        for attr in ("base_model", "model"):
            if hasattr(backbone, attr) and not hasattr(backbone, "set_transforms"):
                backbone = getattr(backbone, attr)
        backbone.set_transforms(**build_kwargs)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(self, pixel_values, gene_indices=None):
        """Predict expression from image tiles.

        ``gene_indices`` (optional 1-D LongTensor / list of ints) restricts the
        prediction to a subset of genes. With the cross-attention decoder this
        also reduces compute, since only the selected gene queries are built and
        attended. The output columns follow ``gene_indices`` order.
        """
        if self.hparams.use_cross_attention:
            patch_tokens = self.backbone.forward_patch_tokens(pixel_values)
            need_weights = False
            expression, pooled, attn_weights = self.gene_decoder(
                patch_tokens, need_weights=need_weights, gene_indices=gene_indices,
            )
        else:
            pooled = self.backbone(pixel_values)
            expression = self.gene_head(pooled)
            if gene_indices is not None:
                idx = torch.as_tensor(gene_indices, dtype=torch.long, device=expression.device)
                expression = expression.index_select(1, idx)
            attn_weights = None
        return expression, pooled, attn_weights

    def genes_to_indices(self, genes) -> "torch.Tensor":
        """Map a gene symbol or list of symbols to a LongTensor of column
        indices. Raises KeyError with the offending names if any are not in the
        model's panel (``model.gene_names``)."""
        names = [genes] if isinstance(genes, str) else list(genes)
        missing = [g for g in names if g not in self._gene_to_idx]
        if missing:
            raise KeyError(
                f"{len(missing)} gene(s) not in the model panel: {missing[:10]}"
                f"{' ...' if len(missing) > 10 else ''}"
            )
        return torch.tensor([self._gene_to_idx[g] for g in names], dtype=torch.long)

    @torch.no_grad()
    def predict_genes(self, pixel_values, genes):
        """Predict expression for specific gene(s) only.

        Args:
            pixel_values: (B, 3, H, W) preprocessed image tensor.
            genes: a gene symbol (str) or list of symbols. Only these genes are
                computed, so this is faster than predicting the full panel.
        Returns:
            (B, len(genes)) tensor of predicted expression, columns ordered as
            ``genes``.
        """
        idx = self.genes_to_indices(genes).to(pixel_values.device)
        expression, _, _ = self.forward(pixel_values, gene_indices=idx)
        return expression

    # ------------------------------------------------------------------ #
    #  Interpretability                                                    #
    # ------------------------------------------------------------------ #
    def get_gene_attention_maps(self, pixel_values, gene_indices=None):
        """Return per-gene attention over spatial patches.

        Args:
            pixel_values: (B, C, H, W) input images.
            gene_indices: optional list/tensor of gene indices to return.

        Returns:
            Tensor of shape (B, G_selected, N_patches) with attention weights
            averaged over heads from the last cross-attention layer.
        """
        if not self.hparams.use_cross_attention:
            raise ValueError("Cross-attention is disabled — no attention maps available.")

        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                patch_tokens = self.backbone.forward_patch_tokens(pixel_values)
                _, _, attn_weights_all = self.gene_decoder(patch_tokens, need_weights=True)
        finally:
            if was_training:
                self.train()

        # Last layer: (B, H, G, N_patches) → average over heads
        attn = attn_weights_all[-1].mean(dim=1)  # (B, G, N_patches)
        if gene_indices is not None:
            attn = attn[:, gene_indices, :]
        return attn

import importlib.resources
from functools import partial
from typing import Mapping, Optional, Sequence

import pandas as pd
import torch
import torch.nn as nn
from .config import config


class StructureExpression:
    """Maps a per-sample {gene: value} dict into a fixed-length tensor.

    Expression values are expected to be pre-normalized at the shard level
    (e.g. log1p(x / total * 1e4) applied upstream during shard creation), so
    this class does no normalization — it just performs the gene-name lookup
    and tracks which genes were measured.
    """

    def __init__(self, token_path: str = None):
        df = pd.read_csv(token_path) if token_path is not None else pd.read_csv(config.ALPHABET_PATH)
        self.gene_names_ordered = df.query("token_type == 'gene'").token.values

    def structure_expression(self, expression: dict):
        gene_mask = torch.tensor(
            [g in expression for g in self.gene_names_ordered], dtype=torch.bool,
        )
        expression_tensor = torch.tensor(
            [expression.get(g, 0.0) for g in self.gene_names_ordered], dtype=torch.float32,
        )
        expression_tensor.clamp_(min=0)
        return expression_tensor, gene_mask


# -----------------------
# Cross-Attention Gene Decoder
# -----------------------
class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, gene_queries, patch_kv, need_weights=False):
        attn_out, attn_weights = self.cross_attn(
            gene_queries, patch_kv, patch_kv,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        gene_queries = self.norm1(gene_queries + attn_out)
        gene_queries = self.norm2(gene_queries + self.ffn(gene_queries))
        return gene_queries, attn_weights


# ─────────────────────────────────────────────────────────────────────────
# Modality vocabulary
# ─────────────────────────────────────────────────────────────────────────
# Canonical ordering — DON'T reorder; the index is what the frozen one-hot
# encoding indexes into. New modalities should be APPENDED at the end so
# existing checkpoints remain valid.
MODALITY_VOCABULARY: tuple = ("dna", "rna", "protein", "single_cell", "text")

# Mapping from gene-embedding source name to the modality it represents.
# Multiple sources can share the same modality (e.g. evo2 + a future
# hyenaDNA would both map to "dna"); they'd then share the same modality
# token and the cross-attention could leverage modality-level invariances.
# To add a new source, register both its loader (in _GENE_EMB_LOADERS below)
# and its modality here.
MODALITY_BY_SOURCE: dict = {
    "evo2": "dna",
    "orthrus": "rna",
    "prott5": "protein",
    "esm2": "protein",
    "scgpt": "single_cell",
    "geneformer": "single_cell",
    "geneformer_cancer": "single_cell",
    "apertus": "text",
    "genept_ada": "text",
    "genept_m3": "text",
    # Other registered sources kept in _GENE_EMB_LOADERS without an entry
    # here can only be used with use_modality_encoding=False.
}


def _load_gene_embeddings(filename: str):
    """Load pre-computed gene embeddings from assets."""
    try:
        with importlib.resources.path("deepspotm.assets", filename) as p:
            return torch.load(p, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Gene embedding file '{filename}' not found in deepspotm.assets. "
            f"Run the corresponding notebook in notebooks/ to generate it."
        )


# -----------------------
# Gene Router (Hypernetwork)
# -----------------------
class GeneRouter(nn.Module):
    """Hypernetwork that generates per-gene output projections from biological
    embeddings, enabling zero-shot prediction for unseen genes.

    Instead of a fixed shared ``nn.Linear`` or a per-gene weight lookup, the
    router produces gene-specific projection weights on the fly from any
    gene's biological embedding (scGPT, Geneformer, ESM-2, etc.).
    """

    def __init__(self, gene_bio_dim: int, gene_embed_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or gene_embed_dim * 2
        # Generates the per-gene projection vector
        self.weight_net = nn.Sequential(
            nn.Linear(gene_bio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, gene_embed_dim),
        )
        # Generates the per-gene bias scalar
        self.bias_net = nn.Sequential(
            nn.Linear(gene_bio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gene_bio_embeddings: torch.Tensor, gene_queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_bio_embeddings: (G, bio_dim) — raw biological embeddings
            gene_queries: (B, G, D) — cross-attention refined queries
        Returns:
            expression: (B, G)
        """
        proj = self.weight_net(gene_bio_embeddings)              # (G, D)
        bias = self.bias_net(gene_bio_embeddings).squeeze(-1)    # (G,)
        return torch.einsum('bgd,gd->bg', gene_queries, proj) + bias


class MultiSourceGeneEmbedding(nn.Module):
    """Per-source biological gene-token embedding + per-source projection adapter.

    Each source has a **trainable** ``(n_genes, bio_dim_i)`` gene-token
    parameter, registered as ``bio_{source}``, initialized from the loaded
    biological prior (scGPT / ProtT5 / evo2 / etc.) and free to drift during
    training. This matches the capacity of the single-source design where
    ``gene_embeddings`` is a trainable ``nn.Embedding`` initialized directly
    from the bio prior (raw values, no rescaling) — preserving that
    capacity is what gives the model per-gene learnable degrees of freedom.

    Each source also has a trainable adapter (``Linear(bio_dim_i,
    gene_embed_dim) → LayerNorm``) that projects the prior into the
    cross-attention's query space. The LayerNorm normalizes per gene so
    cross-attention sees comparably-scaled queries regardless of source.

    On top, an optional **frozen modality encoding** (one-hot over a small
    modality vocabulary — DNA, RNA, protein, single-cell, text — registered
    as a buffer) is added to every gene query before cross-attention. This
    gives the shared cross-attention an explicit, stable signal about which
    *modality* of prior produced the current queries. The encoding is
    keyed by *modality*, not by source name, so two sources sharing the
    same modality (e.g. evo2 and a future hyenaDNA) would share their
    modality token — useful for generalization. The encoding is frozen
    on purpose: modality identity is categorical and has no value in
    being "learned" beyond its discrete signature.

    Dispatches on a runtime ``active_source`` string set via
    :meth:`set_active`. ``active_source`` is a plain Python attribute —
    *not* part of the state dict — so it does not survive a checkpoint
    round-trip. Set it explicitly before each forward pass (typically once
    per training step via :class:`MultiSourceRoundRobinCallback`, once per
    inference call).
    """

    def __init__(
        self,
        sources: Mapping[str, torch.Tensor],
        gene_embed_dim: int,
        use_modality_encoding: bool = True,
    ):
        """
        Args:
            sources: mapping {name: (n_genes, bio_dim_i) tensor}. All tensors
                must share the same first-dim ``n_genes``.
            gene_embed_dim: output dim of each adapter — must equal the
                surrounding decoder's ``gene_embed_dim``.
            use_modality_encoding: if True (default), add a frozen one-hot
                modality vector to every gene query before returning. Each
                source must appear in :data:`MODALITY_BY_SOURCE`; otherwise
                ``ValueError`` at construction. Set False to ablate the
                modality signal cleanly without removing the buffer
                infrastructure.
        """
        super().__init__()
        if not sources:
            raise ValueError("MultiSourceGeneEmbedding requires at least one source.")
        n_genes_set = {t.shape[0] for t in sources.values()}
        if len(n_genes_set) > 1:
            raise ValueError(
                f"All source embeddings must share n_genes; got {n_genes_set}"
            )
        self.gene_embed_dim = gene_embed_dim
        self.source_names = list(sources.keys())
        self.n_genes = next(iter(n_genes_set))

        # bio_dims is derived metadata, not state — adapter input shapes carry
        # the same information and are checked by strict state_dict load.
        self.bio_dims = {}
        for name, emb in sources.items():
            # Cast to float32 only (matches adapter Linear dtype; defensive
            # against future fp16/bf16 assets). Initialize the trainable
            # parameter directly from the raw bio prior — no rescaling —
            # mirroring the single-source ``gene_embeddings.weight.data.copy_``
            # path. AdamW's per-element second-moment estimate handles the
            # cross-source scale differences without an explicit normalization.
            t = emb.detach().clone().to(torch.float32)
            self.register_parameter(f"bio_{name}", nn.Parameter(t))
            self.bio_dims[name] = int(emb.shape[1])

        self.adapters = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.bio_dims[name], gene_embed_dim),
                nn.LayerNorm(gene_embed_dim),
            )
            for name in self.source_names
        })

        # ── Modality encoding (frozen, optional) ──
        self.use_modality_encoding = use_modality_encoding
        self.source_to_modality_idx: dict = {}
        if use_modality_encoding:
            self._init_modality_encoding()

        # Plain Python attribute — intentionally NOT a buffer/parameter so it
        # doesn't end up in state_dict. Caller must set before forward().
        self.active_source: Optional[str] = None

    def _init_modality_encoding(self) -> None:
        """Build the frozen modality buffer + per-source lookup.

        Layout: ``modality_emb`` is shape (n_modalities, gene_embed_dim) with
        ones on the diagonal of the first n_modalities columns and zeros
        elsewhere — i.e. canonical basis vectors e_0 … e_{n-1} of D-space.
        Per-source mapping translates ``active_source`` → modality row index.
        """
        n_mod = len(MODALITY_VOCABULARY)
        if self.gene_embed_dim < n_mod:
            raise ValueError(
                f"gene_embed_dim ({self.gene_embed_dim}) must be >= "
                f"{n_mod} (number of modalities) to fit a one-hot modality "
                f"encoding in the first n_modalities dims."
            )
        for name in self.source_names:
            if name not in MODALITY_BY_SOURCE:
                raise ValueError(
                    f"Source {name!r} has no modality mapping in "
                    f"MODALITY_BY_SOURCE. Either register it there "
                    f"(picking one of {list(MODALITY_VOCABULARY)}) or pass "
                    f"use_modality_encoding=False."
                )
        # Identity-on-first-n_mod-dims encoding. Registered as a buffer
        # because modality identity is a fixed categorical signal — there's
        # nothing meaningful for the model to "learn" about it, only how to
        # USE it via cross-attention.
        table = torch.zeros(n_mod, self.gene_embed_dim)
        for i in range(n_mod):
            table[i, i] = 1.0
        self.register_buffer("modality_emb", table)
        self.source_to_modality_idx = {
            name: MODALITY_VOCABULARY.index(MODALITY_BY_SOURCE[name])
            for name in self.source_names
        }

    @classmethod
    def from_registry(
        cls,
        names: Sequence[str],
        n_genes: int,
        gene_embed_dim: int,
        use_modality_encoding: bool = True,
    ) -> "MultiSourceGeneEmbedding":
        """Build by loading each named source from ``_GENE_EMB_LOADERS``."""
        sources = {}
        for name in names:
            if name not in _GENE_EMB_LOADERS:
                raise ValueError(
                    f"Unknown gene embedding source: {name!r}. "
                    f"Available: {list(_GENE_EMB_LOADERS.keys())}"
                )
            emb = _GENE_EMB_LOADERS[name]()
            if emb.shape[0] != n_genes:
                raise ValueError(
                    f"{name} embeddings have {emb.shape[0]} genes, expected {n_genes}"
                )
            sources[name] = emb
        return cls(sources, gene_embed_dim, use_modality_encoding=use_modality_encoding)

    @classmethod
    def from_shapes(
        cls,
        bio_dims: dict,
        n_genes: int,
        gene_embed_dim: int,
        use_modality_encoding: bool = True,
    ) -> "MultiSourceGeneEmbedding":
        """Build with zero-initialised bio parameters of the given shapes.

        Used when loading a checkpoint that already carries the trained
        ``bio_{source}`` tensors: the raw gene-embedding ``.pt`` assets are not
        needed, only the per-source ``bio_dim`` (from ``config.json``). The
        zeros are overwritten by ``load_state_dict``.
        """
        sources = {
            name: torch.zeros(n_genes, int(dim)) for name, dim in bio_dims.items()
        }
        return cls(sources, gene_embed_dim, use_modality_encoding=use_modality_encoding)

    def set_active(self, name: str) -> None:
        if name not in self.source_names:
            raise ValueError(
                f"Unknown source {name!r}. Available: {self.source_names}"
            )
        self.active_source = name

    def collapse_to_source(self, name: str, new_bio_emb: torch.Tensor) -> None:
        """Drop all sources except ``name`` and swap its bio parameter.

        Used at fine-tune time to specialize the multi-source decoder to one
        pathway (e.g. orthrus for STRS cross-species). The kept source's
        adapter (and the surrounding decoder's per-source router) keep their
        trained weights — only ``bio_{name}`` is replaced, and only if
        ``new_bio_emb`` has a different shape than the existing parameter.

        ``new_bio_emb`` must have the same ``bio_dim`` as the existing source
        (the adapter's input dim is fixed). ``n_genes`` may change — this is
        the cross-species use case: orthrus (human, ~20k) → mouse-orthrus
        (74,815) at 512-d both.

        After collapse, ``source_names == [name]`` and ``active_source ==
        name``. The frozen modality buffer is kept as-is; only the
        per-source modality-index mapping is pruned.
        """
        if name not in self.source_names:
            raise ValueError(
                f"Unknown source {name!r}. Available: {self.source_names}"
            )
        expected_bio_dim = self.bio_dims[name]
        if new_bio_emb.shape[1] != expected_bio_dim:
            raise ValueError(
                f"new_bio_emb bio_dim ({new_bio_emb.shape[1]}) must match "
                f"the existing adapter input dim for {name!r} "
                f"({expected_bio_dim})."
            )

        # Drop other sources' bio Parameters and adapters.
        for src in list(self.source_names):
            if src != name:
                del self._parameters[f"bio_{src}"]
        for src in list(self.adapters.keys()):
            if src != name:
                del self.adapters[src]

        # Replace the kept source's bio param. Re-register because n_genes
        # may change — a plain ``.data.copy_`` would refuse a shape change.
        del self._parameters[f"bio_{name}"]
        t = new_bio_emb.detach().clone().to(torch.float32)
        self.register_parameter(f"bio_{name}", nn.Parameter(t))
        self.n_genes = int(new_bio_emb.shape[0])

        # Update metadata.
        self.source_names = [name]
        self.bio_dims = {name: expected_bio_dim}
        if self.use_modality_encoding:
            self.source_to_modality_idx = {
                name: self.source_to_modality_idx[name]
            }
        self.active_source = name

    def adapter_parameters(self):
        """Iterator over adapter params — used to split weight-decay groups."""
        return self.adapters.parameters()

    @property
    def current_bio(self) -> torch.Tensor:
        """Bio gene-token tensor for the active source (trainable Parameter).

        Used by the per-source router (when the surrounding decoder is built
        with ``use_router=True`` and multi-source mode) to pull the matching
        gene-token embedding — same source identity as the active adapter,
        so the invariant "same embedding feeds adapter + router" extends
        per-source. Raises if no source has been selected.

        The returned tensor is a registered ``nn.Parameter`` (not a buffer);
        backward through any consumer accumulates gradient on it.
        """
        if self.active_source is None:
            raise RuntimeError(
                "MultiSourceGeneEmbedding.active_source is None — call "
                "set_active(name) before reading current_bio."
            )
        return getattr(self, f"bio_{self.active_source}")

    def forward(self, gene_indices=None) -> torch.Tensor:
        """Return the per-gene query template for the active source.

        Output shape: ``(n_genes, gene_embed_dim)`` (or ``(len(gene_indices),
        gene_embed_dim)`` when ``gene_indices`` is given to restrict to a gene
        subset). The surrounding decoder is responsible for expanding to the
        batch dimension — this mirrors the single-source path where
        ``gene_embeddings(gene_ids)`` returns ``(G, bio_dim)`` and
        ``.expand(B, -1, -1)`` happens in the decoder.

        Doing the adapter projection here (before the expand) is also more
        efficient than the single-source order (project after expand): cost
        is ``G * bio_dim * D`` instead of ``B * G * bio_dim * D``.

        When ``use_modality_encoding`` is True, the active source's frozen
        modality vector is added to every gene query before returning —
        giving the cross-attention an explicit signal about which modality
        is being asked from.
        """
        if self.active_source is None:
            raise RuntimeError(
                "MultiSourceGeneEmbedding.active_source is None — call "
                "set_active(name) before forward."
            )
        bio = getattr(self, f"bio_{self.active_source}")    # (G, bio_dim_i)
        if gene_indices is not None:
            bio = bio[gene_indices]                         # (G_sel, bio_dim_i)
        queries = self.adapters[self.active_source](bio)    # (G, gene_embed_dim)
        if self.use_modality_encoding:
            modality_idx = self.source_to_modality_idx[self.active_source]
            queries = queries + self.modality_emb[modality_idx]
        return queries

    def extra_repr(self) -> str:
        modality_summary = (
            {n: MODALITY_BY_SOURCE.get(n) for n in self.source_names}
            if self.use_modality_encoding else "disabled"
        )
        return (
            f"sources={self.source_names}, "
            f"bio_dims={self.bio_dims}, "
            f"gene_embed_dim={self.gene_embed_dim}, "
            f"modalities={modality_summary}, "
            f"active_source={self.active_source!r}"
        )


_GENE_EMB_LOADERS = {
    "scgpt": partial(_load_gene_embeddings, "scgpt_gene_embeddings.pt"),
    "geneformer": partial(_load_gene_embeddings, "geneformer_gene_embeddings.pt"),
    "geneformer_cancer": partial(_load_gene_embeddings, "geneformer_cancer_gene_embeddings.pt"),
    "string_network": partial(_load_gene_embeddings, "string_network_gene_embeddings.pt"),
    "prott5": partial(_load_gene_embeddings, "prott5_gene_embeddings.pt"),
    "genept_ada": partial(_load_gene_embeddings, "genept_ada_gene_embeddings.pt"),
    "genept_m3": partial(_load_gene_embeddings, "genept_m3_gene_embeddings.pt"),
    "esm2": partial(_load_gene_embeddings, "esm2_gene_embeddings.pt"),
    "apertus": partial(_load_gene_embeddings, "apertus_gene_embeddings.pt"),
    "evo2": partial(_load_gene_embeddings, "evo2_7b_gene_embeddings.pt"),
    "orthrus": partial(_load_gene_embeddings, "orthrus_gene_embeddings.pt"),
}


class CrossAttentionGeneDecoder(nn.Module):
    def __init__(self, n_genes, patch_dim, gene_embed_dim=256,
                 num_heads=8, num_layers=2, dropout=0.1,
                 init_gene_embeddings="xavier",
                 use_router: bool = False,
                 router_hidden_dim: int = None,
                 use_modality_encoding: bool = True,
                 bio_dims: dict = None):
        """
        Args:
            init_gene_embeddings: one of:

                * ``"xavier"`` — random init.
                * A single source name from ``_GENE_EMB_LOADERS``
                  (e.g. ``"scgpt"``, ``"prott5"``) — load that embedding and
                  use it both to initialise gene queries and (if enabled) to
                  feed the router.
                * A **list of source names** — build a
                  :class:`MultiSourceGeneEmbedding` that holds one frozen
                  embedding + one trainable adapter per source, and dispatch
                  at forward-time on a runtime ``active_source`` selector
                  (set via the surrounding model's ``set_source`` API).
                  Combine with ``use_router=True`` to also build one
                  :class:`GeneRouter` per source — each router reads its
                  own source's frozen bio buffer, preserving the
                  "same embedding feeds adapter + router" invariant on a
                  per-source basis.
            use_router: if True, replace the shared output projection with a
                ``GeneRouter`` hypernetwork that derives per-gene projections
                from the same biological embeddings used for query init —
                enabling zero-shot gene prediction. In multi-source mode this
                becomes one router per source (an ``nn.ModuleDict`` at
                ``self.routers``).
            router_hidden_dim: hidden size inside the router MLPs (default:
                ``gene_embed_dim * 2``).
            use_modality_encoding: only used when ``init_gene_embeddings`` is
                a list (multi-source). Forwarded to
                :class:`MultiSourceGeneEmbedding`; controls whether a frozen
                one-hot modality vector is added to gene queries. Ignored in
                single-source mode.
        """
        super().__init__()
        self.use_router = use_router
        self.patch_proj = nn.Linear(patch_dim, gene_embed_dim)
        self.layers = nn.ModuleList([
            CrossAttentionLayer(gene_embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # ── Multi-source branch ──────────────────────────────────────────
        # init_gene_embeddings as a list/tuple means "build the multi-source
        # gene-query stack instead of a single Embedding". The remaining
        # architectural pieces (cross-attention, patch_proj) are identical to
        # the single-source path. The output head is either a shared linear
        # (use_router=False) or an nn.ModuleDict of per-source routers.
        self.multi_source: Optional[MultiSourceGeneEmbedding] = None
        if isinstance(init_gene_embeddings, (list, tuple)):
            if bio_dims is not None:
                # Load-from-checkpoint path: build zero bio params by shape
                # (the trained values arrive via load_state_dict); avoids
                # reading the raw gene-embedding .pt assets.
                self.multi_source = MultiSourceGeneEmbedding.from_shapes(
                    bio_dims={n: bio_dims[n] for n in init_gene_embeddings},
                    n_genes=n_genes,
                    gene_embed_dim=gene_embed_dim,
                    use_modality_encoding=use_modality_encoding,
                )
            else:
                self.multi_source = MultiSourceGeneEmbedding.from_registry(
                    names=list(init_gene_embeddings),
                    n_genes=n_genes,
                    gene_embed_dim=gene_embed_dim,
                    use_modality_encoding=use_modality_encoding,
                )
            if use_router:
                # Per-source routers (Plan §6.2): one GeneRouter per source,
                # each reading its corresponding bio buffer via
                # multi_source.current_bio. Preserves the single-source
                # design intent — same biological embedding feeds adapter
                # (query init) and router (output hypernet) — extended to a
                # runtime-dispatched per-source variant.
                self.routers = nn.ModuleDict({
                    name: GeneRouter(
                        gene_bio_dim=self.multi_source.bio_dims[name],
                        gene_embed_dim=gene_embed_dim,
                        hidden_dim=router_hidden_dim,
                    )
                    for name in self.multi_source.source_names
                })
            else:
                self.output_proj = nn.Linear(gene_embed_dim, 1)
            return  # skip the single-source initialisation below

        # ── Single-source branch (legacy, behaviour unchanged) ───────────
        bio_emb = None
        if init_gene_embeddings != "xavier":
            if init_gene_embeddings not in _GENE_EMB_LOADERS:
                raise ValueError(
                    f"Unknown gene embedding source: {init_gene_embeddings!r}. "
                    f"Available: {['xavier'] + list(_GENE_EMB_LOADERS.keys())}"
                )
            bio_emb = _GENE_EMB_LOADERS[init_gene_embeddings]()
            assert bio_emb.shape[0] == n_genes, (
                f"{init_gene_embeddings} embeddings have {bio_emb.shape[0]} genes, "
                f"expected {n_genes}"
            )

        # --- Output head: shared linear OR router ---
        if use_router:
            if bio_emb is None:
                raise ValueError(
                    "use_router=True requires init_gene_embeddings to be set "
                    "to a biological embedding source (not 'xavier')."
                )
            self.register_buffer("_router_bio_emb", bio_emb)
            self.router = GeneRouter(bio_emb.shape[1], gene_embed_dim, router_hidden_dim)
        else:
            self.output_proj = nn.Linear(gene_embed_dim, 1)

        # --- Initialise gene query embeddings ---
        self.gene_query_proj = None
        if bio_emb is not None:
            emb_dim = bio_emb.shape[1]
            self.gene_embeddings = nn.Embedding(n_genes, emb_dim)
            self.gene_embeddings.weight.data.copy_(bio_emb)
            if emb_dim != gene_embed_dim:
                self.gene_query_proj = nn.Linear(emb_dim, gene_embed_dim, bias=False)
        else:
            self.gene_embeddings = nn.Embedding(n_genes, gene_embed_dim)
            nn.init.xavier_uniform_(self.gene_embeddings.weight)

    @property
    def gene_embed_dim(self) -> int:
        return self.patch_proj.out_features

    def swap_vocabulary(
        self,
        bio_emb: torch.Tensor,
        use_router: bool = None,
        router_hidden_dim: int = None,
        source: Optional[str] = None,
    ) -> None:
        """Replace the gene vocabulary with a new set of biological embeddings.

        Rebuilds ``gene_embeddings``, ``gene_query_proj`` (if dim changes), and
        the router (or shared output head) to match the new vocab. Use this to
        retarget a checkpoint to a different vocabulary (e.g. mouse, custom
        panel) before loading vocab-specific weights.

        Args:
            bio_emb: (n_genes_new, bio_dim_new) biological embeddings tensor.
            use_router: if not None, override the router on/off state. ``True``
                requires that bio_emb is set (which it always is here); ``False``
                replaces the router with a shared ``Linear(gene_embed_dim, 1)``.
                Defaults to the existing setting on this decoder.
            router_hidden_dim: hidden size when (re)building the router. If
                None, reuse the current router's hidden dim (or
                ``gene_embed_dim * 2`` when enabling the router for the first time).
            source: required when this decoder is multi-source. Names the
                source pathway to keep; all other sources' bio params,
                adapters, and routers are dropped. The kept source's
                adapter weights are preserved (bio_dim is unchanged). The
                ``use_router`` and ``router_hidden_dim`` arguments are
                ignored on the multi-source path — the existing per-source
                router is kept verbatim for the retained source.
        """
        if self.multi_source is not None:
            if source is None:
                raise ValueError(
                    "swap_vocabulary on a multi-source decoder requires a "
                    "``source`` argument naming which source pathway to "
                    f"keep. Available: {self.multi_source.source_names}."
                )
            # Collapse to the named source with the new vocabulary
            # embeddings. bio_dim must match the existing source's dim
            # (e.g. orthrus → mouse-orthrus, both 512-d) so the trained
            # adapter Linear and per-source router are preserved verbatim.
            self.multi_source.collapse_to_source(source, bio_emb)
            # Drop the other sources' routers if present.
            if getattr(self, "routers", None) is not None:
                for other in [n for n in list(self.routers.keys()) if n != source]:
                    del self.routers[other]
            return

        n_genes_new, bio_dim_new = bio_emb.shape
        gene_embed_dim = self.gene_embed_dim

        # Replace the gene query embedding lookup (always — vocab size changed
        # or content changed).
        new_embeddings = nn.Embedding(n_genes_new, bio_dim_new)
        new_embeddings.weight.data.copy_(bio_emb)
        self.gene_embeddings = new_embeddings

        # Rebuild gene_query_proj ONLY when its input shape must change. This
        # preserves trained weights when only the embedding contents change
        # (e.g. STRS prott5 → mouse-prott5, both 1024-d).
        needed_proj_in = None if bio_dim_new == gene_embed_dim else bio_dim_new
        current_proj_in = (
            self.gene_query_proj.in_features
            if self.gene_query_proj is not None else None
        )
        if needed_proj_in is None:
            self.gene_query_proj = None  # passthrough
        elif current_proj_in != needed_proj_in:
            self.gene_query_proj = nn.Linear(needed_proj_in, gene_embed_dim, bias=False)
        # else: keep the existing trained projection.

        # Resolve the target router state.
        if use_router is None:
            use_router = self.use_router

        if use_router:
            # The bio_emb buffer reflects the NEW vocab regardless.
            self.register_buffer("_router_bio_emb", bio_emb.clone())
            # Rebuild router only when (a) it doesn't exist (switching from
            # no-router) or (b) its input dim changes. Otherwise keep weights.
            existing_router = getattr(self, "router", None)
            existing_router_in = (
                existing_router.weight_net[0].in_features
                if existing_router is not None else None
            )
            if existing_router is None or existing_router_in != bio_dim_new:
                if router_hidden_dim is None and existing_router is not None:
                    router_hidden_dim = existing_router.weight_net[0].out_features
                self.router = GeneRouter(bio_dim_new, gene_embed_dim, router_hidden_dim)
            # Drop the shared head if we just switched from no-router.
            if hasattr(self, "output_proj"):
                del self.output_proj
        else:
            if not hasattr(self, "output_proj"):
                self.output_proj = nn.Linear(gene_embed_dim, 1)
            for stale in ("router", "_router_bio_emb"):
                if hasattr(self, stale):
                    if stale in self._buffers:
                        del self._buffers[stale]
                    else:
                        delattr(self, stale)

        self.use_router = use_router

    def forward(self, patch_tokens, need_weights=False, gene_bio_embeddings=None,
                gene_indices=None):
        """
        Args:
            patch_tokens: (B, N_patches, patch_dim)
            need_weights: return attention weights for interpretability
            gene_bio_embeddings: optional (G, bio_dim) override for zero-shot
                inference on unseen genes. If None, uses the stored buffer.
            gene_indices: optional 1-D LongTensor selecting a subset of genes to
                predict. When given, only those gene queries are built and
                cross-attended, so cost scales with len(gene_indices) instead of
                the full panel. The output is ordered to match gene_indices.
        Returns:
            expression: (B, G) — G == len(gene_indices) if subsetting
            gene_features: (B, gene_embed_dim) — mean-pooled gene queries
            attn_weights: list of attention weight tensors or None
        """
        B = patch_tokens.shape[0]
        patch_kv = self.patch_proj(patch_tokens)

        if gene_indices is not None and not torch.is_tensor(gene_indices):
            gene_indices = torch.as_tensor(gene_indices, dtype=torch.long)
        if gene_indices is not None:
            gene_indices = gene_indices.to(patch_tokens.device)

        if self.multi_source is not None:
            # Multi-source mode: the active source's adapter produces (G, D),
            # then we expand to (B, G, D) — symmetric with the single-source
            # path below.
            gene_queries = self.multi_source(gene_indices=gene_indices).unsqueeze(0).expand(B, -1, -1)
        else:
            if gene_indices is None:
                gene_ids = torch.arange(
                    self.gene_embeddings.num_embeddings, device=patch_tokens.device,
                )
            else:
                gene_ids = gene_indices
            gene_queries = self.gene_embeddings(gene_ids).unsqueeze(0).expand(B, -1, -1)
            if self.gene_query_proj is not None:
                gene_queries = self.gene_query_proj(gene_queries)

        attn_weights_all = []
        for layer in self.layers:
            gene_queries, attn_weights = layer(gene_queries, patch_kv, need_weights=need_weights)
            if need_weights and attn_weights is not None:
                attn_weights_all.append(attn_weights)

        # Output projection
        if self.use_router:
            if self.multi_source is not None:
                # Multi-source: dispatch to the active source's router.
                # ``current_bio.detach()`` mirrors single-source's
                # ``_router_bio_emb`` (which is a frozen buffer): the router
                # sees the current bio value but the gradient does NOT flow
                # back into bio through the router pathway. Bio still gets
                # gradient via the adapter pathway (the queries below).
                # Without this detach, bio would be co-optimized by three
                # consumers simultaneously — adapter, router.weight_net, and
                # router.bias_net — yielding the unstable dynamics that hurt
                # validation Pearson in the previous run.
                active = self.multi_source.active_source
                router_bio = self.multi_source.current_bio.detach()
                if gene_indices is not None:
                    router_bio = router_bio[gene_indices]
                expression = self.routers[active](router_bio, gene_queries)
            else:
                bio_emb = (
                    gene_bio_embeddings if gene_bio_embeddings is not None
                    else self._router_bio_emb
                )
                if gene_indices is not None and gene_bio_embeddings is None:
                    bio_emb = bio_emb[gene_indices]
                expression = self.router(bio_emb, gene_queries)
        else:
            expression = self.output_proj(gene_queries).squeeze(-1)  # (B, G)

        gene_features = gene_queries.mean(dim=1)  # (B, gene_embed_dim)
        return expression, gene_features, attn_weights_all if need_weights else None

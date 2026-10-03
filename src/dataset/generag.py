import os

import numpy as np
import torch
from scipy import sparse

from dataset.base_dataset import STDataset
from dataset.path_utils import emb_dir as resolve_emb_dir
from dataset.path_utils import st_dir as resolve_st_dir


class GeneRAGDataset(STDataset):
    """STDataset + GeneRAG's anchor genes and reference bank.

    * Train phase: ``label`` is narrowed to the anchor genes (the linear head
      only predicts those).
    * Val/test/predict: ``label`` stays the full `DATA` gene panel;
      ``get_bank()`` returns the retrieval bank built from the training spots.

    Anchor genes come from ``{anchor_gene_type}_{num_anchor_genes}genes.json``
    (same lookup as the panel JSON) and must be a subset of the training gene
    panel.
    """

    def __init__(self,
                 mode: str,
                 phase: str,
                 fold: int,
                 data_dir: str,
                 meta_dir: str = None,
                 ref_data_dir: str = None,
                 ref_asset_dir: str = None,
                 wsi_dir: str = None,
                 gene_type: str = 'mean',
                 num_genes: int = 1000,
                 num_outputs: int = 300,
                 normalize: bool = True,
                 cpm: bool = False,
                 smooth: bool = False,
                 data_id: str = None,
                 model_name: str = 'uni_v2',
                 load_level: str = 'patch',
                 use_emb: bool = True,
                 genes_override: list = None,
                 anchor_gene_type: str = 'hmhvg',
                 num_anchor_genes: int = 200,
                 ):
        # Internal evaluation sets ref_data_dir == data_dir; STDataset must
        # see None so it only loads the fold's own test split (as EGNDataset).
        if ref_data_dir is not None and os.path.normpath(ref_data_dir) == os.path.normpath(data_dir):
            ref_data_dir = None

        super().__init__(mode=mode,
                         phase=phase,
                         fold=fold,
                         data_dir=data_dir,
                         meta_dir=meta_dir,
                         ref_data_dir=ref_data_dir,
                         wsi_dir=wsi_dir,
                         gene_type=gene_type,
                         num_genes=num_genes,
                         num_outputs=num_outputs,
                         normalize=normalize,
                         cpm=cpm,
                         smooth=smooth,
                         data_id=data_id,
                         model_name=model_name,
                         load_level=load_level,
                         use_emb=use_emb,
                         genes_override=genes_override)

        self.fold = fold
        self.ref_data_dir = ref_data_dir
        self.ref_asset_dir = ref_asset_dir or ref_data_dir
        self.gene_type = gene_type
        self.num_genes = num_genes
        self.num_outputs = num_outputs
        self.anchor_gene_type = anchor_gene_type
        self.num_anchor_genes = num_anchor_genes
        self._bank = None

        # The model output (and the bank) always covers the *training* gene
        # panel; for external evaluation self.genes may be a narrower
        # genes_override, which BaseModule applies afterwards through
        # gene_output_indices.
        self.panel_genes = self._resolve_genes(gene_type, num_genes, num_outputs, ref_data_dir, None)
        self.anchor_idx = self._resolve_anchor_idx()

    # ---------------------------------------------------------------- anchors

    def _resolve_anchor_idx(self):
        import json
        path = self._resolve_gene_path(self.anchor_gene_type, self.num_anchor_genes, self.ref_data_dir)
        if not os.path.isfile(path):
            raise ValueError(f"{path} is not found")
        with open(path, 'r') as f:
            anchors = json.load(f)['genes']
        if self.anchor_gene_type in ['mean', 'hmhvg', 'total']:
            anchors = anchors[:self.num_anchor_genes]
        if len(anchors) != self.num_anchor_genes:
            raise ValueError(f"{path} has {len(anchors)} genes, expected num_anchor_genes={self.num_anchor_genes}")
        pos = {g: i for i, g in enumerate(self.panel_genes)}
        missing = [g for g in anchors if g not in pos]
        if missing:
            raise ValueError(
                f"{len(missing)} anchor genes are not in the {self.gene_type}_{self.num_genes} "
                f"gene panel (first few: {missing[:5]}); anchors must be a subset of the panel."
            )
        return np.array([pos[g] for g in anchors], dtype=np.int64)

    # -------------------------------------------------------------------- data

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if self.phase == 'train' and 'label' in data:
            data['label'] = data['label'][torch.as_tensor(self.anchor_idx)]
        return data

    def get_bank(self):
        """(expression (N, G), embeddings (N, d), anchor_idx) from the training spots.

        Built with the dataset's own load_st/load_emb so normalization matches
        the labels exactly. For external evaluation the bank comes from the
        training run's data (ref_data_dir / ref_asset_dir), never the
        external cohort. Cached: identical for every slide.
        """
        if self._bank is not None:
            return self._bank

        if self.ref_data_dir is not None:
            ids = self._get_ids(phase='train', fold=self.fold, ids_dir=self.ref_data_dir)
            st_dir, emb_dir = resolve_st_dir(self.ref_asset_dir), resolve_emb_dir(self.ref_asset_dir)
        else:
            ids = self._get_ids(phase='train', fold=self.fold)
            st_dir, emb_dir = self.st_dir, self.emb_dir

        exprs, embs = [], []
        for _id in ids:
            adata = self.load_st(_id, self.panel_genes, st_dir=st_dir, **self.norm_param)
            if list(adata.var_names) != list(self.panel_genes):
                raise ValueError(f"{_id}: bank slide is missing panel genes; cannot build a consistent bank.")
            x = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
            e = self.load_emb(_id, emb_dir=emb_dir)
            if len(x) != len(e):
                raise ValueError(f"{_id}: {len(x)} expression rows vs {len(e)} embeddings.")
            exprs.append(torch.from_numpy(x.astype(np.float32)))
            embs.append(e.float())

        self._bank = (torch.cat(exprs, 0), torch.cat(embs, 0), self.anchor_idx)
        return self._bank

"""GeneRAG refinement of an already-predicted ST panel, for any baseline model.

The baseline's saved prediction is the anchor query of GeneRAG's Dual-Constrained Retrieval
(`model/generag/generag.py`): each spot's predicted panel ``y_anchor`` and its frozen patch
embedding ``f_img`` are matched against a reference bank of the training run's own training
spots (expression + embedding), and the solved coefficients reconstruct the panel from the bank::

    a = argmin  w||f_img - D_img a||^2 + (1-w)||y_anchor - D_anchor a||^2
                + (1-l1_ratio)*alpha/2 ||a||^2 + l1_ratio*alpha ||a||_1      (a >= 0)
    y_refined = D_full a

Expected `cfg` shape (an addict.Dict, built by `STPred._run_single_refine_action`)::

    cfg.DATA.pred_path_fold      # baseline predictions "{output_dir}/{data}/{model}/[{train_data}/]fold{k}"
    cfg.DATA.out_path_fold       # where refined predictions are written
    cfg.DATA.data_dir / meta_dir / fold    # target data (internal: test split of fold k)
    cfg.DATA.external            # True when the target data is not the training data
    cfg.DATA.eval_genes          # external only: genes the data measures (None/unset = whole panel)
    cfg.TRAIN.data_dir / meta_dir           # training data: bank = fold k's training spots
    cfg.TRAIN.gene_type / num_genes / normalize / cpm / smooth / model_name
    cfg.REFINE                   # omega, alpha, l1_ratio, positive, max_iter, tol, max_batch_size
    cfg.GENERAL.gpu_id
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from downstream.common import list_fold_samples, load_pred_adata


def _gene_pcc_mae(pred: np.ndarray, gt: np.ndarray):
    """Per-gene Pearson across spots (NaN for constant genes) and per-gene MAE."""
    p = pred - pred.mean(0, keepdims=True)
    t = gt - gt.mean(0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        pcc = (p * t).sum(0) / np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    return pcc, np.abs(pred - gt).mean(0)


def _target_samples(cfg) -> List[str]:
    """Internal: fold k's test split (`list_fold_samples`). External: every sample of the target
    data -- its own ids.csv may carry fold columns of an unrelated CV split (e.g. hest/LUAD), which
    must not filter an external evaluation of training fold k."""
    if not cfg.DATA.get("external"):
        return list_fold_samples(cfg)
    import pandas as pd

    ids = pd.read_csv(os.path.join(cfg.DATA.meta_dir, "ids.csv"))["sample_id"].astype(str)
    return [s for s in ids if os.path.isfile(os.path.join(cfg.DATA.pred_path_fold, f"{s}.h5ad"))]


def run_generag_refine(cfg) -> Dict[str, Any]:
    import pandas as pd
    import torch

    from dataset.generag import GeneRAGDataset
    from dataset.path_utils import emb_dir as resolve_emb_dir
    from dataset.path_utils import st_dir as resolve_st_dir
    from model.generag.generag import GeneRAG

    params = cfg.REFINE
    train = cfg.TRAIN
    device = torch.device(f"cuda:{cfg.GENERAL.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Bank = this fold's training spots of the training run, normalized exactly like its labels.
    # Every panel gene is an anchor: the baseline predicts the whole panel.
    bank_ds = GeneRAGDataset(
        mode="cv", phase="test", fold=cfg.DATA.fold,
        data_dir=train.data_dir, meta_dir=train.meta_dir,
        gene_type=train.gene_type, num_genes=train.num_genes, num_outputs=train.num_genes,
        normalize=train.normalize, cpm=train.cpm, smooth=train.smooth, model_name=train.model_name,
        anchor_gene_type=train.gene_type, num_anchor_genes=train.num_genes,
    )
    expression, embeddings, anchor_idx = bank_ds.get_bank()
    panel = list(bank_ds.panel_genes)

    model = GeneRAG(
        num_genes=len(panel), img_embedding_dim=embeddings.shape[1], num_anchor_genes=len(panel),
        alpha=params.alpha, l1_ratio=params.l1_ratio, embedding_ratio=params.omega,
        positive=params.positive, max_iter=params.max_iter, tol=params.tol,
        max_batch_size=params.max_batch_size,
    ).to(device)
    model.set_bank(expression, embeddings, anchor_idx)

    target_emb_dir = resolve_emb_dir(cfg.DATA.data_dir)
    target_st_dir = resolve_st_dir(cfg.DATA.data_dir)
    eval_genes = list(cfg.DATA.eval_genes) if cfg.DATA.get("eval_genes") else panel
    eval_cols = [panel.index(g) for g in eval_genes]
    os.makedirs(cfg.DATA.out_path_fold, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for sample_id in _target_samples(cfg):
        pred = load_pred_adata(cfg.DATA.pred_path_fold, sample_id)
        missing = [g for g in panel if g not in set(pred.var_names)]
        if missing:
            raise ValueError(
                f"{cfg.DATA.pred_path_fold}/{sample_id}.h5ad lacks {len(missing)} of the {len(panel)} "
                "training-panel genes GeneRAG uses as anchors (external evaluation only saves the genes "
                "the external dataset measures). Re-predict the full panel with "
                "stp.predict(data, train_data=..., folds=...) after removing those files."
            )
        y_anchor = np.asarray(pred[:, panel].X, dtype=np.float32)

        emb_path = f"{target_emb_dir}/global/features_{train.model_name}/{sample_id}.h5"
        if not os.path.isfile(emb_path):
            raise FileNotFoundError(
                f"{emb_path} not found: GeneRAG needs the target data's global {train.model_name} "
                "embeddings -- run stp.preprocess(data, models=['GeneRAG']) first."
            )
        f_img = bank_ds.load_emb(sample_id, emb_dir=target_emb_dir).float()
        if len(f_img) != len(y_anchor):
            raise ValueError(f"{sample_id}: {len(y_anchor)} predicted spots vs {len(f_img)} embeddings.")

        refined = model.retrieve(torch.from_numpy(y_anchor).to(device), f_img.to(device)).cpu().numpy()
        out = pred[:, panel].copy()
        out.X = refined
        out.write_h5ad(os.path.join(cfg.DATA.out_path_fold, f"{sample_id}.h5ad"))

        if cfg.DATA.get("evaluate_against_gt", True):
            gt = bank_ds.load_st(sample_id, eval_genes, st_dir=target_st_dir, **bank_ds.norm_param)
            gt_x = gt[:, eval_genes].X
            gt_x = gt_x.toarray() if hasattr(gt_x, "toarray") else np.asarray(gt_x)
            for name, values in ((cfg.DATA.model, y_anchor), (f"{cfg.DATA.model}+GeneRAG", refined)):
                pcc, mae = _gene_pcc_mae(values[:, eval_cols], gt_x)
                rows.append({
                    "model": name, "sample": sample_id, "n_genes": len(eval_genes),
                    "pearson": float(np.nanmean(pcc)), "mae": float(mae.mean()),
                })

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(cfg.DATA.out_path_fold, "refine_metrics.csv"), index=False)
    return {"per_sample": rows, "out_path_fold": cfg.DATA.out_path_fold}

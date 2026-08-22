"""cell2location spatial deconvolution for one sample, ported from
src/deconvolution/run_cell2location_xenium.py::run_spatial_sample in the
main STpredBench repo.

The caller (run.py/evaluate.py) is responsible for bringing `adata` to a
count-like scale via `downstream.common.to_count_scale` first (pred:
expm1_round_clip when cpm=False; GT: round/clip only) — this module casts
that float32-but-integer-valued array to strict int64 via
`reference.coerce_count_matrix` right before handing it to cell2location.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from .reference import coerce_count_matrix, import_cell2location


def run_cell2location_sample(
    adata,
    signatures: pd.DataFrame,
    sample_name: str,
    n_cells_per_location: float = 30,
    detection_alpha: float = 200,
    spatial_epochs: int = 30000,
    spatial_batch_size: int = 2048,
    spatial_posterior_samples: int = 1000,
    accelerator: str = "cpu",
) -> Tuple[Any, pd.DataFrame]:
    """Returns (adata_with_posterior, q05_cell_abundance_w_sf DataFrame)."""
    Cell2location, _ = import_cell2location()

    common_genes = [g for g in signatures.index if g in adata.var_names]
    if not common_genes:
        raise ValueError(f"No signature genes found in sample {sample_name!r}.")

    adata = adata[:, common_genes].copy()
    adata.X = coerce_count_matrix(adata.X)
    adata.obs["sample"] = sample_name
    cell_state_df = signatures.loc[common_genes].copy()

    Cell2location.setup_anndata(adata=adata, batch_key="sample")
    model = Cell2location(
        adata, cell_state_df=cell_state_df,
        N_cells_per_location=n_cells_per_location, detection_alpha=detection_alpha,
    )
    model.train(max_epochs=spatial_epochs, batch_size=spatial_batch_size, accelerator=accelerator)
    adata = model.export_posterior(
        adata, sample_kwargs={"num_samples": spatial_posterior_samples, "batch_size": spatial_batch_size},
    )

    abundance_key = "q05_cell_abundance_w_sf"
    abundance = adata.obsm.get(abundance_key)
    if abundance is not None and not hasattr(abundance, "to_csv"):
        abundance = pd.DataFrame(abundance, index=adata.obs_names)
    if abundance is not None:
        # cell2location names these columns "q05cell_abundance_w_sf_<cell_type>"
        # — strip the prefix so cell-type names are directly comparable
        # between GT- and prediction-derived abundance, same convention
        # analysis/src/figure_deconvolution.py::load_data() uses in the
        # main repo.
        abundance = abundance.rename(columns=lambda c: c.replace("q05cell_abundance_w_sf_", ""))
    return adata, abundance

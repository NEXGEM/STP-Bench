import pandas as pd
import tangram as tg
import numpy as np
import torch
import anndata
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors



def generate_feature_ad(ad_expr, feature_path, sc=False):
    """
    Generates an AnnData object with OmiCLIP text or image embeddings.

    :param ad_expr: AnnData object containing metadata for the dataset.
    :param feature_path: Path to the CSV file containing the features to be loaded.
    :param sc: Boolean flag indicating whether to copy single-cell metadata or ST metadata. Default is False (ST).
    :return: A new AnnData object with the loaded features and relevant metadata from ad_expr.
    """
    
    # Load features from the CSV file. The index should match the cells/spots in ad_expr.obs.index.
    features = pd.read_csv(feature_path, index_col=0)[ad_expr.obs.index]
    
    # Create a new AnnData object with the features, transposing them to have cells/spots as rows
    feature_ad = anndata.AnnData(features[ad_expr.obs.index].T)
    
    # Copy relevant metadata from ad_expr based on the sc flag
    if sc:
        # If the data is single-cell (sc), copy the metadata from ad_expr.obs
        feature_ad.obs = ad_expr.obs.copy()
    else:
        # If the data is spatial, copy the 'cell_num', 'spatial' info, and spatial coordinates
        feature_ad.obs['cell_num'] = ad_expr.obs['cell_num'].copy()
        feature_ad.uns['spatial'] = ad_expr.uns['spatial'].copy()
        feature_ad.obsm['spatial'] = ad_expr.obsm['spatial'].copy()

    return feature_ad



def normalize_percentile(df, cols, min_percentile=5, max_percentile=95):
    """
    Clips and normalizes the specified columns of a DataFrame based on percentile thresholds,
    transforming their values to the [0, 1] range.

    :param df: A pandas DataFrame containing the columns to normalize.
    :type df: pandas.DataFrame
    :param cols: A list of column names in `df` that should be normalized.
    :type cols: list[str]
    :param min_percentile: The lower percentile used for clipping (defaults to 5).
    :type min_percentile: float
    :param max_percentile: The upper percentile used for clipping (defaults to 95).
    :type max_percentile: float
    :return: The same DataFrame with specified columns clipped and normalized.
    :rtype: pandas.DataFrame
    """

    # Iterate over each column that needs to be normalized
    for col in cols:
        # Compute the lower and upper values at the given percentiles
        min_val = np.percentile(df[col], min_percentile)
        max_val = np.percentile(df[col], max_percentile)

        # Clip the column's values between these percentile thresholds
        df[col] = np.clip(df[col], min_val, max_val)

        # Perform min-max normalization to scale the clipped values to the [0, 1] range
        df[col] = (df[col] - min_val) / (max_val - min_val)

    return df



def cell_type_decompose(sc_ad, st_ad, cell_type_col='cell_type', NMS_mode=False, major_types=None, min_percentile=5, max_percentile=95):
    """
    Performs cell type decomposition on spatial data (ST or image) with single-cell data .
    
    :param sc_ad: AnnData object containing single-cell meta data.
    :param st_ad: AnnData object containing spatial data (ST or image) meta data.
    :param cell_type_col: The column name in `sc_ad.obs` that contains cell type annotations. Default is 'cell_type'.
    :param NMS_mode: Boolean flag to apply Non-Maximum Suppression (NMS) mode. Default is False.
    :param major_types: Major cell types used for NMS mode. Default is None.
    :param min_percentile: The lower percentile used for clipping (defaults to 5).
    :param max_percentile: The upper percentile used for clipping (defaults to 95).
    :return: The spatial AnnData object with projected cell type annotations.
    """
    
    # Preprocess the data for decomposition using tangram (tg)
    tg.pp_adatas(sc_ad, st_ad, genes=None)  # Preprocessing: match genes between single-cell and spatial data
    

    # Map single-cell data to spatial data using Tangram's "map_cells_to_space" function
    ad_map = tg.map_cells_to_space(
        sc_ad, st_ad, 
        mode="clusters",  # Map based on clusters (cell types)
        cluster_label=cell_type_col,  # Column in `sc_ad.obs` representing cell type
        device='cpu',  # Run on CPU (or 'cuda' if GPU is available)
        scale=False,  # Don't scale data (can be set to True if needed)
        density_prior='uniform',  # Use prior information for cell densities
        random_state=10,  # Set random state for reproducibility
        verbose=False,  # Disable verbose output for cleaner logging
    )
    
    # Project cell type annotations from the single-cell data to the spatial data
    tg.project_cell_annotations(ad_map, st_ad, annotation=cell_type_col)


    if NMS_mode:
        major_types = major_types
        st_ad.obs = normalize_percentile(st_ad.obsm['tangram_ct_pred'], major_types, min_percentile, max_percentile)

        st_ad_binary = st_ad.obsm['tangram_ct_pred'][major_types].copy()
        # Retain the max value in each row and set the rest to 0
        st_ad.obs[major_types] = st_ad_binary.where(st_ad_binary.eq(st_ad_binary.max(axis=1), axis=0), other=0)

    return st_ad  # Return the spatial AnnData object with the projected annotations



def assign_cells_to_spots(cell_locs, spot_locs, patch_size=16):
    """
    Assigns cells to spots based on their spatial coordinates. Each cell within the specified patch size (radius)
    of a spot will be assigned to that spot.

    :param cell_locs: Numpy array of shape (n_cells, 2) with the x, y coordinates of the cells.
    :param spot_locs: Numpy array of shape (n_spots, 2) with the x, y coordinates of the spots.
    :param patch_size: The diameter of the spot patch. The radius used for assignment will be half of this value.
    :return: A sparse matrix where each row corresponds to a cell and each column corresponds to a spot.
                        The value is 1 if the cell is assigned to that spot, 0 otherwise.
    """
    # Initialize the NearestNeighbors model with a radius equal to half the patch size
    neigh = NearestNeighbors(radius=patch_size * 0.5)
    
    # Fit the model on the spot locations
    neigh.fit(spot_locs)
    
    # Create the radius neighbors graph which will assign cells to spots based on proximity
    # This graph is a sparse matrix where rows are cells and columns are spots, with a 1 indicating assignment
    A = neigh.radius_neighbors_graph(cell_locs, mode='connectivity')
    
    return A



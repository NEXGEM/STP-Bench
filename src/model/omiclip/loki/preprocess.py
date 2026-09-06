import scanpy as sc
import numpy as np
import pandas as pd
import json
import os
from PIL import Image



def generate_gene_df(ad, house_keeping_genes, todense=True):
    """
    Generates a DataFrame with the top 50 genes for each observation in an AnnData object.
    It removes genes containing '.' or '-' in their names, as well as genes listed in
    the provided `house_keeping_genes` DataFrame/Series under the 'genesymbol' column.

    :param ad: An AnnData object containing gene expression data.
    :type ad: anndata.AnnData
    :param house_keeping_genes: DataFrame or Series with a 'genesymbol' column listing housekeeping genes to exclude.
    :type house_keeping_genes: pandas.DataFrame or pandas.Series
    :param todense: Whether to convert the sparse matrix (ad.X) to a dense matrix before creating a DataFrame.
    :type todense: bool
    :return: A DataFrame (`top_k_genes_str`) that contains a 'label' column. Each row in 'label' is a string
             with the top 50 gene names (space-separated) for that observation.
    :rtype: pandas.DataFrame
    """

    # Remove genes containing '.' in their names
    ad = ad[:, ~ad.var.index.str.contains('.', regex=False)]
    # Remove genes containing '-'
    ad = ad[:, ~ad.var.index.str.contains('-', regex=False)]
    # Exclude housekeeping genes
    ad = ad[:, ~ad.var.index.isin(house_keeping_genes['genesymbol'])]

    # Convert to dense if requested; otherwise use the data as-is
    if todense:
        expr = pd.DataFrame(ad.X.todense(), index=ad.obs.index, columns=ad.var.index)
    else:
        expr = pd.DataFrame(ad.X, index=ad.obs.index, columns=ad.var.index)

    # For each row (observation), find the top 50 genes with the highest expression
    top_k_genes = expr.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=50)

    # Create a new DataFrame to store the labels (space-separated top gene names)
    top_k_genes_str = pd.DataFrame()
    top_k_genes_str['label'] = top_k_genes[top_k_genes.columns].astype(str) \
        .apply(lambda x: ' '.join(x), axis=1)

    return top_k_genes_str



def segment_patches(img_array, coord, patch_dir, height=20, width=20):
    """
    Extracts small image patches centered at specified coordinates and saves them as individual PNG files.

    :param img_array: A NumPy array representing the full-resolution image. Shape is expected to be (H, W[, C]).
    :type img_array: numpy.ndarray
    :param coord: A pandas DataFrame containing patch center coordinates in columns "pixel_x" and "pixel_y".
                  The index corresponds to spot IDs. Example columns: ["pixel_x", "pixel_y"].
    :type coord: pandas.DataFrame
    :param patch_dir: Directory path where the patch images will be saved.
    :type patch_dir: str
    :param height: The patch's height in pixels (distance in the y-direction).
    :type height: int
    :param width: The patch's width in pixels (distance in the x-direction).
    :type width: int
    :return: None. The function saves image patches to `patch_dir` but does not return anything.
    """

    # Ensure the output directory exists; create it if it doesn't
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)

    # Extract the overall height and width of the image
    yrange, xrange = img_array.shape[:2]

    # Iterate through each coordinate in the DataFrame
    for spot_idx in coord.index:
        # Retrieve the center x and y coordinates for the current spot
        ycenter, xcenter = coord.loc[spot_idx, ["pixel_x", "pixel_y"]]

        # Compute the top-left (x1, y1) and bottom-right (x2, y2) boundaries of the patch
        x1 = round(xcenter - width / 2)
        y1 = round(ycenter - height / 2)
        x2 = x1 + width
        y2 = y1 + height

        # Check if the patch boundaries go outside the image
        if x1 < 0 or y1 < 0 or x2 > xrange or y2 > yrange:
            print(f"Patch {spot_idx} is out of range and will be skipped.")
            continue

        # Extract the patch and convert to a PIL Image; cast to uint8 if needed
        patch_img = Image.fromarray(img_array[y1:y2, x1:x2].astype(np.uint8))

        # Create a filename for the patch image (e.g., "0_hires.png")
        patch_name = f"{spot_idx}_hires.png"
        patch_path = os.path.join(patch_dir, patch_name)

        # Save the patch image to disk
        patch_img.save(patch_path)



def read_gct(file_path):
    """
    Reads a GCT file, parses its dimensions, and returns the data as a pandas DataFrame.

    :param file_path: The path to the GCT file to be read.
    :return: A pandas DataFrame containing the GCT data, where the first two columns represent gene names and descriptions,
                  and the subsequent columns contain the expression data.
    """
    
    # Open the GCT file for reading
    with open(file_path, 'r') as file:
        # Read and ignore the first line (GCT version line)
        file.readline()
        
        # Read the second line which contains the dimensions of the data matrix
        dims = file.readline().strip().split()  # Split the dimensions line by whitespace
        num_rows = int(dims[0])  # Number of data rows (genes)
        num_cols = int(dims[1])  # Number of data columns (samples + metadata)
        
        # Read the data starting from the third line, using pandas for tab-delimited data
        # The first two columns in GCT files are "Name" and "Description" (gene identifiers and annotations)
        data = pd.read_csv(file, sep='\t', header=0, nrows=num_rows)
        
    # Return the loaded data as a pandas DataFrame
    return data



def get_library_id(adata):
    """
    Retrieves the library ID from the AnnData object, assuming it contains spatial data.
    The function will return the first library ID found in `adata.uns['spatial']`.

    :param adata: AnnData object containing spatial information in `adata.uns['spatial']`.
    :return: The first library ID found in `adata.uns['spatial']`.
    :raises: 
            AssertionError: If 'spatial' is not present in `adata.uns`.
            Logs an error if no library ID is found.
    """
    
    # Check if 'spatial' is present in adata.uns; raises an error if not found
    assert 'spatial' in adata.uns, "spatial not present in adata.uns"
    
    # Retrieve the list of library IDs (which are keys in the 'spatial' dictionary)
    library_ids = adata.uns['spatial'].keys()
    
    try:
        # Attempt to return the first library ID (converting the keys object to a list)
        library_id = list(library_ids)[0]
        return library_id
    except IndexError:
        # If no library IDs exist, log an error message
        logger.error('No library_id found in adata')



def get_scalefactors(adata, library_id=None):
    """
    Retrieves the scalefactors from the AnnData object for a given library ID. If no library ID is provided, 
    the function will automatically retrieve the first available library ID.

    :param adata: AnnData object containing spatial data and scalefactors in `adata.uns['spatial']`.
    :param library_id: The library ID for which the scalefactors are to be retrieved. If not provided, it defaults to the first available ID.
    :return: A dictionary containing scalefactors for the specified library ID.
    """
    
    # If no library_id is provided, retrieve the first available library ID
    if library_id is None:
        library_id = get_library_id(adata)
    
    try:
        # Attempt to retrieve the scalefactors for the specified library ID
        scalef = adata.uns['spatial'][library_id]['scalefactors']
        return scalef
    except KeyError:
        # Log an error if the scalefactors or library ID is not found
        logger.error('scalefactors not found in adata')



def get_spot_diameter_in_pixels(adata, library_id=None):
    """
    Retrieves the spot diameter in pixels from the AnnData object's scalefactors for a given library ID.
    If no library ID is provided, the function will automatically retrieve the first available library ID.

    :param adata: AnnData object containing spatial data and scalefactors in `adata.uns['spatial']`.
    :param library_id: The library ID for which the spot diameter is to be retrieved. If not provided, defaults to the first available ID.
    
    :return: The spot diameter in full resolution pixels, or None if not found.
    """
    
    # Get the scalefactors for the specified or default library ID
    scalef = get_scalefactors(adata, library_id=library_id)
    
    try:
        # Attempt to retrieve the spot diameter in full resolution from the scalefactors
        spot_diameter = scalef['spot_diameter_fullres']
        return spot_diameter    
    except TypeError:
        # Handle case where `scalef` is None or invalid (if get_scalefactors returned None)
        pass
    except KeyError:
        # Log an error if the 'spot_diameter_fullres' key is not found in the scalefactors
        logger.error('spot_diameter_fullres not found in adata')



def prepare_data_for_alignment(data_path, scale_type='tissue_hires_scalef'):
    """
    Prepares data for alignment by reading an AnnData object and preparing the high-resolution tissue image.

    :param data_path: The path to the AnnData (.h5ad) file containing the Visium data.
    :param scale_type: The type of scale factor to use (`tissue_hires_scalef` by default).
    
    :return:
        - ad: AnnData object containing the spatial transcriptomics data.
        - ad_coor: Numpy array of scaled spatial coordinates (adjusted for the specified resolution).
        - img: High-resolution tissue image, normalized to 8-bit unsigned integers.
    
    :raises: 
            ValueError: If required data (e.g., scale factors, spatial coordinates, or images) is missing.
    """
    
    # Load the AnnData object from the specified file path
    ad = sc.read_h5ad(data_path)
    
    # Ensure the variable (gene) names are unique to avoid potential conflicts
    ad.var_names_make_unique()
    
    try:
        # Retrieve the specified scale factor for spatial coordinates
        scalef = get_scalefactors(ad)[scale_type]
    except KeyError:
        raise ValueError(f"Scale factor '{scale_type}' not found in ad.uns['spatial']")
    
    # Scale the spatial coordinates using the specified scale factor
    try:
        ad_coor = np.array(ad.obsm['spatial']) * scalef
    except KeyError:
        raise ValueError("Spatial coordinates not found in ad.obsm['spatial']")
    
    # Retrieve the high-resolution tissue image
    try:
        img = ad.uns['spatial'][get_library_id(ad)]['images']['hires']
    except KeyError:
        raise ValueError("High-resolution image not found in ad.uns['spatial']")
    
    # If the image values are normalized to [0, 1], convert to 8-bit format for compatibility
    if img.max() < 1.1:
        img = (img * 255).astype('uint8')
    
    return ad, ad_coor, img



def load_data_for_annotation(st_data_path, json_path, in_tissue=True):
    """
    Loads spatial transcriptomics (ST) data from an .h5ad file and prepares it for annotation.

    :param sample_type: The type or category of the sample (used to locate the data in the directory structure).
    :param sample_name: The name of the sample (used to locate specific files).
    :param in_tissue: Boolean flag to filter the data to include only spots that are in tissue. Default is True.
    
    :return:
        - st_ad: AnnData object containing the spatial transcriptomics data, with spatial coordinates in `obs`.
        - library_id: The library ID associated with the spatial data.
        - roi_polygon: Region of interest polygon loaded from a JSON file for further annotation or analysis.
    """

    # Load the spatial transcriptomics data into an AnnData object
    st_ad = sc.read_h5ad(st_data_path)
    
    # Optionally filter the data to include only spots that are within the tissue
    if in_tissue:
        st_ad = st_ad[st_ad.obs['in_tissue'] == 1]
    
    # Initialize pixel coordinates for spatial information
    st_ad.obs[["pixel_y", "pixel_x"]] = None  # Ensure the columns exist
    st_ad.obs[["pixel_y", "pixel_x"]] = st_ad.obsm['spatial']  # Copy spatial coordinates into obs
    
    # Retrieve the library ID associated with the spatial data
    library_id = get_library_id(st_ad)
    
    # Load the region of interest (ROI) polygon from a JSON file
    with open(json_path) as f:
        roi_polygon = json.load(f)

    return st_ad, library_id, roi_polygon



def read_polygons(file_path, slide_id):
    """
    Reads polygon data from a JSON file for a specific slide ID, extracting coordinates, colors, and thickness.

    :param file_path: Path to the JSON file containing polygon configurations.
    :param slide_id: Identifier for the specific slide whose polygon data is to be extracted.
    :return: 
        - polygons: A list of numpy arrays, where each array contains the coordinates of a polygon.
        - polygon_colors: A list of color values corresponding to each polygon.
        - polygon_thickness: A list of thickness values for each polygon's border.
    """

    # Open the JSON file and load the polygon configurations into a Python dictionary
    with open(file_path, 'r') as f:
        polygons_configs = json.load(f)

    # Check if the given slide_id exists in the polygon configurations
    if slide_id not in polygons_configs:
        return None, None, None  # If slide_id is not found, return None for all outputs

    # Extract the polygon coordinates, colors, and thicknesses for the given slide_id
    polygons = [np.array(poly['coords']) for poly in polygons_configs[slide_id]]  # Convert polygon coordinates to numpy arrays
    polygon_colors = [poly['color'] for poly in polygons_configs[slide_id]]  # Extract the color for each polygon
    polygon_thickness = [poly['thickness'] for poly in polygons_configs[slide_id]]  # Extract the thickness for each polygon

    # Return the polygons, their colors, and their thicknesses
    return polygons, polygon_colors, polygon_thickness



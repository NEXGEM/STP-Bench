import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import json
import cv2
from sklearn.decomposition import PCA
from open_clip import create_model_from_pretrained, get_tokenizer



import os
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pandas as pd

# --- Model loading --------------------------------------------------------

def load_model(
    model_path: str,
    device: Union[str, torch.device]
) -> Tuple[torch.nn.Module, callable, callable]:
    """
    Load pretrained OmiCLIP (COCA ViT‑L‑14) model, its image preprocess, and tokenizer.
    """
    model, preprocess = create_model_from_pretrained(
        "coca_ViT-L-14", device=device, pretrained=model_path
    )
    tokenizer = get_tokenizer("coca_ViT-L-14")
    model.to(device).eval()
    return model, preprocess, tokenizer

# --- Image encoding -------------------------------------------------------

def encode_images(
    model: torch.nn.Module,
    preprocess: callable,
    image_paths: List[str],
    device: Union[str, torch.device]
) -> torch.Tensor:
    """
    Batch–encode a list of image file paths into L2‑normalized embeddings.
    Returns a tensor of shape (N, D).
    """

    # Prepare a list to hold each image's feature embedding
    image_embeddings = []

    # Loop through each image name in the provided list
    for image_path in image_paths:
        # Build the path to the patch image and open it
        image = Image.open(image_path)

        # Preprocess the image, then stack to create a batch of size 1
        image_input = torch.stack([preprocess(image)]).to(device)

        # Generate the image features without gradient tracking
        with torch.no_grad():
            image_features = model.encode_image(image_input)

        # Accumulate the feature embeddings in the list
        image_embeddings.append(image_features)

    # Convert the list of embeddings to a NumPy array, then to a PyTorch tensor
    image_embeddings = torch.cat(image_embeddings, dim=0)

    # Normalize all embeddings across the feature dimension (L2 normalization)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    return image_embeddings


def encode_images_from_h5(
    model: torch.nn.Module,
    preprocess: callable,
    image_path: str,
    device: Union[str, torch.device]
) -> torch.Tensor:
    """
    Batch–encode a list of image file paths into L2‑normalized embeddings.
    Returns a tensor of shape (N, D).
    """

    # Prepare a list to hold each image's feature embedding
    image_embeddings = []
    
    import h5py
    with h5py.File(image_path, 'r') as f:
        num_patches = len(f['img'])
        
        for idx in range(num_patches):
            image = f['img'][idx]
            image = Image.fromarray(image)
            image_input = torch.stack([preprocess(image)]).to(device)
            
            # Generate the image features without gradient tracking
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                
            # Accumulate the feature embeddings in the list
            image_embeddings.append(image_features)
            
    # Convert the list of embeddings to a NumPy array, then to a PyTorch tensor
    image_embeddings = torch.cat(image_embeddings, dim=0)

    # Normalize all embeddings across the feature dimension (L2 normalization)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    
    return image_embeddings

# --- Text encoding --------------------------------------------------------

def encode_texts(
    model: torch.nn.Module,
    tokenizer: callable,
    texts: List[str],
    device: Union[str, torch.device]
) -> torch.Tensor:
    """
    Batch–encode a list of strings into L2‑normalized embeddings.
    Returns a tensor of shape (N, D).
    """
    # Tokenizer returns a dict of tensors
    text_inputs = tokenizer(texts)
    text_inputs = text_inputs.to(device)
    
    with torch.no_grad():
        feats = model.encode_text(text_inputs).to(device)             # (N, D)
    return F.normalize(feats, p=2, dim=-1)                # (N, D)


def encode_text_df(
    model: torch.nn.Module,
    tokenizer: callable,
    df: pd.DataFrame,
    col_name: str,
    device: Union[str, torch.device]
) -> torch.Tensor:
    """
    Encodes an entire DataFrame column into (N, D) embeddings.
    """
    texts = df[col_name].astype(str).tolist()
    return encode_texts(model, tokenizer, texts, device)




def get_pca_by_fit(tar_features, src_features):
    """
    Applies PCA to target features and transforms both target and source features using the fitted PCA model.
    Combines the PCA-transformed features from both target and source datasets and returns the combined data 
    along with batch labels indicating the origin of each sample.

    :param tar_features: Numpy array of target features (samples by features).
    :param src_features: Numpy array of source features (samples by features).
    :return: 
        - pca_comb_features: A numpy array containing PCA-transformed target and source features combined.
        - pca_comb_features_batch: A numpy array of batch labels indicating which samples are from target (0) and source (1).
    """

    pca = PCA(n_components=3)
    
    # Fit the PCA model on the target features (transposed to fit on features)
    pca_fit_tar = pca.fit(tar_features.T)
    
    # Transform the target and source features using the fitted PCA model
    pca_tar = pca_fit_tar.transform(tar_features.T)  # Transform target features
    pca_src = pca_fit_tar.transform(src_features.T)  # Transform source features using the same PCA fit
    
    # Combine the PCA-transformed target and source features
    pca_comb_features = np.concatenate((pca_tar, pca_src))
    
    # Create a batch label array: 0 for target features, 1 for source features
    pca_comb_features_batch = np.array([0] * len(pca_tar) + [1] * len(pca_src))

    return pca_comb_features, pca_comb_features_batch



def cap_quantile(weight, cap_max=None, cap_min=None):
    """
    Caps the values in the 'weight' array based on the specified quantile thresholds for maximum and minimum values.
    If the quantile thresholds are provided, the function will replace values above or below these thresholds 
    with the corresponding quantile values.

    :param weight: Numpy array of weights to be capped.
    :param cap_max: Quantile threshold for the maximum cap. Values above this quantile will be capped. 
                    If None, no maximum capping will be applied.
    :param cap_min: Quantile threshold for the minimum cap. Values below this quantile will be capped. 
                    If None, no minimum capping will be applied.
    :return: Numpy array with the values capped at the specified quantiles.
    """
    
    # If a maximum cap is specified, calculate the value at the specified cap_max quantile
    if cap_max is not None:
        cap_max = np.quantile(weight, cap_max)  # Get the value at the cap_max quantile
    
    # If a minimum cap is specified, calculate the value at the specified cap_min quantile
    if cap_min is not None:
        cap_min = np.quantile(weight, cap_min)  # Get the value at the cap_min quantile
    
    # Cap the values in 'weight' array to not exceed the maximum cap (cap_max)
    weight = np.minimum(weight, cap_max)
    
    # Cap the values in 'weight' array to not go below the minimum cap (cap_min)
    weight = np.maximum(weight, cap_min)
    
    return weight



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



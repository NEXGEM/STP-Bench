import numpy as np
import torch
from torch.nn import functional as F
import os
import scanpy as sc
import json
import cv2



def annotate_with_bulk(img_features, bulk_features, normalize=True, T=1, tensor=False):
    """
    Annotates tissue image with similarity scores between image features and bulk RNA-seq features.

    :param img_features: Feature matrix representing histopathology image features.
    :param bulk_features: Feature vector representing bulk RNA-seq features.
    :param normalize: Whether to normalize similarity scores, default=True.
    :param T: Temperature parameter to control the sharpness of the softmax distribution. Higher values result in a smoother distribution.
    :param tensor: Feature format in torch tensor or not, default=False.

    :return: An array or tensor containing the normalized similarity scores.
    """
    
    if tensor:
        # Compute similarity between image features and bulk RNA-seq features
        cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cosine_similarity(img_features, bulk_features.unsqueeze(0))  # Shape: [n]

        # Optional normalization using the feature vector's norm
        if normalize:
            normalization_factor = torch.sqrt(torch.tensor([bulk_features.shape[0]], dtype=torch.float))  # sqrt(768)
            similarity = similarity / normalization_factor

        # Reshape and apply temperature scaling for softmax
        similarity = similarity.unsqueeze(0)  # Shape: [1, n]
        similarity = similarity / T  # Control distribution sharpness

        # Convert similarity scores to probability distribution using softmax
        similarity = torch.nn.functional.softmax(similarity, dim=-1)  # Shape: [1, n]

    else:
        # Compute similarity for non-tensor mode
        similarity = np.dot(img_features.T, bulk_features)

        # Apply a softmax-like normalization for numerical stability
        max_similarity = np.max(similarity)  # Maximum value for stability
        similarity = np.exp(similarity - max_similarity) / np.sum(np.exp(similarity - max_similarity))

        # Normalize similarity scores to [0, 1] range for interpretation
        similarity = (similarity - np.min(similarity)) / (np.max(similarity) - np.min(similarity))

    return similarity



def annotate_with_marker_genes(classes, image_embeddings, all_text_embeddings):
    """
    Annotates tissue image with similarity scores between image features and marker gene features.

    :param classes: A list or array of tissue type labels.
    :param image_embeddings: A numpy array or torch tensor of image embeddings (shape: [n_images, embedding_dim]).
    :param all_text_embeddings: A numpy array or torch tensor of text embeddings of the marker genes 
                                (shape: [n_classes, embedding_dim]).

    :return: 
        - dot_similarity: The matrix of dot product similarities between image embeddings and text embeddings.
        - pred_class: The predicted tissue type for the image based on the highest similarity score.
    """
    
    # Calculate dot product similarity between image embeddings and text embeddings
    # This results in a similarity matrix of shape [n_images, n_classes]
    dot_similarity = image_embeddings @ all_text_embeddings.T

    # Find the class with the highest similarity for each image
    # Use argmax to identify the index of the highest similarity score
    pred_class = classes[dot_similarity.argmax()]

    return dot_similarity, pred_class



def load_image_annotation(image_path):
    """
    Loads an image with annotation.

    :param image_path: The file path to the image.
    
    :return: The processed image, converted to BGR color space and of type uint8.
    """
    
    # Load the image from the specified file path using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the color from RGB (OpenCV loads as BGR by default) to BGR (which matches common color standards)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Ensure the image is of type uint8 for proper handling in OpenCV and other image processing libraries
    image = image.astype(np.uint8)

    return image



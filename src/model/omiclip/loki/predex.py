import pandas as pd



def predict_st_gene_expr(image_text_similarity, train_data):
    """
    Predicts ST gene expression by H&E image.

    :param image_text_similarity: Numpy array of similarities between images and text features (shape: [n_samples, n_genes]).
    :param train_data: Numpy array or DataFrame of training data used for making predictions (shape: [n_genes, n_shared_genes]).
    :return: Numpy array or DataFrame containing the predicted gene expression levels for the samples.
    """
    
    # Compute the weighted sum of the train_data using image_text_similarity
    weighted_sum = image_text_similarity @ train_data
    
    # Compute the normalization factor (sum of the image-text similarities for each sample)
    weights = image_text_similarity.sum(axis=1, keepdims=True)
    
    # Normalize the predicted matrix to get weighted gene expression predictions
    predicted_image_text_matrix = weighted_sum / weights

    return predicted_image_text_matrix



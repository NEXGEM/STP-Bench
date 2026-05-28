import torch



def retrieve_st_by_image(image_embeddings, all_text_embeddings, dataframe, k=3):
    """
    Retrieves the top-k most similar ST based on the similarity between ST embeddings and image embeddings.

    :param image_embeddings: A numpy array or torch tensor containing image embeddings (shape: [1, embedding_dim]).
    :param all_text_embeddings: A numpy array or torch tensor containing ST embeddings (shape: [n_samples, embedding_dim]).
    :param dataframe: A pandas DataFrame containing information about the ST samples, specifically the image indices in the 'img_idx' column.
    :param k: The number of top similar samples to retrieve. Default is 3.
    :return: A list of the filenames or indices corresponding to the top-k similar samples.
    """
    
    # Compute the dot product (similarity) between the image embeddings and all ST embeddings
    dot_similarity = image_embeddings @ all_text_embeddings.T
    
    # Retrieve the top-k most similar samples by similarity score (dot product)
    values, indices = torch.topk(dot_similarity.squeeze(0), k)
    
    # Extract the image filenames or indices from the DataFrame based on the top-k matches
    image_filenames = dataframe['img_idx'].values
    matches = [image_filenames[idx] for idx in indices]

    return matches



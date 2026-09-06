import matplotlib.pyplot as plt
from pathlib import Path
import json
import cv2
from matplotlib import cm
import pandas as pd
import numpy as np
from tqdm import tqdm


def plot_alignment(
    ad_tar_coor: np.ndarray,
    ad_src_coor: np.ndarray,
    homo_coor: np.ndarray,
    pca_hex_comb: np.ndarray,
    tar_features: np.ndarray,
    shift: float = 300,
    s: float = 0.8,
    boundary_line: bool = True
) -> None:
    """
    Optimized plot: target, source, and aligned coordinates with titles.
    """
    # Determine common limits
    coords = np.vstack([ad_tar_coor, ad_src_coor, homo_coor])
    x_min, x_max = coords[:,0].min() - shift, coords[:,0].max() + shift
    y_min, y_max = coords[:,1].min() - shift, coords[:,1].max() + shift

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=150)
    titles = ["Target ST", "Source ST", "Aligned Source ST"]
    splits = [len(ad_tar_coor), len(ad_tar_coor)+len(ad_src_coor)]

    for ax, title, data_slice in zip(
        axes,
        titles,
        [(ad_tar_coor, pca_hex_comb[:splits[0]]),
         (ad_src_coor, pca_hex_comb[splits[0]:splits[1]]),
         (homo_coor, pca_hex_comb[splits[0]:splits[1]])]
    ):
        coords_arr, colors = data_slice
        ax.scatter(coords_arr[:,0], coords_arr[:,1], s=s, c=colors, marker='o')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        if boundary_line:
            ax.axvline(x=ad_tar_coor[:,0].min(), color='black', linewidth=1)
            ax.axhline(y=ad_tar_coor[:,1].min(), color='black', linewidth=1)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()



def plot_alignment_with_img(
    ad_tar_coor: np.ndarray,
    ad_src_coor: np.ndarray,
    homo_coor: np.ndarray,
    tar_img,
    src_img,
    aligned_image,
    pca_hex_comb: np.ndarray,
    tar_features: np.ndarray,
    s: float = 1.0
) -> None:
    """
    Optimized plot with images in the background and subplot titles.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    titles = ["Target + Image", "Source + Image", "Aligned + Image"]
    splits = [len(tar_features.T), len(tar_features.T) * 2]

    # Data slices for each subplot
    data_slices = [
        (ad_tar_coor, pca_hex_comb[:splits[0]], tar_img),
        (ad_src_coor, pca_hex_comb[splits[0]:splits[1]], src_img),
        (np.vstack([ad_tar_coor, homo_coor]), 
         np.concatenate([pca_hex_comb[:splits[0]], pca_hex_comb[splits[0]:splits[1]]]),
         aligned_image)
    ]

    for ax, title, (coords_arr, colors, img) in zip(axes, titles, data_slices):
        ax.imshow(img, origin='lower', alpha=0.3)
        ax.scatter(coords_arr[:,0], coords_arr[:,1], s=s, c=colors, marker='o')
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_image(img, title: str = "Aligned Source Image", origin: str = "lower", cmap=None):
    """
    Display a single image with no axes and a title.

    :param img: The image to display (NumPy array, PIL Image, etc.).
    :param title: Title to show above the image.
    :param origin: Origin parameter passed to plt.imshow (e.g. 'lower' or 'upper').
    :param cmap: Optional colormap for grayscale or other single‑channel data.
    """
    plt.imshow(img, origin=origin, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def draw_polygon(image, polygon, color='k', thickness=2):
    """
    Draws one or more polygons on the given image.

    :param image: The image on which to draw the polygons (as a numpy array).
    :param polygon: A list of polygons, where each polygon is a list of (x, y) coordinate tuples.
    :param color: A string or list of strings representing the color(s) for each polygon.
                  If a single color is provided, it will be applied to all polygons. Default is 'k' (black).
    :param thickness: An integer or a list of integers representing the thickness of the polygon borders.
                      If a single value is provided, it will be applied to all polygons. Default is 2.
    
    :return: The image with the polygons drawn on it.
    """
    
    # If the provided `color` is a single value (string), convert it to a list of the same color for each polygon
    if not isinstance(color, list):
        color = [color] * len(polygon)  # Create a list where each polygon gets the same color
    
    # Loop through each polygon in the list, along with its corresponding color
    for i, poly in enumerate(polygon):
        # Get the color for the current polygon
        c = color[i]
        
        # Convert the color from a string format (e.g., 'k' or '#ff0000') to an RGB tuple
        c = color_string_to_rgb(c)
        
        # Get the thickness value for the current polygon (if a list is provided, use the corresponding value)
        t = thickness[i] if isinstance(thickness, list) else thickness

        # Convert the polygon coordinates to a numpy array of integers
        poly = np.array(poly, np.int32)

        # Reshape the polygon array to match OpenCV's expected input format: (number of points, 1, 2)
        poly = poly.reshape((-1, 1, 2))

        # Draw the polygon on the image using OpenCV's `cv2.polylines` function
        # `isClosed=True` indicates that the polygon should be closed (start and end points are connected)
        image = cv2.polylines(image, [poly], isClosed=True, color=c, thickness=t)

    return image



def blend_images(image1, image2, alpha=0.5):
    """
    Blends two images together.

    :param image1: Background image, a numpy array of shape (H, W, 3), where H is height, W is width, and 3 represents the RGB color channels.
    :param image2: Foreground image, a numpy array of shape (H, W, 3), same dimensions as image1.
    :param alpha: Blending factor, a float between 0 and 1. The value of alpha determines the weight of image1 in the blend,
                  where 0 means only image2 is shown, and 1 means only image1 is shown. Default is 0.5 (equal blending).
    
    :return: A blended image, where each pixel is a weighted combination of the corresponding pixels from image1 and image2.
            The blending is computed as: `blended = alpha * image1 + (1 - alpha) * image2`.
    """
    
    # Use cv2.addWeighted to blend the two images.
    # The first image (image1) is weighted by 'alpha', and the second image (image2) is weighted by '1 - alpha'.
    blended = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    
    # Return the resulting blended image.
    return blended



def color_string_to_rgb(color_string):
    """
    Converts a color string to an RGB tuple.
    
    :param color_string: A string representing the color. This can be in hexadecimal form (e.g., '#ff0000') or 
                         a shorthand character for basic colors (e.g., 'k' for black, 'r' for red, etc.).
    :return: 
            A tuple (r, g, b) representing the RGB values of the color, where each value is an integer between 0 and 255.
    :raises: 
            ValueError: If the color string is not recognized.
    """
    
    # Remove any spaces in the color string
    color_string = color_string.replace(' ', '')
    
    # If the string starts with a '#', it's a hexadecimal color, so we remove the '#'
    if color_string.startswith('#'):
        color_string = color_string[1:]
    else:
        # Handle shorthand single-letter color codes by converting them to hex values
        # 'k' -> black, 'r' -> red, 'g' -> green, 'b' -> blue, 'w' -> white
        if color_string == 'k':  # Black
            color_string = '000000'
        elif color_string == 'r':  # Red
            color_string = 'ff0000'
        elif color_string == 'g':  # Green
            color_string = '00ff00'
        elif color_string == 'b':  # Blue
            color_string = '0000ff'
        elif color_string == 'w':  # White
            color_string = 'ffffff'
        else:
            # Raise an error if the color string is not recognized
            raise ValueError(f"Unknown color string {color_string}")
    
    # Convert the first two characters to the red (R) value
    r = int(color_string[:2], 16)
    
    # Convert the next two characters to the green (G) value
    g = int(color_string[2:4], 16)
    
    # Convert the last two characters to the blue (B) value
    b = int(color_string[4:], 16)
    
    # Return the RGB values as a tuple
    return (r, g, b)



def plot_heatmap(
    coor,
    similairty,
    image_path=None,
    polygons=None,
    polygons_color='k',
    polygons_thickness=2,
    patch_size=(256, 256),
    save_path=None,
    downsize=32,
    cmap='turbo',
    smooth=False,
    boxes=None,
    box_color='k',
    box_thickness=2,
    image_alpha=0.5
):
    """
    Plots a heatmap overlaid on an image based on given coordinates and similairty.

    :param coor: Array of coordinates (N, 2) where N is the number of patches to place on the heatmap.
    :param similairty: Array of similairty (N,) corresponding to the coordinates. These similairties are mapped to colors using a colormap.
    :param image_path: Path to the background image on which the heatmap will be overlaid. If None, a blank white background is used.
    :param patch_size: Size of each patch in pixels (default is 256x256).
    :param save_path: Path to save the heatmap image. If None, the heatmap is returned instead of being saved.
    :param downsize: Factor to downsize the image and patches for faster processing. Default is 32.
    :param cmap: Colormap to map the similairties to colors. Default is 'turbo'.
    :param smooth: Boolean to indicate if the heatmap should be smoothed. Not implemented in this version.
    :param boxes: List of boxes in (x, y, w, h) format. If provided, boxes will be drawn on the heatmap.
    :param box_color: Color of the boxes. Default is black ('k').
    :param box_thickness: Thickness of the box outlines.
    :param polygons: List of polygons (N, 2) to draw on the heatmap.
    :param polygons_color: Color of the polygon outlines. Default is black ('k').
    :param polygons_thickness: Thickness of the polygon outlines.
    :param image_alpha: Transparency value (0 to 1) for blending the heatmap with the original image. Default is 0.5.
    
    :return: 
        - heatmap: The generated heatmap as a numpy array (RGB).
        - image: The original image with overlaid polygons if provided.
    """

    # Read the background image (if provided), otherwise a blank image
    image = cv2.imread(image_path)
    image_size = (image.shape[0], image.shape[1])  # Get the size of the image
    coor = [(x // downsize, y // downsize) for x, y in coor]  # Downsize the coordinates for faster processing
    patch_size = (patch_size[0] // downsize, patch_size[1] // downsize)  # Downsize the patch size

    # Convert similairties to colors using the provided colormap
    cmap = plt.get_cmap(cmap)  # Get the colormap object
    norm = plt.Normalize(vmin=similairty.min(), vmax=similairty.max())  # Normalize similairties to map to color range
    colors = cmap(norm(similairty))  # Convert the normalized similairties to RGB colors

    # Initialize a blank white heatmap the size of the image
    heatmap = np.ones((image_size[0], image_size[1], 3)) * 255  # Start with a white background

    # Place the colored patches on the heatmap according to the coordinates and patch size
    for i in range(len(coor)):
        x, y = coor[i]
        w = colors[i][:3] * 255  # Get the RGB color for the patch, scaling from [0, 1] to [0, 255]
        w = w.astype(np.uint8)  # Convert the color to uint8
        heatmap[y:y + patch_size[0], x:x + patch_size[1], :] = w  # Place the patch on the heatmap

    # If the image_alpha is greater than 0, blend the heatmap with the original image
    if image_alpha > 0:
        image = np.array(image)

        # Pad the image if necessary to match the heatmap size
        if image.shape[0] < heatmap.shape[0]:
            pad = heatmap.shape[0] - image.shape[0]
            image = np.pad(image, ((0, pad), (0, 0), (0, 0)), mode='constant', constant_values=255)
        if image.shape[1] < heatmap.shape[1]:
            pad = heatmap.shape[1] - heatmap.shape[1]
            image = np.pad(image, ((0, 0), (0, pad), (0, 0)), mode='constant', constant_values=255)

        # Convert the image to BGR (for OpenCV compatibility) and blend with the heatmap
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image.astype(np.uint8)
        heatmap = heatmap.astype(np.uint8)
        heatmap = blend_images(heatmap, image, alpha=image_alpha)  # Blend the heatmap and the image

    # If polygons are provided, draw them on the heatmap and image
    if polygons is not None:
        polygons = [poly // downsize for poly in polygons]  # Downsize the polygon coordinates
        image_polygons = draw_polygon(image, polygons, color=polygons_color, thickness=polygons_thickness)  # Draw polygons on the original image
        heatmap_polygons = draw_polygon(heatmap, polygons, color=polygons_color, thickness=polygons_thickness)  # Draw polygons on the heatmap
        
        return heatmap_polygons, image_polygons  # Return the heatmap and image with polygons drawn on them
    else:
        return heatmap, image  # Return the heatmap and image



def show_images_side_by_side(image1, image2, title1='Annotated H&E Image', title2='Similatrity Heatmap'):
    """
    Displays two images side by side in a single figure.

    :param image1: The first image to display (as a numpy array).
    :param image2: The second image to display (as a numpy array).
    :param title1: The title for the first image. Default is None (no title).
    :param title2: The title for the second image. Default is None (no title).
    :return: Displays the images side by side.
    """
    
    # Create a figure with 2 subplots (1 row, 2 columns), and set the figure size
    fig, ax = plt.subplots(1, 2, figsize=(8,6), dpi=150)
    
    # Display the first image on the first subplot
    ax[0].imshow(image1)
    
    # Display the second image on the second subplot
    ax[1].imshow(image2)
    
    # Set the title for the first image (if provided)
    ax[0].set_title(title1)
    
    # Set the title for the second image (if provided)
    ax[1].set_title(title2)
    
    # Remove axis labels and ticks for both images to give a cleaner look
    ax[0].axis('off')
    ax[1].axis('off')
    
    # Show the final figure with both images displayed side by side
    plt.show()



def plot_img_with_annotation(fullres_img, roi_polygon, linewidth, xlim, ylim):
    """
    Plots image with polygons.

    :param fullres_img: The full-resolution image to display (as a numpy array).
    :param roi_polygon: A list of polygons, where each polygon is a list of (x, y) coordinate tuples.
    :param linewidth: The thickness of the lines used to draw the polygons.
    :param xlim: A tuple (xmin, xmax) defining the x-axis limits for zooming in on a specific region of the image.
    :param ylim: A tuple (ymin, ymax) defining the y-axis limits for zooming in on a specific region of the image.
    :return: Displays the image with ROI polygons overlaid.
    """
    
    # Create a new figure with a fixed size for displaying the image and annotations
    plt.figure(figsize=(12, 12), dpi=150)
    
    # Display the full-resolution image
    plt.imshow(fullres_img)
    
    # Loop through each polygon in roi_polygon and plot them on the image
    for polygon in roi_polygon:
        x, y = zip(*polygon)  # Unzip the list of (x, y) tuples into separate x and y coordinate lists
        plt.plot(x, y, color='black', linewidth=linewidth)  # Plot the polygon using the specified linewidth
    
    # Set the x-axis limits based on the provided tuple (xlim)
    plt.xlim(xlim)
    
    # Set the y-axis limits based on the provided tuple (ylim)
    plt.ylim(ylim)
    
    # Invert the y-axis to match the typical image display convention (origin at the top-left)
    plt.gca().invert_yaxis()
    
    # Turn off the axis to give a cleaner image display without ticks or labels
    plt.axis('off')



def plot_annotation_heatmap(st_ad, roi_polygon, s, linewidth, xlim, ylim):
    """
    Plots tissue type annotation heatmap.

    :param st_ad: AnnData object containing coordinates in `obsm['spatial']`
                  and similarity scores in `obs['bulk_simi']`.
    :param roi_polygon: A list of polygons, where each polygon is a list of (x, y) coordinate tuples.
    :param s: The size of the scatter plot markers representing each spatial transcriptomics spot.
    :param linewidth: The thickness of the lines used to draw the polygons.
    :param xlim: A tuple (xmin, xmax) defining the x-axis limits for zooming in on a specific region of the image.
    :param ylim: A tuple (ymin, ymax) defining the y-axis limits for zooming in on a specific region of the image.
    :return: Displays the heatmap with polygons overlaid.
    """
    
    # Create a new figure with a fixed size for displaying the heatmap and annotations
    plt.figure(figsize=(12, 12), dpi=150)
    
    # Scatter plot for the spatial transcriptomics data.
    # The 'spatial' coordinates are plotted with color intensity based on 'bulk_simi' values.
    plt.scatter(
        st_ad.obsm['spatial'][:, 0], st_ad.obsm['spatial'][:, 1],  # x and y coordinates
        c=st_ad.obs['bulk_simi'],  # Color values based on 'bulk_simi'
        s=s,  # Size of each marker
        vmin=0.1, vmax=0.95,  # Set the range for the color normalization
        cmap='turbo'  # Use the 'turbo' colormap for the heatmap
    )
    
    # Loop through each polygon in roi_polygon and plot them on the image
    for polygon in roi_polygon:
        x, y = zip(*polygon)  # Unzip the list of (x, y) tuples into separate x and y coordinate lists
        plt.plot(x, y, color='black', linewidth=linewidth)  # Plot the polygon using the specified linewidth
    
    # Set the x-axis limits based on the provided tuple (xlim)
    plt.xlim(xlim)
    
    # Set the y-axis limits based on the provided tuple (ylim)
    plt.ylim(ylim)
    
    # Invert the y-axis to match the typical image display convention (origin at the top-left)
    plt.gca().invert_yaxis()
    
    # Turn off the axis to give a cleaner image display without ticks or labels
    plt.axis('off')



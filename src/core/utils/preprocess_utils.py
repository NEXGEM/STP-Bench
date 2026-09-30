import json
import os
import warnings
from enum import Enum
from typing import Tuple

import cv2
import h5py
import numpy as np
import openslide
import pandas as pd
import tifffile
from PIL import Image

from torchvision import transforms
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from hest import (
    STReader,
    VisiumHDReader,
)
from hestcore.segmentation import apply_otsu_thresholding, mask_to_gdf, segment_tissue_deep
from hestcore.wsi import NumpyWSI, OpenSlideWSI, WSI


def load_st(path, platform, meta_dir=None):
    """Read one sample directory into an object with `adata`, `wsi`, `pixel_size`,
    `segment_tissue()` and `dump_patches()`.

    'visium' (raw SpaceRanger directory) and 'xenium' (Xenium_aligned/<sample>) follow HEST 1.1.x
    (see the section below); 'st' and 'visium-hd' still use the installed HEST readers.
    `meta_dir`: where `pixel_sizes.csv` is looked up for 'visium' (see `pixel_size_um`).
    """
    assert platform in ['st', 'visium', 'visium-hd', 'xenium'], \
        "platform must be one of ['st', 'visium', 'visium-hd', 'xenium']"

    if platform == 'st':
        return STReader().auto_read(path)
    if platform == 'visium':
        adata, img_path = _read_visium_spots(path)
        sample_id = os.path.basename(os.path.normpath(path))
        return STSample(adata, open_slide_image(img_path), pixel_size_um(sample_id, adata.obs, meta_dir))
    if platform == 'visium-hd':
        return VisiumHDReader().auto_read(path)
    if platform == 'xenium':
        adata, img_path, pixel_size = _read_xenium_aligned(path)
        return STSample(adata, open_slide_image(img_path), pixel_size)


def map_values(arr, step_size=256):
    """
    Map NumPy array values to integers such that:
    1. The minimum value is mapped to 0
    2. Values within 256 of each other are mapped to the same integer
    
    Args:
    arr (np.ndarray): Input NumPy array of numeric values
    
    Returns:
    tuple: 
        - NumPy array of mapped integer values 
    """
    if arr.size == 0:
        return np.array([]), {}
    
    # Sort the unique values
    unique_values = np.sort(np.unique(arr))
    
    mapping = {}
    current_key = 0
    
    mapping[unique_values[0]] = 0
    current_value = unique_values[0]

    for i in range(1, len(unique_values)):
        if unique_values[i] - current_value > step_size:
            current_key += 1
            current_value = unique_values[i] 
        
        mapping[unique_values[i]] = current_key
    
    mapped_arr = np.vectorize(mapping.get)(arr)
    
    return mapped_arr

def pxl_to_array(pixel_crds, step_size):
    x_crds = map_values(pixel_crds[:,0], step_size)
    y_crds = map_values(pixel_crds[:,1], step_size)
    dst = np.stack((x_crds, y_crds), axis=1)
    return dst


def save_hdf5(output_fpath, 
                      asset_dict, 
                      attr_dict= None, 
                      mode='a', 
                      auto_chunk = True,
                      chunk_size = None):
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if len(data_shape) == 1:
                val = np.expand_dims(val, axis=1)
                data_shape = val.shape

            # Determine if the data is of string type
            if np.issubdtype(val.dtype, np.bytes_) or np.issubdtype(val.dtype, np.str_):
                data_type = h5py.string_dtype(encoding='utf-8')
            else:
                data_type = val.dtype

            if key not in f:  # if key does not exist, create dataset
                if auto_chunk:
                    chunks = True  # let h5py decide chunk size
                else:
                    chunks = (chunk_size,) + data_shape[1:]
                dset = f.create_dataset(
                    key,
                    shape=data_shape,
                    chunks=chunks,
                    maxshape=(None,) + data_shape[1:],
                    dtype=data_type
                )
                # Save attribute dictionary
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
                dset[:] = val
            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                if dset.dtype != data_type:
                    raise TypeError(f"Data type mismatch for key '{key}'. Dataset dtype: {dset.dtype}, value dtype: {data_type}")
                dset[-data_shape[0]:] = val


def get_transforms(mean, std, target_img_size = -1, center_crop = False, transform_type = 'eval'):
    trsforms = []
    
    # Apply specific transformation based on transform_type
    if transform_type == 'hori':
        # Horizontal flip
        trsforms.append(transforms.RandomHorizontalFlip(p=1.0))
    elif transform_type == 'vert':
        # Vertical flip
        trsforms.append(transforms.RandomVerticalFlip(p=1.0))
    elif transform_type == 'rot_90':
        # 90-degree rotation
        trsforms.append(transforms.RandomRotation((90, 90)))
    elif transform_type == 'rot_180':
        # 180-degree rotation
        trsforms.append(transforms.RandomRotation((180, 180)))
    elif transform_type == 'rot_270':
        # 270-degree rotation
        trsforms.append(transforms.RandomRotation((270, 270)))
    elif transform_type == 'tp':
        # Transpose (90-degree rotation + horizontal flip)
        class Transpose(object):
            def __call__(self, img):
                return transforms.functional.hflip(transforms.functional.rotate(img, 90))
        trsforms.append(Transpose())
    elif transform_type == 'tv':
        # Transverse (90-degree rotation + vertical flip)
        class Transverse(object):
            def __call__(self, img):
                return transforms.functional.vflip(transforms.functional.rotate(img, 90))
        trsforms.append(Transverse())
        
    elif transform_type == 'eval':
        # Default 'eval' mode has no augmentations
        pass
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size))
    if center_crop:
        assert target_img_size > 0, "target_img_size must be set if center_crop is True"
        trsforms.append(transforms.CenterCrop(target_img_size))
        
    trsforms.append(transforms.ToTensor())
    if mean is not None and std is not None:
        trsforms.append(transforms.Normalize(mean, std))
    trsforms = transforms.Compose(trsforms)

    return trsforms


def add_augmentation_to_transform(existing_transform, transform_type='eval'):
    """
    Adds the specified augmentation to an existing transform pipeline.
    
    Args:
        existing_transform (transforms.Compose): Existing transformation pipeline
        transform_type (str): Type of augmentation to add ('hori', 'vert', 'rot_90', 
                            'rot_180', 'rot_270', 'tp', 'tv', or 'eval')
    
    Returns:
        transforms.Compose: New transformation pipeline with added augmentation
    """
    from torchvision import transforms
    
    # Create augmentation based on transform_type
    augmentation = None
    if transform_type == 'hori':
        # Horizontal flip
        augmentation = transforms.RandomHorizontalFlip(p=1.0)
    elif transform_type == 'vert':
        # Vertical flip
        augmentation = transforms.RandomVerticalFlip(p=1.0)
    elif transform_type == 'rot_90':
        # 90-degree rotation
        augmentation = transforms.RandomRotation((90, 90))
    elif transform_type == 'rot_180':
        # 180-degree rotation
        augmentation = transforms.RandomRotation((180, 180))
    elif transform_type == 'rot_270':
        # 270-degree rotation
        augmentation = transforms.RandomRotation((270, 270))
    elif transform_type == 'tp':
        # Transpose (90-degree rotation + horizontal flip)
        class Transpose(object):
            def __call__(self, img):
                return transforms.functional.hflip(transforms.functional.rotate(img, 90))
        augmentation = Transpose()
    elif transform_type == 'tv':
        # Transverse (90-degree rotation + vertical flip)
        class Transverse(object):
            def __call__(self, img):
                return transforms.functional.vflip(transforms.functional.rotate(img, 90))
        augmentation = Transverse()
    elif transform_type == 'eval':
        # No augmentation in eval mode
        return existing_transform
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    # Extract transforms from the existing pipeline
    transform_list = list(existing_transform.transforms)
    
    # Insert the augmentation at the beginning of the pipeline
    transform_list.insert(0, augmentation)
    
    # Return new transform pipeline
    return transforms.Compose(transform_list)


def compute_mini_tiles(image, n_tiles):
    D = image.shape[-1]  # assuming the image is a square, so width = height = D
    n = int(np.sqrt(n_tiles))  # number of squares along one dimension
    square_size = D // n
    D, n, square_size

    # List to hold the split images
    squares = []

    # Loop to crop the image into n x n squares
    for i in range(n):
        for j in range(n):
            left = j * square_size
            right = left + square_size

            lower = i * square_size
            upper = lower + square_size

            if image.ndim == 3:
                # Crop the image
                crop = image[:, lower:upper, left:right]
            elif image.ndim == 4:
                # Crop the image
                crop = image[:, :, lower:upper, left:right]
                
            squares.append(crop)

    return squares


# ---------------------------------------------------------------------------------------------
# Visium / Xenium_aligned reading, tissue segmentation and patch dumping as done by HEST 1.1.x
# (mahmoodlab/HEST, commit 162c42f) on top of hestcore. These are the steps that produced the
# STP-Bench reference patches; the pinned HEST re-implements segmentation/patching on TRIDENT and
# does not reproduce them. Image classes, the segmentation model and the patcher come from the
# `hestcore` package (requirements/core/preprocess.txt).
#
# Differences from the original: the readers only read the input directory (the original moves
# files inside it), samples that would take HEST's alignment-file branch raise instead of being
# processed differently, the image backend is OpenSlide for pyramidal slides and a full in-memory
# array otherwise, and the segmentation checkpoint lives in a user cache (see
# `default_weights_dir`) and is only downloaded when missing.
#
# Original code: Copyright the HEST / hestcore authors, licensed CC BY-NC-SA 4.0
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
# ---------------------------------------------------------------------------------------------

class SpotPacking(Enum):
    """Types of ST spots disposition,
    for Orange Crate Packing see:
    https://kb.10xgenomics.com/hc/en-us/articles/360041426992-Where-can-I-find-the-Space-Ranger-barcode-whitelist-and-their-coordinates-on-the-slide
    """
    ORANGE_CRATE_PACKING = 0
    GRID_PACKING = 1


def find_pixel_size_from_spot_coords(my_df: pd.DataFrame, inter_spot_dist: float = 100.,
                                     packing: SpotPacking = SpotPacking.ORANGE_CRATE_PACKING) -> Tuple[float, int]:
    """Estimate the pixel size of an image in um/px given a dataframe containing the spot coordinates in that image

    Args:
        my_df (pd.DataFrame): must contain the columns
            ['pxl_row_in_fullres', 'pxl_col_in_fullres', 'array_col', 'array_row'].
        inter_spot_dist (float, optional): distance in um between two spots on the same row. Defaults to 100..
        packing (SpotPacking, optional): disposition of the spots on the slide.

    NOTE: the estimate only looks at the first few rows (sorted by array_row) that
    hold two or more spots, so it depends on WHICH spots are passed. Pass the full,
    unfiltered spot table (as the raw SpaceRanger output provides) to reproduce the
    reference patches; a table already filtered to the kept spots can give a
    slightly different value.

    Returns:
        Tuple[float, int]: approximation of the pixel size in um/px and over how many spots it was estimated
    """
    def _cart_dist(start_spot, end_spot):
        """cartesian distance in pixel between two spots"""
        d = np.sqrt((start_spot['pxl_col_in_fullres'] - end_spot['pxl_col_in_fullres']) ** 2
                    + (start_spot['pxl_row_in_fullres'] - end_spot['pxl_row_in_fullres']) ** 2)
        return d

    df = my_df.copy()

    max_dist_col = 0
    approx_nb = 0
    best_approx = 0
    df = df.sort_values('array_row')
    for _, row in df.iterrows():
        y = row['array_col']
        x = row['array_row']
        if len(df[df['array_row'] == x]) > 1:
            b = df[df['array_row'] == x]['array_col'].idxmax()
            start_spot = row
            end_spot = df.loc[b]
            dist_px = _cart_dist(start_spot, end_spot)

            div = 1 if packing == SpotPacking.GRID_PACKING else 2
            dist_col = abs(df.loc[b, 'array_col'] - y) // div

            approx_nb += 1

            if dist_col > max_dist_col:
                max_dist_col = dist_col
                best_approx = inter_spot_dist / (dist_px / dist_col)
            if approx_nb > 3:
                break

    if approx_nb == 0:
        raise Exception("Couldn't find two spots on the same row")

    return best_approx, max_dist_col


_SPOT_COLS = ['pxl_row_in_fullres', 'pxl_col_in_fullres', 'array_col', 'array_row']


def pixel_size_um(name, obs, meta_dir):
    """Pixel size (um/px) used to cut and segment Visium patches.

    Preferably read from `<meta_dir>/pixel_sizes.csv` (columns sample_id, pixel_size_um).
    Otherwise it is estimated from the spot grid (HEST's find_pixel_size_from_spot_coords).
    That estimator only looks at the first few rows of the table sorted by array_row, so
    its value depends on WHICH spots are given (a table already filtered to the kept spots
    can differ from the raw one) and, through the tie order of an unstable sort, on the
    numpy version -- differences show up around the 5th digit and can move spots that sit
    right at the tissue-overlap threshold.
    """
    csv = os.path.join(meta_dir, 'pixel_sizes.csv') if meta_dir else None
    if csv and os.path.isfile(csv):
        df = pd.read_csv(csv)
        hit = df.loc[df['sample_id'].astype(str) == str(name), 'pixel_size_um']
        if len(hit):
            return float(hit.iloc[0])
    missing = [c for c in _SPOT_COLS if c not in obs.columns]
    if missing:
        raise KeyError(f"{name}: cannot estimate the pixel size, spot table lacks {missing} "
                       f"and no entry was found in {csv}")
    px, _ = find_pixel_size_from_spot_coords(obs[_SPOT_COLS].astype(int))
    warnings.warn(
        f"{name}: pixel size {px:.6f} um/px was estimated from {len(obs)} spots (no entry in "
        f"{csv}); the estimate depends on the spot subset and numpy version, so it can differ "
        "from the value the reference patches were made with.", stacklevel=2)
    return px


def default_weights_dir() -> str:
    """Where the tissue-segmentation checkpoint (deeplabv3_seg_v4.ckpt, MahmoodLab/hest-tissue-seg)
    is kept. Override with the STPBENCH_TISSUE_SEG_DIR environment variable."""
    return os.environ.get('STPBENCH_TISSUE_SEG_DIR') or os.path.expanduser('~/.cache/stpbench/hest-tissue-seg')


def open_slide_image(img_path: str) -> WSI:
    """Open a slide with the backend chosen by its format (see module docstring)."""
    img_path = str(img_path)
    try:
        return OpenSlideWSI(openslide.OpenSlide(img_path))
    except openslide.OpenSlideError:
        pass

    if img_path.endswith('.png') or img_path.endswith('.jpg'):
        img = np.array(Image.open(img_path))
    else:
        img = tifffile.imread(img_path)

    # sometimes the RGB axis are inverted
    if img.shape[0] == 3 or img.shape[0] == 4:
        img = np.transpose(img, axes=(1, 2, 0))
    if img.shape[2] == 4:  # RGBA to RGB
        img = img[:, :, :3]
    if np.max(img) > 1000:
        img = img.astype(np.float32)
        img /= 2**8
        img = img.astype(np.uint8)

    return NumpyWSI(img)


class STSample:
    """The subset of HESTData (HEST 1.1.x) used to segment tissue and dump patches."""

    def __init__(self, adata, wsi: WSI, pixel_size: float):
        self.adata = adata
        self.wsi = wsi
        self.pixel_size = pixel_size
        self._tissue_contours = None

    @property
    def tissue_contours(self):
        """ Geodataframe of tissue contours polygons also contains a tissue_id column """
        if self._tissue_contours is None:
            raise Exception("No tissue segmentation attached to this sample, segment tissue first by calling `segment_tissue()` for this object")
        return self._tissue_contours

    def segment_tissue(self, fast_mode=False, target_pxl_size=1, patch_size_um=512,
                       model_name='deeplabv3_seg_v4.ckpt', batch_size=8, auto_download=True,
                       num_workers=8, thumbnail_width=2000, method: str = 'deep', weights_dir=None):
        """ Compute tissue mask and stores it in the current object (HESTData.segment_tissue, HEST 1.1.x) """
        if method not in ('deep', 'otsu'):
            raise ValueError(f"method must be 'deep' or 'otsu', got {method!r}")

        if method == 'deep':
            if weights_dir is None:
                weights_dir = default_weights_dir()
            # hestcore contacts the Hugging Face hub whenever auto_download is set, even if
            # the checkpoint is already there; only download when it is missing.
            auto_download = auto_download and not os.path.isfile(os.path.join(weights_dir, model_name))
            self._tissue_contours = segment_tissue_deep(
                self.wsi,
                self.pixel_size,
                fast_mode,
                target_pxl_size,
                patch_size_um,
                model_name,
                batch_size,
                auto_download,
                num_workers,
                weights_dir
            )
        elif method == 'otsu':
            width, height = self.wsi.get_dimensions()
            scale = thumbnail_width / width
            thumbnail = self.wsi.get_thumbnail(round(width * scale), round(height * scale))
            mask = apply_otsu_thresholding(thumbnail).astype(np.uint8)
            mask = 1 - mask
            tissue_mask = np.round(cv2.resize(mask, (width, height))).astype(np.uint8)
            self._tissue_contours = mask_to_gdf(tissue_mask, pixel_size=self.pixel_size)

        return self.tissue_contours

    def dump_patches(self, patch_save_dir: str, name: str = 'patches', target_patch_size: int = 224,
                     target_pixel_size: float = 0.5, verbose=0, dump_visualization=True,
                     use_mask=True, threshold=0.15, coords_only=False):
        """ Dump H&E patches centered around ST spots to a .h5 file (HESTData.dump_patches, HEST 1.1.x).

        Each patch is rescaled to `target_pixel_size` um/px; a crop of
        `target_patch_size` x `target_patch_size` pixels around each spot (centres taken
        from adata.obsm['spatial']) is kept when at least `threshold` of it lies on tissue.
        """
        os.makedirs(patch_save_dir, exist_ok=True)

        dst_pixel_size = target_pixel_size

        adata = self.adata.copy()

        for index in adata.obs.index:
            if len(index) != len(adata.obs.index[0]):
                warnings.warn("indices of adata.obs should all have the same length to avoid problems when saving to h5", UserWarning)

        src_pixel_size = self.pixel_size

        patch_count = 0
        h5_path = os.path.join(patch_save_dir, name + '.h5')

        assert len(adata.obs) == len(adata.obsm['spatial'])

        patch_size_src = target_patch_size * (dst_pixel_size / src_pixel_size)
        coords_center = adata.obsm['spatial']
        coords_topleft = coords_center - patch_size_src // 2
        len_tmp = len(coords_topleft)
        in_slide_mask = (0 <= coords_topleft[:, 0] + patch_size_src) & (coords_topleft[:, 0] < self.wsi.width) & (0 <= coords_topleft[:, 1] + patch_size_src) & (coords_topleft[:, 1] < self.wsi.height)
        coords_topleft = coords_topleft[in_slide_mask]
        if len(coords_topleft) < len_tmp:
            warnings.warn(f"Filtered {len_tmp - len(coords_topleft)} spots outside the WSI")

        barcodes = np.array(adata.obs.index)
        barcodes = barcodes[in_slide_mask]
        mask = self.tissue_contours if use_mask else None
        coords_topleft = np.array(coords_topleft).astype(int)
        patcher = self.wsi.create_patcher(target_patch_size, src_pixel_size, dst_pixel_size,
                                          mask=mask, custom_coords=coords_topleft, threshold=threshold, coords_only=coords_only)

        if mask is not None:
            valid_barcodes = barcodes[patcher.valid_mask]
        else:
            valid_barcodes = barcodes

        patcher.to_h5(h5_path, extra_assets={'barcode': valid_barcodes})

        if dump_visualization:
            patcher.save_visualization(os.path.join(patch_save_dir, name + '_patch_vis.png'), dpi=400)

        if verbose:
            print(f'found {patch_count} valid patches')


ACCEPTED_IMG_FORMATS = ['.tif', '.jpg', '.btf', '.png', '.tiff', '.TIF', 'ndpi', 'nd2']
EXCLUDED_IMGS = ['aligned_fullres_HE.ome.tif', 'morphology.ome.tif',
                 'morphology_focus.ome.tif', 'morphology_mip.ome.tif']


def _find_first_file_endswith(dir: str, suffix: str, exclude: str = '', anywhere: bool = False):
    """first entry of `dir` (top level only) whose name ends with `suffix`"""
    if dir is None:
        return None
    files_dir = os.listdir(dir)
    if anywhere:
        matching = [f for f in files_dir if suffix in f and f != exclude]
    else:
        matching = [f for f in files_dir if f.endswith(suffix) and f != exclude]
    return os.path.join(dir, matching[0]) if matching else None


def _find_biggest_img(path: str) -> str:
    """filename of the biggest image (of an accepted format) in the `path` directory"""
    biggest_size = -1
    biggest = None
    for file in os.listdir(path):
        if any(file.endswith(s) for s in ACCEPTED_IMG_FORMATS) and file not in EXCLUDED_IMGS:
            size = os.path.getsize(os.path.join(path, file))
            if size > biggest_size:
                biggest, biggest_size = file, size
    if biggest is None:
        raise Exception(f"Couldn't find an image automatically, make sure that the folder {path} "
                        f"contains an image of one of these types: {ACCEPTED_IMG_FORMATS}")
    return biggest


def _read_positions_old(tissue_position_list_path):
    tp = pd.read_csv(tissue_position_list_path, header=None, sep=",", na_filter=False, index_col=0)
    return tp.rename(columns={1: "in_tissue", 2: "array_row", 3: "array_col",
                              4: "pxl_row_in_fullres", 5: "pxl_col_in_fullres"})


def _detect_alignment_file(path):
    """The files whose presence makes HEST 1.1.x leave the plain tissue-positions branch."""
    p = _find_first_file_endswith(path, 'alignment_file.json')
    if p is None:
        p = _find_first_file_endswith(path, 'alignment.json')
    if p is None:
        p = _find_first_file_endswith(path, 'alignment', anywhere=True)
    spatial = _find_first_file_endswith(path, 'spatial')
    if p is None and spatial is not None and os.path.exists(spatial):
        p = _find_first_file_endswith(spatial, 'autoalignment.json')
    if p is None:
        j = _find_first_file_endswith(path, '.json')
        if j is not None:
            with open(j) as f:
                if 'oligo' in json.load(f):
                    p = j
    return p


def _read_visium_spots(path: str):
    """Read expression + spot table of a raw Visium (SpaceRanger) sample directory.

    Returns:
        (adata, image_path): `adata.obs` holds in_tissue/array_row/array_col/pxl_row_in_fullres/
        pxl_col_in_fullres (int), `adata.obsm['spatial']` = (pxl_col, pxl_row), rows follow the
        expression matrix barcodes.
    """
    import scanpy as sc

    img_path = os.path.join(path, _find_biggest_img(path))

    align = _detect_alignment_file(path)
    if align is not None:
        raise NotImplementedError(
            f"{path}: found alignment file {align}; HEST 1.1.x takes a different (alignment-based) "
            "branch for such samples, which is not ported. Refusing to process it differently.")

    spatial_dir = _find_first_file_endswith(path, 'spatial')
    search = [d for d in (spatial_dir, path) if d is not None]

    def _first(suffix, exclude=''):
        for d in search:
            p = _find_first_file_endswith(d, suffix, exclude=exclude)
            if p is not None:
                return p
        return None

    tp_new = _first('tissue_positions.csv', exclude='aligned_tissue_positions.csv')
    tp_old = _first('tissue_positions_list.csv')
    if tp_new is None and tp_old is None:
        raise NotImplementedError(
            f"{path}: no tissue_positions(.csv/_list.csv); HEST 1.1.x would auto-align the fiducials, "
            "which is not ported.")

    # --- expression (same precedence as HEST: filtered h5, raw h5, mex)
    filtered = _find_first_file_endswith(path, 'filtered_feature_bc_matrix.h5')
    raw_h5 = _find_first_file_endswith(path, 'raw_feature_bc_matrix.h5')
    mex = _find_first_file_endswith(path, 'mex')
    if filtered is not None:
        adata = sc.read_10x_h5(filtered)
    elif raw_h5 is not None:
        adata = sc.read_10x_h5(raw_h5)
    else:
        if mex is None:
            has_top = any(_find_first_file_endswith(path, s) for s in ('matrix.mtx.gz', 'matrix.mtx'))
            mex = path if has_top else None
        if mex is None:
            raise ValueError(f"Couldn't find gene expressions in {path}: expected filtered/raw "
                             "feature_bc_matrix.h5 or a mex folder")
        adata = sc.read_10x_mtx(mex)
    adata.var_names_make_unique()

    # --- barcodes
    adata.obs.index = [idx[:18] for idx in adata.obs.index]
    if not adata.obs.index[0].endswith('-1'):
        adata.obs.index = [idx + '-1' for idx in adata.obs.index]

    # --- tissue positions
    if tp_new is not None:                                    # SpaceRanger >= 2.0
        tissue_positions = pd.read_csv(tp_new, sep=",", na_filter=False, index_col=0)
    else:                                                     # SpaceRanger < 2.0
        tissue_positions = _read_positions_old(tp_old)
    tissue_positions.index = [idx[:18] for idx in tissue_positions.index]

    spatial_aligned = tissue_positions.loc[adata.obs.index]
    assert np.array_equal(spatial_aligned.index, adata.obs.index)

    spatial_aligned = spatial_aligned.astype(int)  # header rows / stray dtypes: force int, as HEST 1.1.x
    adata.obsm['spatial'] = np.vstack((spatial_aligned['pxl_col_in_fullres'].values,
                                       spatial_aligned['pxl_row_in_fullres'].values)).T
    adata.obs = spatial_aligned
    return adata, img_path


def _read_xenium_aligned(path: str):
    """Returns (adata, image_path, pixel_size_um)."""
    import scanpy as sc

    adata_path = os.path.join(path, 'aligned_adata.h5ad')
    img_path = os.path.join(path, 'aligned_fullres_HE.tif')
    metrics_path = os.path.join(path, 'metrics.json')
    for p in (adata_path, img_path, metrics_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} not found (expected a Xenium_aligned sample directory)")

    adata = sc.read_h5ad(adata_path)
    if 'spatial' not in adata.obsm:
        raise KeyError(f"{adata_path} has no obsm['spatial']")
    with open(metrics_path) as f:
        metrics = json.load(f)
    return adata, img_path, float(metrics['pixel_size_um_estimated'])


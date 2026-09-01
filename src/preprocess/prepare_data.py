import os
import sys
import warnings
from glob import glob
from tqdm import tqdm

import argparse
import numpy as np
import pandas as pd
import h5py
from openslide import OpenSlide
import scanpy as sc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.utils.preprocess_utils import load_st, save_hdf5


MPP_TO_LEVEL = {
    0.5: 0,
    1.0: 1,
    2.0: 2,
    4.0: 3,
}


def _ensure_tif_aliases(hest_dir):
    """hest._read_st() hardcodes the WSI path as wsis/{id}.tif, with no
    fallback for other TIFF extensions. Some samples in shared datasets
    are stored as .tiff -- symlink a .tif alias next to each one so
    hest's lookup succeeds without duplicating the (often multi-GB) file."""
    wsi_dir = os.path.join(hest_dir, 'wsis')
    if not os.path.isdir(wsi_dir):
        return
    for entry in os.listdir(wsi_dir):
        if entry.lower().endswith('.tiff'):
            tif_alias = os.path.join(wsi_dir, entry[:-len('.tiff')] + '.tif')
            if not os.path.exists(tif_alias):
                os.symlink(entry, tif_alias)


def _iter_hest(*args, **kwargs):
    try:
        from hest import iter_hest
    except ImportError as exc:
        raise ImportError(
            "HEST preprocessing requires the 'hest' package. "
            "Install preprocessing dependencies with: pip install -r requirements/preprocess.txt"
        ) from exc
    hest_dir = args[0] if args else kwargs.get('hest_dir')
    if hest_dir:
        # hest._read_st raises UnboundLocalError for datasets without a
        # tissue_seg dir (tissue_contours_path is only assigned inside that
        # branch). An empty dir makes it resolve to None instead of crashing.
        os.makedirs(os.path.join(hest_dir, 'tissue_seg'), exist_ok=True)
        _ensure_tif_aliases(hest_dir)
    return iter_hest(*args, **kwargs)


def _load_hest_dataset(*args, **kwargs):
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(
            "Downloading HEST patches requires the Hugging Face 'datasets' package. "
            "Install preprocessing dependencies with: pip install -r requirements/preprocess.txt"
        ) from exc
    return datasets.load_dataset(*args, **kwargs)


def _read_barcodes(f):
    key = 'barcodes' if 'barcodes' in f else 'barcode'
    return key, f[key][:].squeeze()


def match_to_target(target_path, source_path):
    """Align source patches to target barcodes via intersection."""
    with h5py.File(target_path, 'r') as f:
        _, barcode_target = _read_barcodes(f)

    with h5py.File(source_path, 'r+') as f:
        has_img = 'img' in f
        source_img = f['img'][:] if has_img else None
        crd_source = f['coords'][:]
        coords_attrs = dict(f['coords'].attrs)
        key, barcode_source = _read_barcodes(f)

        _, idx_in_source = np.intersect1d(
            barcode_target, barcode_source, return_indices=True
        )[1:3]

        del f['coords']
        ds = f.create_dataset('coords', data=crd_source[idx_in_source])
        for k, v in coords_attrs.items():
            ds.attrs[k] = v

        if has_img:
            del f['img']
            f.create_dataset('img', data=source_img[idx_in_source])

        del f[key]
        f.create_dataset(key, data=np.expand_dims(barcode_source[idx_in_source], axis=-1))
        f.attrs['matched_to_target'] = True


def _native_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with all Arrow-backed dtypes converted to native numpy.

    pandas 3.0+ with infer_string=True re-infers string columns as StringDtype
    (Arrow-backed) both during DataFrame construction AND during column assignment
    from a numpy array.  The only reliable escape is:
      1. Convert each column to a numpy-typed array up front.
      2. Construct a fresh DataFrame (pandas re-infers strings → StringDtype here).
      3. Post-process: force any remaining non-numpy columns to object via
         Series.astype(object) — assigning a *Series* with explicit object dtype
         is respected without re-inference.
    """
    cols = {}
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, np.dtype):
            cols[c] = s.values
        else:
            try:
                cols[c] = s.to_numpy(dtype=s.dtype.numpy_dtype, na_value=0)
            except (AttributeError, NotImplementedError):
                cols[c] = s.to_numpy(dtype=object, na_value=None)
            except Exception:
                cols[c] = np.array(s.tolist(), dtype=object)
    result = pd.DataFrame(cols, index=pd.Index(df.index.tolist(), dtype=object))
    for c in list(result.columns):
        if not isinstance(result[c].dtype, np.dtype):
            result[c] = result[c].astype(object)
    return result


def _patch_barcodes(output_dir, name):
    """Return patch barcodes for a sample as a list of str, or None if not found."""
    for fname in (f"{name}.h5", f"{name}_patches.h5"):
        path = os.path.join(output_dir, "patches", fname)
        if os.path.isfile(path):
            with h5py.File(path, "r") as f:
                _, bc = _read_barcodes(f)
            return bc.astype(str).tolist()
    return None


def _n_obs_h5ad(h5ad_path):
    """Return number of observations in an h5ad file without loading X."""
    with h5py.File(h5ad_path, "r") as f:
        obs = f["obs"]
        idx_key = obs.attrs.get("_index", "_index")
        if idx_key in obs:
            return len(obs[idx_key])
        for key in obs.keys():
            if isinstance(obs[key], h5py.Dataset):
                return len(obs[key])
    return None


def _h5ad_obs_names(h5ad_path) -> list:
    """Read obs index (barcode list) from h5ad without loading X or var."""
    with h5py.File(h5ad_path, "r") as f:
        obs = f["obs"]
        idx_key = obs.attrs.get("_index", "_index")
        if idx_key in obs:
            return obs[idx_key][:].astype(str).tolist()
    return []


def _is_aligned(h5ad_path, barcodes: list) -> bool:
    """Return True if the h5ad obs index already exactly matches barcodes."""
    existing = _h5ad_obs_names(h5ad_path)
    return existing == barcodes


def _write_aligned_adata(adata, save_dir):
    try:
        sc.AnnData(
            X=adata.X,
            obs=_native_df(adata.obs),
            var=_native_df(adata.var),
        ).write(save_dir)
    except Exception:
        if os.path.exists(save_dir):
            os.remove(save_dir)
        raise


def preprocess_st(name, adata, output_dir):
    """Filter and save ST data aligned to patch barcodes."""
    os.makedirs(f"{output_dir}/st", exist_ok=True)
    save_dir = f"{output_dir}/st/{name}.h5ad"

    barcodes = _patch_barcodes(output_dir, name)

    if os.path.exists(save_dir):
        if barcodes is None:
            print(f"ST data already exists for {name}. Skipping...")
            return None
        n_existing = _n_obs_h5ad(save_dir)
        if n_existing == len(barcodes):
            print(f"ST data already exists for {name}. Skipping...")
            return None
        print(f"ST data exists but spot count mismatch ({n_existing} vs {len(barcodes)} patches). Re-aligning {name}...")
        adata = sc.read_h5ad(save_dir)
    else:
        if barcodes is None:
            raise FileNotFoundError(f"Patch file not found for {name} in {output_dir}/patches/")
        print(f"Saving ST data for {name}...")

    valid = [b for b in barcodes if b in adata.obs_names]
    adata = adata[valid].copy()
    _write_aligned_adata(adata, save_dir)
    return adata


def align_st_to_patches(output_dir, sample_ids=None, overwrite=False):
    """Align h5ad files in output_dir/st/ to their patch barcodes and re-save in-place.

    sample_ids: if given, only process those samples; otherwise process all *.h5ad files.
    """
    st_dir_path = os.path.join(output_dir, "st")
    if not os.path.isdir(st_dir_path):
        return

    if sample_ids is not None:
        h5ad_files = [
            os.path.join(st_dir_path, f"{sid}.h5ad")
            for sid in sample_ids
            if os.path.isfile(os.path.join(st_dir_path, f"{sid}.h5ad"))
        ]
    else:
        h5ad_files = sorted(glob(os.path.join(st_dir_path, "*.h5ad")))
    if not h5ad_files:
        return

    aligned = 0
    skipped = 0
    for h5ad_path in tqdm(h5ad_files, desc="Aligning ST to patches"):
        name = os.path.splitext(os.path.basename(h5ad_path))[0]
        barcodes = _patch_barcodes(output_dir, name)
        if barcodes is None:
            skipped += 1
            continue

        if not overwrite and _is_aligned(h5ad_path, barcodes):
            skipped += 1
            continue

        adata = sc.read_h5ad(h5ad_path)
        valid = [b for b in barcodes if b in adata.obs_names]
        adata = adata[valid].copy()
        _write_aligned_adata(adata, h5ad_path)
        aligned += 1

    print(f"Aligned {aligned} ST files ({skipped} already aligned / no patch file).")


def _read_sample_ids(path):
    if not path or not os.path.isfile(path):
        return None
    return pd.read_csv(path)['sample_id'].dropna().astype(str).tolist()


def _sample_ids_from_patch_dir(patch_dir):
    if not patch_dir or not os.path.isdir(patch_dir):
        return None
    patch_paths = glob(f"{patch_dir}/*.h5")
    if not patch_paths:
        return None
    return [
        os.path.splitext(os.path.basename(p))[0].replace("_patches", "")
        for p in patch_paths
    ]


def _resolve_hest_ids(input_dir, output_dir, meta_dir=None):
    """Resolve sample IDs for stpbench-mode preprocessing — requires ids.csv in meta_dir."""
    ids_path = f"{meta_dir}/ids.csv" if meta_dir else None
    if not ids_path or not os.path.isfile(ids_path):
        raise FileNotFoundError(
            "[stpbench mode] ids.csv must be prepared before running preprocessing.\n"
            f"  Expected: {ids_path or '<meta_dir>/ids.csv'}\n"
            "  Create a CSV with a 'sample_id' column listing the samples for this dataset."
        )
    ids = _read_sample_ids(ids_path)
    if not ids:
        raise ValueError(
            f"ids.csv exists but contains no sample IDs: {ids_path}\n"
            "Ensure the file has a 'sample_id' column with at least one entry."
        )
    return ids


def _read_hest_adata(sample_id, input_dir):
    """Read ST h5ad from a hest-format directory."""
    for subdir in ['st', 'adata']:
        path = f"{input_dir}/{subdir}/{sample_id}.h5ad"
        if os.path.exists(path):
            return sc.read_h5ad(path)
    raise FileNotFoundError(f"ST file not found for {sample_id} in {input_dir}")


def _segment_tissue_safe(st):
    """Segment tissue, preferring the deep-learning model but falling back
    to Otsu thresholding when the WSI backend can't support it.

    cuCIM's read_region (used when the `cucim` package is installed and a
    slide can't be opened by OpenSlide, e.g. a non-tiled/non-pyramidal
    TIFF) refuses any read that extends past the slide's bounds, unlike
    OpenSlide's own read_region which auto-pads. Deep tissue segmentation
    tiles the whole slide on a fixed grid, so its last row/column of tiles
    routinely overruns the slide edge by design -- this is normal and
    OpenSlide silently handles it, but cuCIM raises `ValueError: Cannot
    handle the out-of-boundary cases`. Otsu thresholding works from a
    downsampled thumbnail instead of full-res tiles, so it doesn't hit
    this path at all.

    Separately, trident's CuCIMWSI.segment_tissue() calls self.close()
    once segmentation finishes (trident's own pipeline always reopens the
    WSI fresh for each stage), which sets wsi.img = None. HESTData reuses
    the same wsi object for dump_patches() right after segmentation, so
    that later read crashes with `AttributeError: 'NoneType' object has
    no attribute 'read_region'` unless the handle is reopened here first.
    close() also resets _initialized, so _lazy_initialize() cleanly
    reopens it; it's a no-op for backends (e.g. OpenSlideWSI) that don't
    close on segment_tissue().
    """
    try:
        st.segment_tissue(method='deep')
        st.wsi._lazy_initialize()
    except ValueError as exc:
        if 'out-of-boundary' not in str(exc):
            raise
        warnings.warn(
            "Deep tissue segmentation failed with a cuCIM out-of-boundary "
            "read (see docstring of _segment_tissue_safe) -- falling back "
            "to Otsu thresholding for this slide.",
            stacklevel=2,
        )
        st.segment_tissue(method='otsu')


def save_patches(name, input_dir, output_dir, platform='visium',
                 save_targets=True, save_neighbors=False,
                 num_n=25, dst_pixel_size=0.5, save_neighbor_imgs=False):
    """Extract and save patches for a single sample."""
    print("Loading ST data...")
    if platform == 'hest':
        st = [st for st in _iter_hest(input_dir, id_list=[name])][0]
    else:
        try:
            st = load_st(f"{input_dir}/{name}", platform=platform)
        except Exception as exc:
            print(f"Failed to load ST data for {name}: {exc}. Skipping.")
            return None

    level = MPP_TO_LEVEL[dst_pixel_size]

    if save_targets:
        if level == 0 and os.path.exists(f"{output_dir}/patches/{name}.h5"):
            print("Target patches already exist. Skipping...")
        else:
            if st._tissue_contours is None:
                print("Segmenting tissue...")
                _segment_tissue_safe(st)

            if level == 0:
                print("Dumping target patches...")
                st.dump_patches(
                    f"{output_dir}/patches",
                    name=name,
                    target_patch_size=224,
                    target_pixel_size=dst_pixel_size,
                    dump_visualization=False,
                )
            else:
                st.dump_patches(
                    f"{output_dir}/patches/level{level}",
                    name=name,
                    target_patch_size=224,
                    target_pixel_size=dst_pixel_size,
                    use_mask=False,
                    dump_visualization=False,
                )
                target_path = f"{output_dir}/patches/{name}.h5"
                if not os.path.exists(target_path):
                    raise FileNotFoundError(f"Target patch file not found: {target_path}")
                source_path = f"{output_dir}/patches/level{level}/{name}.h5"
                print("Matching lower resolution patches to target patches...")
                match_to_target(target_path, source_path)

    if save_neighbors:
        if os.path.exists(f"{output_dir}/patches/neighbor/{name}.h5"):
            print("Neighbor patches already exist. Skipping...")
        else:
            if st._tissue_contours is None:
                print("Segmenting tissue...")
                _segment_tissue_safe(st)

            n = int(np.sqrt(num_n))
            print("Dumping neighbor patches...")
            st.dump_patches(
                f"{output_dir}/patches/neighbor",
                name=name,
                target_patch_size=224 * n,
                target_pixel_size=dst_pixel_size,
                use_mask=False,
                dump_visualization=False,
                # When the images aren't going to be kept, skip creating
                # them in the first place (coords_only=True) instead of
                # reading/writing the full image array here and then
                # deleting it right after -- for a neighbor grid this is a
                # large array (target_patch_size**2 per spot), so building
                # and immediately discarding it wastes significant time and
                # memory for no benefit.
                coords_only=not save_neighbor_imgs,
            )
            target_path = f"{output_dir}/patches/{name}.h5"
            neighbor_path = f"{output_dir}/patches/neighbor/{name}.h5"
            print("Matching neighbor patches to target patches...")
            match_to_target(target_path, neighbor_path)

    return st


def save_image(slide_path, patch_path, slide_level=0, patch_size=256):
    """Read image patches from a WSI and append to an existing patch file."""
    with h5py.File(patch_path, 'r') as f:
        coords = f['coords'][:]

    wsi = OpenSlide(slide_path)
    imgs = np.stack([
        np.array(wsi.read_region(coord, slide_level, (patch_size, patch_size)).convert('RGB'))
        for coord in coords
    ])
    save_hdf5(patch_path, asset_dict={'img': imgs}, mode='a')


_WSI_EXTENSIONS = (
    '.svs', '.ndpi', '.tif', '.tiff', '.btf', '.mrxs', '.scn', '.vms',
    '.vmu', '.bif', '.qptiff',
)


def discover_wsi_files(input_dir, wsi_ext=_WSI_EXTENSIONS):
    """Return sorted absolute paths of WSI files directly under input_dir."""
    paths = []
    for entry in sorted(os.listdir(input_dir)):
        if entry.lower().endswith(tuple(e.lower() for e in wsi_ext)):
            paths.append(os.path.join(input_dir, entry))
    return paths


def extract_patches_from_wsi(wsi_path, output_dir, name=None, dst_pixel_size=0.5,
                              patch_size=224, seg_model_name='hest', seg_target_mag=10,
                              device='cuda:0', overwrite=False):
    """Segment tissue and extract a coords-only patch grid from a bare WSI
    file with no ST companion data.

    Unlike save_patches() (which always goes through a HEST ST reader —
    hest.HESTData.dump_patches() is spot-centric and needs adata.obsm['spatial'],
    so it cannot tile a spot-free slide), this calls trident's own
    WSI.segment_tissue()/extract_tissue_coords() directly. The resulting
    patches/<name>_patches.h5 carries the same patch_size/level0_magnification/
    target_magnification attrs that STDataset._get_patcher() already reads via
    trident's own read_coords() as its primary (non-legacy) path, so nothing
    downstream (feature extraction, model-specific extra_preprocess) needs to
    know this patch file came from a WSI-only source.
    """
    from trident import load_wsi
    from trident.segmentation_models import segmentation_model_factory

    wsi = load_wsi(slide_path=wsi_path, lazy_init=False)
    name = name or wsi.name
    out_path = os.path.join(output_dir, 'patches', f'{name}_patches.h5')
    if os.path.isfile(out_path) and not overwrite:
        print(f"{out_path} exists, skip!")
        return out_path

    # 20x <-> 0.5 um/px is the scanner convention already implicit in
    # MPP_TO_LEVEL; derive trident's magnification-based target from the
    # repo's own pixel-size convention rather than hardcoding a magnification.
    target_mag = round(20 * (0.5 / dst_pixel_size))

    seg_model = segmentation_model_factory(seg_model_name)
    job_dir = os.path.join(output_dir, '_seg_job')
    wsi.segment_tissue(seg_model, target_mag=seg_target_mag, job_dir=job_dir, device=device)
    coords_path = wsi.extract_tissue_coords(
        target_mag=target_mag, patch_size=patch_size, save_coords=output_dir,
    )
    return coords_path


def _load_patch_centers(coords_path):
    """Read user-supplied patch-CENTER coordinates from an .h5ad
    (obsm['spatial']) or .csv (x, y columns) file."""
    ext = os.path.splitext(coords_path)[1].lower()
    if ext == '.h5ad':
        adata = sc.read_h5ad(coords_path)
        if 'spatial' not in adata.obsm:
            raise ValueError(
                f"{coords_path} has no obsm['spatial'] — cannot read patch-center coordinates from it."
            )
        return np.asarray(adata.obsm['spatial'], dtype=np.float64)
    if ext == '.csv':
        df = pd.read_csv(coords_path)
        if not {'x', 'y'}.issubset(df.columns):
            raise ValueError(
                f"{coords_path} must have 'x' and 'y' columns; found: {list(df.columns)}"
            )
        return df[['x', 'y']].to_numpy(dtype=np.float64)
    raise ValueError(f"Unsupported coordinates file type {ext!r} for {coords_path}; use .h5ad or .csv.")


def extract_patches_from_coords_file(wsi_path, output_dir, coords_path, name=None,
                                      dst_pixel_size=0.5, patch_size=224, overwrite=False):
    """Skip tissue segmentation entirely — crop patches centered at
    user-supplied coordinates instead of an automatically tiled tissue grid.

    `coords_path` gives patch CENTERS (an .h5ad's obsm['spatial'], or a .csv
    with x/y columns), converted here to the level-0 top-left-corner format
    trident's coords .h5 files store. Writes the exact same shape
    extract_patches_from_wsi does (same 'coords' dataset + patch_size/
    level0_magnification/target_magnification/... attrs, via trident's own
    coords_to_h5), so nothing downstream needs to know the difference.
    """
    from trident import load_wsi
    from trident.IO import coords_to_h5

    wsi = load_wsi(slide_path=wsi_path, lazy_init=False)
    name = name or wsi.name
    out_path = os.path.join(output_dir, 'patches', f'{name}_patches.h5')
    if os.path.isfile(out_path) and not overwrite:
        print(f"{out_path} exists, skip!")
        return out_path

    # Same convention as extract_patches_from_wsi.
    target_mag = round(20 * (0.5 / dst_pixel_size))
    # Matches coords_to_h5's own 'patch_size_level0' derivation exactly, so
    # the top-left corners land where the caller's centers actually are.
    patch_size_src = patch_size * wsi.mag // target_mag

    centers = _load_patch_centers(coords_path)
    top_left = np.round(centers - patch_size_src / 2).astype(np.int64)

    os.makedirs(os.path.join(output_dir, 'patches'), exist_ok=True)
    coords_to_h5(
        top_left, out_path, patch_size, wsi.mag, target_mag,
        output_dir, wsi.width, wsi.height, name, overlap=0,
    )
    return out_path


def extract_neighbor_patches_from_wsi(wsi_path, output_dir, name, dst_pixel_size=0.5,
                                       patch_size=224, num_n=5, overwrite=False):
    """Build a neighbor-grid coords file for a slide whose target patches
    were just extracted from a raw WSI (mode='inference', extract_from_wsi=
    True) — the 'raw'/'stpbench' modes get this via HEST's own dump_patches(),
    but a bare WSI has no HESTData/adata to drive that, so neighbor centers
    are re-derived here directly from the target patch file's own coords.

    Mirrors extract_patches_from_coords_file's coords_to_h5 call shape, just
    with centers/size computed from the existing target patches instead of
    an external coords file.
    """
    from trident.IO import read_coords, coords_to_h5

    out_path = os.path.join(output_dir, 'patches', 'neighbor', f'{name}_patches.h5')
    if os.path.isfile(out_path) and not overwrite:
        print(f"{out_path} exists, skip!")
        return out_path

    target_path = os.path.join(output_dir, 'patches', f'{name}_patches.h5')
    attrs, coords = read_coords(target_path)

    n = int(np.sqrt(num_n))
    patch_size_level0 = attrs['patch_size_level0']
    patch_size_level0_neighbor = patch_size_level0 * n

    center = coords + patch_size_level0 / 2
    top_left_neighbor = np.round(center - patch_size_level0_neighbor / 2).astype(np.int64)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    coords_to_h5(
        top_left_neighbor, out_path, patch_size * n,
        attrs['level0_magnification'], attrs['target_magnification'],
        output_dir, attrs['level0_width'], attrs['level0_height'], name, overlap=0,
    )
    return out_path


def _write_ids_from_patch_dir(output_dir, meta_dir=None, sample_ids=None):
    """Write ids.csv from patch filenames.

    Shared by mode='inference' with extract_from_wsi=False (patches already
    extracted by some other means — pass sample_ids=None to scan
    output_dir/patches/*.h5 for everything present, since that whole
    directory IS the target) and mode='inference' with extract_from_wsi=True
    (patches just extracted by extract_patches_from_wsi/
    extract_patches_from_coords_file, above). For the latter, the caller
    MUST pass the explicit sample_ids this call actually processed —
    output_dir/patches/ is commonly reused across separate predict() calls
    on different slides, so scanning the whole directory would silently
    pull in unrelated samples left over from an earlier call.
    """
    manifest_dir = meta_dir or output_dir
    if sample_ids is None:
        sample_ids = _sample_ids_from_patch_dir(f"{output_dir}/patches") or []
    os.makedirs(manifest_dir, exist_ok=True)
    pd.DataFrame(sample_ids, columns=['sample_id']).to_csv(f"{manifest_dir}/ids.csv", index=False)
    return sample_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--meta_dir", type=str, default=None)
    parser.add_argument("--platform", type=str, default='visium')
    parser.add_argument("--prefix", type=str, default='')
    parser.add_argument("--mode", type=str, default='raw', choices=['raw', 'stpbench', 'inference'])
    parser.add_argument("--extract_from_wsi", action='store_true', default=False)
    parser.add_argument("--overwrite", action='store_true', default=False)
    parser.add_argument("--slide_level", type=int, default=0)
    parser.add_argument("--slide_ext", type=str, default='.svs')
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--num_n", type=int, default=5)
    parser.add_argument("--dst_pixel_size", type=float, default=0.5)
    parser.add_argument("--save_neighbors", action='store_true', default=False)
    parser.add_argument("--save_neighbor_imgs", action='store_true', default=False)
    parser.add_argument("--coords_path", type=str, default=None)

    args = parser.parse_args()

    mode = args.mode
    input_dir = args.input_dir
    output_dir = args.output_dir
    meta_dir = args.meta_dir
    platform = args.platform
    prefix = args.prefix
    dst_pixel_size = args.dst_pixel_size
    level = MPP_TO_LEVEL[dst_pixel_size]

    if mode == 'raw':
        manifest_dir = meta_dir or output_dir
        os.makedirs(manifest_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/patches", exist_ok=True)
        if args.save_neighbors:
            os.makedirs(f"{output_dir}/patches/neighbor", exist_ok=True)
        os.makedirs(f"{output_dir}/st", exist_ok=True)

        if not os.path.exists(f"{manifest_dir}/ids.csv"):
            ids = [os.path.basename(p) for p in glob(f"{input_dir}/{prefix}*")]
            pd.DataFrame(ids, columns=['sample_id']).to_csv(f"{manifest_dir}/ids.csv", index=False)
        else:
            ids = pd.read_csv(f"{manifest_dir}/ids.csv")['sample_id'].tolist()

        for name in tqdm(ids):
            name = os.path.basename(name)
            st = save_patches(name, input_dir, output_dir,
                              platform=platform,
                              save_neighbors=args.save_neighbors,
                              dst_pixel_size=dst_pixel_size,
                              save_neighbor_imgs=args.save_neighbor_imgs)
            if st is not None:
                preprocess_st(name, st.adata, output_dir)

    elif mode == 'stpbench':
        os.makedirs(f"{output_dir}/patches", exist_ok=True)
        os.makedirs(f"{output_dir}/st", exist_ok=True)

        manifest_dir = meta_dir or output_dir
        sample_ids = _resolve_hest_ids(input_dir, output_dir, meta_dir=manifest_dir)
        os.makedirs(manifest_dir, exist_ok=True)
        pd.DataFrame(sample_ids, columns=['sample_id']).to_csv(f"{manifest_dir}/ids.csv", index=False)

        for name in tqdm(sample_ids):
            if level != 0:
                os.makedirs(f"{output_dir}/patches/level{level}", exist_ok=True)
                save_patches(name, input_dir, output_dir,
                             platform='hest',
                             dst_pixel_size=dst_pixel_size)

            if args.save_neighbors:
                os.makedirs(f"{output_dir}/patches/neighbor", exist_ok=True)
                save_patches(name, input_dir, output_dir,
                             platform='hest',
                             save_targets=False,
                             save_neighbors=True,
                             num_n=args.num_n,
                             save_neighbor_imgs=args.save_neighbor_imgs)

            adata = _read_hest_adata(name, input_dir)
            preprocess_st(name, adata, output_dir)

    elif mode == 'inference' and not args.extract_from_wsi:
        # Patches already exist under input_dir (that's the whole premise
        # of this case — nothing gets extracted here). output_dir is a
        # separate, writable location for the refreshed ids.csv (and any
        # later feature extraction) — it may not contain the patches at
        # all, e.g. when input_dir is a shared/read-only asset dir.
        _write_ids_from_patch_dir(input_dir, meta_dir=meta_dir)

    elif mode == 'inference' and args.extract_from_wsi:
        os.makedirs(f"{output_dir}/patches", exist_ok=True)
        if args.save_neighbors:
            os.makedirs(f"{output_dir}/patches/neighbor", exist_ok=True)

        if os.path.isfile(input_dir) or input_dir.lower().endswith(_WSI_EXTENSIONS):
            wsi_paths = [input_dir]
        else:
            wsi_paths = discover_wsi_files(input_dir)
            if not wsi_paths:
                raise FileNotFoundError(f"No WSI files found under {input_dir}")

        if args.coords_path and len(wsi_paths) > 1:
            raise ValueError(
                "--coords_path only applies to a single WSI file, not a directory "
                f"of {len(wsi_paths)} slides — coordinates are tied to one slide's pixel space."
            )

        # Collect exactly the sample IDs THIS call processed, rather than
        # scanning output_dir/patches/* afterward — that directory is
        # commonly reused across separate predict() calls on different
        # slides, and a directory-wide scan would silently pull in
        # unrelated samples left over from an earlier call.
        sample_ids = []
        for wsi_path in tqdm(wsi_paths):
            if args.coords_path:
                out_path = extract_patches_from_coords_file(
                    wsi_path, output_dir, args.coords_path,
                    dst_pixel_size=dst_pixel_size,
                    patch_size=args.patch_size,
                    overwrite=args.overwrite,
                )
            else:
                out_path = extract_patches_from_wsi(
                    wsi_path, output_dir,
                    dst_pixel_size=dst_pixel_size,
                    patch_size=args.patch_size,
                    overwrite=args.overwrite,
                )
            name = os.path.splitext(os.path.basename(out_path))[0].replace('_patches', '')
            sample_ids.append(name)

            if args.save_neighbors:
                extract_neighbor_patches_from_wsi(
                    wsi_path, output_dir, name,
                    dst_pixel_size=dst_pixel_size,
                    patch_size=args.patch_size,
                    num_n=args.num_n,
                    overwrite=args.overwrite,
                )

        _write_ids_from_patch_dir(output_dir, meta_dir=meta_dir, sample_ids=sample_ids)


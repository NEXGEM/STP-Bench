import warnings
import os 
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py
import cv2
import tifffile as tifi

from trident.IO import read_coords, read_coords_legacy
from trident.wsi_objects.WSIFactory import load_wsi




class _TiffWSI:
    """Minimal stand-in for plain (single-level, non-pyramidal, uncompressed) TIFF slides that OpenSlide cannot open."""
    mag = None
    level_downsamples = None

    def __init__(self, path):
        import tifffile
        try:
            self.arr = tifffile.memmap(path)
        except Exception:
            self.arr = tifffile.imread(path)
        self.height, self.width = self.arr.shape[:2]

    def get_dimensions(self):
        return (self.width, self.height)

    def create_patcher(self, patch_size, src_mag=None, dst_mag=None, custom_coords=None, coords_only=False, pil=True, **kw):
        # trident coords format: patch_size is at dst_mag, level 0 is src_mag -> crop patch_size * src_mag / dst_mag px, resize to patch_size
        src = int(round(patch_size * (float(src_mag) / float(dst_mag)))) if src_mag and dst_mag else int(patch_size)
        return _TiffPatcher(self, custom_coords, src, patch_size)


class _TiffPatcher:
    """patcher[i] -> (PIL tile of size target, x, y); coords are level-0 top-left, src crop size patch_size_src -> resized to patch_size_target"""
    def __init__(self, wsi, coords, src, target):
        self.wsi, self.coords, self.src, self.target = wsi, np.asarray(coords), int(src), int(target)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        x, y = int(self.coords[i][0]), int(self.coords[i][1])
        a = self.wsi.arr
        H, W = a.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + self.src, W), min(y + self.src, H)
        tile = np.full((self.src, self.src, 3), 255, dtype=np.uint8)
        if x1 > x0 and y1 > y0:
            tile[y0 - y:y1 - y, x0 - x:x1 - x] = np.asarray(a[y0:y1, x0:x1, :3])
        im = Image.fromarray(tile)
        if self.src != self.target:
            im = im.resize((self.target, self.target), Image.BILINEAR)
        return im, x, y

class H5TileDataset(Dataset):
    def __init__(self, 
                h5_path, 
                mode='inference',
                sample_id=None,
                wsi_dir=None, 
                ext='.tif', 
                level=1, 
                img_transform=None, 
                num_n=1, 
                feature_type='global',
                radius=112, 
                chunk_size=1000, 
                num_workers=6):

        self.h5_path = h5_path
        self.chunk_size = chunk_size
        self.level = level
        self.feature_type = feature_type
        self.wsi_loaded = 0
        self.num_workers = num_workers
        self.use_openslide = False if 'tif' in ext else False
        
        if wsi_dir.lower() == 'none':
            wsi_dir = None 
        
        n = int(np.sqrt(num_n))
        self.n = n
        assert n % 2 == 1, "n must be odd number"
        
        if sample_id is None:
            sample_id = os.path.basename(h5_path).split('.h5')[0]
        
        with h5py.File(h5_path, 'r') as _f:
            _has_img = 'img' in _f

        if wsi_dir is not None and not _has_img:
            # WSI is only needed when patch images are not pre-saved in the h5
            wsi_path = self.get_wsi_path(wsi_dir, sample_id, ext)
            if wsi_path is None:
                wsi_path = self.get_wsi_path(wsi_dir, sample_id, '.tif')

            if wsi_path is None:
                raise FileNotFoundError(f"WSI file for sample_id {sample_id} not found in {wsi_dir} with extension {ext} or .tif")

            try:
                self.wsi = load_wsi(wsi_path, lazy_init=False)
            except Exception as _e:   # plain TIFF that OpenSlide cannot read
                if not wsi_path.lower().endswith(('.tif', '.tiff')):
                    raise
                self.wsi = _TiffWSI(wsi_path)
            if feature_type == 'neighbor':
                self.patcher = self._get_patcher(h5_path, n=self.n)
            elif mode == 'inference':
                # No companion ST data, so the patch h5 may be coords-only
                # (no embedded 'img') -- a real patcher is needed to re-crop
                # pixels from the WSI.
                self.patcher = self._get_patcher(h5_path)
            else:
                self.patcher = None
            self._sync_wsi_metadata()
        else:
            self.wsi = None
            self.patcher = None
        
        self.img_transform = img_transform
        self.transformed = False
        
        if feature_type == 'neighbor':
            assert n > 1, "n must be greater than 1 for neighbor feature type"
        
        self.r = radius

        with h5py.File(h5_path, 'r') as f:
            self.total_length = len(f['coords'])
            self.n_chunks = int(np.ceil(self.total_length / chunk_size))

    def get_wsi_path(self, wsi_dir, sample_id, ext):
        if wsi_dir is not None:
            if os.path.isfile(f"{wsi_dir}/{sample_id}{ext}"):
                wsi_path = f"{wsi_dir}/{sample_id}{ext}"
            else:
                wsi_path = glob(f"{wsi_dir}/{sample_id}/*{ext}*")
                if len(wsi_path) == 0:
                    return None
                else:
                    wsi_path = wsi_path[0]

        return wsi_path

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = (idx + 1) * self.chunk_size
        if end_idx > self.total_length:
            end_idx = self.total_length

        with h5py.File(self.h5_path, 'r') as f:
            if 'barcodes' in f.keys():
                barcodes = f['barcodes'][start_idx:end_idx].flatten().tolist()
            elif 'barcode' in f.keys():
                barcodes = f['barcode'][start_idx:end_idx].flatten().tolist()
            else:
                barcodes = torch.zeros(self.n_chunks)

            coords = f['coords'][start_idx:end_idx]

            if 'img' in f.keys():
                imgs = f['img'][start_idx:end_idx]
            else:
                if self.feature_type in ['global', 'target']:
                    imgs = [self.patcher[idx][0] for idx in range(start_idx, end_idx)]
                    imgs = np.stack(imgs)
                else:
                    patches = [self.patcher[idx] for idx in range(start_idx, end_idx)]
                    imgs, mask_tb = self.get_neighbor(patches)
                    self.transformed = True


        if self.feature_type in ['global', 'target']:
            if not self.transformed:
                imgs = torch.stack([self.img_transform(Image.fromarray(img)) for img in imgs])
            return {'imgs': imgs, 'barcodes': barcodes, 'coords': coords}
        else:
            if not self.transformed:
                
                imgs_transformed = torch.zeros((imgs.shape[0], imgs.shape[3], imgs.shape[1], imgs.shape[2]))
                for i in range(imgs.shape[0]):
                    img = imgs[i]
                    for x in range(0, img.shape[0], self.r*2):
                        for y in range(0, img.shape[1], self.r*2):
                            imgs_transformed[i, :, x:x+self.r*2, y:y+self.r*2] = self.img_transform(Image.fromarray(img[x:x+self.r*2, y:y+self.r*2]))
                imgs = imgs_transformed
                
                if self.wsi is not None:
                    mask_tb = self.get_mask_tables(coords, self.wsi)
                else:
                    mask_tb = torch.ones((imgs.shape[0], self.n ** 2))
                
            # imgs = torch.stack([self.img_transform(Image.fromarray(img)) for img in imgs])
            # imgs, mask_tb = self.get_neighbor(coords, self.wsi) 
            
            return {'imgs': imgs, 'barcodes': barcodes, 'coords': coords, 'mask_tb':mask_tb}
        
    def _load_wsi(self, wsi_path):
        return load_wsi(wsi_path, lazy_init=False)

    def _sync_wsi_metadata(self):
        self.mag = getattr(self.wsi, 'mag', None)
        self.level_downsamples = getattr(self.wsi, 'level_downsamples', None)
    
    def _get_patcher(self, coords_path, n=None):
        try:
            coords_attrs, coords = read_coords(coords_path)
            patch_size = coords_attrs.get('patch_size', None)

            if n is not None and patch_size is not None:
                patch_size = patch_size * n
            
            level0_magnification = coords_attrs.get('level0_magnification', None)
            target_magnification = coords_attrs.get('target_magnification', None)            
            if None in (patch_size, level0_magnification, target_magnification):
                raise KeyError('Missing attributes in coords_attrs.')
        except (KeyError, FileNotFoundError, ValueError) as e:
            # HEST dump_patches format (img dropped): attrs patch_size_src / patch_size_target / pixel_size, coords = top-left (level 0)
            with h5py.File(coords_path, 'r') as _f:
                _a = dict(_f['coords'].attrs); _coords = _f['coords'][:]
            if all(k in _a for k in ('patch_size_src', 'patch_size_target', 'pixel_size')):
                if isinstance(self.wsi, _TiffWSI):
                    return _TiffPatcher(self.wsi, _coords, _a['patch_size_src'], _a['patch_size_target'])
                dst_px = float(_a['patch_size_src']) * float(_a['pixel_size']) / float(_a['patch_size_target'])
                return self.wsi.create_patcher(int(_a['patch_size_target']), float(_a['pixel_size']), dst_px,
                                               custom_coords=_coords, coords_only=False, pil=True)
            warnings.warn(f"Cannot read using Trident coords format ({str(e)}). Trying with CLAM/Fishing-Rod.")
            patch_size, patch_level, custom_downsample, coords = read_coords_legacy(coords_path)

            if n is not None:
                patch_size = patch_size * n

                _, patch_level, custom_downsample, coords = read_coords_legacy(coords_path)
            level0_magnification = self.mag
            target_magnification = int(self.mag / (self.level_downsamples[patch_level] * custom_downsample))
        
        patcher = self.wsi.create_patcher(
            patch_size=patch_size,
            src_mag=level0_magnification,
            dst_mag=target_magnification,
            custom_coords=coords,
            coords_only=False,
            pil=True,
        )
        
        return patcher
    
    def make_masking_table(self, x: int, y: int, img_shape: tuple):
        """Generate masking table for neighbor encoder.

        Args:
            x (int): x coordinate of target spot
            y (int): y coordinate of target spot
            img_shape (tuple): Shape of whole slide image

        Raises:
            Exception: if self.n is bigger than 5, raise error.

        Returns:
            torch.Tensor: masking table
        """
        
        # Make masking table for neighbor encoding module
        mask_tb = torch.ones(self.n**2)
        
        def create_mask(ind, mask_tb, window):
            if x-self.r*window < 0:
                mask_tb[self.n*ind:self.n*ind+self.n] = 0 
            if x+self.r*window > img_shape[0]:
                mask_tb[(self.n**2-self.n*(ind+1)):(self.n**2-self.n*ind)] = 0 
            if y-self.r*window < 0:
                mask = [i+ind for i in range(self.n**2) if i % self.n == 0]
                mask_tb[mask] = 0 
            if y+self.r*window > img_shape[1]:
                mask = [i-ind for i in range(self.n**2) if i % self.n == (self.n-1)]
                mask_tb[mask] = 0 
                
            return mask_tb
        
        ind = 0
        window = self.n
        while window >= 3: 
            mask_tb = create_mask(ind, mask_tb, window)
            ind += 1 
            window -= 2   

        return mask_tb
    
    def get_mask_tables(self, coords, wsi):
        n_patches = len(coords)
        
        mask_tb = torch.ones((n_patches, self.n**2))
        
        for i in range(n_patches):
            x, y = coords[i]

            wsi_shape = wsi.get_dimensions()
            mask = self.make_masking_table(x, y, wsi_shape)
            mask_tb[i] = mask

        return mask_tb
    
    def get_neighbor(self, patches):
        n_patches = len(patches)
        
        patch_size = self.r * 2 * self.n
        neighbor_patches = torch.zeros((n_patches, 3, patch_size, patch_size))
        mask_tb = torch.ones((n_patches, self.n**2))
        
        k_offsets = torch.arange(self.n) * self.r * 2
        m_offsets = torch.arange(self.n) * self.r * 2

        for i in range(n_patches):
            img, x, y = patches[i]
            img = np.array(img)

            wsi_shape = self.wsi.get_dimensions()
            mask = self.make_masking_table(x, y, wsi_shape)
            mask_tb_i = mask.clone()

            # x_start = x - self.r * self.n
            # y_start = y - self.r * self.n

            # Initialize empty patch
            neighbor_patch = torch.zeros((3, patch_size, patch_size))

            for k in range(self.n):
                for m in range(self.n):
                    n = k * self.n + m
                    if mask_tb_i[n] != 0:
                        # tmp = img[:, k * self.r * 2 : (k + 1) * self.r * 2, m * self.r * 2 : (m + 1) * self.r * 2]
                        tmp = img[k * self.r * 2 : (k + 1) * self.r * 2, m * self.r * 2 : (m + 1) * self.r * 2, :]
                        # current_x = x_start + k_offsets[k]
                        # current_y = y_start + m_offsets[m]
                        # tmp = self.wsi.read_region((current_x, current_y), self.level, (self.r * 2, self.r * 2))
                        tmp = self.img_transform(Image.fromarray(tmp))
                        neighbor_patch[:, k * self.r * 2 : (k + 1) * self.r * 2, m * self.r * 2 : (m + 1) * self.r * 2] = tmp
                        
            neighbor_patches[i] = neighbor_patch
            mask_tb[i] = mask

        return neighbor_patches, mask_tb
        
    # def get_neighbor(self, coords, wsi):
    #     n_patches = len(coords)
        
    #     patch_size = self.r * 2 * self.n
    #     neighbor_patches = torch.zeros((n_patches, 3, patch_size, patch_size))
    #     mask_tb = torch.ones((n_patches, self.n**2))
        
        
    #     k_offsets = torch.arange(self.n) * self.r * 2
    #     m_offsets = torch.arange(self.n) * self.r * 2

        
    #     for i in range(n_patches):
    #         x, y = coords[i]

    #         wsi_shape = wsi.get_dimensions()
    #         mask = self.make_masking_table(x, y, wsi_shape)
    #         mask_tb_i = mask.clone()

    #         x_start = x - self.r * self.n
    #         y_start = y - self.r * self.n

    #         # Initialize empty patch
    #         neighbor_patch = torch.zeros((3, patch_size, patch_size))

    #         for k in range(self.n):
    #             for m in range(self.n):
    #                 n = k * self.n + m
    #                 if mask_tb_i[n] != 0:
    #                     current_x = x_start + k_offsets[k]
    #                     current_y = y_start + m_offsets[m]
                        
    #                     tmp = wsi.read_region((current_x, current_y), self.level, (self.r * 2, self.r * 2))
    #                     tmp = self.img_transform(Image.fromarray(tmp))
                        
    #                     neighbor_patch[:, k * self.r * 2 : (k + 1) * self.r * 2, m * self.r * 2 : (m + 1) * self.r * 2] = tmp
                        
    #         neighbor_patches[i] = neighbor_patch
    #         mask_tb[i] = mask

    #     return neighbor_patches, mask_tb

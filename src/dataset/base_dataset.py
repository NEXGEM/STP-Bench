
from glob import glob
import os
import json
import warnings

import numpy as np
from scipy import sparse
import pandas as pd
import h5py
import scanpy as sc
import torch
import torchvision.transforms as transforms
from trident.IO import read_coords, read_coords_legacy
from trident.wsi_objects.WSIFactory import load_wsi

from core import normalize_adata
from dataset.path_utils import emb_dir, patch_dir, st_dir


class STDataset(torch.utils.data.Dataset):
    def __init__(self,
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                meta_dir: str = None,
                ref_data_dir: str = None,
                wsi_dir: str = None,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False,
                data_id: str = None,
                model_name: str = 'uni_v2',
                load_level: str = 'patch', # 'patch' or 'slide'
                use_emb: bool = True,
                genes_override: list = None,
                ):
        super(STDataset, self).__init__()
        
        if mode not in ['cv', 'eval', 'inference', 'prep', 'nested_cv']:
            raise ValueError(f"mode must be 'cv' or 'eval' or 'inference' or 'prep' or 'nested_cv', but got {mode}")
        if phase not in ['train', 'test']:
            raise ValueError(f"phase must be 'train' or 'test', but got {phase}")
        if mode in ['eval', 'inference'] and phase == 'train':
            print(f"mode is {mode} but phase is 'train', so phase is changed to 'test'")
            phase = 'test'
        
        if (mode == 'cv' and phase == 'test') or (mode == 'eval'):
            self.load_level = 'slide'
        else:
            self.load_level = load_level
        if self.load_level not in ['patch', 'slide']:
            raise ValueError(f"load_level must be 'patch' or 'slide', but got {self.load_level}")

        if gene_type not in ['var', 'mean', 'hmhvg', 'hmhvg_imm', 'total', 'all']:
            raise ValueError(f"gene_type must be 'var' or 'mean' or 'total', 'all' but got {gene_type}")
        
        self.data_dir = data_dir
        self.meta_dir = meta_dir or data_dir
        self.asset_dir = data_dir
        self.wsi_dir = wsi_dir
        self.img_dir = patch_dir(data_dir)
        self.st_dir = st_dir(data_dir)
        self.emb_dir = emb_dir(data_dir)
        self.model_name = model_name
        self.use_emb = use_emb

        self.mode = mode
        self.phase = phase
        self.norm_param = {'normalize': normalize, 'cpm': cpm, 'smooth': smooth}
        
        if mode == 'inference':
            if wsi_dir is not None:
                wsi_path = glob(f"{wsi_dir}/{data_id}.*")[0]
                self.wsi = load_wsi(wsi_path, lazy_init=False)
                
                self.name = data_id
                # self.img = self.load_img(data_id)
                h5_path = f"{self.img_dir}/{self.name}_patches.h5"
                if not os.path.isfile(h5_path):
                    raise FileNotFoundError(f"{h5_path} is not found.")

                self.patcher = self._get_patcher(h5_path)

                with h5py.File(h5_path, 'r') as f:
                    self.length = len(f['coords'])
            else:
                self.name = data_id
                img = self.load_img(data_id)
                self.length = len(img)
                
            if ref_data_dir is not None:
                self.ids = self._get_ids()
                self.genes = self._resolve_genes(gene_type, num_genes, num_outputs, ref_data_dir, genes_override)

        else:
            if ref_data_dir is not None:
                self.ids = self._get_ids()
            else:
                self.ids = self._get_ids(phase=phase, fold=fold)

            self.int2id = dict(enumerate(self.ids))
            self.id2int = {v: k for k, v in self.int2id.items()}

            self.genes = self._resolve_genes(gene_type, num_genes, num_outputs, ref_data_dir, genes_override)

        if phase == 'train':
            self.adata_dict = {
                _id: self.load_st(_id, self.genes, **self.norm_param)
                for _id in self.ids
            }

            self.lengths = [len(adata) for adata in self.adata_dict.values()]
            self.cumlen = np.cumsum(self.lengths)
            
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation((90, 90))]),
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        
    def __getitem__(self, index):
        data = {}
        
        if self.load_level == 'patch':
            if self.mode == 'inference':
                img = self.load_img(self.name, idx=index)
                img = self.transforms(img)
                if self.use_emb:
                    img_emb = self.load_emb(self.name, idx=index, emb_name='global')

            else:
                i = 0
                while index >= self.cumlen[i]:
                    i += 1
                idx = index
                if i > 0:
                    idx = index - self.cumlen[i-1]

                name = self.int2id[i]
                img = self.load_img(name, idx)
                img = self.transforms(img)
                if self.use_emb:
                    img_emb = self.load_emb(name, idx=idx, emb_name='global')

                adata = self.adata_dict[name]
                expression = adata[idx].X
                expression = expression.toarray().squeeze(0) \
                    if sparse.issparse(expression) else expression.squeeze(0)

                data['label'] = torch.FloatTensor(expression)

            data['img'] = img
            if self.use_emb:
                data['img_emb'] = img_emb

        elif self.load_level == 'slide':
            if self.mode == 'inference':
                img = self.load_img(self.name, idx=index)
                img = self.transforms(img)
                if self.use_emb:
                    img_emb = self.load_emb(self.name, idx=index, emb_name='global')
            else:
                name = self.int2id[index]
                img = self.load_img(name)
                img = torch.stack([self.transforms(im) for im in img], dim=0)
                if self.use_emb:
                    img_emb = self.load_emb(name, emb_name='global')

                if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                    adata = self.load_st(name, self.genes, **self.norm_param)
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression)

            data['img'] = img
            if self.use_emb:
                data['img_emb'] = img_emb

        return data
        
    def __len__(self):
        if self.load_level == 'patch':
            if self.mode == 'inference':
                return self.length
            else:
                return self.cumlen[-1]
            
        elif self.load_level == 'slide':
            if self.mode == 'inference':
                return 1
            else:
                return len(self.int2id)
        
    def _resolve_gene_path(self, gene_type: str, num_genes: int, ref_data_dir: str = None) -> str:
        """Return the first existing gene JSON path, checking ref_data_dir → meta_dir → data_dir.

        ref_data_dir (the training run's own meta_dir) takes priority when set:
        a model's output columns are fixed to the gene panel it was trained on,
        so external/eval-time gene resolution must reuse that panel rather than
        the eval dataset's own independently-computed HVG list, which is
        typically a different set of genes entirely.
        """
        candidates = []
        if ref_data_dir:
            candidates.append(f"{ref_data_dir}/{gene_type}_{num_genes}genes.json")
        candidates.append(f"{self.meta_dir}/{gene_type}_{num_genes}genes.json")
        if self.data_dir != self.meta_dir:
            candidates.append(f"{self.data_dir}/{gene_type}_{num_genes}genes.json")
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]  # return primary path so the caller can raise a clear error

    def _resolve_genes(self, gene_type: str, num_genes: int, num_outputs: int,
                        ref_data_dir: str = None, genes_override: list = None) -> list:
        """Return the target gene list, or raise if the gene JSON is missing.

        genes_override lets the caller (external evaluation with a partial
        gene-panel overlap) inject an explicit gene list directly, bypassing
        file-based resolution entirely.
        """
        if genes_override is not None:
            return list(genes_override)
        gene_path = self._resolve_gene_path(gene_type, num_genes, ref_data_dir)
        if not os.path.isfile(gene_path):
            raise ValueError(f"{gene_path} is not found")
        with open(gene_path, 'r') as f:
            genes = json.load(f)['genes']
        return genes[:num_outputs] if gene_type in ['mean', 'hmhvg', 'total'] else genes

    def _get_ids(self, phase=None, fold=None, ids_dir=None):
        """Load sample IDs from ids.csv, optionally filtered by fold/phase."""
        ids_path = os.path.join(ids_dir or self.meta_dir, "ids.csv")
        if not os.path.isfile(ids_path):
            raise FileNotFoundError(f"{ids_path} is not found.")
        data = pd.read_csv(ids_path)
        if "sample_id" not in data.columns:
            raise ValueError(f"{ids_path} must contain a sample_id column.")
        if phase is None or fold is None:
            return data["sample_id"].tolist()
        fold_col = f"fold_{fold}"
        if fold_col not in data.columns:
            raise FileNotFoundError(
                f"Data split CSV not found — run preprocessing (split_data step) first.\n"
                f"  hint: {fold_col} is not present in {ids_path}."
            )
        return data.loc[data[fold_col].astype(str).str.lower() == phase, "sample_id"].tolist()
        
    def _align_adata_to_patches(self, name: str, adata):
        """Filter and reorder adata rows to match the patch h5 barcode ordering.

        Patch extraction may drop spots that fall outside the tissue boundary or
        fail quality checks, so the patch h5 barcode list is a strict subset of
        the adata obs_names. Reindexing adata to the patch barcode order ensures
        that adata[i] and h5['img'][i] always refer to the same spot.
        """
        img_path = os.path.join(self.img_dir, f"{name}.h5")
        if not os.path.isfile(img_path):
            img_path = os.path.join(self.img_dir, f"{name}_patches.h5")
        if not os.path.isfile(img_path):
            return adata
        with h5py.File(img_path, 'r') as f:
            if 'barcode' not in f:
                return adata
            patch_barcodes = f['barcode'][:].flatten().astype(str).tolist()
        # Subset and reorder adata to match patch order
        valid = [b for b in patch_barcodes if b in adata.obs_names]
        if len(valid) == len(adata):
            return adata  # already aligned
        return adata[valid]

    def load_img(self, name: str, idx: int = None, level: int = 0):
        """Load whole slide image of a sample.

        Args:
            name (str): name of a sample
            idx (int): index of a patch.

        Returns:
            numpy.array: return whole slide image.
        """
        if level == 0:
            path = f"{self.img_dir}/{name}.h5"
            if not os.path.isfile(path):
                path = f"{self.img_dir}/{name}_patches.h5"

                if not os.path.isfile(path):
                    raise FileNotFoundError(f"{path} is not found.")

        else:
            path = f"{self.img_dir}/level{level}/{name}.h5"
            if not os.path.isfile(path):
                path = f"{self.img_dir}/level{level}/{name}_patches.h5"
                
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"{path} is not found.")

        if self.wsi_dir is not None:
            if idx is not None:
                img = self.patcher[idx][0]
            else:
                img = [patch[0] for patch in self.patcher]
                img = np.stack(img, axis=0)
            # patcher = self._get_patcher(path)
            # img = [patch[0] for patch in patcher]
            # img = np.stack(img, axis=0)
        else:
            if idx is not None:
                with h5py.File(path, 'r') as f:
                    img = f['img'][idx]
            else:
                with h5py.File(path, 'r') as f:
                    img = f['img'][:]
            
        return img

    def _get_patcher(self, coords_path, n=None):
        try:
            coords_attrs, coords = read_coords(coords_path)
            patch_size = coords_attrs.get('patch_size', None)
            
            if n is not None:
                patch_size = patch_size * n
            
            level0_magnification = coords_attrs.get('level0_magnification', None)
            target_magnification = coords_attrs.get('target_magnification', None)            
            if None in (patch_size, level0_magnification, target_magnification):
                raise KeyError('Missing attributes in coords_attrs.')
        except (KeyError, FileNotFoundError, ValueError) as e:
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
    
    def load_st(self, 
                name: str, 
                genes, 
                normalize: bool = True, 
                cpm=False, 
                smooth=False,
                st_dir=None,
                return_total_genes=False):
        """Load gene expression data of a sample.

        Args:
            name (str): name of a sample
            normalize (bool): whether to normalize gene expression data.
            cpm (bool): whether to conduct CPM while normalizing gene expression data.
            smooth (bool): whether to smooth gene expression data.
            
        Returns:
            annData: return adata of st data. 
        """
        if st_dir is None:
            st_dir = self.st_dir
        
        path = f"{st_dir}/{name}.h5ad"
        if not os.path.isfile(path) and st_dir.endswith("/st"):
            fallback = f"{st_dir[:-3]}/adata/{name}.h5ad"
            if os.path.isfile(fallback):
                path = fallback
        adata = sc.read_h5ad(path)
        
        total_genes = adata.var_names.tolist()
        
        # common_genes = list(set(genes).intersection(set(adata.var_names)))
        if adata.var_names.isin(genes).sum() < len(genes):
            common_genes = list(set(genes).intersection(set(adata.var_names)))
            warnings.warn(f"Some genes are not found in {name}.h5ad. Use {len(common_genes)} / {len(genes)} genes.")
            adata = adata[:, common_genes]
        else:
            adata = adata[:, genes]
        
        if normalize:
            adata = normalize_adata(adata, cpm=cpm, smooth=smooth)
    
        if return_total_genes:
            return adata, total_genes
        else:
            return adata
    
    def load_emb(self, name: str, 
                 emb_name: str = 'global', 
                 idx: int = None, 
                 return_crds=False,
                 emb_dir=None,
                 model_name: str = None):
        if emb_name not in ['global', 'neighbor', 'target']:
            raise ValueError(f"emb_name must be 'global' or 'neighbor' or 'target', but got {emb_name}")
        
        if emb_dir is None:
            emb_dir = self.emb_dir
        
        if model_name is None:
            model_name = self.model_name
            
        path = f"{emb_dir}/{emb_name}/features_{model_name}/{name}.h5"
        
        with h5py.File(path, 'r') as f:
            if 'features' in f.keys():
                emb_key = 'features'
            elif 'embeddings' in f.keys():
                emb_key = 'embeddings'
            else:
                raise KeyError(f"Neither 'features' nor 'embeddings' found in {path}")

            if 'virchow' in model_name:
                if emb_name == 'global':
                    emb = f[emb_key][idx,:1280] if idx is not None else f[emb_key][:,:1280]
                else:
                    emb = f[emb_key][idx,:,:1280] if idx is not None else f[emb_key][:,:,:1280]

            else:
                emb = f[emb_key][idx] if idx is not None else f[emb_key][:]

            emb = torch.Tensor(emb)
            
            if emb_name == 'neighbor':
                mask = f['mask_tb'][idx] if idx is not None else f['mask_tb'][:]
                mask = torch.LongTensor(mask)
                # return emb, mask
                if return_crds:
                    coords = f['coords'][idx] if idx is not None else f['coords'][:]
                    coords = torch.Tensor(coords)
                    return emb, mask, coords
                else:
                    return emb, mask
            else:
                if return_crds:
                    coords = f['coords'][idx] if idx is not None else f['coords'][:]
                    coords = torch.Tensor(coords)
                    return emb, coords
                else:
                    return emb

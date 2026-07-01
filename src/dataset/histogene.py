import os

import json
import numpy as np
from scipy import sparse
import h5py
import torch
import torchvision.transforms as transforms

from dataset.base_dataset import STDataset


class HistDataset(STDataset):
    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                meta_dir: str = None,
                ref_data_dir: str = None,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False,
                data_id: str = None,
                img_size = 112,
                model_name: str = 'uni_v2',
                neighs: int = 4,
                prune: str = 'Grid',
                load_level: str = 'slide' # 'patch' or 'slide'
                ):
        super(HistDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                meta_dir=meta_dir,
                                ref_data_dir=ref_data_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth,
                                data_id=data_id,
                                model_name=model_name,
                                load_level=load_level)
        if phase == 'train':
            self.transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomApply([transforms.RandomRotation((90, 90))]),
                    transforms.ToTensor(),
                    transforms.CenterCrop((img_size, img_size)),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
        else: 
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((img_size, img_size)),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        self.neighs = neighs
        self.prune = prune
            
        
    def __getitem__(self, index):
        data = {}
        
        if self.mode == 'inference':
            # img = self.load_img(self.name, idx=index)
            img = self.load_img(self.name)
            img = torch.stack([self.transforms(im) for im in img], dim=0)
            # img = self.img[index]
            # img = self.transforms(img)
            img_emb, centers = self.load_emb(self.name, emb_name='global', return_crds=True)
            # img_emb, centers = self.load_emb(self.name, idx=index, emb_name='global', return_crds=True)
            # neighbor_emb, mask = self.load_emb(self.name, emb_name='neighbor', idx=index)
            
            # global_emb, centers = self.load_emb(self.name, emb_name='global', return_crds=True)
            # pos = np.load(f"{self.data_dir}/pos/{self.name}.npy")
            data['img'] = img
            data['img_emb'] = img_emb
            data['centers'] = centers.long()
            # data['sid'] = torch.LongTensor([index])
        else:
            name = self.int2id[index]
            img = self.load_img(name)
            img = torch.stack([self.transforms(im) for im in img], dim=0)
            
            # neighbor_emb, mask = self.load_emb(name, emb_name='neighbor')
            img_emb, centers = self.load_emb(name, emb_name='global', return_crds=True)
            grid_pos = self.get_normalized_pos(centers, rounding_factor=20)
            
            if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                adata = self.load_st(name, self.genes, **self.norm_param)
                oris = self.load_st(name, self.genes, normalize=False)
                oris = oris.X.toarray() if sparse.issparse(oris.X) else oris.X
                n_counts=oris.sum(1)
                sfs = n_counts / np.median(n_counts)
                
                # pos = adata.obs[['array_row', 'array_col']].values
                expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                data['label'] = torch.FloatTensor(expression) 

                adj = self.calcADJ(grid_pos, self.neighs, pruneTag = self.prune)
                
            data['img'] = img
            data['img_emb'] = img_emb
            data['centers'] = centers.long()
            data['adj'] = adj
            data['oris'] = oris
            data['sfs'] = sfs
            # data['img'] = img
            # data['mask'] = mask
            # data['neighbor_emb'] = neighbor_emb
            
        return data

    @staticmethod
    def calcADJ(coord, k=8, distanceType='euclidean', pruneTag='NA'):
        r"""
        Calculate spatial Matrix directly use X/Y coordinates
        """
        from scipy.spatial import distance

        spatialMatrix=coord#.cpu().numpy()
        nodes=spatialMatrix.shape[0]
        Adj=torch.zeros((nodes,nodes))
        for i in np.arange(spatialMatrix.shape[0]):
            tmp=spatialMatrix[i,:].reshape(1,-1)
            distMat = distance.cdist(tmp,spatialMatrix, distanceType)
            if k == 0:
                k = spatialMatrix.shape[0]-1
            res = distMat.argsort()[:k+1]
            tmpdist = distMat[0,res[0][1:k+1]]
            boundary = np.mean(tmpdist)+np.std(tmpdist) #optional
            for j in np.arange(1,k+1):
                # No prune
                if pruneTag == 'NA':
                    Adj[i][res[0][j]]=1.0
                elif pruneTag == 'STD':
                    if distMat[0,res[0][j]]<=boundary:
                        Adj[i][res[0][j]]=1.0
                # Prune: only use nearest neighbor as exact grid: 6 in cityblock, 8 in euclidean
                elif pruneTag == 'Grid':
                    if distMat[0,res[0][j]]<=2.0:
                        Adj[i][res[0][j]]=1.0
        return Adj
    
    def get_normalized_pos(self, pos, rounding_factor=None):
        W,H = self.infer_grid_size(pos, rounding_factor=rounding_factor)

        pos_min = pos.min(dim=0, keepdim=True)[0]
        pos_max = pos.max(dim=0, keepdim=True)[0]
        pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-5)

        grid_pos = pos_norm * torch.tensor([W - 1, H - 1])
        grid_pos = grid_pos.round().long()
        
        return grid_pos

    def infer_grid_size(self, pos, rounding_factor=None):
        """
        pos: (N, 2) tensor
        """
        if rounding_factor is None:
            rounding_factor = self.dynamic_rounding_factor(pos)
        
        pos_rounded = (pos / rounding_factor).round() * rounding_factor
        unique_x = torch.unique(pos_rounded[:, 0])
        unique_y = torch.unique(pos_rounded[:, 1])
        W = unique_x.numel()
        H = unique_y.numel()
        return (W, H)
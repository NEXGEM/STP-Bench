import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from dataset.base_dataset import STDataset

# Mirrors DeepSpotM's MidnightEncoder normalization and eval transform
# (deepspotm package -- pip install -r requirements/models/DeepSpotM.txt --
# image_encoder.py::MidnightEncoder, utils.py::get_eval_transforms with
# center_crop=True).
_MIDNIGHT_MEAN = (0.5, 0.5, 0.5)
_MIDNIGHT_STD = (0.5, 0.5, 0.5)


class DeepSpotMDataset(STDataset):
    """Raw-pixel dataset for DeepSpotM.

    DeepSpotM predicts directly from a raw 224x224 tile rather than a
    precomputed patch embedding, so it needs its own resize/crop/normalize
    transform instead of STDataset's hardcoded ImageNet one.
    """

    def __init__(self,
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                meta_dir: str = None,
                ref_data_dir: str = None,
                genes_override: list = None,
                wsi_dir: str = None,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False,
                data_id: str = None,
                model_name: str = 'uni_v2',
                load_level: str = 'patch',
                use_emb: bool = True,
                ):
        super(DeepSpotMDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                meta_dir=meta_dir,
                                ref_data_dir=ref_data_dir,
                                genes_override=genes_override,
                                wsi_dir=wsi_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth,
                                data_id=data_id,
                                model_name=model_name,
                                load_level=load_level,
                                use_emb=use_emb)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=_MIDNIGHT_MEAN, std=_MIDNIGHT_STD),
        ])

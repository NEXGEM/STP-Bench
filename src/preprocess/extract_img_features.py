import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torchvision
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

from trident.patch_encoder_models.load import encoder_factory
from trident.patch_encoder_models.utils.transform_utils import get_eval_transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import H5TileDataset
from core.utils.preprocess_utils import add_augmentation_to_transform, save_hdf5, compute_mini_tiles


class InferenceEncoder(nn.Module):
    def __init__(self, weights_path=None, **build_kwargs):
        super().__init__()
        self.weights_path = weights_path
        self.model, self.eval_transforms, self.precision = self._build(weights_path, **build_kwargs)

    def _build(self, weights_path=None, **build_kwargs):
        raise NotImplementedError

    def forward(self, x):
        return self.model(x)


class CigarInferenceEncoder(InferenceEncoder):
    def __init__(self):
        super().__init__(weights_path=None)

    def _build(self, weights_path=None, **build_kwargs):
        import wget

        model = torchvision.models.__dict__['resnet18'](weights=None)

        ckpt_dir = './weights/cigar'
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f'{ckpt_dir}/tenpercent_resnet18.ckpt'

        if not os.path.exists(ckpt_path):
            ckpt_url = 'https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt'
            wget.download(ckpt_url, out=ckpt_dir)

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        model.fc = nn.Identity()

        eval_transform = get_eval_transforms((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), target_img_size=224)
        return model, eval_transform, torch.float32

    def forward(self, x):
        return self.model(x)


def post_collate_fn(batch):
    if batch["imgs"].dim() == 5:
        assert batch["imgs"].size(0) == 1
        batch["imgs"] = batch["imgs"].squeeze(0)
    if batch["coords"].dim() == 3:
        assert batch["coords"].size(0) == 1
        batch["coords"] = batch["coords"].squeeze(0)
    if "mask_tb" in batch and batch["mask_tb"].dim() == 3:
        assert batch["mask_tb"].size(0) == 1
        batch["mask_tb"] = batch["mask_tb"].squeeze(0)
    return batch


def embed_tiles(dataloader, model, embedding_save_path, device,
                precision=torch.float32, transform_type=None,
                num_tiles=1, feature_type='global'):
    """Extract embeddings from tiles using encoder and save to h5 file."""
    model.eval()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="  batches", leave=False):
        batch = post_collate_fn(batch)
        imgs = batch['imgs']

        if feature_type == 'global':
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
                embeddings = model(imgs.to(device))
        else:
            if num_tiles == 1:
                raise ValueError("num_tiles must be > 1 for neighbor or target mode")

            if feature_type == 'target':
                img_size = imgs.shape[-1]

            imgs = compute_mini_tiles(imgs, n_tiles=num_tiles)
            embeddings = []
            for img in imgs:
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
                    if feature_type == 'target':
                        img = torchvision.transforms.Resize(img_size, antialias=True)(img)
                    embeddings.append(model(img.to(device)).cpu())
            embeddings = torch.stack(embeddings, dim=1)

        asset_dict = {'features': embeddings.cpu().numpy()}
        if transform_type == 'eval':
            asset_dict.update({
                key: val.cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
                for key, val in batch.items() if key != 'imgs'
            })

        save_hdf5(embedding_save_path, asset_dict=asset_dict, mode='w' if batch_idx == 0 else 'a')

    return embedding_save_path


def get_bench_weights(weights_root, name):
    local_ckpt_registry = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_ckpts.json')
    with open(local_ckpt_registry, 'r') as f:
        ckpt_registry = json.load(f)
    if name not in ckpt_registry:
        raise ValueError(f"Please specify the weights path to {name} in {local_ckpt_registry}")
    path = ckpt_registry[name]
    return path if os.path.isabs(path) else os.path.join(weights_root, path)


def main(args, device):
    embedding_dir = os.path.join(args.embed_dataroot, f"features_{args.patch_encoder}")
    if args.transform_type != 'eval':
        embedding_dir = os.path.join(embedding_dir, args.transform_type)
    os.makedirs(embedding_dir, exist_ok=True)

    if args.id_path is not None:
        if not os.path.isfile(args.id_path):
            raise ValueError(f"{args.id_path} doesn't exist")
        if 'sample_id' not in pd.read_csv(args.id_path).columns:
            raise ValueError("Column 'sample_id' not found in id file")
        ids = pd.read_csv(args.id_path)['sample_id'].tolist()
    else:
        ids = [
            os.path.splitext(f)[0].replace('_patches', '')
            for f in os.listdir(args.patch_dataroot) if f.endswith('.h5')
        ]

    print(f"Extracting features with {args.patch_encoder} ({len(ids)} samples)")
    if args.patch_encoder == 'cigar':
        encoder = CigarInferenceEncoder()
    else:
        encoder = encoder_factory(args.patch_encoder, weights_path=args.patch_encoder_ckpt_path)

    precision = encoder.precision

    for sample_id in tqdm(ids, desc="samples"):
        start = time.time()

        tile_h5_path = f"{args.patch_dataroot}/{sample_id}.h5"
        if not os.path.isfile(tile_h5_path):
            tile_h5_path = f"{args.patch_dataroot}/{sample_id}_patches.h5"
        if not os.path.isfile(tile_h5_path):
            tqdm.write(f"  [skip] {sample_id}: patch file not found")
            continue

        embed_path = os.path.join(embedding_dir, f'{sample_id}.h5')
        if os.path.isfile(embed_path) and not args.overwrite:
            tqdm.write(f"  [skip] {sample_id}: already exists")
            continue

        encoder.eval().to(device)
        transforms = add_augmentation_to_transform(encoder.eval_transforms, transform_type=args.transform_type)

        tile_dataset = H5TileDataset(
            tile_h5_path,
            sample_id=sample_id,
            mode=args.mode,
            wsi_dir=args.wsi_dataroot,
            ext=args.slide_ext,
            level=args.level,
            img_transform=transforms,
            num_n=args.num_tiles,
            feature_type=args.feature_type,
            chunk_size=args.batch_size,
            num_workers=args.num_workers,
        )

        tile_dataloader = torch.utils.data.DataLoader(
            tile_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
        )

        embed_tiles(
            tile_dataloader, encoder, embed_path, device,
            precision=precision,
            transform_type=args.transform_type,
            num_tiles=args.num_tiles,
            feature_type=args.feature_type,
        )

        tqdm.write(f"  {sample_id}: {time.time() - start:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract image features from ST patches')

    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--patch_dataroot', type=str, default='input/ST/andrew/patches')
    parser.add_argument('--embed_dataroot', type=str, default='input/ST/andrew/emb/global')
    parser.add_argument('--wsi_dataroot', type=str, default='input/ST/andrew/wsis')
    parser.add_argument('--id_path', type=str, default=None)
    parser.add_argument('--slide_ext', type=str, default='.tif')
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--weights_root', type=str, default='fm_v1')
    parser.add_argument('--transform_type', type=str, default='eval')
    parser.add_argument('--mode', type=str, default='inference')
    parser.add_argument('--patch_encoder_ckpt_path', type=str, default=None)
    parser.add_argument(
        '--patch_encoder', type=str, default='conch_v15',
        choices=[
            'conch_v1', 'uni_v1', 'uni_v2', 'ctranspath', 'phikon',
            'resnet50', 'gigapath', 'virchow', 'virchow2',
            'hoptimus0', 'hoptimus1', 'phikon_v2', 'conch_v15', 'musk', 'hibou_l',
            'kaiko-vits8', 'kaiko-vits16', 'kaiko-vitb8', 'kaiko-vitb16',
            'kaiko-vitl14', 'lunit-vits8', 'midnight12k', 'cigar',
        ],
    )
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--feature_type', type=str, default='global')
    parser.add_argument('--num_tiles', type=int, default=1)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args, device)

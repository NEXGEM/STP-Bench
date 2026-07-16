#!/usr/bin/env python
"""Predict spatial gene expression for a whole-slide H&E image with DeepSpot-M.

Unlike ``predict.py`` (which scores pre-cut 224x224 tiles), this reads a
whole-slide image, tiles it on a 224-px grid at native resolution, drops
background tiles, predicts expression per tile, and writes a spatial AnnData
(``.h5ad``) with one row per tile — the same format as the TCGA virtual spatial
transcriptomics atlas, ready for scanpy / squidpy.

Tiles are cut at the slide's native pixel resolution, so the input should be a
~20x H&E whole-slide image (the magnification DeepSpot-M was trained on).

Needs ``pyvips`` (libvips) and ``anndata`` in addition to the core deps:
    pip install pyvips anndata

Examples:
  # one slide -> one .h5ad (full ~19k-gene panel)
  python examples/predict_wsi.py slide.svs -o slide.h5ad

  # a folder of slides -> one .h5ad each in out/, scoring three marker genes
  python examples/predict_wsi.py slides/ -o out/ --genes EPCAM CD3D PTPRC
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WSI_SUFFIXES = {".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs", ".svslide", ".bif"}


def collect_slides(paths: list[str], recursive: bool) -> list[Path]:
    """Expand files and folders into a sorted list of whole-slide images."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            globber = p.rglob("*") if recursive else p.glob("*")
            found += [f for f in globber if f.suffix.lower() in WSI_SUFFIXES]
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
    return sorted({f.resolve() for f in found})


def tile_slide(slide, image_processor, tile, white_mean):
    """Yield (PIL tile, x, y) for tissue tiles on a `tile`-px grid.

    Two-stage background filter: a one-pixel-per-tile thumbnail flags candidate
    tissue tiles cheaply, then each candidate is read at full resolution and
    re-checked against `white_mean` before it is kept.
    """
    import numpy as np
    from PIL import Image

    W, H = slide.width, slide.height
    nx, ny = W // tile, H // tile
    if nx == 0 or ny == 0:
        return

    import pyvips
    thumb = pyvips.Image.thumbnail_image(slide, nx)
    tg = np.ndarray(buffer=thumb.write_to_memory(), dtype=np.uint8,
                    shape=[thumb.height, thumb.width, thumb.bands])[..., :3]
    grid_mean = np.asarray(Image.fromarray(tg).resize((nx, ny))).astype(np.float32).mean(2)
    candidates = np.argwhere(grid_mean <= white_mean + 10)   # (row, col)

    for gy, gx in candidates:
        x, y = int(gx) * tile, int(gy) * tile
        buf = slide.crop(x, y, tile, tile).write_to_memory()
        arr = np.ndarray(buffer=buf, dtype=np.uint8,
                         shape=[tile, tile, slide.bands])[..., :3]
        if arr.mean() > white_mean:          # exact background check
            continue
        yield Image.fromarray(arr), x, y


def predict_slide(model, image_processor, slide_path, genes, tile, batch_size,
                  white_mean, device):
    """Return (expr (n_tiles, n_genes) float32, coords (n_tiles, 2), gene names)."""
    import numpy as np
    import pyvips
    import torch

    slide = pyvips.Image.new_from_file(str(slide_path), access="random")
    columns = list(genes) if genes else list(model.gene_names)

    coords, chunks = [], []
    buf_imgs, buf_xy = [], []

    def flush():
        if not buf_imgs:
            return
        px = torch.stack([image_processor(im) for im in buf_imgs]).to(device)
        with torch.no_grad():
            if genes:
                out = model.predict_genes(px, genes)
            else:
                out, _, _ = model(px)
        chunks.append(out.float().cpu().numpy())
        coords.extend(buf_xy)
        buf_imgs.clear(); buf_xy.clear()

    for pil, x, y in tile_slide(slide, image_processor, tile, white_mean):
        buf_imgs.append(pil); buf_xy.append((x, y))
        if len(buf_imgs) >= batch_size:
            flush()
    flush()

    if not coords:
        return np.empty((0, len(columns)), np.float32), np.empty((0, 2), int), columns
    return np.concatenate(chunks, 0), np.asarray(coords, int), columns


def to_anndata(expr, coords, genes, slide_path, tile, source):
    import anndata as ad
    import numpy as np
    import pandas as pd

    obs = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1]},
        index=[f"{slide_path.stem}_{x}_{y}" for x, y in coords],
    )
    adata = ad.AnnData(
        X=expr.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=list(genes)),
    )
    # tile-center coordinates, the squidpy/scanpy spatial convention
    adata.obsm["spatial"] = (coords + tile / 2).astype(float)
    adata.uns["deepspotm"] = {
        "slide": slide_path.name, "tile_px": tile, "source": source,
        "expression": "log1p-CPM",
    }
    return adata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="predict_wsi.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", metavar="PATH",
                        help="whole-slide image(s) or folder(s) of slides")
    parser.add_argument("--output", "-o", metavar="PATH", default=".",
                        help="output .h5ad (single slide) or output directory "
                             "(default: current directory)")
    parser.add_argument("--repo", default="ratschlab/DeepSpotM",
                        help="HF repo id or a local model directory (default: %(default)s)")
    parser.add_argument("--source", default="scgpt",
                        choices=["evo2", "orthrus", "prott5", "scgpt", "apertus"],
                        help="gene-embedding source (default: %(default)s)")
    parser.add_argument("--genes", nargs="+", metavar="GENE",
                        help="score only these genes instead of the full panel")
    parser.add_argument("--tile", type=int, default=224, metavar="PX",
                        help="tile size in pixels (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="tiles per forward pass (default: %(default)s)")
    parser.add_argument("--white-mean", type=float, default=220.0, metavar="V",
                        help="mean-pixel threshold above which a tile is background "
                             "(default: %(default)s)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="do not descend into sub-folders when a folder is given")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="compute device (default: cuda when available)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        import pyvips  # noqa: F401
        import anndata  # noqa: F401
    except ImportError as err:
        print(f"error: {err}. This script needs libvips + anndata: "
              f"pip install pyvips anndata", file=sys.stderr)
        return 1

    try:
        slides = collect_slides(args.inputs, recursive=not args.no_recursive)
    except FileNotFoundError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    if not slides:
        print("error: no whole-slide images found in the given paths", file=sys.stderr)
        return 1

    out = Path(args.output)
    single = len(slides) == 1 and out.suffix.lower() == ".h5ad"
    if not single:
        out.mkdir(parents=True, exist_ok=True)

    import torch
    from deepspotm import DeepSpotM

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.repo} (source={args.source}) on {device} ...", file=sys.stderr)
    model, image_processor = DeepSpotM.from_pretrained(args.repo, source=args.source)
    model = model.to(device).eval()

    rc = 0
    for slide_path in slides:
        print(f"tiling + predicting {slide_path.name} ...", file=sys.stderr)
        try:
            expr, coords, genes = predict_slide(
                model, image_processor, slide_path, args.genes, args.tile,
                args.batch_size, args.white_mean, device,
            )
        except KeyError as err:
            print(f"  skipped: unknown gene symbol(s): {err}", file=sys.stderr)
            rc = 1
            continue
        if len(coords) == 0:
            print(f"  skipped: no tissue tiles found (try raising --white-mean)",
                  file=sys.stderr)
            rc = 1
            continue

        adata = to_anndata(expr, coords, genes, slide_path, args.tile, args.source)
        dest = out if single else out / f"{slide_path.stem}.h5ad"
        adata.write_h5ad(dest)
        print(f"  wrote {dest}  ({adata.n_obs} tiles x {adata.n_vars} genes)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

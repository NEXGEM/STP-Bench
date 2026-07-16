#!/usr/bin/env python
"""Predict spatial gene expression from H&E image tiles with DeepSpot-M.

Run it on a single tile or a folder of tiles. For each tile the model predicts
expression across its gene panel. By default the highest-expression genes are
printed per tile. Pass --genes to score specific genes, or --output to write all
predictions to a CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SOURCES = ["evo2", "orthrus", "prott5", "scgpt", "apertus"]

EXAMPLES = """\
examples:
  # one tile, print the 10 highest-expression genes
  python examples/predict.py tile.png

  # every tile in a folder, score three marker genes, write a CSV
  python examples/predict.py tiles/ --genes EPCAM CD3D PTPRC --output preds.csv

  # a different gene-embedding source, forced onto CPU
  python examples/predict.py tile.png --source evo2 --device cpu
"""


def collect_images(paths: list[str], recursive: bool) -> list[Path]:
    """Expand the given files and folders into a sorted list of image tiles."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            globber = p.rglob("*") if recursive else p.glob("*")
            found += [f for f in globber if f.suffix.lower() in IMAGE_SUFFIXES]
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
    return sorted({f.resolve() for f in found})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=__doc__,
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="PATH",
        help="one or more H&E tiles, or folders of tiles (RGB, resized to 224x224)",
    )
    parser.add_argument(
        "--repo", default="ratschlab/DeepSpotM",
        help="Hugging Face repo id or a local directory with the model files "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--source", default="scgpt", choices=SOURCES,
        help="gene-embedding source used by the decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--genes", nargs="+", metavar="GENE",
        help="score only these gene symbols instead of the full panel",
    )
    parser.add_argument(
        "--top", type=int, default=10, metavar="N",
        help="how many highest-expression genes to print per tile, "
             "ignored when --genes is set (default: %(default)s)",
    )
    parser.add_argument(
        "--output", "-o", metavar="CSV",
        help="write every prediction to this CSV (rows are tiles, columns are genes)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, metavar="N",
        help="number of tiles per forward pass (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-background", action="store_true",
        help="skip near-white background tiles (mean pixel above --white-mean)",
    )
    parser.add_argument(
        "--white-mean", type=float, default=220.0, metavar="V",
        help="mean-pixel threshold above which a tile counts as background, "
             "used with --skip-background (default: %(default)s)",
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="do not descend into sub-folders when a folder is given",
    )
    parser.add_argument(
        "--device", default=None, choices=["cuda", "cpu"],
        help="compute device (default: cuda when available, else cpu)",
    )
    return parser.parse_args(argv)


def predict(model, image_processor, images, genes, batch_size, device, white_mean=None):
    """Predict expression for the tiles, skipping background when white_mean is set.

    Returns the (n_kept, n_genes) CPU prediction tensor, the gene order, and the
    list of tiles that were actually scored.
    """
    import numpy as np
    import torch
    from PIL import Image

    columns = list(genes) if genes else list(model.gene_names)
    kept, rows = [], []
    batch_tiles, batch_paths = [], []

    def flush():
        if not batch_tiles:
            return
        pixel_values = torch.stack(batch_tiles).to(device)
        with torch.no_grad():
            if genes:
                out = model.predict_genes(pixel_values, genes)
            else:
                out, _, _ = model(pixel_values)
        rows.append(out.float().cpu())
        kept.extend(batch_paths)
        batch_tiles.clear(); batch_paths.clear()

    for path in images:
        image = Image.open(path).convert("RGB")
        if white_mean is not None and float(np.asarray(image).mean()) > white_mean:
            continue
        batch_tiles.append(image_processor(image))
        batch_paths.append(path)
        if len(batch_tiles) >= batch_size:
            flush()
    flush()

    preds = torch.cat(rows, dim=0) if rows else torch.empty(0, len(columns))
    return preds, columns, kept


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        images = collect_images(args.inputs, recursive=not args.no_recursive)
    except FileNotFoundError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    if not images:
        print("error: no image tiles found in the given paths", file=sys.stderr)
        return 1

    import torch
    from deepspotm import DeepSpotM

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.repo} (source={args.source}) on {device} ...", file=sys.stderr)
    model, image_processor = DeepSpotM.from_pretrained(args.repo, source=args.source)
    model = model.to(device).eval()

    white_mean = args.white_mean if args.skip_background else None
    try:
        preds, columns, kept = predict(
            model, image_processor, images, args.genes, args.batch_size, device, white_mean
        )
    except KeyError as err:
        print(f"error: unknown gene symbol(s): {err}", file=sys.stderr)
        return 1

    skipped = len(images) - len(kept)
    if not kept:
        print("error: every tile was filtered as background; "
              "lower --white-mean or drop --skip-background", file=sys.stderr)
        return 1

    note = f" ({skipped} background tile(s) skipped)" if skipped else ""
    print(f"predicted expression for {len(kept)} tile(s) over {len(columns)} gene(s){note}.")
    print("values are the model's predicted expression (log1p-CPM); "
          "higher means more expressed.\n")

    for path, row in zip(kept, preds):
        values = row.tolist()
        if args.genes:
            scored = ", ".join(f"{g}={v:.3f}" for g, v in zip(columns, values))
            print(f"{path.name}: {scored}")
        else:
            ranked = sorted(zip(columns, values), key=lambda kv: kv[1], reverse=True)
            print(f"{path.name}  (top {args.top} highest-expression genes)")
            for gene, val in ranked[:args.top]:
                print(f"  {gene:<12s} {val:.3f}")
            print()

    if args.output:
        import pandas as pd
        frame = pd.DataFrame(preds.numpy(), index=[p.name for p in kept],
                             columns=columns)
        frame.index.name = "tile"
        frame.to_csv(args.output)
        print(f"wrote {args.output}  ({frame.shape[0]} tiles x {frame.shape[1]} genes)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

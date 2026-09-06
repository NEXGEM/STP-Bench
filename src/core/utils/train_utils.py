import os
import random
import re
import sys
import yaml
from datetime import datetime
from glob import glob
from pathlib import Path

from addict import Dict
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

import scanpy as sc
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class STPBenchProgressBar(TQDMProgressBar):
    """Progress bar that writes to sys.__stdout__ to survive stdout redirection.

    tqdm stores the file in two places: `bar.fp` (used for misc writes) and
    `bar.sp` (a closure created by status_printer() that captures the file at
    construction time and is used for all display updates).  Patching only
    `bar.fp` is not enough — we must also rebuild `bar.sp` with the real file.
    """

    def _out(self):
        return sys.__stdout__ if sys.__stdout__ is not None else sys.stderr

    def _redirect(self, bar):
        out = self._out()
        bar.fp = out
        bar.sp = bar.status_printer(out)
        return bar

    def init_train_tqdm(self):
        return self._redirect(super().init_train_tqdm())

    def init_validation_tqdm(self):
        return self._redirect(super().init_validation_tqdm())

    def init_test_tqdm(self):
        return self._redirect(super().init_test_tqdm())

    def init_predict_tqdm(self):
        return self._redirect(super().init_predict_tqdm())

    def init_sanity_tqdm(self):
        return self._redirect(super().init_sanity_tqdm())


def _stpbench_info(msg: str) -> None:
    from api.benchmark_logger import format_stpbench_line
    line = format_stpbench_line(msg, level="INFO")
    out = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
    print(line, flush=True, file=out)


# Short display names for known metric keys
_METRIC_LABELS = {
    "PearsonCorrCoef":    "PCC",
    "ConcordanceCorrCoef": "CCC",
    "MeanSquaredError":   "MSE",
    "MeanAbsoluteError":  "MAE",
    "ExplainedVariance":  "ExpVar",
    "RVDMetric":          "RVD",
}

def _print_eval_table(ckpt_name: str, metrics: dict) -> None:
    """Print a formatted evaluation results table to the real terminal."""
    from api.benchmark_logger import _color_ok, _B, _R, _GRN, _GRY, _D, _CYN

    out = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
    color = _color_ok()

    # Build rows: strip 'test_' prefix, apply short labels, skip 'epoch'
    rows = []
    epoch = None
    for key, val in metrics.items():
        if key == "epoch":
            epoch = val
            continue
        bare = key.removeprefix("test_").removeprefix("val_")
        label = _METRIC_LABELS.get(bare, bare)
        # suffix _hpg if present
        if bare.endswith("_hpg"):
            bare2 = bare[:-4]
            label = _METRIC_LABELS.get(bare2, bare2) + " (HPG)"
        rows.append((label, f"{val:.4f}" if isinstance(val, float) else str(val)))

    col1 = max(len(r[0]) for r in rows) if rows else 6
    col2 = max(len(r[1]) for r in rows) if rows else 6
    col1 = max(col1, 6)
    col2 = max(col2, 6)

    # Header line
    ep_str = f"  epoch={int(epoch)}" if epoch is not None else ""
    ckpt_short = ckpt_name if len(ckpt_name) <= 50 else "..." + ckpt_name[-47:]

    if color:
        sep   = f"{_D}{_GRY}{'─' * (col1 + col2 + 5)}{_R}"
        hdr   = f"  {_B}{'Metric':<{col1}}  {'Value':>{col2}}{_R}"
        title = f"{_B}{_CYN}[STPBench]{_R} {_D}{_GRY}Eval{_R}  {ckpt_short}{_D}{_GRY}{ep_str}{_R}"
        def row_str(label, val):
            return f"  {_GRY}{label:<{col1}}{_R}  {_GRN}{val:>{col2}}{_R}"
    else:
        sep   = "  " + "─" * (col1 + col2 + 3)
        hdr   = f"  {'Metric':<{col1}}  {'Value':>{col2}}"
        title = f"[STPBench] Eval  {ckpt_short}{ep_str}"
        def row_str(label, val):
            return f"  {label:<{col1}}  {val:>{col2}}"

    lines = ["", title, hdr, sep]
    for label, val in rows:
        lines.append(row_str(label, val))
    lines.append("")

    print("\n".join(lines), flush=True, file=out)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize_adata(adata: sc.AnnData, cpm=False, smooth=False) -> sc.AnnData:
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    """

    normed_adata = adata.copy()

    if cpm:
        sc.pp.normalize_total(normed_adata, target_sum=1e4)

    sc.pp.log1p(normed_adata)

    if smooth:
        new_X = []
        for _, df_row in normed_adata.obs.iterrows():
            row = int(df_row['array_row'])
            col = int(df_row['array_col'])

            neighbors_index = normed_adata.obs[
                ((normed_adata.obs['array_row'] >= row - 1) & (normed_adata.obs['array_row'] <= row + 1))
                & ((normed_adata.obs['array_col'] >= col - 1) & (normed_adata.obs['array_col'] <= col + 1))
            ].index
            neighbors = normed_adata[neighbors_index]
            nb_neighbors = len(neighbors)

            avg = neighbors.X.sum(0) / nb_neighbors
            new_X.append(avg)

        new_X = np.stack(new_X)
        normed_adata.X = new_X

    return normed_adata


def load_config(config_path: str):
    """Load a YAML config file into an addict Dict."""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return Dict(config)


def _merge_config(default_cfg: Dict, override_cfg: Dict) -> Dict:
    """Recursively merge override_cfg into default_cfg."""
    for key, value in override_cfg.items():
        if key in default_cfg and isinstance(default_cfg[key], dict) and isinstance(value, dict):
            _merge_config(default_cfg[key], value)
        else:
            default_cfg[key] = value
    return default_cfg


def load_config_with_default(config_path, default_config_path=None):
    """Load config with optional default config fallback."""
    if default_config_path is None:
        default_config_path = os.path.join(os.path.dirname(config_path), "default.yaml")

    if default_config_path and os.path.exists(default_config_path):
        default_cfg = load_config(default_config_path)
        cfg = load_config(config_path)
        return _merge_config(default_cfg, cfg)

    return load_config(config_path)


def load_configs(prefix_or_data_config, dataset=None, model=None):
    """Load data and model configs, merging model into data config.

    Supports two call patterns:
    - load_configs(<prefix>, <dataset>, <model>)
    - load_configs(<data_config_path>, <model_config_path>)
    """
    if dataset is not None and model is not None:
        data_config_path = f"./config/{prefix_or_data_config}/data/{dataset}/default.yaml"
        model_config_path = f"./config/{prefix_or_data_config}/model/{model}.yaml"
    else:
        data_config_path = prefix_or_data_config
        model_config_path = dataset

    if not data_config_path or not os.path.exists(data_config_path):
        raise ValueError(f"Data config not found: {data_config_path}")
    if not model_config_path or not os.path.exists(model_config_path):
        raise ValueError(f"Model config not found: {model_config_path}")

    default_cfg = load_config(data_config_path)
    model_cfg = load_config(model_config_path)
    return _merge_config(default_cfg, model_cfg)


def load_loggers(cfg: Dict, fold_info=None):
    """Return Logger instances for Trainer."""
    log_root_dir = cfg.GENERAL.log_path
    current_time = cfg.GENERAL.timestamp

    data_name = getattr(cfg.DATA, "name", None) or str(Path(cfg.config).parent)
    model_name = getattr(cfg.MODEL, "name", None) or Path(cfg.config).name

    base_log_dir = getattr(cfg.GENERAL, "log_dir", getattr(cfg, "log_dir", None))
    if base_log_dir is None:
        base_log_dir = f"{log_root_dir}/{data_name}/{model_name}/{current_time}"
    cfg.GENERAL.log_dir = base_log_dir

    fold_identifier = fold_info or f"fold{cfg.DATA.fold}"
    cfg.GENERAL.log_dir_fold = f"{base_log_dir}/{fold_identifier}"
    _stpbench_info(f"Log dir: {cfg.GENERAL.log_dir}")

    csv_logger = pl_loggers.CSVLogger(
        save_dir=cfg.GENERAL.log_dir_fold,
        name="",
        version=""
    )
    loggers = [csv_logger]

    if cfg.GENERAL.get("use_wandb", False):
        os.environ["WANDB_DIR"] = cfg.GENERAL.log_dir_fold
        os.makedirs(f'{cfg.GENERAL.log_dir_fold}/wandb', exist_ok=True)
        wandb_logger = pl_loggers.WandbLogger(
            save_dir=cfg.GENERAL.log_dir_fold,
            name=f'{data_name}-{model_name}-{current_time}-{fold_identifier}',
            project=cfg.GENERAL.get("wandb_project", "ST_prediction"),
        )
        loggers.insert(0, wandb_logger)

    return loggers


def load_callbacks(cfg: Dict):
    """Return EarlyStopping and Checkpoint callbacks."""
    callbacks = []

    monitor = cfg.TRAINING.monitor
    # target = f'val_{monitor}'
    target = 'val_target'
    patience = cfg.TRAINING.early_stopping.patience
    mode = cfg.TRAINING.mode
    save_best_only = cfg.TRAINING.save_best_only
    every_n_epochs = cfg.TRAINING.get('every_n_epochs', 10)
    log_name = f'{{epoch:02d}}-{{{target}:.4f}}'

    if mode == 'max':
        callbacks.append(
            EarlyStopping(
                monitor=target,
                min_delta=0.00,
                patience=patience,
                verbose=True,
                mode=mode
            )
        )
        checkpoint = ModelCheckpoint(
            monitor=target,
            dirpath=cfg.GENERAL.log_dir_fold,
            filename=log_name,
            verbose=True,
            save_last=False,
            save_top_k=1,
            mode=mode,
            save_weights_only=True
        )
    else:
        if save_best_only:
            callbacks.append(
                EarlyStopping(
                    monitor=target,
                    min_delta=0.00,
                    patience=patience,
                    verbose=True,
                    mode=mode
                )
            )
        checkpoint = ModelCheckpoint(
            monitor=target,
            dirpath=cfg.GENERAL.log_dir_fold,
            filename=log_name,
            verbose=True,
            save_last=True,
            save_top_k=1 if save_best_only else -1,
            every_n_epochs=1 if save_best_only else every_n_epochs,
            mode=mode,
            save_weights_only=True
        )

    callbacks.append(checkpoint)
    return callbacks


def get_best_epoch(metrics, target='test_PearsonCorrCoef', get_last=False, mode='max'):
    metrics = metrics.iloc[:-1]
    idx = metrics[target].idxmax() if mode == 'max' else metrics[target].idxmin()
    best_score = metrics[target][idx]
    best_epoch = metrics['epoch'][idx]

    def format_epoch(epoch):
        epoch = str(int(epoch))
        if len(epoch) == 1:
            epoch = f"0{epoch}"
        return epoch

    best_epoch = format_epoch(best_epoch)

    if get_last:
        last_epoch = metrics['epoch'].iloc[-1]
        last_epoch = format_epoch(last_epoch)
        return best_epoch, best_score, last_epoch
    return best_epoch, best_score


def get_ckpt_path(fold_dir):
    ckpt_paths = glob(f"{fold_dir}/*.ckpt")

    metrics = f"{fold_dir}/eval/metrics.csv"
    metrics = pd.read_csv(metrics)

    if len(ckpt_paths) > 1:
        best_epoch, best_score = get_best_epoch(metrics)
        ckpt_path = [ckpt for ckpt in ckpt_paths if f"epoch={best_epoch}" in ckpt][0]
    elif len(ckpt_paths) == 1:
        best_score = metrics['test_PearsonCorrCoef'].max()
        ckpt_path = ckpt_paths[0]
    else:
        _stpbench_info(f"Warning: no checkpoint found in {fold_dir}")
        return None, None

    return ckpt_path, best_score


def create_fresh_run(cfg):
    """Create a fresh training run with new timestamp and directories."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg.GENERAL.timestamp = timestamp
    cfg.log_dir = f"{cfg.log_dir_parent}/{timestamp}"
    cfg.GENERAL.log_dir = cfg.log_dir
    cfg.GENERAL.log_dir_parent = cfg.log_dir_parent
    os.makedirs(cfg.log_dir, exist_ok=True)

    with open(f"{cfg.log_dir}/config.yaml", 'w') as f:
        yaml.dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    return cfg


# ---- New utilities aligned with STHR
def create_data_module(cfg):
    """Create BaseDataModule instance from config."""
    from core import BaseDataModule
    return BaseDataModule(
        dataset_name=cfg.DATA.dataset_name,
        data_config=cfg.DATA
    )


def create_model_module(cfg, ckpt_path=None):
    """Create BaseModule instance from config."""
    from core import BaseModule
    if ckpt_path:
        return BaseModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    return BaseModule(cfg=cfg)


def setup_trainer(cfg, logger=False, callbacks=None, mode='cv'):
    """Setup PyTorch Lightning Trainer."""
    accelerator = cfg.GENERAL.get('accelerator', 'gpu' if torch.cuda.is_available() else 'cpu')
    devices = 1 if accelerator == 'cpu' or mode == 'inference' else cfg.GENERAL.gpu
    precision = '16-mixed' if accelerator != 'cpu' and cfg.GENERAL.get('use_amp', True) else '32'

    cb_list = list(callbacks) if callbacks else []
    if not any(isinstance(cb, TQDMProgressBar) for cb in cb_list):
        cb_list.append(STPBenchProgressBar())

    return pl.Trainer(
        accelerator=accelerator,
        strategy="auto",
        devices=devices,
        max_epochs=cfg.TRAINING.num_epochs if cfg.DATA.mode == 'cv' else None,
        check_val_every_n_epoch=1 if cfg.DATA.mode == 'cv' else None,
        logger=logger,
        callbacks=cb_list,
        precision=precision,
        num_sanity_val_steps=2 if cfg.GENERAL.get('debug', False) else 0
    )


def get_checkpoint_paths(ckpt_dir, fold):
    """Get sorted checkpoint paths from directory."""
    ckpt_paths = glob(f"{ckpt_dir}/fold{fold}/*.ckpt")
    ckpt_paths = [ckpt for ckpt in ckpt_paths if 'last' not in ckpt]
    ckpt_paths = sorted(ckpt_paths, key=lambda x: int(re.search(r"epoch=(\d+)", x).group(1)))

    last_ckpt = f"{ckpt_dir}/fold{fold}/last.ckpt"
    if os.path.exists(last_ckpt):
        ckpt_paths.append(last_ckpt)

    return ckpt_paths


def run_cross_validation(cfg, dm):
    """Run cross-validation training."""
    debug = cfg.GENERAL.get('debug', False)

    loggers = load_loggers(cfg) if not debug else False
    callbacks = load_callbacks(cfg) if not debug else None

    model = create_model_module(cfg)
    trainer = setup_trainer(cfg, logger=loggers, callbacks=callbacks, mode='cv')
    trainer.fit(model, datamodule=dm)


def run_evaluation(cfg, dm):
    """Run model evaluation on checkpoints."""
    pred_path_fold = f"{cfg.DATA.pred_path}/fold{cfg.DATA.fold}"
    os.makedirs(pred_path_fold, exist_ok=True)
    cfg.DATA.pred_path_fold = pred_path_fold
    
    log_path = cfg.GENERAL.log_path
    ckpt_dir = f'{log_path}/{cfg.config}/{cfg.GENERAL.timestamp}'
    # ckpt_dir = f'{log_path}/{data_name}/{model_name}/{cfg.GENERAL.timestamp}'
    ckpt_paths = get_checkpoint_paths(ckpt_dir, cfg.DATA.fold)

    csv_logger = pl_loggers.CSVLogger(
        save_dir=f"{log_path}/{cfg.config}/{cfg.GENERAL.timestamp}/fold{cfg.DATA.fold}/eval",
        name=f"",
        version=f""
    )

    trainer = setup_trainer(cfg, logger=csv_logger, mode='eval')

    if len(ckpt_paths) > 1:
        best_score = -np.inf
        best_outputs = None
        best_ckpt_name = None
        patience = cfg.TRAINING.early_stopping.patience
        patience_counter = 0
        save_pred_flag = cfg.GENERAL.get('save_predictions', False)

        for i, ckpt_path in enumerate(ckpt_paths):
            ckpt_name = os.path.basename(ckpt_path)
            match = re.search(r"epoch=(\d+)", ckpt_name)
            step_epoch = int(match.group(1)) if match else 0

            # Temporarily disable save_predictions for non-best models
            if save_pred_flag:
                cfg.GENERAL.save_predictions = False

            model = create_model_module(cfg, ckpt_path)
            model.step_epoch = step_epoch

            outputs = trainer.test(model, datamodule=dm, verbose=False)[0]
            current_pcc = outputs.get('test_PearsonCorrCoef', -np.inf)

            if current_pcc > best_score:
                _stpbench_info(f"New best: {ckpt_name}  PCC={current_pcc:.4f}  (prev={best_score:.4f})")
                best_score = current_pcc
                best_outputs = outputs
                best_ckpt_name = ckpt_name
                patience_counter = 0

                # Save predictions for best model
                if save_pred_flag and hasattr(model, 'preds'):
                    for batch_idx, pred in model.preds.items():
                        if hasattr(model, 'save_predictions'):
                            model.save_predictions(pred, batch_idx)

            else:
                patience_counter += 1
                _stpbench_info(f"No improvement (patience {patience_counter}/{patience})")

            # patience reached → early stop
            if patience_counter >= patience:
                _stpbench_info(f"Early stopping triggered after {patience} checkpoints without improvement")
                break

        if best_outputs is not None:
            _print_eval_table(best_ckpt_name, best_outputs)

        # Restore original save_predictions flag
        if save_pred_flag:
            cfg.GENERAL.save_predictions = True
            
    elif len(ckpt_paths) == 1:
        ckpt_path = ckpt_paths[0]
        ckpt_name = os.path.basename(ckpt_path)
        match = re.search(r"epoch=(\d+)", ckpt_name)
        step_epoch = int(match.group(1)) if match else 0
        
        model = create_model_module(cfg, ckpt_path)
        model.step_epoch = step_epoch
        outputs = trainer.test(model, datamodule=dm, verbose=False)
        _print_eval_table(ckpt_name, outputs[0])

    else:
        _stpbench_info("No checkpoint found — evaluating with the default model weights.")
        model = create_model_module(cfg)
        trainer.test(model, datamodule=dm, verbose=False)


def run_inference(cfg):
    """Run inference on all samples."""
    pred_path_fold = f"{cfg.DATA.pred_path}/fold{cfg.DATA.fold}"
    
    os.makedirs(pred_path_fold, exist_ok=True)
    cfg.DATA.pred_path_fold = pred_path_fold
    
    trainer = setup_trainer(cfg, mode='inference')
    model = create_model_module(cfg, cfg.MODEL.ckpt_path)

    data_dir = cfg.DATA.data_dir
    meta_dir = cfg.DATA.get('meta_dir', data_dir)
    ids = pd.read_csv(f"{meta_dir}/ids.csv")['sample_id'].tolist()
    
    ids_to_predict = []
    for _id in ids:
        if os.path.isfile(f"{pred_path_fold}/{_id}.h5ad"):
            _stpbench_info(f"{_id} is already predicted — skipping")
        else:
            ids_to_predict.append(_id)
    ids = ids_to_predict

    if len(ids) == 0:
        _stpbench_info("All samples are already predicted. Skipping inference.")
        return

    cfg.DATA.ids = ids
    gpus = cfg.GENERAL.gpu
    gpu_id = cfg.GENERAL.gpu_id

    if gpus > 1:
        if gpu_id < 0 or gpu_id >= gpus:
            raise ValueError(f"Invalid gpu_id={gpu_id}. Must be in range [0, {gpus-1}]")

        _stpbench_info(f"Distributing {len(ids)} samples across {gpus} GPUs (current: GPU {gpu_id})")

        gpu_samples = {i: [] for i in range(gpus)}
        for i, sample_id in enumerate(ids):
            gpu_idx = i % gpus
            gpu_samples[gpu_idx].append(sample_id)

        ids = gpu_samples[gpu_id]
        _stpbench_info(f"GPU {gpu_id} assigned {len(ids)} samples: {ids[:5]}{'...' if len(ids) > 5 else ''}")

        if len(ids) == 0:
            _stpbench_info(f"No samples assigned to GPU {gpu_id}. Skipping inference.")
            return
        
    for _id in tqdm(ids):
        output_file = f"{pred_path_fold}/{_id}.h5ad"

        cfg.DATA.data_id = _id
        cfg.DATA.output_path = output_file
        dm = create_data_module(cfg)
        trainer.predict(model, datamodule=dm, return_predictions=False)


def setup_env(cfg):
    fix_seed(cfg.GENERAL.seed)
    torch.set_float32_matmul_precision('high')


def setup_config(args):
    """Setup configuration based on CLI args."""
    repo_root = getattr(args, "repo_root", ".")
    default_data_name = None
    default_model_name = None
    if hasattr(args, "config_prefix"):
        config_dir = os.path.join(repo_root, 'config', args.config_prefix)
        data_config_path = os.path.join(config_dir, 'data', args.config_data, 'default.yaml')
        model_config_path = os.path.join(config_dir, 'model', f'{args.config_model}.yaml')
        cfg = load_configs(data_config_path, model_config_path)
        default_data_name = f"{args.config_prefix}/{args.config_data}"
        default_model_name = args.config_model
        cfg.DATA.name = default_data_name
        cfg.MODEL.name = default_model_name
        config_key = f"{default_data_name}/{default_model_name}"
        cfg.config = config_key
        mode = getattr(args, 'mode', 'cv')
        debug = getattr(args, 'debug', False)
        gpu = getattr(args, 'gpu', 1)
        gpu_id = getattr(args, 'gpu_id', 0)
    else:
        config_dir = os.path.join(repo_root, 'configs')
        data_config_path = os.path.join(config_dir, 'data', f'{args.data}.yaml')
        model_config_path = os.path.join(config_dir, 'model', f'{args.model}.yaml')
        cfg = load_configs(data_config_path, model_config_path)
        default_data_name = args.data
        default_model_name = args.model
        cfg.DATA.name = default_data_name
        cfg.MODEL.name = default_model_name
        config_key = f"{default_data_name}/{default_model_name}"
        cfg.config = config_key
        mode = args.mode
        debug = args.debug
        gpu = args.gpu
        gpu_id = args.gpu_id
    
    if mode == 'preprocess':
        return cfg

    cfg.GENERAL.log_path = os.path.join(repo_root, cfg.GENERAL.log_path)

    cfg.DATA.data_dir = os.path.join(repo_root, cfg.DATA.data_dir) if not os.path.isabs(cfg.DATA.data_dir) else cfg.DATA.data_dir
    cfg.DATA.output_dir = os.path.join(repo_root, cfg.DATA.output_dir)
    if cfg.DATA.get('meta_dir', None):
        cfg.DATA.meta_dir = os.path.join(repo_root, cfg.DATA.meta_dir) if not os.path.isabs(cfg.DATA.meta_dir) else cfg.DATA.meta_dir
        os.makedirs(cfg.DATA.meta_dir, exist_ok=True)
    else:
        cfg.DATA.meta_dir = cfg.DATA.data_dir
    if cfg.DATA.get('wsi_dir', None):
        cfg.DATA.wsi_dir = os.path.join(repo_root, cfg.DATA.wsi_dir)

    log_path = cfg.GENERAL.log_path
    os.makedirs(log_path, exist_ok=True)
    log_dir = os.path.join(log_path, cfg.DATA.name, cfg.MODEL.name)
    cfg.DATA.name = cfg.DATA.name.replace(f"{args.config_prefix}/", '')
    
    if mode == 'cv':
        
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        cfg.GENERAL.timestamp = timestamp

        if not debug:
            cfg.GENERAL.log_dir = os.path.join(log_dir, timestamp)
            os.makedirs(cfg.GENERAL.log_dir, exist_ok=True)

            with open(os.path.join(cfg.GENERAL.log_dir, "config.yaml"), 'w') as f:
                yaml.dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    elif mode == 'eval':
        timestamp = getattr(args, 'timestamp', None)
        if timestamp is None:
            if os.path.exists(log_dir) and os.listdir(log_dir):
                timestamp = sorted(os.listdir(log_dir))[-1]
            else:
                _stpbench_info("Log folder not found — evaluating without pre-trained weights.")
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                cfg.GENERAL.timestamp = timestamp
                cfg.GENERAL.log_dir = os.path.join(log_dir, timestamp)
                os.makedirs(cfg.GENERAL.log_dir, exist_ok=True)

        cfg.GENERAL.timestamp = timestamp

        data_dir = cfg.DATA.data_dir
        output_dir = cfg.DATA.output_dir
        wsi_dir = cfg.DATA.get('wsi_dir', None)
        batch_size = cfg.DATA.test_dataloader.batch_size
        load_level = cfg.DATA.get('load_level', None)

        config_path = os.path.join(log_dir, timestamp, "config.yaml")
        if os.path.exists(config_path):
            cfg = load_config(config_path)
            cfg.GENERAL.log_path = log_path
            cfg.GENERAL.timestamp = timestamp
        else:
            _stpbench_info("Config file not found in log dir — using default config.")
            cfg.GENERAL.log_dir = os.path.join(log_dir, timestamp)
            os.makedirs(cfg.GENERAL.log_dir, exist_ok=True)
            with open(os.path.join(cfg.GENERAL.log_dir, "config.yaml"), 'w') as f:
                yaml.dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        cfg.DATA.train_data_dir = cfg.DATA.data_dir
        cfg.DATA.data_dir = data_dir
        cfg.DATA.output_dir = output_dir
        if wsi_dir:
            cfg.DATA.wsi_dir = wsi_dir
        cfg.DATA.test_dataloader.batch_size = batch_size
        if load_level:
            cfg.DATA.load_level = load_level

        train_meta_dir = cfg.DATA.get('meta_dir', cfg.DATA.train_data_dir)
        cfg.MODEL.gene_path = f"{train_meta_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
        if not os.path.isfile(cfg.MODEL.gene_path):
            cfg.MODEL.gene_path = f"{cfg.DATA.train_data_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"

        if getattr(args, 'fold', None) is not None:
            cfg.DATA.fold = args.fold
        cfg.DATA.name = cfg.DATA.get('name', default_data_name)
        cfg.MODEL.name = cfg.MODEL.get('name', default_model_name)
        cfg.config = cfg.get('config', config_key)

        cfg.DATA.pred_path = f"{cfg.DATA.output_dir}/{cfg.DATA.name}/{cfg.MODEL.name}"

    elif mode == 'inference':
        fine_tune = cfg.MODEL.get('fine_tune', True)
        if fine_tune:
            def _select_ckpt_in_fold(fold_dir):
                ckpt_paths = glob(f"{fold_dir}/*.ckpt")
                ckpt_paths = [ckpt for ckpt in ckpt_paths if os.path.isfile(ckpt)]
                if not ckpt_paths:
                    return None, None

                def _score(ckpt):
                    name = os.path.basename(ckpt)
                    # match = re.search(r"val_target=([-0-9.]+)", name)
                    match = re.search(r"val_target=(-?\d+\.?\d*)", name)
                    if match:
                        return float(match.group(1))
                    match = re.search(r"epoch=(\\d+)", name)
                    if match:
                        return float(match.group(1))
                    return float(os.path.getmtime(ckpt))

                if len(ckpt_paths) == 1:
                    return ckpt_paths[0], _score(ckpt_paths[0])

                ckpt_paths = sorted(ckpt_paths, key=_score)
                return ckpt_paths[-1], _score(ckpt_paths[-1])

            ckpt_path = getattr(args, 'ckpt_path', None)
            if ckpt_path is None:
                raise ValueError("Inference mode requires --ckpt_path.")

            fold = getattr(args, 'fold', None)
            if os.path.isdir(ckpt_path):
                ckpt_dirs = sorted(glob(f"{ckpt_path}/*"))
                if not ckpt_dirs:
                    raise ValueError(f"Checkpoint directory {ckpt_path} is empty.")
                ckpt_dir = ckpt_dirs[-1]

                if fold is not None:
                    fold_dir = f"{ckpt_dir}/fold{fold}"
                    try:
                        ckpt_path, _ = get_ckpt_path(fold_dir)
                    except Exception:
                        ckpt_path, _ = _select_ckpt_in_fold(fold_dir)
                else:
                    best_score = -np.Inf
                    for fold_dir in glob(f"{ckpt_dir}/fold*"):
                        fold_num = int(re.search(r"fold(\\d+)", fold_dir).group(1))
                        try:
                            ckpt_path_fold, best_score_fold = get_ckpt_path(fold_dir)
                        except Exception:
                            ckpt_path_fold, best_score_fold = _select_ckpt_in_fold(fold_dir)
                        if ckpt_path_fold is None:
                            continue
                        if best_score_fold > best_score:
                            best_score = best_score_fold
                            ckpt_path = ckpt_path_fold
                            fold = fold_num
            else:
                if fold is None:
                    match = re.search(r"fold(\\d+)", str(Path(ckpt_path).parent))
                    if match:
                        fold = int(match.group(1))

            if not ckpt_path or not os.path.exists(ckpt_path):
                raise ValueError(f"Checkpoint path {ckpt_path} does not exist for inference.")
            _stpbench_info(f"Using checkpoint: {ckpt_path}")

            config_path = f"{Path(ckpt_path).parent.parent}/config.yaml"
            if os.path.exists(config_path):
                data_dir = cfg.DATA.data_dir
                output_dir = cfg.DATA.output_dir
                wsi_dir = cfg.DATA.get('wsi_dir', None)
                batch_size = cfg.DATA.test_dataloader.batch_size
                load_level = cfg.DATA.get('load_level', None)
                data_name = cfg.DATA.name
                save_predictions = cfg.GENERAL.get('save_predictions', True)

                cfg = load_config(config_path)
                timestamp = Path(config_path).parent.name
                cfg.GENERAL.log_path = log_path
                cfg.GENERAL.timestamp = timestamp

                cfg.DATA.train_data_dir = cfg.DATA.data_dir
                cfg.DATA.ref_data_dir = cfg.DATA.data_dir
                ref_data_dir = str(cfg.DATA.ref_data_dir).replace('/bench_data', '')
                ref_data_config = '/'.join(ref_data_dir.split('/')[-2:])
                cfg.DATA.output_dir = output_dir
                # cfg.DATA.output_dir = f"{output_dir}/{ref_data_config}"
                cfg.DATA.data_dir = data_dir
                cfg.DATA.name = data_name
                cfg.DATA.save_predictions = save_predictions
                if load_level:
                    cfg.DATA.load_level = load_level
                if wsi_dir:
                    cfg.DATA.wsi_dir = wsi_dir
                cfg.DATA.test_dataloader.batch_size = batch_size

                # cfg.DATA.name = cfg.DATA.get('name', cfg.DATA.name)
                # cfg.MODEL.name = cfg.MODEL.get('name', default_model_name)
                cfg.config = cfg.get('config', config_key)

            cfg.DATA.fold = fold
            cfg.MODEL.ckpt_path = ckpt_path
            train_data_dir = cfg.DATA.get('train_data_dir', cfg.DATA.get('ref_data_dir', cfg.DATA.data_dir))
            train_meta_dir = cfg.DATA.get('meta_dir', train_data_dir)
            cfg.MODEL.gene_path = f"{train_meta_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
            if not os.path.isfile(cfg.MODEL.gene_path):
                cfg.MODEL.gene_path = f"{train_data_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"

        else:
            cfg.MODEL.ckpt_path = None
            config_train_data = '/'.join(str(getattr(args, 'ckpt_path', '')).split('/')[-4:-2])
            if config_train_data:
                train_data_config_path = os.path.join(repo_root, "config", "bench", "data", "no_cpm", config_train_data, "default.yaml")
                if os.path.exists(train_data_config_path):
                    train_data_cfg = load_config(train_data_config_path)
                    for key, value in train_data_cfg.DATA.items():
                        if key in ['num_outputs', 'num_genes', 'gene_type']:
                            cfg.DATA[key] = value

                    cfg.DATA.train_data_dir = os.path.join(repo_root, train_data_cfg.DATA.data_dir)
                    cfg.DATA.ref_data_dir = cfg.DATA.train_data_dir
                    ref_data_dir = str(cfg.DATA.ref_data_dir).replace('/bench_data', '')
                    ref_data_config = '/'.join(ref_data_dir.split('/')[-2:])
                    # cfg.DATA.output_dir = f"{cfg.DATA.output_dir}/{ref_data_config}"
                else:
                    raise ValueError(f"Training data config file {train_data_config_path} does not exist.")
            else:
                raise ValueError("Please provide the training data config for inference when not fine-tuning.")

            if getattr(args, 'fold', None) is not None:
                cfg.DATA.fold = args.fold
            else:
                raise ValueError("Please specify the fold for inference when not fine-tuning.")

            train_meta_dir = cfg.DATA.get('meta_dir', cfg.DATA.train_data_dir)
            cfg.MODEL.gene_path = f"{train_meta_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
            if not os.path.isfile(cfg.MODEL.gene_path):
                cfg.MODEL.gene_path = f"{cfg.DATA.train_data_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"

        # cfg.DATA.name = cfg.DATA.get('name', cfg.DATA.name)
        # cfg.MODEL.name = cfg.MODEL.get('name', cfg.MODEL.name)
        cfg.MODEL.name = cfg.MODEL.get('name', default_model_name)
        cfg.MODEL.ref_data_dir = cfg.DATA.get('ref_data_dir', cfg.DATA.data_dir)
        
        if len(cfg.MODEL.name.split('/')) == 1:
            fm_encoder = cfg.MODEL.ckpt_path.split('/')[-5]
            cfg.MODEL.name = f"{fm_encoder}/{cfg.MODEL.name}"
        cfg.config = cfg.get('config', config_key)
        cfg.DATA.pred_path = f"{cfg.DATA.output_dir}/{cfg.DATA.name}/{cfg.MODEL.name}/{ref_data_config}"
        
    else:
        raise ValueError(f"Invalid mode: {mode}")

    cfg.GENERAL.debug = debug
    cfg.GENERAL.config_dir = config_dir
    cfg.GENERAL.log_path = log_path
    cfg.MODEL.num_genes = cfg.DATA.num_genes
    
    cfg.GENERAL.debug = debug
    cfg.GENERAL.config_dir = config_dir
    cfg.GENERAL.gpu = gpu
    cfg.GENERAL.gpu_id = gpu_id
    # cfg.DATA.pred_path = f"{cfg.DATA.output_dir}/{cfg.DATA.name}/{cfg.MODEL.name}"
    
    cfg.DATA.mode = mode
    return cfg

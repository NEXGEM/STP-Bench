"""Public Python API for running STPBench workflows."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import csv
import hashlib
import platform
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
import pandas as pd

from api.benchmark_logger import BenchmarkLogger
from api.config_resolver import NamedConfig, data_config_exists, resolve_data_config, resolve_model_config
from api.output_control import suppress_library_output
from api.result import BenchmarkResult


class _repomethod:
    """Descriptor that works like @classmethod but injects repo_root from the
    instance when called on one, so both STPred.list_data(repo_root="x") and
    stp.list_data() (using stp.repo_root automatically) work correctly."""

    def __init__(self, func):
        self.__func__ = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __get__(self, obj, objtype=None):
        if objtype is None:
            objtype = type(obj)
        func = self.__func__
        if obj is None:
            # Called on the class: STPred.list_data(repo_root="x")
            def _class_call(*args, **kwargs):
                return func(objtype, *args, **kwargs)
            _class_call.__name__ = self.__name__
            return _class_call
        else:
            # Called on an instance: stp.list_data()  — inject repo_root
            def _instance_call(*args, **kwargs):
                kwargs.setdefault("repo_root", obj.repo_root)
                return func(type(obj), *args, **kwargs)
            _instance_call.__name__ = self.__name__
            return _instance_call


class _model_list_method:
    """Descriptor for list_models.

    Class call: STPred.list_models(repo_root=".") lists available config names.
    Instance call: stp.list_models() lists models configured on that instance.
    """

    def __get__(self, obj, objtype=None):
        if objtype is None:
            objtype = type(obj)
        if obj is None:
            def _class_call(repo_root: str = "."):
                return objtype._list_named_configs("model", repo_root=repo_root)
            _class_call.__name__ = "list_models"
            return _class_call

        def _instance_call():
            return list(obj.models)
        _instance_call.__name__ = "list_models"
        return _instance_call


FEATURE_ORDER = ("global", "neighbor", "target")
REQUIRED_RUNTIME_FIELDS = {
    "GENERAL": ("seed", "log_path"),
    "TRAINING": ("num_k", "learning_rate", "num_epochs", "monitor", "mode", "early_stopping", "lr_scheduler"),
    "DATA": ("data_dir", "output_dir", "dataset_name", "gene_type", "num_genes", "num_outputs", "train_dataloader", "test_dataloader"),
    "MODEL": ("model_name",),
}


def _as_list(value: Optional[Sequence[str] | str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _feature_set(feature_type) -> Tuple[str, ...]:
    if feature_type in (None, "none"):
        return ()
    if isinstance(feature_type, list):
        result = []
        for ft in feature_type:
            if ft in FEATURE_ORDER:
                result.append(ft)
            else:
                raise ValueError(f"Invalid feature_type in list: {ft!r}")
        return tuple(f for f in FEATURE_ORDER if f in result)
    if feature_type == "all":
        return FEATURE_ORDER
    if feature_type in FEATURE_ORDER:
        return (feature_type,)
    raise ValueError(
        "feature_type must be one of 'global', 'neighbor', 'target', 'all', 'none', or a list thereof, "
        f"but got {feature_type!r}."
    )


def _coalesced_feature_type(features: Iterable[str]):
    feature_tuple = tuple(feature for feature in FEATURE_ORDER if feature in set(features))
    if not feature_tuple:
        return "none"
    if feature_tuple == FEATURE_ORDER:
        return "all"
    if len(feature_tuple) == 1:
        return feature_tuple[0]
    return list(feature_tuple)


def _load_yaml_file(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Legacy config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _nested_get(mapping: Dict[str, Any], dotted_key: str) -> Any:
    value = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _validate_runtime_config(data_cfg: NamedConfig, model_cfg: NamedConfig, cfg: Dict[str, Any]) -> None:
    missing = []
    for section, keys in REQUIRED_RUNTIME_FIELDS.items():
        section_value = cfg.get(section)
        if not isinstance(section_value, dict):
            missing.append(section)
            continue
        for key in keys:
            if _nested_get(cfg, f"{section}.{key}") is None:
                missing.append(f"{section}.{key}")

    for dataloader_key in ("DATA.train_dataloader", "DATA.test_dataloader"):
        dataloader = _nested_get(cfg, dataloader_key)
        if not isinstance(dataloader, dict):
            continue
        for key in ("batch_size", "num_workers", "pin_memory", "shuffle"):
            if key not in dataloader:
                missing.append(f"{dataloader_key}.{key}")

    if missing:
        raise ValueError(
            "Invalid STPred config pair.\n"
            f"Data config: {os.path.relpath(data_cfg.path)}\n"
            f"Model config: {os.path.relpath(model_cfg.path)}\n"
            "Missing required fields:\n"
            + "\n".join(f"- {field}" for field in missing)
        )

    feature_type = cfg.get("DATA", {}).get("feature_type", "global")
    _feature_set(feature_type)


def _read_last_csv_row(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[-1]
    parsed = {}
    for key, value in row.items():
        if value is None:
            parsed[key] = value
            continue
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def _list_checkpoints(fold_dir: str) -> List[str]:
    if not os.path.isdir(fold_dir):
        return []
    return sorted(path for path in glob(f"{fold_dir}/*.ckpt") if os.path.isfile(path))


def _sha256_file(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _write_manifest(cfg, payload: Dict[str, Any], action: str, manifest_path: str) -> str:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    manifest = {
        "api": "STPred",
        "action": action,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": payload["repo_root"],
        "git_commit": _git_commit(payload["repo_root"]),
        "python": sys.version,
        "platform": platform.platform(),
        "model": payload["model"],
        "data": payload["data"],
        "train_data": payload.get("train_data", payload["data"]),
        "config_key": payload["config_key"],
        "gpu_id": payload["gpu_id"],
        "timestamp": getattr(cfg.GENERAL, "timestamp", None),
        "data_config": payload.get("data_config"),
        "model_config": payload.get("model_config"),
        "data_config_sha256": _sha256_file(payload.get("data_config")),
        "model_config_sha256": _sha256_file(payload.get("model_config")),
    }
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return manifest_path


def _payload_logger(payload: Dict[str, Any]) -> BenchmarkLogger:
    return BenchmarkLogger(
        enabled=payload.get("verbose", True),
        log_file=payload.get("log_file"),
    )


def _finish_wandb(payload: Dict[str, Any]) -> None:
    if not payload.get("use_wandb", False):
        return
    import wandb

    wandb.finish()


def _collect_run_artifacts(cfg, payload: Dict[str, Any], action: str, folds: Sequence[int]) -> Dict[str, Any]:
    artifact: Dict[str, Any] = {
        "log_dir": getattr(cfg.GENERAL, "log_dir", None),
        "timestamp": getattr(cfg.GENERAL, "timestamp", None),
        "folds": {},
    }

    if action in {"train", "evaluate"}:
        for fold in folds:
            fold_dir = os.path.join(cfg.GENERAL.log_dir, f"fold{fold}")
            fold_artifact: Dict[str, Any] = {
                "log_dir": fold_dir,
                "checkpoints": _list_checkpoints(fold_dir),
            }
            metrics_path = os.path.join(fold_dir, "eval", "metrics.csv")
            metrics = _read_last_csv_row(metrics_path)
            if metrics is not None:
                fold_artifact["metrics_path"] = metrics_path
                fold_artifact["metrics"] = metrics
            pred_path = getattr(cfg.DATA, "pred_path", None)
            if pred_path:
                fold_artifact["prediction_dir"] = os.path.join(pred_path, f"fold{fold}")
            artifact["folds"][fold] = fold_artifact

    if action == "predict":
        pred_path = getattr(cfg.DATA, "pred_path", None)
        fold = getattr(cfg.DATA, "fold", payload.get("fold", 0))
        artifact["checkpoint"] = getattr(cfg.MODEL, "ckpt_path", None)
        if pred_path:
            artifact["prediction_dir"] = os.path.join(pred_path, f"fold{fold}")

    return artifact


def _run_single_model_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process-worker entrypoint for one model/data action."""

    from core import create_data_module, run_cross_validation, run_evaluation, run_inference, setup_env

    action = payload["action"]
    logger = _payload_logger(payload)
    action_start = time.perf_counter()
    if action == "sepal_train":
        with logger.section(
            "model_action",
            action=action,
            model=payload["model"],
            data=payload["data"],
            gpu_id=payload["gpu_id"],
        ):
            _run_sepal_train(payload)
            return {
                "model": payload["model"],
                "data": payload["data"],
                "train_data": payload.get("train_data", payload["data"]),
                "action": action,
                "config": payload["config_key"],
                "gpu_id": payload["gpu_id"],
                "timestamp": None,
                "elapsed_sec": round(time.perf_counter() - action_start, 3),
            }

    with suppress_library_output(), logger.section(
        "model_action",
        action=action,
        model=payload["model"],
        data=payload["data"],
        train_data=payload.get("train_data", payload["data"]),
        gpu_id=payload["gpu_id"],
    ):
        cfg = _build_runtime_cfg(payload)
        if action in {"train", "evaluate"}:
            manifest_path = _write_manifest(cfg, payload, action, os.path.join(cfg.GENERAL.log_dir, "manifest.yaml"))
        elif cfg.DATA.name.startswith("_wsi_predict/"):
            # WSI-predict output has no per-target directory (see
            # cfg.DATA.pred_path below), and output_dir is commonly reused
            # across separate predict() calls on different slides/batches —
            # write one manifest per SAMPLE (not one per call) so visualize()
            # can always find the right one by sample name alone, and a
            # later call can't silently overwrite an earlier one's
            # provenance record for a still-relevant sample.
            manifests_dir = os.path.join(cfg.DATA.output_dir, "_wsi_predict", "manifests")
            try:
                sample_ids = pd.read_csv(os.path.join(cfg.DATA.meta_dir, "ids.csv"))["sample_id"].dropna().astype(str).tolist()
            except Exception:
                sample_ids = []
            if not sample_ids:
                sample_ids = [cfg.DATA.name.split("/", 1)[1]]
            manifest_path = None
            for sample_id in sample_ids:
                manifest_path = _write_manifest(cfg, payload, action, os.path.join(manifests_dir, f"{sample_id}.yaml"))
        else:
            manifest_path = _write_manifest(
                cfg, payload, action,
                os.path.join(cfg.DATA.pred_path, f"fold{cfg.DATA.fold}", "manifest.yaml"),
            )
        setup_env(cfg)

        if action == "train":
            if cfg.MODEL.get("skip_train", False):
                logger.info(
                    "train_skipped",
                    model=payload["model"],
                    data=payload["data"],
                    reason="MODEL.skip_train=true",
                )
                folds = []
                return {
                    "model": payload["model"],
                    "data": payload["data"],
                    "train_data": payload.get("train_data", payload["data"]),
                    "action": action,
                    "config": payload["config_key"],
                    "gpu_id": payload["gpu_id"],
                    "timestamp": getattr(cfg.GENERAL, "timestamp", None),
                    "elapsed_sec": round(time.perf_counter() - action_start, 3),
                    "artifacts": _collect_run_artifacts(cfg, payload, action, folds),
                    "manifest": manifest_path,
                    "skipped": True,
                    "skip_reason": "MODEL.skip_train=true",
                }
            folds = list(range(cfg.TRAINING.num_k))
            for fold in folds:
                with logger.section("fold", action=action, model=payload["model"], data=payload["data"], fold=fold):
                    cfg.DATA.fold = fold
                    dm = create_data_module(cfg)
                    run_cross_validation(cfg, dm)
                    _finish_wandb(payload)
        elif action == "evaluate":
            folds = [payload.get("fold")] if payload.get("fold") is not None else list(range(cfg.TRAINING.num_k))
            for fold in folds:
                with logger.section("fold", action=action, model=payload["model"], data=payload["data"], fold=fold):
                    cfg.DATA.fold = fold
                    dm = create_data_module(cfg)
                    run_evaluation(cfg, dm)
        elif action == "predict":
            with logger.section("fold", action=action, model=payload["model"], data=payload["data"], fold=cfg.DATA.fold):
                run_inference(cfg)
            folds = [cfg.DATA.fold]
        else:
            raise ValueError(f"Unsupported action: {action}")

    return {
        "model": payload["model"],
        "data": payload["data"],
        "train_data": payload.get("train_data", payload["data"]),
        "action": action,
        "config": payload["config_key"],
        "gpu_id": payload["gpu_id"],
        "timestamp": getattr(cfg.GENERAL, "timestamp", None),
        "elapsed_sec": round(time.perf_counter() - action_start, 3),
        "artifacts": _collect_run_artifacts(cfg, payload, action, folds),
        "manifest": manifest_path,
    }


def _run_sepal_train(payload: Dict[str, Any]) -> None:
    from core import create_data_module, run_cross_validation, setup_env

    logger = _payload_logger(payload)

    localnet_payload = dict(payload)
    localnet_payload["action"] = "train"
    localnet_payload["runtime_config"] = payload["localnet_runtime_config"]
    localnet_payload["model"] = payload["localnet_model"]
    localnet_payload["config_key"] = f"{payload['data']}/{payload['localnet_model']}"

    localnet_cfg = _build_runtime_cfg(localnet_payload)
    setup_env(localnet_cfg)
    with logger.section("sepal_stage", stage="localnet", model=payload["localnet_model"], data=payload["data"]):
        for fold in range(localnet_cfg.TRAINING.num_k):
            with logger.section("fold", action="train", model=payload["localnet_model"], data=payload["data"], fold=fold):
                localnet_cfg.DATA.fold = fold
                dm = create_data_module(localnet_cfg)
                run_cross_validation(localnet_cfg, dm)
                _finish_wandb(payload)

    with logger.section("sepal_stage", stage="preprocess", model=payload["model"], data=payload["data"]):
        _run_command(
            payload["sepal_preprocess_command"],
            cwd=payload["sepal_preprocess_cwd"],
            log_path=payload.get("sepal_preprocess_log"),
        )

    sepal_payload = dict(payload)
    sepal_payload["action"] = "train"
    sepal_cfg = _build_runtime_cfg(sepal_payload)
    setup_env(sepal_cfg)
    with logger.section("sepal_stage", stage="sepal", model=payload["model"], data=payload["data"]):
        for fold in range(sepal_cfg.TRAINING.num_k):
            with logger.section("fold", action="train", model=payload["model"], data=payload["data"], fold=fold):
                sepal_cfg.DATA.fold = fold
                dm = create_data_module(sepal_cfg)
                run_cross_validation(sepal_cfg, dm)
                _finish_wandb(payload)


def _run_command(command: List[str], cwd: Optional[str] = None, log_path: Optional[str] = None) -> None:
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        try:
            with open(log_path, "w") as f:
                subprocess.run(command, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            detail = _tail_nonempty_lines(log_path, max_lines=12)
            if detail:
                raise RuntimeError(
                    f"Command failed with exit status {exc.returncode}: {command}\n"
                    f"Log tail from {log_path}:\n{detail}"
                ) from exc
            raise
    else:
        subprocess.run(command, cwd=cwd, check=True)


def _tail_nonempty_lines(path: str, max_lines: int = 12) -> str:
    try:
        with open(path, "r") as f:
            lines = [line.rstrip() for line in f if line.strip()]
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def _external_gene_overlap(train_gene_path: str, ext_meta_dir: str, ext_data_dir: str) -> Optional[Dict[str, Any]]:
    """Determine which of the training gene panel's genes are actually
    measured in an external dataset.

    A model's output columns are fixed to the gene panel it was trained on,
    but an external dataset (different platform/experiment) may not measure
    every one of those genes. Returns None when the full panel is available
    (no restriction needed) or the check can't be performed; otherwise
    {'genes': [...], 'indices': [...]} covering just the overlapping subset,
    in training-panel order — 'indices' are each gene's position in the
    original training panel, used to slice a fixed-width model output down
    to the genes that can actually be evaluated.
    """
    if not os.path.isfile(train_gene_path):
        return None
    import json as _json

    with open(train_gene_path) as f:
        train_genes = _json.load(f)['genes']

    ids_path = os.path.join(ext_meta_dir, "ids.csv")
    if not os.path.isfile(ids_path):
        return None
    sample_ids = pd.read_csv(ids_path)["sample_id"].dropna().astype(str).tolist()

    from dataset.path_utils import st_dir as _resolve_st_dir
    st_root = _resolve_st_dir(ext_data_dir)

    measured = None
    try:
        import scanpy as sc
        for name in sample_ids:
            path = os.path.join(st_root, f"{name}.h5ad")
            if not os.path.isfile(path):
                continue
            var_names = set(sc.read_h5ad(path, backed='r').var_names)
            measured = var_names if measured is None else (measured & var_names)
    except Exception:
        return None
    if not measured:
        return None

    overlap_indices = [i for i, g in enumerate(train_genes) if g in measured]
    if len(overlap_indices) == len(train_genes):
        return None
    return {
        "genes": [train_genes[i] for i in overlap_indices],
        "indices": overlap_indices,
    }


def _user_gene_overlap(
    train_gene_path: str, requested_genes: Sequence[str]
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Restrict a model's fixed-width output to a user-requested gene subset.

    Same {'genes': [...], 'indices': [...]} shape/training-panel-order
    contract as _external_gene_overlap, but driven by an explicit
    predict(gene_list=...) instead of by what an external dataset happens
    to measure. Returns (None, requested_genes) if the training panel
    can't be read, or (None, missing) if none of the requested genes are
    in the training panel. 'missing' always lists requested genes not
    found in the training panel, for the caller to warn about.
    """
    requested_genes = list(requested_genes)
    if not os.path.isfile(train_gene_path):
        return None, requested_genes
    import json as _json

    with open(train_gene_path) as f:
        train_genes = _json.load(f)['genes']
    train_index = {gene: i for i, gene in enumerate(train_genes)}

    missing = [gene for gene in requested_genes if gene not in train_index]
    overlap_indices = sorted({train_index[gene] for gene in requested_genes if gene in train_index})
    if not overlap_indices:
        return None, missing
    return {
        "genes": [train_genes[i] for i in overlap_indices],
        "indices": overlap_indices,
    }, missing


def _classify_predict_target(data: str, repo_root: str = ".") -> str:
    """Classify predict()'s `data` argument.

    Checks whether a real, on-disk named config exists first — it always
    wins outright, so this can't be confused by a coincidentally-named
    relative directory/file sitting under repo_root. This is a pure
    existence check (no YAML parsing), so a malformed-but-present config
    doesn't get silently reinterpreted as a WSI path — it's classified as
    "named_config" here and lets resolve_data_config's real parse error
    surface later, instead of masking it. Only falls through to filesystem
    inspection (resolved relative to repo_root, matching every other path
    in this file) when no such config exists at all.
    """
    if data_config_exists(data, repo_root=repo_root):
        return "named_config"

    from preprocess.prepare_data import _WSI_EXTENSIONS, discover_wsi_files

    abs_data = _abs_path(repo_root, data)
    if os.path.isdir(abs_data):
        if glob(os.path.join(abs_data, "patches", "*.h5")):
            return "asset_dir"
        if discover_wsi_files(abs_data):
            return "wsi_dir"
        return "named_config"
    if os.path.isfile(abs_data) and abs_data.lower().endswith(_WSI_EXTENSIONS):
        return "wsi_file"
    return "named_config"


def _build_runtime_cfg(payload: Dict[str, Any]):
    from addict import Dict
    from core import load_config

    repo_root = payload["repo_root"]
    action = payload["action"]
    train_data = payload.get("train_data", payload["data"])
    cfg = Dict(payload["runtime_config"])
    cfg.DATA.name = payload["data"]
    cfg.MODEL.name = payload["model"]
    cfg.config = payload["config_key"]
    cfg.GENERAL.debug = payload["debug"]
    cfg.GENERAL.use_wandb = payload.get("use_wandb", False)
    cfg.GENERAL.wandb_project = payload.get("wandb_project", "ST_prediction")
    cfg.GENERAL.gpu = 1
    cfg.GENERAL.gpu_id = payload["gpu_id"]
    cfg.GENERAL.config_dir = os.path.join(repo_root, "config")
    cfg.GENERAL.log_path = _abs_path(repo_root, cfg.GENERAL.get("log_path", "./logs"))

    if "save_predictions" not in cfg.GENERAL:
        cfg.GENERAL.save_predictions = True
    if "num_k" not in cfg.TRAINING:
        cfg.TRAINING.num_k = 1

    cfg.DATA.data_dir = _abs_path(repo_root, cfg.DATA.data_dir)
    cfg.DATA.output_dir = _abs_path(repo_root, cfg.DATA.get("output_dir", "output/pred"))
    if cfg.DATA.get("meta_dir", None):
        cfg.DATA.meta_dir = _abs_path(repo_root, cfg.DATA.meta_dir)
    if cfg.DATA.get("wsi_dir", None):
        cfg.DATA.wsi_dir = _abs_path(repo_root, cfg.DATA.wsi_dir)

    log_dir_parent = os.path.join(cfg.GENERAL.log_path, train_data, cfg.MODEL.name)

    if action == "train":
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        cfg.GENERAL.timestamp = timestamp
        cfg.GENERAL.log_dir = os.path.join(log_dir_parent, timestamp)
        cfg.MODEL.data_dir = cfg.DATA.get("meta_dir", cfg.DATA.data_dir)
        if not payload["debug"]:
            os.makedirs(cfg.GENERAL.log_dir, exist_ok=True)
            with open(os.path.join(cfg.GENERAL.log_dir, "config.yaml"), "w") as f:
                yaml.dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    elif action == "evaluate":
        timestamp = payload.get("timestamp") or _latest_timestamp(log_dir_parent)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        cfg.GENERAL.timestamp = timestamp
        cfg.GENERAL.log_dir = os.path.join(log_dir_parent, timestamp)
        saved_cfg_path = os.path.join(cfg.GENERAL.log_dir, "config.yaml")
        train_meta_dir_from_ckpt = None
        if os.path.exists(saved_cfg_path):
            current_data = cfg.DATA
            cfg = load_config(saved_cfg_path)
            cfg.GENERAL.log_path = _abs_path(repo_root, cfg.GENERAL.get("log_path", "./logs"))
            cfg.GENERAL.timestamp = timestamp
            cfg.GENERAL.debug = payload["debug"]
            cfg.GENERAL.use_wandb = payload.get("use_wandb", False)
            cfg.GENERAL.wandb_project = payload.get("wandb_project", "ST_prediction")
            cfg.GENERAL.gpu = 1
            cfg.GENERAL.gpu_id = payload["gpu_id"]
            train_data_name = cfg.DATA.get("name", train_data)
            cfg.DATA.train_data_dir = _abs_path(repo_root, cfg.DATA.data_dir)
            # Capture the train run's own meta_dir *before* it's overwritten
            # below with the eval data's meta_dir — otherwise external eval
            # silently re-reads the training data's ids.csv/genes.json
            # instead of the eval dataset's. Kept as a local var, not
            # cfg.DATA.ref_data_dir, so it doesn't leak into the Dataset
            # constructor for internal eval (see below).
            train_meta_dir_from_ckpt = cfg.DATA.get("meta_dir", cfg.DATA.train_data_dir)
            cfg.DATA.data_dir = current_data.data_dir
            cfg.DATA.output_dir = current_data.output_dir
            cfg.DATA.name = current_data.get("name", payload["data"])
            cfg.DATA.meta_dir = current_data.get("meta_dir", current_data.data_dir)
            cfg.DATA.train_data_name = train_data_name
            if current_data.get("wsi_dir", None):
                cfg.DATA.wsi_dir = current_data.wsi_dir
            if current_data.get("test_dataloader", None):
                cfg.DATA.test_dataloader = current_data.test_dataloader
        else:
            os.makedirs(cfg.GENERAL.log_dir, exist_ok=True)
        train_data_dir = cfg.DATA.get("train_data_dir", cfg.DATA.data_dir)
        train_meta_dir = train_meta_dir_from_ckpt or cfg.DATA.get("meta_dir", train_data_dir)
        if payload["data"] != train_data:
            # External evaluation only: point the dataset at the training
            # run's own meta_dir/data_dir so it resolves the model's trained
            # gene panel and loads the (unsplit) external samples. For
            # internal CV evaluation, ref_data_dir must stay unset —
            # STDataset only applies its phase/fold test-split filter when
            # ref_data_dir is None, so setting it here would evaluate every
            # fold's checkpoint against the *entire* dataset instead of just
            # its held-out test split.
            cfg.DATA.ref_data_dir = train_meta_dir
            cfg.DATA.ref_asset_dir = train_data_dir
        gene_path = f"{train_meta_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
        cfg.MODEL.gene_path = gene_path if os.path.isfile(gene_path) else f"{train_data_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
        cfg.MODEL.ref_data_dir = train_meta_dir
        if payload["data"] != train_data:
            overlap = _external_gene_overlap(cfg.MODEL.gene_path, cfg.DATA.meta_dir, cfg.DATA.data_dir)
            if overlap is not None:
                cfg.DATA.genes_override = overlap["genes"]
                cfg.DATA.num_outputs = len(overlap["genes"])
                cfg.DATA.gene_output_indices = overlap["indices"]
        if payload["data"] == train_data:
            cfg.DATA.pred_path = f"{cfg.DATA.output_dir}/{cfg.DATA.name}/{cfg.MODEL.name}"
        else:
            cfg.DATA.pred_path = f"{cfg.DATA.output_dir}/{cfg.DATA.name}/{cfg.MODEL.name}/{train_data}"

    elif action == "predict":
        if cfg.MODEL.get("skip_train", False):
            ckpt_path = None
            fold = payload.get("fold") or 0
        else:
            ckpt_path, fold = _resolve_predict_checkpoint(payload["ckpt_path"], payload.get("fold"))
        cfg.DATA.fold = fold
        cfg.MODEL.ckpt_path = ckpt_path
        config_path = f"{Path(ckpt_path).parent.parent}/config.yaml" if ckpt_path else None
        ref_data_config = cfg.DATA.name
        if config_path and os.path.exists(config_path):
            current_data = cfg.DATA
            save_predictions = cfg.GENERAL.get("save_predictions", True)
            cfg = load_config(config_path)
            cfg.GENERAL.log_path = _abs_path(repo_root, cfg.GENERAL.get("log_path", "./logs"))
            cfg.GENERAL.debug = payload["debug"]
            cfg.GENERAL.use_wandb = payload.get("use_wandb", False)
            cfg.GENERAL.wandb_project = payload.get("wandb_project", "ST_prediction")
            cfg.GENERAL.gpu = 1
            cfg.GENERAL.gpu_id = payload["gpu_id"]
            cfg.DATA.train_data_dir = _abs_path(repo_root, cfg.DATA.data_dir)
            cfg.DATA.ref_data_dir = cfg.DATA.get("meta_dir", cfg.DATA.train_data_dir)
            cfg.DATA.ref_asset_dir = cfg.DATA.train_data_dir
            ref_data_config = cfg.DATA.get("name", payload["data"])
            cfg.DATA.data_dir = current_data.data_dir
            cfg.DATA.meta_dir = current_data.get("meta_dir", current_data.data_dir)
            cfg.DATA.output_dir = current_data.output_dir
            cfg.DATA.name = payload["data"]
            cfg.DATA.save_predictions = save_predictions
            if current_data.get("wsi_dir", None):
                cfg.DATA.wsi_dir = current_data.wsi_dir
            if current_data.get("test_dataloader", None):
                cfg.DATA.test_dataloader = current_data.test_dataloader
        cfg.DATA.fold = fold
        cfg.MODEL.ckpt_path = ckpt_path
        if payload.get("train_data_config"):
            train_cfg = load_config(payload["train_data_config"])
            train_data_section = train_cfg.DATA
            train_data_dir = _abs_path(repo_root, train_data_section.data_dir)
            train_meta_dir = _abs_path(
                repo_root,
                train_data_section.get("meta_dir", train_data_section.data_dir),
            )
            cfg.DATA.train_data_name = train_data_section.get("name", train_data)
        else:
            train_data_dir = cfg.DATA.get("train_data_dir", cfg.DATA.get("ref_data_dir", cfg.DATA.data_dir))
            train_meta_dir = cfg.DATA.get("meta_dir", train_data_dir)
        cfg.DATA.train_data_dir = train_data_dir
        cfg.DATA.ref_data_dir = train_meta_dir
        cfg.DATA.ref_asset_dir = train_data_dir
        gene_path = f"{train_meta_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
        cfg.MODEL.gene_path = gene_path if os.path.isfile(gene_path) else f"{train_data_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
        cfg.MODEL.ref_data_dir = train_meta_dir
        if payload.get("gene_list"):
            overlap, missing = _user_gene_overlap(cfg.MODEL.gene_path, payload["gene_list"])
            if missing:
                # Use the payload's own logger, not warnings.warn(): this runs
                # inside _run_single_model_action's suppress_library_output(),
                # which filters out UserWarning by category and would
                # otherwise silently swallow this.
                _payload_logger(payload).warning(
                    "gene_list_genes_missing",
                    genes=",".join(missing),
                    gene_path=cfg.MODEL.gene_path,
                )
            if overlap is None:
                raise ValueError(
                    "None of the requested gene_list genes are in the training "
                    f"gene panel ({cfg.MODEL.gene_path})."
                )
            cfg.DATA.genes_override = overlap["genes"]
            cfg.DATA.num_outputs = len(overlap["genes"])
            cfg.DATA.gene_output_indices = overlap["indices"]
        if payload.get("batch_size"):
            cfg.DATA.test_dataloader.batch_size = payload["batch_size"]
        if cfg.DATA.name.startswith("_wsi_predict/"):
            # No per-target (slide-name) directory for WSI-predict output —
            # samples are already distinguished by their own filename, and
            # output_dir is commonly reused across separate predict() calls
            # on different slides/assets.
            cfg.DATA.pred_path = f"{cfg.DATA.output_dir}/_wsi_predict/predictions/{cfg.MODEL.name}/{ref_data_config}"
        else:
            cfg.DATA.pred_path = f"{cfg.DATA.output_dir}/{cfg.DATA.name}/{cfg.MODEL.name}/{ref_data_config}"

    cfg.MODEL.num_genes = cfg.DATA.get("num_genes", cfg.MODEL.get("num_genes", None))
    cfg.DATA.mode = {"train": "cv", "evaluate": "eval", "predict": "inference"}[action]
    return cfg


def _abs_path(repo_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def _estimate_patch_spacing(coords) -> Optional[float]:
    """Fallback patch footprint when trident's own `patch_size_level0` attr
    isn't available (e.g. a legacy/non-trident coords file): the median
    nearest-neighbor distance between patch centers is a reasonable stand-in
    for the true non-overlapping patch size."""
    import numpy as np
    if len(coords) < 2:
        return None
    try:
        from scipy.spatial import cKDTree
        dists, _ = cKDTree(coords).query(coords, k=2)
        nn_dist = dists[:, 1]
    except Exception:
        return None
    nn_dist = nn_dist[nn_dist > 0]
    return float(np.median(nn_dist)) if len(nn_dist) else None


def _draw_patches(ax, coords, values, patch_px: Optional[float]):
    """Render one square per patch, sized to its true non-overlapping
    footprint (`patch_px`), colored by `values`. Falls back to small dot
    markers only if no footprint size could be determined at all — patches
    are a dense grid, not sparse spots, and a fixed small dot misleadingly
    looks like sparse Visium-style spots regardless of true patch density."""
    import numpy as np
    import matplotlib.colors as mcolors
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    if not patch_px or patch_px <= 0:
        return ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=8, alpha=0.8)

    rects = [
        Rectangle((x - patch_px / 2, y - patch_px / 2), patch_px, patch_px)
        for x, y in coords
    ]
    norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
    collection = PatchCollection(rects, cmap="viridis", norm=norm)
    collection.set_array(np.asarray(values))
    ax.add_collection(collection)
    return collection


def _latest_timestamp(log_dir_parent: str) -> Optional[str]:
    if os.path.isdir(log_dir_parent) and os.listdir(log_dir_parent):
        return sorted(os.listdir(log_dir_parent))[-1]
    return None


def _resolve_predict_checkpoint(ckpt_path: str, fold: Optional[int]) -> Tuple[str, int]:
    if not ckpt_path:
        raise ValueError("predict() requires ckpt_path or an existing default checkpoint directory.")
    if os.path.isfile(ckpt_path):
        if fold is None:
            match = re.search(r"fold(\d+)", str(Path(ckpt_path).parent))
            fold = int(match.group(1)) if match else 0
        return ckpt_path, fold
    if not os.path.isdir(ckpt_path):
        raise ValueError(f"Checkpoint path does not exist: {ckpt_path}")

    if glob(f"{ckpt_path}/fold*"):
        # ckpt_path already points at a specific run directory (contains fold* dirs directly).
        search_root = ckpt_path
    else:
        timestamp_dirs = sorted(path for path in glob(f"{ckpt_path}/*") if os.path.isdir(path))
        search_root = timestamp_dirs[-1] if timestamp_dirs else ckpt_path
    if fold is not None:
        fold_dirs = [f"{search_root}/fold{fold}"]
    else:
        fold_dirs = sorted(glob(f"{search_root}/fold*"))
    for fold_dir in fold_dirs:
        if not os.path.isdir(fold_dir):
            continue
        ckpts = sorted(path for path in glob(f"{fold_dir}/*.ckpt") if "last" not in os.path.basename(path))
        if not ckpts:
            ckpts = sorted(glob(f"{fold_dir}/*.ckpt"))
        if ckpts:
            match = re.search(r"fold(\d+)", fold_dir)
            return ckpts[-1], int(match.group(1)) if match else (fold or 0)
    raise ValueError(f"No checkpoint found under {ckpt_path}.")


class STPred:
    """User-facing API for the STPBench multi-model benchmark."""

    def __init__(
        self,
        models: Sequence[str] | str,
        repo_root: str = ".",
        gpu: int = 1,
        gpu_id: int = 0,
        debug: bool = False,
        dry_run: bool = False,
        preprocess_overrides: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        log_file: Optional[str] = None,
        wandb: bool = False,
        wandb_project: str = "ST_prediction",
    ):
        self.repo_root = os.path.abspath(repo_root)
        self.models = _as_list(models)
        if not self.models:
            raise ValueError("At least one model must be provided.")
        if gpu < 1:
            raise ValueError("gpu must be at least 1.")
        self.gpu = gpu
        self.gpu_id = gpu_id
        self.debug = debug
        self.dry_run = dry_run
        self.preprocess_overrides = preprocess_overrides or {}
        self.verbose = verbose
        self.log_file = log_file
        self.use_wandb = wandb
        self.wandb_project = wandb_project
        self.logger = BenchmarkLogger(enabled=verbose, log_file=log_file)
        self.state: Dict[str, Any] = {
            "internal_data": None,
            "external_data": None,
            "train_results": {},
            "eval_results": {},
            "predict_results": {},
            "run_timestamps": {},
            "checkpoint_roots": {},
        }
        self.logger.info(
            "initialized",
            models=",".join(self.models),
            gpu=self.gpu,
            gpu_id=self.gpu_id,
            dry_run=self.dry_run,
            wandb=self.use_wandb,
            wandb_project=self.wandb_project if self.use_wandb else None,
        )

    @_repomethod
    def list_data(cls, repo_root: str = ".") -> List[str]:
        """List data config names available under config/data."""
        return cls._list_named_configs("data", repo_root=repo_root)

    list_models = _model_list_method()

    @_repomethod
    def list_available_models(cls, repo_root: str = ".") -> List[str]:
        """List model config names available under config/model."""
        return cls._list_named_configs("model", repo_root=repo_root)

    @_repomethod
    def describe_data(cls, name: str, repo_root: str = ".") -> Dict[str, Any]:
        cfg = resolve_data_config(name, repo_root=repo_root)
        data = cfg.config.get("DATA", {})
        preprocess = cfg.config.get("preprocess", {})
        return {
            "name": cfg.name,
            "path": cfg.path,
            "data_dir": data.get("data_dir"),
            "output_dir": data.get("output_dir"),
            "gene_type": data.get("gene_type"),
            "num_genes": data.get("num_genes"),
            "num_outputs": data.get("num_outputs"),
            "technology": data.get("tech"),
            "preprocess": preprocess,
        }

    @_repomethod
    def describe_model(cls, name: str, repo_root: str = ".") -> Dict[str, Any]:
        cfg = resolve_model_config(name, repo_root=repo_root)
        data = cfg.config.get("DATA", {})
        model = cfg.config.get("MODEL", {})
        return {
            "name": cfg.name,
            "path": cfg.path,
            "model_name": model.get("model_name"),
            "dataset_name": data.get("dataset_name"),
            "feature_type": data.get("feature_type", "global"),
            "patch_encoder": data.get("model_name"),
        }


    @classmethod
    def from_run(
        cls,
        data: str,
        models: Sequence[str] | str,
        timestamp: Optional[Dict[str, str] | str] = None,
        state_path: Optional[str] = None,
        **kwargs,
    ) -> "STPred":
        """Create a STPred instance from a known training run or saved state."""

        stp = cls(models=models, **kwargs)
        if state_path is not None:
            stp.load_state(state_path)
            return stp
        stp.state["internal_data"] = data
        if timestamp is None:
            timestamps = {
                model: _latest_timestamp(stp._default_ckpt_root(data, model))
                for model in stp.models
            }
            timestamps = {model: value for model, value in timestamps.items() if value}
        elif isinstance(timestamp, str):
            timestamps = {model: timestamp for model in stp.models}
        else:
            timestamps = dict(timestamp)
        stp.state["run_timestamps"][data] = timestamps
        stp.state["checkpoint_roots"][data] = {
            model: os.path.join(stp._default_ckpt_root(data, model), ts)
            for model, ts in timestamps.items()
        }
        return stp

    @staticmethod
    def _data_config_template(name: str) -> Dict[str, Any]:
        return {
            "GENERAL": {
                "seed": 2021,
                "log_path": "./logs",
                "save_predictions": True,
                "use_wandb": False,
                "wandb_project": "ST_prediction",
            },
            "TRAINING": {
                "num_k": 5,
                "learning_rate": 1.0e-4,
                "num_epochs": 200,
                "monitor": "PearsonCorrCoef",
                "mode": "max",
                "save_best_only": True,
                "early_stopping": {"patience": 20},
                "lr_scheduler": {"patience": 5, "factor": 0.1},
                "every_n_epochs": 1,
            },
            "DATA": {
                "data_dir": "/path/to/processed_data",
                "meta_dir": f"input/{name}",
                "output_dir": "output/pred",
                "dataset_name": "STDataset",
                "gene_type": "hmhvg",
                "num_genes": 200,
                "num_outputs": 200,
                "normalize": True,
                "cpm": False,
                "smooth": False,
                "model_name": "uni_v2",
                "load_level": "patch",
                "tech": "Visium",
                "train_dataloader": {"batch_size": 128, "num_workers": 4, "pin_memory": False, "shuffle": True},
                "test_dataloader": {"batch_size": 1, "num_workers": 4, "pin_memory": False, "shuffle": False},
            },
            "preprocess": {
                "mode": "raw",
                "platform": "visium",
                "input_dir": "/path/to/raw_data",
                "output_dir": "/path/to/processed_data",
                "meta_dir": f"input/{name}",
                "overwrite": False,
            },
        }

    @classmethod
    def init_data_config(cls, name: str, output: Optional[str] = None, repo_root: str = ".") -> str:
        """Write a minimal editable data config template and return its path."""

        output = output or os.path.join(repo_root, "config", "data", f"{name}.yaml")
        cls._write_yaml_template(output, cls._data_config_template(name))
        return output

    @classmethod
    def init_model_config(cls, name: str, output: Optional[str] = None, repo_root: str = ".") -> str:
        """Write a minimal editable model config template and return its path."""

        output = output or os.path.join(repo_root, "config", "model", f"{name}.yaml")
        config = {
            "MODEL": {
                "model_name": f"{name}.{name}",
                "extra_preprocess": [],
                "adapter": "default",
            },
            "DATA": {
                "dataset_name": "STDataset",
                "feature_type": "global",
            },
        }
        cls._write_yaml_template(output, config)
        return output

    @staticmethod
    def _write_yaml_template(path: str, config: Dict[str, Any], overwrite: bool = False) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"Config already exists: {path}")
        with open(path, "w") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    @staticmethod
    def _list_named_configs(kind: str, repo_root: str = ".") -> List[str]:
        config_dir = os.path.join(os.path.abspath(repo_root), "config", kind)
        if not os.path.isdir(config_dir):
            return []
        names = []
        for root, _, files in os.walk(config_dir):
            for filename in files:
                if not filename.endswith(".yaml"):
                    continue
                path = os.path.join(root, filename)
                rel = os.path.relpath(path, config_dir)
                names.append(os.path.splitext(rel)[0].replace(os.sep, "/"))
        return sorted(names)

    def benchmark(self, internal_data: str, external_data: Optional[str] = None, **kwargs) -> BenchmarkResult:
        """Run preprocess, train, internal evaluation, and optional external prediction."""

        with self.logger.section("benchmark", internal_data=internal_data, external_data=external_data):
            summary: Dict[str, Any] = {}
            summary["preprocess_internal"] = self.preprocess(internal_data, **kwargs)
            summary["train"] = self.train(internal_data)
            summary["evaluate_internal"] = self.evaluate(mode="int", data=internal_data)
            if external_data is not None:
                # Use the external dataset's own configured preprocess.mode
                # (e.g. 'stpbench' for a fully HEST-onboarded labeled dataset)
                # rather than forcing 'inference' — 'inference' mode is for
                # genuinely H&E-only, ground-truth-free prediction (stp.predict
                # on a dataset whose own config sets preprocess.mode: inference),
                # not for a complete external dataset like hest/LUAD that we
                # also evaluate against known labels.
                ext_kwargs = dict(kwargs)
                ext_kwargs.setdefault("skip_model_preprocess", True)
                summary["preprocess_external"] = self.preprocess(external_data, **ext_kwargs)
                # preprocess_external above may have overwritten
                # state["internal_data"] with `external_data` (its
                # state-write heuristic treats any non-inference/wsi_only
                # mode as "internal", which is wrong here) — pin train_data
                # explicitly instead of letting predict()/evaluate() fall
                # back to state.
                summary["predict_external"] = self.predict(external_data, train_data=internal_data)
                summary["evaluate_external"] = self.evaluate(
                    mode="ext", data=internal_data, external_data=external_data
                )
            return BenchmarkResult({"steps": summary})

    def preprocess(self, data: str, **overrides) -> Dict[str, Any]:
        """Run one deduplicated preprocessing plan for all configured models."""

        # skip_model_preprocess: used when this data will be used as an
        # EXTERNAL eval set for a different training run — model-specific
        # extra_preprocess scripts (EGN/EGGN graph building, OmiCLIP
        # similarity matrices, ...) behave differently for external data
        # (train-vs-external namespacing, using the *training* set's key
        # samples) and must be run later via _run_external_model_preprocess
        # with that context, not here where `data` is treated as a
        # standalone/primary dataset.
        skip_model_preprocess = overrides.pop("skip_model_preprocess", False)
        models = overrides.pop("models", None)
        data_cfg = self._resolve_data(data)
        model_cfgs = self._resolve_models(models)
        plan = self._build_preprocess_plan(
            data_cfg, model_cfgs, overrides, skip_model_preprocess=skip_model_preprocess,
        )

        if self.dry_run or overrides.get("dry_run", False):
            self.logger.info(
                "preprocess",
                status="dry_run",
                data=data_cfg.name,
                models=",".join(plan["models"]),
                feature_tasks=len(plan["feature_tasks"]),
                model_preprocess_tasks=len(plan["model_preprocess_tasks"]),
            )
            return {"dry_run": True, "plan": plan}

        with self.logger.section(
            "preprocess",
            data=data_cfg.name,
            models=",".join(plan["models"]),
            feature_tasks=len(plan["feature_tasks"]),
            model_preprocess_tasks=len(plan["model_preprocess_tasks"]),
        ):
            config = dict(plan["base_config"])
            config["feature_type"] = plan["raw_feature_type"]
            config["repo_root"] = self.repo_root
            config["python"] = sys.executable
            from api.data_pipeline import DataPipeline

            pipeline = DataPipeline(config=config)
            with suppress_library_output():
                with self.logger.section("preprocess_raw", data=data_cfg.name, mode=pipeline.mode):
                    pipeline.preprocess()
                if pipeline.mode in ["raw", "stpbench"]:
                    with self.logger.section("preprocess_genesets", data=data_cfg.name):
                        pipeline.prepare_genesets()
                    with self.logger.section("preprocess_splits", data=data_cfg.name):
                        pipeline.split_data()

            for task in plan["feature_tasks"]:
                task_config = dict(config)
                task_config.update(task)
                with self.logger.section(
                    "feature_extraction",
                    data=data_cfg.name,
                    patch_encoder=task["patch_encoder"],
                    feature_type=task["feature_type"],
                ):
                    DataPipeline(config=task_config).run_extraction()

            for task in plan["model_preprocess_tasks"]:
                with self.logger.section(
                    "model_preprocess",
                    data=data_cfg.name,
                    model=task["model"],
                    kind=task["kind"],
                    log_path=task.get("log_path"),
                ):
                    _run_command(task["command"], cwd=task.get("cwd"), log_path=task.get("log_path"))

            if plan["base_config"].get("mode") in ("inference", "wsi_only"):
                self.state["external_data"] = data
            else:
                self.state["internal_data"] = data
            return {"dry_run": False, "plan": plan}

    def train(self, data: Optional[str] = None, folds: Optional[Sequence[int]] = None) -> BenchmarkResult:
        """Train all models on an internal dataset."""

        if folds is not None:
            raise NotImplementedError("train(folds=...) is not supported by the current runner.")
        data = data or self.state.get("internal_data")
        if data is None:
            raise ValueError("data must be provided for train() before internal state is set.")
        results = self._run_models(action="train", data=data, folds=folds)
        self.state["internal_data"] = data
        self.state["train_results"][data] = results
        self._remember_run(data, results)
        return results

    def evaluate(
        self,
        mode: str = "int",
        data: Optional[str] = None,
        external_data: Optional[str] = None,
        folds: Optional[Sequence[int]] = None,
        timestamps: Optional[Dict[str, str] | str] = None,
    ) -> BenchmarkResult:
        """Evaluate all models on internal or external data."""

        if mode not in ["int", "ext"]:
            raise ValueError("mode must be 'int' or 'ext'.")
        eval_data = data if mode == "int" else external_data
        train_data = data if mode == "int" else (data or self.state.get("internal_data"))
        if eval_data is None:
            state_key = "internal_data" if mode == "int" else "external_data"
            eval_data = self.state.get(state_key)
        if eval_data is None:
            raise ValueError(f"{'data' if mode == 'int' else 'external_data'} must be provided for mode {mode!r}.")
        if train_data is None:
            raise ValueError("internal training data must be known before evaluating external data.")
        if timestamps is None:
            timestamps = self._timestamps_for_data(train_data)
        if mode == "ext":
            model_cfgs = self._resolve_models()
            ext_data_cfg = self._resolve_data(eval_data)
            train_data_cfg = self._resolve_data(train_data)
            if self._has_missing_external_base_artifacts(ext_data_cfg, model_cfgs):
                with self.logger.section("preprocess_external", data=ext_data_cfg.name):
                    self.preprocess(data=eval_data, skip_model_preprocess=True)
            self._run_external_model_preprocess(ext_data_cfg, train_data_cfg, model_cfgs)
        results = self._run_models(
            action="evaluate",
            data=eval_data,
            folds=folds,
            timestamps=timestamps,
            train_data=train_data,
        )
        self.state["eval_results"][(mode, eval_data)] = results
        return results

    def evaluate_internal(
        self,
        data: Optional[str] = None,
        folds: Optional[Sequence[int]] = None,
        timestamps: Optional[Dict[str, str] | str] = None,
    ) -> BenchmarkResult:
        """Evaluate trained models on the internal test folds."""

        return self.evaluate(mode="int", data=data, folds=folds, timestamps=timestamps)

    def evaluate_external(
        self,
        data: str,
        train_data: Optional[str] = None,
        folds: Optional[Sequence[int]] = None,
        timestamps: Optional[Dict[str, str] | str] = None,
    ) -> BenchmarkResult:
        """Evaluate trained models on an external labeled dataset."""

        return self.evaluate(
            mode="ext",
            data=train_data,
            external_data=data,
            folds=folds,
            timestamps=timestamps,
        )

    def predict(
        self,
        data: str,
        ckpt_path: Optional[Dict[str, str] | str] = None,
        folds: Optional[Sequence[int]] = None,
        timestamps: Optional[Dict[str, str] | str] = None,
        train_data: Optional[str] = None,
        models: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        gene_list: Optional[Sequence[str]] = None,
        wsi_dir: Optional[str] = None,
        overwrite: Optional[bool] = None,
        coordinates: Optional[str] = None,
        batch_size: Optional[int] = 32,
    ) -> BenchmarkResult:
        """Run prediction/inference for all (or selected) models.

        `data` accepts a named data config (`data="ncche/xenium"`), a
        single WSI file (single-slide prediction), a directory of WSI
        files (batch prediction, one row per slide), or an
        already-preprocessed asset directory (one already containing
        `patches/*.h5`). For the latter three, `output_dir` is required
        to say where extracted patches/embeddings/predictions get
        written, and a checkpoint must be resolvable via `ckpt_path` or
        `train_data` (or a prior `train()`/`from_run()` call).

        `models` selects a per-call subset of models to run, overriding
        (not mutating) the models this `STPred` instance was constructed
        with. `gene_list` restricts predicted genes to a user-supplied
        subset of the training gene panel. `overwrite` forces WSI
        patch/embedding re-extraction for this call only; left unset (the
        default), it falls back to this instance's `preprocess_overrides`
        (if any set `"overwrite"`) rather than silently forcing `False`.

        `coordinates` — path to an `.h5ad` (patch centers read from
        `obsm['spatial']`) or `.csv` (`x`/`y` columns) file of patch-CENTER
        coordinates — crops patches at exactly those locations instead of
        running tissue segmentation + automatic tiling. Only valid when
        `data` is a single WSI file (raises otherwise). If patches were
        already extracted into `output_dir` by an earlier call (e.g. a
        tissue-seg run), pass `overwrite=True` too, or the old patches are
        reused unchanged.

        `batch_size` overrides how many patches are batched together per
        model forward pass during prediction — defaults to 32 (the data
        config's own `DATA.test_dataloader.batch_size`, usually 1, is used
        only if `batch_size=None` is passed explicitly). Raise or lower it
        depending on GPU memory and slide size.
        """

        train_data = train_data or self.state.get("internal_data")
        kind = _classify_predict_target(data, repo_root=self.repo_root)
        if coordinates is not None and kind != "wsi_file":
            raise ValueError(
                f"predict(coordinates=...) is only supported when data is a single "
                f"WSI file — {data!r} was classified as {kind!r}. Coordinates are "
                "tied to one slide's own pixel space."
            )
        if kind != "named_config":
            if not output_dir:
                raise ValueError(
                    f"predict(data={data!r}) was classified as a {kind!r} target "
                    "(WSI file/dir or an already-preprocessed asset dir), which "
                    "requires output_dir to say where extracted patches/"
                    "embeddings/predictions are written."
                )
            if ckpt_path is None and train_data is None:
                raise ValueError(
                    f"predict(data={data!r}) requires either ckpt_path or "
                    "train_data (or a prior train()/from_run() call) to resolve "
                    "a checkpoint."
                )
            data = self._materialize_wsi_data_config(
                data, kind, output_dir, wsi_dir=wsi_dir, overwrite=overwrite,
                coords_path=coordinates,
            )
            self.preprocess(data, skip_model_preprocess=True, models=models, overwrite=overwrite)

        if timestamps is None and train_data is not None:
            timestamps = self._timestamps_for_data(train_data)
        model_cfgs = self._resolve_models(models)
        ext_data_cfg = self._resolve_data(data)
        self._warn_missing_external_base_artifacts(ext_data_cfg, model_cfgs)
        if train_data is not None:
            train_data_cfg = self._resolve_data(train_data)
            self._run_external_model_preprocess(ext_data_cfg, train_data_cfg, model_cfgs)
        results = self._run_models(
            action="predict",
            data=data,
            folds=folds,
            timestamps=timestamps,
            ckpt_path=ckpt_path,
            train_data=train_data,
            models=models,
            gene_list=gene_list,
            batch_size=batch_size,
        )
        self.state["external_data"] = data
        self.state["predict_results"][data] = results
        if output_dir:
            self.state["last_output_dir"] = output_dir
        return results

    def visualize(
        self,
        gene: str,
        sample: str,
        output_dir: Optional[str] = None,
        model: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Render a predicted gene's expression for one sample as a heatmap,
        overlaid on the slide's own thumbnail when the original WSI can be
        located, falling back to a plain spatial scatter otherwise.

        `output_dir` defaults to whatever was passed to the most recent
        `predict()` call. Returns the saved PNG path.
        """
        import numpy as np
        import scanpy as sc
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = output_dir or self.state.get("last_output_dir")
        if not output_dir:
            raise ValueError(
                "visualize() needs output_dir — none was given and no prior "
                "predict() call set one."
            )
        output_dir = _abs_path(self.repo_root, output_dir)

        candidates = sorted(glob(os.path.join(output_dir, "**", f"{sample}.h5ad"), recursive=True))
        if model:
            candidates = [path for path in candidates if f"{os.sep}{model}{os.sep}" in path]
        if not candidates:
            raise ValueError(
                f"No prediction .h5ad found for sample={sample!r} under output_dir={output_dir!r}"
                + (f" and model={model!r}" if model else "")
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple predictions found for sample={sample!r} under output_dir={output_dir!r}; "
                "pass model= to disambiguate:\n" + "\n".join(f"- {path}" for path in candidates)
            )
        h5ad_path = candidates[0]

        adata = sc.read_h5ad(h5ad_path)
        if adata.n_obs == 0:
            raise ValueError(f"Predictions at {h5ad_path!r} have zero samples — nothing to visualize.")
        if gene not in adata.var_names:
            available = ", ".join(list(adata.var_names[:10]))
            raise ValueError(
                f"Gene {gene!r} not found in predictions at {h5ad_path!r}. "
                f"Available genes include: {available}..."
            )
        if "spatial" not in adata.obsm:
            raise ValueError(
                f"Predictions at {h5ad_path!r} have no obsm['spatial'] coordinates to plot."
            )
        expr = np.asarray(adata[:, gene].X).ravel()
        coords = np.asarray(adata.obsm["spatial"])

        # Every predict() run writes a manifest with the exact data config
        # name it ran against — read that directly instead of guessing it
        # back from the output directory layout. Named-config predict()
        # writes manifest.yaml into the same fold<k> directory as the
        # sample .h5ad files; WSI-predict output has no per-target
        # directory (output_dir is commonly reused across separate
        # predict() calls), so it writes one manifest per sample instead,
        # under output_dir/_wsi_predict/manifests/<sample>.yaml.
        wsi_path = None
        data_name = None
        data_dir = None
        manifest_path = os.path.join(os.path.dirname(h5ad_path), "manifest.yaml")
        if not os.path.isfile(manifest_path):
            manifest_path = os.path.join(output_dir, "_wsi_predict", "manifests", f"{sample}.yaml")
        if os.path.isfile(manifest_path):
            data_name = _load_yaml_file(manifest_path).get("data")
            if data_name:
                try:
                    data_cfg = resolve_data_config(data_name, repo_root=self.repo_root)
                    data_dir = data_cfg.config.get("DATA", {}).get("data_dir")
                    wsi_dir = data_cfg.config.get("DATA", {}).get("wsi_dir")
                    if wsi_dir:
                        wsi_dir = _abs_path(self.repo_root, wsi_dir)
                        matches = sorted(glob(os.path.join(wsi_dir, f"{sample}.*")))
                        wsi_path = matches[0] if matches else None
                except Exception:
                    pass

        # Patches are extracted as a dense non-overlapping grid, not sparse
        # Visium-style spots — render them at their true footprint (a square
        # the size of one patch) rather than an arbitrary small dot, or the
        # plot misleadingly looks sparse regardless of how densely tiled the
        # patches actually are. `patch_size_level0` is the pixel footprint of
        # one patch at the WSI's level-0 (native) resolution, which trident
        # writes into the same coords .h5 `_read_predict_coords` already
        # reads (src/core/model_adapters/base.py) — same coordinate space as
        # `coords` above, so it scales identically.
        patch_size_level0 = self._patch_footprint_level0(data_dir, sample)
        if patch_size_level0 is None:
            patch_size_level0 = _estimate_patch_spacing(coords)

        fig, ax = plt.subplots(figsize=(8, 8))
        try:
            plotted_on_thumbnail = False
            if wsi_path:
                try:
                    from trident import load_wsi
                    wsi = load_wsi(slide_path=wsi_path, lazy_init=False)
                    max_dimension = 1000
                    if wsi.width > wsi.height:
                        thumb_w = max_dimension
                        thumb_h = int(thumb_w * wsi.height / wsi.width)
                    else:
                        thumb_h = max_dimension
                        thumb_w = int(thumb_h * wsi.width / wsi.height)
                    thumbnail = wsi.get_thumbnail((thumb_w, thumb_h))
                    scale = thumb_w / wsi.width
                    scaled_coords = coords * scale
                    patch_px = (patch_size_level0 * scale) if patch_size_level0 else None
                    ax.imshow(thumbnail)
                    mappable = _draw_patches(ax, scaled_coords, expr, patch_px)
                    plotted_on_thumbnail = True
                except Exception as exc:
                    # Discard any partial thumbnail/patch draw from the
                    # failed attempt above — otherwise the fallback below
                    # would layer unscaled, native-pixel-space patches on
                    # top of a still-visible (and now stale) thumbnail.
                    ax.cla()
                    print(
                        f"[STPBench] WARNING: failed to overlay on WSI thumbnail ({wsi_path!r}): "
                        f"{exc}; falling back to a plain spatial scatter.",
                        file=sys.stderr,
                    )

            if not plotted_on_thumbnail:
                if not wsi_path:
                    print(
                        f"[STPBench] WARNING: could not locate the original WSI for sample={sample!r} "
                        f"(data={data_name!r}); rendering a plain spatial scatter instead.",
                        file=sys.stderr,
                    )
                mappable = _draw_patches(ax, coords, expr, patch_size_level0)
                margin = patch_size_level0 or 0
                ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
                ax.set_ylim(coords[:, 1].max() + margin, coords[:, 1].min() - margin)
                ax.set_aspect("equal")

            ax.set_title(f"{sample} — {gene}")
            ax.axis("off")
            fig.colorbar(mappable, ax=ax, label=gene, fraction=0.046, pad=0.04)

            save_path = save_path or os.path.join(output_dir, "viz", f"{sample}_{gene}.png")
            save_path = _abs_path(self.repo_root, save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            # Show inline when running in a notebook. Reads back the saved
            # PNG (not the live Figure) so this works regardless of the
            # "Agg" backend forced above, and no-ops harmlessly outside a
            # notebook or without IPython installed.
            try:
                from IPython import get_ipython
                from IPython.display import Image as IPImage, display
                if get_ipython() is not None:
                    display(IPImage(filename=save_path))
            except ImportError:
                pass
        finally:
            plt.close(fig)
        return save_path

    def check(
        self,
        data: Optional[str] = None,
        mode: str = "train",
        strict: bool = False,
    ) -> Dict[str, Any]:
        """Validate config shape and expected data artifacts before a heavy run."""

        if mode not in {"train", "eval", "inference"}:
            raise ValueError("mode must be 'train', 'eval', or 'inference'.")
        data = data or self.state.get("internal_data")
        if data is None:
            raise ValueError("data must be provided for check() before internal state is set.")

        data_cfg = self._resolve_data(data)
        model_cfgs = self._resolve_models()
        model_reports = []
        missing_paths = []

        for model_cfg in model_cfgs:
            runtime_cfg = self._merged_runtime_config(data_cfg, model_cfg)
            report = self._check_runtime_artifacts(data_cfg, model_cfg, runtime_cfg, mode=mode)
            missing_paths.extend(report["missing"])
            model_reports.append(report)

        result = {
            "ok": not missing_paths,
            "data": data_cfg.name,
            "models": [model_cfg.name for model_cfg in model_cfgs],
            "mode": mode,
            "reports": model_reports,
            "missing": missing_paths,
        }
        if strict and missing_paths:
            raise FileNotFoundError(
                "STPred preflight check failed. Missing required paths:\n"
                + "\n".join(f"- {path}" for path in missing_paths)
            )
        return result

    preflight = check

    def save_state(self, path: str) -> None:
        """Save lightweight STPred workflow state for later evaluation or prediction."""

        state = {
            "models": self.models,
            "repo_root": self.repo_root,
            "gpu": self.gpu,
            "gpu_id": self.gpu_id,
            "debug": self.debug,
            "dry_run": self.dry_run,
            "verbose": self.verbose,
            "log_file": self.log_file,
            "wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "preprocess_overrides": self.preprocess_overrides,
            "state": self._serialize_value(self.state),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(state, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    def load_state(self, path: str) -> None:
        """Load lightweight STPred workflow state saved by save_state()."""

        if not os.path.isfile(path):
            raise FileNotFoundError(f"STPred state file not found: {path}")
        with open(path, "r") as f:
            payload = yaml.safe_load(f) or {}
        state = payload.get("state")
        if not isinstance(state, dict):
            raise ValueError(f"Invalid STPred state file: {path}")
        self.state.update(state)
        if payload.get("models"):
            self.models = list(payload["models"])
        logger_changed = False
        if "wandb" in payload:
            self.use_wandb = bool(payload["wandb"])
        if payload.get("wandb_project"):
            self.wandb_project = payload["wandb_project"]
        if "verbose" in payload:
            new_verbose = bool(payload["verbose"])
            logger_changed = logger_changed or new_verbose != self.verbose
            self.verbose = new_verbose
        if payload.get("log_file") != self.log_file:
            self.log_file = payload.get("log_file")
            logger_changed = True
        if logger_changed:
            self.logger = BenchmarkLogger(enabled=self.verbose, log_file=self.log_file)

    @staticmethod
    def _serialize_value(value):
        if isinstance(value, BenchmarkResult):
            return value.to_dict()
        if isinstance(value, dict):
            return {str(key): STPred._serialize_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [STPred._serialize_value(item) for item in value]
        if isinstance(value, tuple):
            return [STPred._serialize_value(item) for item in value]
        return value

    def _remember_run(self, data: str, result: Dict[str, Any]) -> None:
        if result.get("dry_run"):
            return
        run_timestamps = self.state.setdefault("run_timestamps", {}).setdefault(data, {})
        checkpoint_roots = self.state.setdefault("checkpoint_roots", {}).setdefault(data, {})
        for item in result.get("results", []):
            model = item.get("model")
            timestamp = item.get("timestamp")
            if not model or not timestamp:
                continue
            run_timestamps[model] = timestamp
            artifacts = item.get("artifacts", {})
            log_dir = artifacts.get("log_dir")
            if log_dir:
                checkpoint_roots[model] = log_dir

    def _timestamps_for_data(self, data: Optional[str]) -> Optional[Dict[str, str]]:
        if data is None:
            return None
        timestamps = self.state.get("run_timestamps", {}).get(data, {})
        return dict(timestamps) if timestamps else None

    def _resolve_data(self, data: str) -> NamedConfig:
        return resolve_data_config(data, repo_root=self.repo_root)

    def _resolve_models(self, models: Optional[Sequence[str]] = None) -> List[NamedConfig]:
        return [resolve_model_config(model, repo_root=self.repo_root) for model in (models or self.models)]

    def _patch_footprint_level0(self, data_dir: Optional[str], sample: str) -> Optional[float]:
        """Read the level0-pixel patch footprint trident stamped onto this
        sample's coords .h5 (`patch_size_level0`) — same file/lookup pattern
        as `ModelAdapter._read_predict_coords` (src/core/model_adapters/
        base.py), so `visualize()` can render patches at their true
        non-overlapping size. Returns None (not an error) if unavailable —
        callers fall back to an estimate."""
        if not data_dir:
            return None
        from dataset.path_utils import patch_dir
        img_dir = patch_dir(_abs_path(self.repo_root, data_dir))
        for candidate_name in (f"{sample}.h5", f"{sample}_patches.h5"):
            path = os.path.join(img_dir, candidate_name)
            if os.path.isfile(path):
                try:
                    from trident.IO import read_coords
                    attrs, _ = read_coords(path)
                    return attrs.get("patch_size_level0")
                except Exception:
                    return None
        return None

    def _materialize_wsi_data_config(
        self,
        data: str,
        kind: str,
        output_dir: str,
        wsi_dir: Optional[str] = None,
        overwrite: Optional[bool] = None,
        coords_path: Optional[str] = None,
    ) -> str:
        """Auto-write config/data/_wsi_predict/<name>.yaml for a WSI-path
        predict() target, so the rest of the pipeline (resolve_data_config
        -> _merged_runtime_config -> _build_runtime_cfg -> _run_models)
        runs completely unmodified downstream.

        Machine-owned namespace, always overwritten: this is regenerated
        fresh on every predict() call, not a config a user is expected to
        hand-edit.
        """
        # normpath so two calls that mean the same directory (trailing
        # slash, "./", ".." segments, ...) canonicalize to the same string.
        output_dir = os.path.normpath(_abs_path(self.repo_root, output_dir))
        # Resolve once, relative to repo_root (not cwd) — matching
        # _classify_predict_target and every other path in this file — so
        # a relative `data` still resolves correctly under the documented
        # repo_root != cwd usage pattern (README's repo_root parameter).
        abs_data = os.path.normpath(_abs_path(self.repo_root, data))
        raw_name = os.path.basename(abs_data.rstrip(os.sep))
        if kind == "wsi_file":
            raw_name = os.path.splitext(raw_name)[0]
        # Named only by the slide's own basename: two predict() calls on
        # same-named slides from different source locations sharing an
        # output_dir will collide on this config (last predict() to run
        # wins) — acceptable given each call regenerates it fresh anyway;
        # keep output_dir per-slide-name if that matters for your workflow.
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("_") or "wsi_predict"
        name = f"_wsi_predict/{safe_name}"
        config_path = os.path.join(self.repo_root, "config", "data", f"{name}.yaml")

        # Both kinds read raw input from `data` and write everything else —
        # refreshed ids.csv, freshly-extracted embeddings a chosen model
        # might still need, and the scratch emb/ subdirectories
        # DataPipeline._setup_dirs() creates unconditionally just from
        # being constructed — to the separate writable `output_dir`, never
        # into `data`, which may be a shared/read-only asset directory.
        # DATA.data_dir is `output_dir` for both kinds (so patch AND
        # embedding reads at predict time agree with wherever extraction
        # actually wrote embeddings); for asset_dir, output_dir/patches is
        # symlinked to the real (possibly read-only) patches below instead
        # of physically extracting them (mode='inference', vs 'wsi_only'
        # which extracts for real).
        preprocess_input_dir = data
        preprocess_output_dir = preprocess_meta_dir = data_dir = meta_dir = output_dir
        mode = "inference" if kind == "asset_dir" else "wsi_only"

        if kind == "asset_dir":
            os.makedirs(output_dir, exist_ok=True)
            patches_link = os.path.join(output_dir, "patches")
            real_patches_dir = os.path.join(data, "patches")
            if not os.path.islink(patches_link) or os.readlink(patches_link) != real_patches_dir:
                if os.path.lexists(patches_link):
                    os.remove(patches_link)
                os.symlink(real_patches_dir, patches_link)

        if kind == "wsi_file":
            resolved_wsi_dir = wsi_dir or os.path.dirname(abs_data)
        elif kind == "wsi_dir":
            resolved_wsi_dir = wsi_dir or data
        else:
            resolved_wsi_dir = wsi_dir

        config = self._data_config_template(safe_name)
        config["DATA"].update({
            "data_dir": data_dir,
            "meta_dir": meta_dir,
            "output_dir": output_dir,
        })
        if resolved_wsi_dir:
            config["DATA"]["wsi_dir"] = resolved_wsi_dir
        config["preprocess"].update({
            "mode": mode,
            "input_dir": preprocess_input_dir,
            "output_dir": preprocess_output_dir,
            "meta_dir": preprocess_meta_dir,
        })
        # Only bake an explicit overwrite in when the caller actually asked
        # for one — leave the template's own default (and, downstream,
        # this instance's preprocess_overrides) in charge otherwise, so an
        # unset predict(overwrite=...) doesn't silently force overwrite=False
        # over a user's preprocess_overrides={"overwrite": True} default.
        if overwrite is not None:
            config["preprocess"]["overwrite"] = overwrite
        if coords_path is not None:
            config["preprocess"]["coords_path"] = _abs_path(self.repo_root, coords_path)
        self._write_yaml_template(config_path, config, overwrite=True)
        return name

    def _legacy_pair(self, data_cfg: NamedConfig, model_cfg: NamedConfig) -> Tuple[str, str, str]:
        data_legacy = data_cfg.config.get("legacy", {})
        model_legacy = model_cfg.config.get("legacy", {})
        if not data_legacy.get("config_prefix") or not data_legacy.get("config_data"):
            raise ValueError(f"{data_cfg.path} must define legacy.config_prefix and legacy.config_data.")

        config_prefix = data_legacy["config_prefix"]
        model_prefix = model_legacy.get("config_prefix")
        if model_prefix and model_prefix != config_prefix:
            raise ValueError(
                f"Model {model_cfg.name!r} uses legacy.config_prefix={model_prefix!r}, "
                f"but data {data_cfg.name!r} uses {config_prefix!r}."
            )

        config_model = self._legacy_model_identifier(model_cfg, config_prefix)
        return config_prefix, data_legacy["config_data"], config_model

    def _legacy_model_identifier(self, model_cfg: NamedConfig, config_prefix: str) -> str:
        legacy = model_cfg.config.get("legacy", {})
        config_models = legacy.get("config_models")
        if isinstance(config_models, dict) and config_prefix in config_models:
            return config_models[config_prefix]
        if legacy.get("config_model"):
            return legacy["config_model"]
        raise ValueError(
            f"{model_cfg.path} must define legacy.config_model or legacy.config_models.{config_prefix}."
        )

    def _load_legacy_cfg(self, data_cfg: NamedConfig, model_cfg: NamedConfig):
        config_prefix, config_data, config_model = self._legacy_pair(data_cfg, model_cfg)
        data_path = os.path.join(self.repo_root, "config", config_prefix, "data", config_data, "default.yaml")
        model_path = os.path.join(self.repo_root, "config", config_prefix, "model", f"{config_model}.yaml")
        return _merge_dicts(_load_yaml_file(data_path), _load_yaml_file(model_path))

    def _merged_runtime_config(self, data_cfg: NamedConfig, model_cfg: NamedConfig) -> Dict[str, Any]:
        has_direct_data = any(key in data_cfg.config for key in ("GENERAL", "TRAINING", "DATA"))
        has_direct_model = any(key in model_cfg.config for key in ("MODEL", "TRAINING", "DATA"))
        if has_direct_data or has_direct_model:
            cfg = _merge_dicts(data_cfg.config, model_cfg.config)
        else:
            cfg = self._load_legacy_cfg(data_cfg, model_cfg)
        _validate_runtime_config(data_cfg, model_cfg, cfg)
        return cfg

    def _gpus(self) -> List[int]:
        return list(range(self.gpu_id, self.gpu_id + self.gpu))

    def _build_preprocess_plan(
        self,
        data_cfg: NamedConfig,
        model_cfgs: Sequence[NamedConfig],
        overrides: Dict[str, Any],
        skip_model_preprocess: bool = False,
    ) -> Dict[str, Any]:
        preprocess_cfg = dict(data_cfg.config.get("preprocess", {}))
        preprocess_cfg.update(self.preprocess_overrides)
        preprocess_cfg.update({k: v for k, v in overrides.items() if v is not None and k != "dry_run"})

        feature_requirements: Dict[str, set[str]] = defaultdict(set)
        model_requirements = []
        platform = preprocess_cfg.get("platform")
        n_splits = preprocess_cfg.get("n_splits")

        for model_cfg in model_cfgs:
            runtime_cfg = self._merged_runtime_config(data_cfg, model_cfg)
            data_section = runtime_cfg.get("DATA", {})
            training_section = runtime_cfg.get("TRAINING", {})
            patch_encoder = data_section.get("model_name", "uni_v2")
            feature_type = data_section.get("feature_type", "global")
            features = _feature_set(feature_type)
            feature_requirements[patch_encoder].update(features)
            if platform is None:
                platform = str(data_section.get("tech", "visium")).lower()
            if n_splits is None:
                n_splits = training_section.get("num_k", 4)

            # Merge model-level preprocess overrides into preprocess_cfg.
            # Boolean flags (e.g. save_neighbor_imgs) are OR'd: if any model
            # requires the option, the pipeline enables it.
            model_preprocess = model_cfg.config.get("preprocess", {})
            for key, value in model_preprocess.items():
                if isinstance(value, bool):
                    preprocess_cfg[key] = preprocess_cfg.get(key, False) or value
                elif key not in preprocess_cfg:
                    preprocess_cfg[key] = value

            model_requirements.append(
                {
                    "model": model_cfg.name,
                    "patch_encoder": patch_encoder,
                    "feature_type": feature_type,
                    "features": list(features),
                }
            )

        required_features = set().union(*feature_requirements.values()) if feature_requirements else {"global"}
        base_config = {
            **preprocess_cfg,
            "platform": platform or "visium",
            "n_splits": n_splits or 4,
            "gpus": self._gpus(),
        }

        feature_tasks = []
        for patch_encoder, features in sorted(feature_requirements.items()):
            for feature in FEATURE_ORDER:
                if feature in features:
                    feature_tasks.append({"patch_encoder": patch_encoder, "feature_type": feature})

        return {
            "data": data_cfg.name,
            "data_config": os.path.relpath(data_cfg.path, self.repo_root),
            "models": [model_cfg.name for model_cfg in model_cfgs],
            "model_requirements": model_requirements,
            "base_config": base_config,
            "raw_feature_type": _coalesced_feature_type(required_features),
            "feature_tasks": feature_tasks,
            "model_preprocess_tasks": (
                [] if skip_model_preprocess
                else self._build_model_preprocess_tasks(data_cfg, model_cfgs, base_config)
            ),
            "gpus": self._gpus(),
        }

    def _build_model_preprocess_tasks(
        self,
        data_cfg: NamedConfig,
        model_cfgs: Sequence[NamedConfig],
        base_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tasks = []
        external_dir = None
        ref_data_dir = None
        if base_config.get("mode") == "inference":
            external_dir = base_config.get("output_dir")
            internal_data = self.state.get("internal_data")
            if internal_data:
                ref_cfg = self._resolve_data(internal_data)
                ref_runtime = self._merged_runtime_config(ref_cfg, model_cfgs[0])
                ref_data = ref_runtime.get("DATA", {})
                ref_data_dir = ref_data.get("data_dir")

        for model_cfg in model_cfgs:
            runtime_cfg = self._merged_runtime_config(data_cfg, model_cfg)
            data_section = runtime_cfg.get("DATA", {})
            model_section = runtime_cfg.get("MODEL", {})
            preprocess_section = runtime_cfg.get("preprocess", {})
            preprocess_specs = self._model_extra_preprocess_specs(model_section)

            data_dir = ref_data_dir if external_dir and ref_data_dir else data_section.get("data_dir")
            asset_dir = ref_data_dir if external_dir and ref_data_dir else data_dir
            external_asset_dir = external_dir
            for spec in preprocess_specs:
                kind = spec["kind"]
                if not data_dir:
                    raise ValueError(f"{data_cfg.path} must define DATA.data_dir for {kind} preprocessing.")
                command = self._extra_preprocess_command(
                    spec=spec,
                    data_dir=data_dir,
                    asset_dir=asset_dir,
                    external_dir=external_dir,
                    external_asset_dir=external_asset_dir,
                    data_section=data_section,
                    model_section=model_section,
                    overwrite=bool(base_config.get("overwrite", False)),
                    preprocess_section=preprocess_section,
                )
                log_dir = os.path.join(self.repo_root, "logs", "_stpbench_commands", data_cfg.name)
                tasks.append(
                    {
                        "model": model_cfg.name,
                        "kind": kind,
                        "command": command,
                        "cwd": self.repo_root,
                        "log_path": os.path.join(log_dir, f"{model_cfg.name}_{kind}_preprocess.log"),
                    }
                )
        return tasks

    def _model_extra_preprocess_specs(self, model_section: Dict[str, Any]) -> List[Dict[str, Any]]:
        specs = model_section.get("extra_preprocess", [])
        if specs is None:
            return []
        if isinstance(specs, dict):
            specs = [specs]
        if not isinstance(specs, list):
            raise ValueError("MODEL.extra_preprocess must be a mapping or list of mappings.")

        model_module = model_section.get("model_name", "").split(".")[0]

        normalized = []
        for spec in specs:
            if not isinstance(spec, dict) or not spec.get("script"):
                raise ValueError("Each MODEL.extra_preprocess item must define 'script'.")
            result = dict(spec)
            script = result["script"]
            if "/" not in script:
                result["script"] = f"src/model/{model_module}/preprocess/{script}"
            if not result.get("kind"):
                result["kind"] = os.path.splitext(os.path.basename(result["script"]))[0]
            normalized.append(result)
        return normalized

    def _extra_preprocess_command(
        self,
        spec: Dict[str, Any],
        data_dir: str,
        asset_dir: Optional[str],
        external_dir: Optional[str],
        external_asset_dir: Optional[str],
        data_section: Dict[str, Any],
        model_section: Dict[str, Any],
        overwrite: bool,
        preprocess_section: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        script = spec["script"]
        preprocess_section = preprocess_section or {}
        context = {
            "model_name": data_section.get("model_name", "uni_v2"),
            "gene_type": data_section.get("gene_type", "hmhvg"),
            "num_genes": str(data_section.get("num_genes", data_section.get("num_outputs", 200))),
            "model_path": data_section.get("model_path", model_section.get("model_path")),
            "external_meta_dir": _abs_path(self.repo_root, preprocess_section["meta_dir"]) if preprocess_section.get("meta_dir") else None,
            # preprocess-section fields available as auto-fill targets
            "input_dir": preprocess_section.get("input_dir"),
            "platform": preprocess_section.get("platform"),
            "mode": preprocess_section.get("mode"),
        }
        command = [sys.executable, script, "--data_dir", data_dir]
        meta_dir = data_section.get("meta_dir")
        if meta_dir:
            command += ["--meta_dir", _abs_path(self.repo_root, meta_dir)]
        if asset_dir:
            command += ["--asset_dir", asset_dir]
        if external_dir:
            command += ["--external_dir", external_dir]
        if external_asset_dir:
            command += ["--external_asset_dir", external_asset_dir]
        for key, value in spec.get("args", {}).items():
            if value is None:
                value = context.get(key)
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    command.append(f"--{key}")
            else:
                command += [f"--{key}", str(value)]
        if spec.get("overwrite", overwrite):
            command.append("--overwrite")
        return command

    def _has_missing_external_base_artifacts(
        self,
        ext_data_cfg: NamedConfig,
        model_cfgs: Sequence[NamedConfig],
    ) -> bool:
        """Return True if any model requires artifacts that are missing in ext_data_cfg."""
        for model_cfg in model_cfgs:
            runtime_cfg = self._merged_runtime_config(ext_data_cfg, model_cfg)
            model_section = runtime_cfg.get("MODEL", {})
            if not self._model_extra_preprocess_specs(model_section):
                continue
            data_section = runtime_cfg.get("DATA", {})
            data_dir = data_section.get("data_dir")
            if not data_dir:
                continue
            asset_dir = _abs_path(self.repo_root, data_dir)
            model_name = data_section.get("model_name", "uni_v2")
            features = _feature_set(data_section.get("feature_type", "global"))
            ids = self._read_sample_ids(asset_dir)
            if not ids:
                continue
            sample = ids[0]
            candidates = [
                os.path.join(asset_dir, "patches", f"{sample}.h5"),
                os.path.join(asset_dir, "patches", f"{sample}_patches.h5"),
            ]
            if not any(os.path.exists(p) for p in candidates):
                return True
            for feature in features:
                emb_path = os.path.join(
                    asset_dir, "emb", feature, f"features_{model_name}", f"{sample}.h5"
                )
                if not os.path.exists(emb_path):
                    return True
        return False

    def _warn_missing_external_base_artifacts(
        self,
        ext_data_cfg: NamedConfig,
        model_cfgs: Sequence[NamedConfig],
    ) -> None:
        """Warn when external prediction may need preprocessing first."""
        if not self._has_missing_external_base_artifacts(ext_data_cfg, model_cfgs):
            return
        self.logger.warning(
            "external_base_artifacts_missing",
            data=ext_data_cfg.name,
            hint="Some external patches/features are missing; run preprocess(data, mode='inference') before predict.",
        )

    def _run_external_model_preprocess(
        self,
        ext_data_cfg: NamedConfig,
        train_data_cfg: NamedConfig,
        model_cfgs: Sequence[NamedConfig],
        overwrite: bool = False,
    ) -> None:
        for model_cfg in model_cfgs:
            train_runtime = self._merged_runtime_config(train_data_cfg, model_cfg)
            ext_runtime = self._merged_runtime_config(ext_data_cfg, model_cfg)
            model_section = train_runtime.get("MODEL", {})
            training_pipeline = model_section.get("training_pipeline", {})
            if isinstance(training_pipeline, str):
                training_pipeline = {"kind": training_pipeline}
            if training_pipeline.get("kind") == "sepal_two_stage":
                # Sepal isn't in the generic extra_preprocess list — its
                # external graph-building needs the trained LocalNet/
                # LinearProb checkpoint (ckpt_path), which the generic
                # extra_preprocess command builder has no notion of.
                self._run_sepal_external_preprocess(
                    ext_data_cfg, train_data_cfg, model_cfg, train_runtime, ext_runtime,
                )
                continue
            preprocess_specs = self._model_extra_preprocess_specs(model_section)
            if not preprocess_specs:
                continue
            train_data_section = train_runtime.get("DATA", {})
            ext_data_section = ext_runtime.get("DATA", {})
            ext_preprocess_section = ext_runtime.get("preprocess", {})
            data_dir = train_data_section.get("data_dir")
            external_dir = ext_data_section.get("data_dir")
            if not data_dir or not external_dir:
                continue
            asset_dir = data_dir
            external_asset_dir = external_dir
            log_dir = os.path.join(
                self.repo_root, "logs", "_stpbench_commands", ext_data_cfg.name
            )
            for spec in preprocess_specs:
                command = self._extra_preprocess_command(
                    spec=spec,
                    data_dir=data_dir,
                    asset_dir=asset_dir,
                    external_dir=external_dir,
                    external_asset_dir=external_asset_dir,
                    data_section=train_data_section,
                    model_section=model_section,
                    overwrite=overwrite,
                    preprocess_section=ext_preprocess_section,
                )
                log_path = os.path.join(
                    log_dir, f"{model_cfg.name}_{spec['kind']}_preprocess.log"
                )
                with self.logger.section(
                    "model_preprocess",
                    data=ext_data_cfg.name,
                    model=model_cfg.name,
                    kind=spec["kind"],
                    log_path=log_path,
                ):
                    _run_command(command, cwd=self.repo_root, log_path=log_path)

    def _check_runtime_artifacts(
        self,
        data_cfg: NamedConfig,
        model_cfg: NamedConfig,
        runtime_cfg: Dict[str, Any],
        mode: str,
    ) -> Dict[str, Any]:
        data_section = runtime_cfg.get("DATA", {})
        training_section = runtime_cfg.get("TRAINING", {})
        model_section = runtime_cfg.get("MODEL", {})
        data_dir = _abs_path(self.repo_root, data_section["data_dir"])
        meta_dir = _abs_path(self.repo_root, data_section.get("meta_dir") or data_section["data_dir"])
        asset_dir = data_dir
        model_name = data_section.get("model_name", "uni_v2")
        gene_type = data_section["gene_type"]
        num_genes = data_section["num_genes"]
        num_k = int(training_section.get("num_k", 1))
        feature_type = data_section.get("feature_type", "global")
        features = _feature_set(feature_type)

        required = [data_dir, os.path.join(meta_dir, "ids.csv")]
        if mode != "inference":
            required.append(os.path.join(meta_dir, f"{gene_type}_{num_genes}genes.json"))
            id_path = os.path.join(meta_dir, "ids.csv")
            try:
                id_cols = pd.read_csv(id_path, nrows=1).columns
            except Exception:
                id_cols = []
            for fold in range(num_k):
                if f"fold_{fold}" not in id_cols:
                    required.append(os.path.join(data_dir, "splits", f"train_{fold}.csv"))
                    required.append(os.path.join(data_dir, "splits", f"test_{fold}.csv"))

        ids = self._read_sample_ids(meta_dir)
        if ids:
            sample_ids = ids[: min(3, len(ids))]
            for sample_id in sample_ids:
                required.append(self._first_existing_candidate([
                    os.path.join(asset_dir, "patches", f"{sample_id}.h5"),
                    os.path.join(asset_dir, "patches", f"{sample_id}_patches.h5"),
                ]))
                for feature in features:
                    required.append(
                        os.path.join(asset_dir, "emb", feature, f"features_{model_name}", f"{sample_id}.h5")
                    )

        for spec in self._model_extra_preprocess_specs(model_section):
            artifact_dir = spec.get("artifact_dir")
            if artifact_dir:
                required.append(_abs_path(data_dir, artifact_dir))

        missing = [path for path in required if path and not os.path.exists(path)]
        return {
            "model": model_cfg.name,
            "data": data_cfg.name,
            "config": {
                "data": os.path.relpath(data_cfg.path, self.repo_root),
                "model": os.path.relpath(model_cfg.path, self.repo_root),
            },
            "checked": required,
            "missing": missing,
        }

    def _read_sample_ids(self, meta_dir: str) -> List[str]:
        ids_path = os.path.join(meta_dir, "ids.csv")
        if not os.path.isfile(ids_path):
            return []
        with open(ids_path, newline="") as f:
            reader = csv.DictReader(f)
            if "sample_id" not in (reader.fieldnames or []):
                return []
            return [row["sample_id"] for row in reader if row.get("sample_id")]

    @staticmethod
    def _first_existing_candidate(paths: Sequence[str]) -> str:
        for path in paths:
            if os.path.exists(path):
                return path
        return paths[0]

    def _run_models(
        self,
        action: str,
        data: str,
        folds: Optional[Sequence[int]] = None,
        timestamps: Optional[Dict[str, str] | str] = None,
        ckpt_path: Optional[Dict[str, str] | str] = None,
        train_data: Optional[str] = None,
        models: Optional[Sequence[str]] = None,
        gene_list: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        data_cfg = self._resolve_data(data)
        train_data = train_data or data
        train_data_cfg = self._resolve_data(train_data)
        model_cfgs = self._resolve_models(models)
        payloads = []
        fold_values = list(folds) if folds is not None else [None]

        for index, model_cfg in enumerate(model_cfgs):
            runtime_config = self._merged_runtime_config(data_cfg, model_cfg)
            self._merged_runtime_config(train_data_cfg, model_cfg)
            config_key = f"{train_data_cfg.name}/{model_cfg.name}"
            for fold in fold_values:
                payload = {
                    "repo_root": self.repo_root,
                    "runtime_config": runtime_config,
                    "config_key": config_key,
                    "model": model_cfg.name,
                    "data": data_cfg.name,
                    "train_data": train_data_cfg.name,
                    "data_config": data_cfg.path,
                    "train_data_config": train_data_cfg.path,
                    "model_config": model_cfg.path,
                    "action": action,
                    "fold": fold,
                    "timestamp": self._value_for_model(timestamps, model_cfg.name),
                    "ckpt_path": self._value_for_model(ckpt_path, model_cfg.name),
                    "gpu_id": self.gpu_id + (index % self.gpu),
                    "debug": self.debug,
                    "verbose": self.verbose,
                    "log_file": self.log_file,
                    "use_wandb": self.use_wandb,
                    "wandb_project": self.wandb_project,
                    "gene_list": list(gene_list) if gene_list is not None else None,
                    "batch_size": batch_size,
                }
                training_pipeline = runtime_config.get("MODEL", {}).get("training_pipeline", {})
                if isinstance(training_pipeline, str):
                    training_pipeline = {"kind": training_pipeline}
                if action == "train" and training_pipeline.get("kind") == "sepal_two_stage":
                    payload.update(self._build_sepal_train_payload(data_cfg, model_cfg, runtime_config))
                if (
                    action == "predict"
                    and not runtime_config.get("MODEL", {}).get("skip_train", False)
                    and not payload["ckpt_path"]
                ):
                    timestamp = payload.get("timestamp")
                    if timestamp:
                        payload["ckpt_path"] = os.path.join(
                            self._default_ckpt_root(train_data_cfg.name, model_cfg.name),
                            timestamp,
                        )
                    else:
                        payload["ckpt_path"] = self._default_ckpt_root(train_data_cfg.name, model_cfg.name)
                payloads.append(payload)

        plan = {
            "action": action,
            "data": data_cfg.name,
            "train_data": train_data_cfg.name,
            "payloads": payloads,
            "parallel": self.gpu > 1,
        }
        if self.dry_run:
            self.logger.info(
                "model_run",
                status="dry_run",
                action=action,
                data=data_cfg.name,
                train_data=train_data_cfg.name,
                models=",".join(self.models),
                payloads=len(payloads),
            )
            return BenchmarkResult({"dry_run": True, "plan": plan})

        with self.logger.section(
            "model_run",
            action=action,
            data=data_cfg.name,
            train_data=train_data_cfg.name,
            models=",".join(self.models),
            payloads=len(payloads),
            parallel=self.gpu > 1 and len(payloads) > 1,
            wandb=self.use_wandb,
        ):
            if self.gpu == 1 or len(payloads) <= 1:
                results = []
                for payload in payloads:
                    self.logger.info(
                        "model_submit",
                        action=action,
                        model=payload["model"],
                        data=payload["data"],
                        gpu_id=payload["gpu_id"],
                    )
                    results.append(_run_single_model_action(payload))
            else:
                results = []
                max_workers = min(self.gpu, len(payloads))
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_payload = {
                        executor.submit(_run_single_model_action, payload): payload
                        for payload in payloads
                    }
                    for payload in payloads:
                        self.logger.info(
                            "model_submit",
                            action=action,
                            model=payload["model"],
                            data=payload["data"],
                            gpu_id=payload["gpu_id"],
                        )
                    for future in as_completed(future_to_payload):
                        payload = future_to_payload[future]
                        result = future.result()
                        self.logger.info(
                            "model_complete",
                            action=action,
                            model=payload["model"],
                            data=payload["data"],
                            gpu_id=payload["gpu_id"],
                        )
                        results.append(result)

        return BenchmarkResult({
            "dry_run": False,
            "plan": plan,
            "results": results,
            "summary": self._summarize_results(results),
        })

    def _summarize_results(self, results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for result in results:
            model = result["model"]
            model_summary = summary.setdefault(
                model,
                {
                    "timestamp": result.get("timestamp"),
                    "elapsed_sec": result.get("elapsed_sec"),
                    "metrics": {},
                    "checkpoints": {},
                    "prediction_dirs": {},
                },
            )
            artifacts = result.get("artifacts", {})
            for fold, fold_artifact in artifacts.get("folds", {}).items():
                if fold_artifact.get("metrics"):
                    model_summary["metrics"][fold] = fold_artifact["metrics"]
                if fold_artifact.get("checkpoints"):
                    model_summary["checkpoints"][fold] = fold_artifact["checkpoints"]
                if fold_artifact.get("prediction_dir"):
                    model_summary["prediction_dirs"][fold] = fold_artifact["prediction_dir"]
            if artifacts.get("checkpoint"):
                model_summary["checkpoint"] = artifacts["checkpoint"]
            if artifacts.get("prediction_dir"):
                model_summary["prediction_dir"] = artifacts["prediction_dir"]
        return summary

    def _default_ckpt_root(self, data: str, model: str) -> str:
        return os.path.join(self.repo_root, "logs", data, model)

    def _build_sepal_train_payload(
        self,
        data_cfg: NamedConfig,
        model_cfg: NamedConfig,
        sepal_runtime_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        training_pipeline = sepal_runtime_config.get("MODEL", {}).get("training_pipeline", {})
        if isinstance(training_pipeline, str):
            training_pipeline = {"kind": training_pipeline}
        localnet_model = training_pipeline.get("localnet_model")
        if not localnet_model:
            raise ValueError(
                f"{model_cfg.path} must define MODEL.training_pipeline.localnet_model "
                "for sepal_two_stage training."
            )
        localnet_cfg = resolve_model_config(localnet_model, repo_root=self.repo_root)
        localnet_runtime_config = self._merged_runtime_config(data_cfg, localnet_cfg)
        data_section = sepal_runtime_config.get("DATA", {})
        model_section = sepal_runtime_config.get("MODEL", {})
        use_pretrained_emb = training_pipeline.get("use_pretrained_emb")
        if use_pretrained_emb is None:
            use_pretrained_emb = self._sepal_uses_pretrained_emb(model_section)
        dataset_path = _abs_path(self.repo_root, data_section["data_dir"])
        command = [
            sys.executable,
            "preprocess.py",
            "--dataset_path",
            dataset_path,
            "--ckpt_path",
            self._default_ckpt_root(data_cfg.name, localnet_model),
            "--mode",
            "train",
            "--local_model",
            training_pipeline.get("local_model", localnet_model),
            "--model_name",
            data_section.get("model_name", "uni_v2"),
            "--gene_type",
            data_section.get("gene_type", "hmhvg"),
            "--num_genes",
            str(data_section.get("num_genes", data_section.get("num_outputs", 200))),
        ]
        if data_section.get("meta_dir"):
            command += ["--meta_dir", _abs_path(self.repo_root, data_section["meta_dir"])]
        if data_section.get("cpm", False):
            command.append("--cpm")
        if data_section.get("smooth", False):
            command.append("--smooth")
        if use_pretrained_emb:
            command.append("--use_pretrained_emb")
        return {
            "action": "sepal_train",
            "localnet_model": localnet_model,
            "localnet_runtime_config": localnet_runtime_config,
            "sepal_preprocess_command": command,
            "sepal_preprocess_cwd": os.path.join(self.repo_root, "src", "model", "sepal"),
            "sepal_preprocess_log": os.path.join(
                self.repo_root,
                "logs",
                "_stpbench_commands",
                data_cfg.name,
                f"{model_cfg.name}_sepal_preprocess.log",
            ),
        }

    def _run_sepal_external_preprocess(
        self,
        ext_data_cfg: NamedConfig,
        train_data_cfg: NamedConfig,
        model_cfg: NamedConfig,
        train_runtime: Dict[str, Any],
        ext_runtime: Dict[str, Any],
    ) -> None:
        """Build Sepal's per-slide graph .pt files for the external test set.

        Sepal's own evaluate() action never generates these (unlike EGN/EGGN's
        generic extra_preprocess) — during training, _run_sepal_train's
        "preprocess" stage only ever covers the training dataset's own
        train+test splits. Mirrors _build_sepal_train_payload's command, but
        with --external_dir/--external_meta_dir pointed at the external data
        and --mode train restricted to its 'test' phase (see
        src/model/sepal/preprocess.py's external branch).
        """
        training_pipeline = train_runtime.get("MODEL", {}).get("training_pipeline", {})
        if isinstance(training_pipeline, str):
            training_pipeline = {"kind": training_pipeline}
        localnet_model = training_pipeline.get("localnet_model")
        if not localnet_model:
            return
        train_data_section = train_runtime.get("DATA", {})
        ext_data_section = ext_runtime.get("DATA", {})
        dataset_path = _abs_path(self.repo_root, train_data_section["data_dir"])
        external_dir = ext_data_section.get("data_dir")
        if not external_dir:
            return
        external_asset_dir = _abs_path(self.repo_root, external_dir)
        external_meta_dir = _abs_path(
            self.repo_root, ext_data_section.get("meta_dir") or external_dir
        )
        use_pretrained_emb = training_pipeline.get("use_pretrained_emb")
        if use_pretrained_emb is None:
            use_pretrained_emb = self._sepal_uses_pretrained_emb(train_runtime.get("MODEL", {}))
        command = [
            sys.executable,
            "preprocess.py",
            "--dataset_path",
            dataset_path,
            "--ckpt_path",
            self._default_ckpt_root(train_data_cfg.name, localnet_model),
            "--mode",
            "train",
            "--external_dir",
            external_asset_dir,
            "--external_meta_dir",
            external_meta_dir,
            "--local_model",
            training_pipeline.get("local_model", localnet_model),
            "--model_name",
            train_data_section.get("model_name", "uni_v2"),
            "--gene_type",
            train_data_section.get("gene_type", "hmhvg"),
            "--num_genes",
            str(train_data_section.get("num_genes", train_data_section.get("num_outputs", 200))),
        ]
        if train_data_section.get("meta_dir"):
            command += ["--meta_dir", _abs_path(self.repo_root, train_data_section["meta_dir"])]
        if train_data_section.get("cpm", False):
            command.append("--cpm")
        if train_data_section.get("smooth", False):
            command.append("--smooth")
        if use_pretrained_emb:
            command.append("--use_pretrained_emb")

        log_path = os.path.join(
            self.repo_root, "logs", "_stpbench_commands", ext_data_cfg.name,
            f"{model_cfg.name}_sepal_preprocess.log",
        )
        with self.logger.section(
            "model_preprocess",
            data=ext_data_cfg.name,
            model=model_cfg.name,
            kind="sepal_preprocess",
            log_path=log_path,
        ):
            _run_command(
                command, cwd=os.path.join(self.repo_root, "src", "model", "sepal"), log_path=log_path,
            )

    @staticmethod
    def _model_class(model_name: str) -> str:
        return str(model_name).split(".")[-1]

    @staticmethod
    def _sepal_uses_pretrained_emb(model_section: Dict[str, Any]) -> bool:
        h_preprocess = str(model_section.get("h_preprocess", ""))
        first_dim = h_preprocess.split(",", 1)[0].strip()
        return first_dim == "1536"

    @staticmethod
    def _value_for_model(value: Optional[Dict[str, str] | str], model: str) -> Optional[str]:
        if value is None or isinstance(value, str):
            return value
        return value.get(model)

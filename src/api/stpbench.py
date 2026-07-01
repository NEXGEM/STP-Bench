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
from api.config_resolver import NamedConfig, resolve_data_config, resolve_model_config
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


def _write_manifest(cfg, payload: Dict[str, Any], action: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
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
    manifest_path = os.path.join(output_dir, "manifest.yaml")
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
        manifest_dir = cfg.GENERAL.log_dir if action in {"train", "evaluate"} else os.path.join(cfg.DATA.pred_path, f"fold{cfg.DATA.fold}")
        manifest_path = _write_manifest(cfg, payload, action, manifest_dir)
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
            cfg.DATA.data_dir = current_data.data_dir
            cfg.DATA.output_dir = current_data.output_dir
            cfg.DATA.name = current_data.get("name", payload["data"])
            cfg.DATA.ref_data_dir = cfg.DATA.get("meta_dir", cfg.DATA.train_data_dir)
            cfg.DATA.ref_asset_dir = cfg.DATA.train_data_dir
            cfg.DATA.train_data_name = train_data_name
            if current_data.get("wsi_dir", None):
                cfg.DATA.wsi_dir = current_data.wsi_dir
            if current_data.get("test_dataloader", None):
                cfg.DATA.test_dataloader = current_data.test_dataloader
        else:
            os.makedirs(cfg.GENERAL.log_dir, exist_ok=True)
        train_data_dir = cfg.DATA.get("train_data_dir", cfg.DATA.data_dir)
        train_meta_dir = cfg.DATA.get("meta_dir", train_data_dir)
        cfg.DATA.ref_data_dir = train_meta_dir
        cfg.DATA.ref_asset_dir = train_data_dir
        gene_path = f"{train_meta_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
        cfg.MODEL.gene_path = gene_path if os.path.isfile(gene_path) else f"{train_data_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
        cfg.MODEL.ref_data_dir = train_meta_dir
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
        cfg.DATA.pred_path = f"{cfg.DATA.output_dir}/{cfg.DATA.name}/{cfg.MODEL.name}/{ref_data_config}"

    cfg.MODEL.num_genes = cfg.DATA.get("num_genes", cfg.MODEL.get("num_genes", None))
    cfg.DATA.mode = {"train": "cv", "evaluate": "eval", "predict": "inference"}[action]
    return cfg


def _abs_path(repo_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


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

    @classmethod
    def init_data_config(cls, name: str, output: Optional[str] = None, repo_root: str = ".") -> str:
        """Write a minimal editable data config template and return its path."""

        output = output or os.path.join(repo_root, "config", "data", f"{name}.yaml")
        config = {
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
                "train_dataloader": {"batch_size": 128, "num_workers": 4, "pin_memory": False, "shuffle": True},
                "test_dataloader": {"batch_size": 1, "num_workers": 4, "pin_memory": False, "shuffle": False},
            },
            "preprocess": {
                "mode": "raw",
                "input_dir": "/path/to/raw_data",
                "output_dir": "/path/to/processed_data",
                "overwrite": False,
            },
        }
        cls._write_yaml_template(output, config)
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
    def _write_yaml_template(path: str, config: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path):
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
                summary["preprocess_external"] = self.preprocess(external_data, mode="inference", **kwargs)
                summary["predict_external"] = self.predict(external_data)
                summary["evaluate_external"] = self.evaluate(mode="ext", external_data=external_data)
            return BenchmarkResult({"steps": summary})

    def preprocess(self, data: str, **overrides) -> Dict[str, Any]:
        """Run one deduplicated preprocessing plan for all configured models."""

        data_cfg = self._resolve_data(data)
        model_cfgs = self._resolve_models()
        plan = self._build_preprocess_plan(data_cfg, model_cfgs, overrides)

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

            if plan["base_config"].get("mode") == "inference":
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
                    self.preprocess(data=eval_data)
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
    ) -> BenchmarkResult:
        """Run prediction/inference for all models on an external dataset."""

        train_data = train_data or self.state.get("internal_data")
        if timestamps is None and train_data is not None:
            timestamps = self._timestamps_for_data(train_data)
        model_cfgs = self._resolve_models()
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
        )
        self.state["external_data"] = data
        self.state["predict_results"][data] = results
        return results

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

    def _resolve_models(self) -> List[NamedConfig]:
        return [resolve_model_config(model, repo_root=self.repo_root) for model in self.models]

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
            "model_preprocess_tasks": self._build_model_preprocess_tasks(data_cfg, model_cfgs, base_config),
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
    ) -> Dict[str, Any]:
        data_cfg = self._resolve_data(data)
        train_data = train_data or data
        train_data_cfg = self._resolve_data(train_data)
        model_cfgs = self._resolve_models()
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

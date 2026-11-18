"""Exact-name config resolution for the public STPBench API."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

from ids_io import HOLDOUT_SUFFIX, ensure_holdout_alias


@dataclass(frozen=True)
class NamedConfig:
    """Resolved config file."""

    name: str
    path: str
    config: Dict[str, Any]


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _named_config_path(repo_root: str, kind: str, name: str) -> Optional[str]:
    """Expected absolute path for a validly-shaped, existing named config,
    or None otherwise. A cheap on-disk existence check — never parses the
    file, so a malformed-but-present config still counts as "exists" here.
    Callers that need to distinguish "not a config" from "a broken config"
    (rather than silently treating both as "not a config") should use this
    instead of catching exceptions from resolve_*_config()."""
    if not name or os.path.isabs(name) or ".." in name.split("/"):
        return None
    expected_abs = os.path.join(os.path.abspath(repo_root), "config", kind, f"{name}.yaml")
    return expected_abs if os.path.isfile(expected_abs) else None


def _resolve_named_config(repo_root: str, kind: str, name: str) -> NamedConfig:
    if not name:
        raise ValueError(f"{kind} name must be provided.")
    if os.path.isabs(name) or ".." in name.split("/"):
        raise ValueError(f"{kind} name must be a relative config name without '..': {name}")

    expected_abs = _named_config_path(repo_root, kind, name)
    if expected_abs is None:
        expected_rel = os.path.join("config", kind, f"{name}.yaml")
        raise FileNotFoundError(
            f"Config not found for {kind}='{name}'.\n"
            f"Expected: {expected_rel}\n"
            "Please create this config before running STPBench."
        )

    return NamedConfig(name=name, path=expected_abs, config=_load_yaml(expected_abs))


def _holdout_base_name(repo_root: str, name: str) -> Optional[str]:
    """Base data name when `name` is `<base>_holdout` and has no config file of its own, else None."""
    if not name.endswith(HOLDOUT_SUFFIX) or _named_config_path(repo_root, "data", name):
        return None
    base = name[: -len(HOLDOUT_SUFFIX)]
    return base if _named_config_path(repo_root, "data", base) else None


def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v


def _holdout_config(base: NamedConfig, name: str, repo_root: str) -> NamedConfig:
    """Config of the held-out samples of `base`: its config with meta_dir -> `<meta_dir>_holdout`.

    The optional top-level `holdout:` block of the base config overrides settings that differ for the held-out
    samples (e.g. `DATA.tech`, `preprocess.platform`).
    """
    cfg = copy.deepcopy(base.config)
    overrides = cfg.pop("holdout", None) or {}
    for section in ("DATA", "preprocess"):
        s = cfg.get(section)
        if isinstance(s, dict) and s.get("meta_dir"):
            s["meta_dir"] = str(s["meta_dir"]).rstrip("/") + HOLDOUT_SUFFIX
            ensure_holdout_alias(os.path.join(os.path.abspath(repo_root), s["meta_dir"]))
    if isinstance(cfg.get("preprocess"), dict):
        cfg["preprocess"].pop("n_top_hmhvg", None)  # gene sets are shared with the base dir
    _merge(cfg, overrides)
    return NamedConfig(name=name, path=base.path, config=cfg)


def resolve_data_config(name: str, repo_root: str = ".") -> NamedConfig:
    """Resolve ``data`` to ``config/data/<name>.yaml``.

    Names may include namespace separators, e.g. ``hest/CCRCC`` resolves to
    ``config/data/hest/CCRCC.yaml``. ``<name>_holdout`` (no file of its own) is the dataset's held-out samples:
    the ``test``-in-every-fold rows of ``input/<name>/ids.csv`` (see ids_io), built from ``<name>.yaml``.
    """

    base_name = _holdout_base_name(repo_root, name)
    if base_name:
        return _holdout_config(_resolve_named_config(repo_root=repo_root, kind="data", name=base_name), name, repo_root)
    cfg = _resolve_named_config(repo_root=repo_root, kind="data", name=name)
    if "holdout" in cfg.config:
        cfg = NamedConfig(name=cfg.name, path=cfg.path, config={k: v for k, v in cfg.config.items() if k != "holdout"})
    return cfg


def resolve_model_config(name: str, repo_root: str = ".") -> NamedConfig:
    """Resolve ``model`` to ``config/model/<name>.yaml`` only."""

    return _resolve_named_config(repo_root=repo_root, kind="model", name=name)


def data_config_exists(name: str, repo_root: str = ".") -> bool:
    """Cheap on-disk existence check for a data config name (no YAML
    parsing — can't raise on a malformed-but-present config). Use instead
    of try/except around resolve_data_config() when only "is this a real
    config name" matters, not its contents."""
    return _named_config_path(repo_root, "data", name) is not None or _holdout_base_name(repo_root, name) is not None

"""Exact-name config resolution for the public STPBench API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml


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


def resolve_data_config(name: str, repo_root: str = ".") -> NamedConfig:
    """Resolve ``data`` to ``config/data/<name>.yaml`` only.

    Names may include namespace separators, e.g. ``hest/CCRCC`` resolves to
    ``config/data/hest/CCRCC.yaml``.
    """

    return _resolve_named_config(repo_root=repo_root, kind="data", name=name)


def resolve_model_config(name: str, repo_root: str = ".") -> NamedConfig:
    """Resolve ``model`` to ``config/model/<name>.yaml`` only."""

    return _resolve_named_config(repo_root=repo_root, kind="model", name=name)


def data_config_exists(name: str, repo_root: str = ".") -> bool:
    """Cheap on-disk existence check for a data config name (no YAML
    parsing — can't raise on a malformed-but-present config). Use instead
    of try/except around resolve_data_config() when only "is this a real
    config name" matters, not its contents."""
    return _named_config_path(repo_root, "data", name) is not None

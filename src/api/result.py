from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict, Iterator, List, Optional


class BenchmarkResult(Mapping):
    """Dictionary-compatible result object returned by STPred run methods."""

    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.payload)

    def summary(self) -> Dict[str, Any]:
        base = self.payload.get("summary", {})

        # Collect metric values per model across folds from to_records()
        metrics_by_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for rec in self.to_records():
            model = rec.get("model")
            if not model:
                continue
            for key, val in rec.items():
                if key.startswith("metric/") and val is not None:
                    try:
                        metrics_by_model[model][key[len("metric/"):]].append(float(val))
                    except (TypeError, ValueError):
                        pass

        if not metrics_by_model:
            return base

        # Merge aggregate stats into base summary, keyed by model
        result = {}
        for model in set(base) | set(metrics_by_model):
            entry = dict(base.get(model, {}))
            vals_by_metric = metrics_by_model.get(model, {})
            if vals_by_metric:
                agg = {}
                for metric, vals in vals_by_metric.items():
                    agg[metric] = {
                        "mean": statistics.mean(vals),
                        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                        "n_folds": len(vals),
                        "per_fold": vals,
                    }
                entry["aggregate"] = agg
            result[model] = entry
        return result

    def results(self) -> List[Dict[str, Any]]:
        return list(self.payload.get("results", []))

    def plan(self) -> Dict[str, Any]:
        return self.payload.get("plan", {})

    def best_checkpoints(self) -> Dict[str, Dict[Any, str]]:
        checkpoints: Dict[str, Dict[Any, str]] = {}
        for model, model_summary in self.summary().items():
            model_ckpts = {}
            for fold, paths in model_summary.get("checkpoints", {}).items():
                if paths:
                    model_ckpts[fold] = paths[-1]
            if model_summary.get("checkpoint"):
                model_ckpts["predict"] = model_summary["checkpoint"]
            checkpoints[model] = model_ckpts
        return checkpoints

    def prediction_dirs(self) -> Dict[str, Any]:
        dirs = {}
        for model, model_summary in self.summary().items():
            if model_summary.get("prediction_dirs"):
                dirs[model] = model_summary["prediction_dirs"]
            elif model_summary.get("prediction_dir"):
                dirs[model] = model_summary["prediction_dir"]
        return dirs

    def to_records(self) -> List[Dict[str, Any]]:
        records = []
        for item in self.results():
            base = {
                "model": item.get("model"),
                "data": item.get("data"),
                "train_data": item.get("train_data"),
                "action": item.get("action"),
                "timestamp": item.get("timestamp"),
                "gpu_id": item.get("gpu_id"),
                "manifest": item.get("manifest"),
            }
            artifacts = item.get("artifacts", {})
            folds = artifacts.get("folds", {})
            if not folds:
                record = dict(base)
                record["checkpoint"] = artifacts.get("checkpoint")
                record["prediction_dir"] = artifacts.get("prediction_dir")
                records.append(record)
                continue
            for fold, fold_artifact in folds.items():
                record = dict(base)
                record["fold"] = fold
                record["log_dir"] = fold_artifact.get("log_dir")
                record["prediction_dir"] = fold_artifact.get("prediction_dir")
                checkpoints = fold_artifact.get("checkpoints") or []
                record["checkpoint"] = checkpoints[-1] if checkpoints else None
                record["metrics_path"] = fold_artifact.get("metrics_path")
                for key, value in (fold_artifact.get("metrics") or {}).items():
                    record[f"metric/{key}"] = value
                records.append(record)
        return records

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("BenchmarkResult.to_dataframe() requires pandas.") from exc
        return pd.DataFrame(self.to_records())

    def save(self, path: str) -> None:
        records = self.to_records()
        if not records:
            with open(path, "w") as f:
                f.write("")
            return
        import csv

        fieldnames = sorted({key for record in records for key in record.keys()})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    @property
    def dry_run(self) -> bool:
        return bool(self.payload.get("dry_run", False))

    def __repr__(self) -> str:
        action = self.payload.get("plan", {}).get("action", "benchmark")
        results = len(self.payload.get("results", []))
        return f"BenchmarkResult(action={action!r}, dry_run={self.dry_run}, results={results})"

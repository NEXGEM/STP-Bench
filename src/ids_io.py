"""Read ids.csv with its train / val / test labels.

One ids.csv per dataset holds every sample. For each fold_<k> column:
  train -- the sample trains the fold-k model
  val   -- cross-validation sample held out in fold k (monitoring and CV evaluation)
  test  -- held-out sample, never trained on (the same in every fold)
Legacy files (no "val"; the CV hold-out fold is labelled "test") are read the same way.
The held-out samples of a dataset are evaluated through the virtual dataset name `<name>_holdout`: its meta_dir
`<meta_dir>_holdout` is a generated alias (symlinks to <name>/ids.csv and <name>/genes.json, see ensure_holdout_alias)
and resolves to the held-out rows of <name>/ids.csv. Nothing named `*_holdout` is stored in the repository.
"""
import os

import pandas as pd

HOLDOUT_SUFFIX = "_holdout"


def _fold_cols(df):
    return [c for c in df.columns if c.startswith("fold_")]


def holdout_mask(df):
    """Rows labelled "test" in every fold column (held-out samples)."""
    cols = _fold_cols(df)
    if not cols:
        return pd.Series(False, index=df.index)
    return (df[cols].astype(str).apply(lambda s: s.str.lower()) == "test").all(axis=1)


def has_val(df):
    cols = _fold_cols(df)
    return bool(cols) and any((df[c].astype(str).str.lower() == "val").any() for c in cols)


def ids_path(meta_dir):
    """Path of the ids.csv that holds `meta_dir`'s samples (a `<name>_holdout` dir may only link to <name>/ids.csv)."""
    own = os.path.join(meta_dir, "ids.csv")
    if os.path.isfile(own):
        return own
    d = os.path.normpath(meta_dir)
    if d.endswith(HOLDOUT_SUFFIX):
        base = os.path.join(d[: -len(HOLDOUT_SUFFIX)], "ids.csv")
        if os.path.isfile(base):
            return base
    raise FileNotFoundError(f"{own} is not found.")


def ensure_holdout_alias(meta_dir):
    """Create `<base>_holdout/{ids.csv,genes.json}` as symlinks to the base dir's files (no-op for other dirs)."""
    d = os.path.normpath(meta_dir)
    if not d.endswith(HOLDOUT_SUFFIX):
        return
    base = d[: -len(HOLDOUT_SUFFIX)]
    if not os.path.isdir(base):
        return
    try:
        os.makedirs(d, exist_ok=True)
        for name in ("ids.csv", "genes.json"):
            src, dst = os.path.join(base, name), os.path.join(d, name)
            if os.path.isfile(src) and not os.path.lexists(dst):
                os.symlink(os.path.join("..", os.path.basename(base), name), dst)
    except OSError:
        pass  # read-only checkout: ids_path() still resolves ids.csv through the base dir


def read_ids(meta_dir, **kwargs):
    """ids.csv rows that belong to `meta_dir`: the held-out rows for a `<name>_holdout` dir, the CV rows otherwise.

    Files without held-out rows (legacy or CV-only) are returned unchanged.
    """
    df = pd.read_csv(ids_path(meta_dir), **kwargs)
    if not _fold_cols(df):
        return df
    mask = holdout_mask(df)
    if not has_val(df):
        return df
    is_holdout_dir = os.path.normpath(meta_dir).endswith(HOLDOUT_SUFFIX)
    return df[mask].reset_index(drop=True) if is_holdout_dir else df[~mask].reset_index(drop=True)


def phase_label(df, phase):
    """Label in fold_<k> that a dataset phase selects: train -> train; test -> val (CV) or test (held-out / legacy)."""
    phase = str(phase).lower()
    if phase == "train":
        return "train"
    if phase in ("test", "val"):
        return "val" if has_val(df) else "test"
    raise ValueError(f"phase must be 'train' or 'test', but got {phase}")


def select_ids(df, fold, phase):
    """sample_id list of fold `fold` for `phase` ('train' or 'test') from a frame returned by read_ids."""
    col = f"fold_{fold}"
    if col not in df.columns:
        raise KeyError(col)
    return df.loc[df[col].astype(str).str.lower() == phase_label(df, phase), "sample_id"].tolist()

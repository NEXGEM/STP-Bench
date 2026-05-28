import argparse
import os

import pandas as pd
from sklearn.model_selection import KFold, GroupKFold


def split_data_cv(df, input_dir, n_splits=5, shuffle=True, random_state=42):
    """Add fold_N columns to ids.csv for cross-validation splitting."""
    fold_cols = [f"fold_{fold}" for fold in range(n_splits)]
    if all(col in df.columns for col in fold_cols):
        print(f"Fold columns already exist in {input_dir}/ids.csv. Exiting.")
        return df

    df = df.drop(columns=[col for col in df.columns if col.startswith("fold_")], errors="ignore")
    for col in fold_cols:
        df[col] = "train"

    if 'case_id' in df.columns:
        split_generator = GroupKFold(n_splits=n_splits).split(df, groups=df['case_id'])
    else:
        split_generator = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(df)

    for fold, (_, test_idx) in enumerate(split_generator):
        df.loc[df.index[test_idx], f"fold_{fold}"] = "test"

    df.to_csv(f'{input_dir}/ids.csv', index=False)
    return df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, required=True, help="Directory containing ids.csv")
    argparser.add_argument("--n_splits", type=int, default=5, help="Number of splits for cross-validation")
    argparser.add_argument("--random_state", type=int, default=42, help="Random seed for splitting")

    args = argparser.parse_args()
    ids_df = pd.read_csv(f'{args.input_dir}/ids.csv')
    split_data_cv(ids_df, args.input_dir, n_splits=args.n_splits, random_state=args.random_state)

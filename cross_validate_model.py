"""
cross_validate_model.py — Standalone GroupKFold cross-validation.

Groups by mrn_id so all slabs from the same shipment stay in the same fold,
preventing the model from memorising shipment-level constants across splits.

Expected range with ~45 shipments:  R² 0.75–0.90,  MAPE 8–15 %

Usage:
    python cross_validate_model.py
    python cross_validate_model.py --schema rrmstock --n-splits 5
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.model_selection import GroupKFold, cross_validate

from db import load_training_tables
from features import MODEL_FEATURES, TARGET_COLUMN, build_model_frame
from pipeline import create_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--schema",   default="rrmstock")
    p.add_argument("--n-splits", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rmi, po, mrn = load_training_tables(schema=args.schema)
    dataset      = build_model_frame(rmi, po=po, mrn=mrn)
    n_rows       = len(dataset)
    n_shipments  = dataset["mrn_id"].nunique()

    if n_rows < 30:
        raise ValueError(f"Too few rows ({n_rows}) to cross-validate.")

    X      = dataset[MODEL_FEATURES].copy()
    y      = dataset[TARGET_COLUMN].copy()
    groups = dataset["mrn_id"].fillna(-1)
    n_splits = min(args.n_splits, groups.nunique())

    scores = cross_validate(
        create_pipeline(), X, y,
        cv=GroupKFold(n_splits=n_splits),
        groups=groups,
        scoring={"mae":  "neg_mean_absolute_error",
                 "mape": "neg_mean_absolute_percentage_error",
                 "r2":   "r2"},
        n_jobs=-1,
        return_train_score=True,
    )

    r2_test  = scores["test_r2"]
    r2_train = scores["train_r2"]
    mae_cv   = -scores["test_mae"]
    mape_cv  = -scores["test_mape"]
    gap      = r2_train.mean() - r2_test.mean()

    print(f"GroupKFold CV  rows={n_rows}  shipments={n_shipments}  folds={n_splits}")
    print(f"  R²   {r2_test.mean():.4f} ± {r2_test.std():.4f}")
    print(f"  MAE  ₹{mae_cv.mean():,.0f} ± ₹{mae_cv.std():,.0f}")
    print(f"  MAPE {mape_cv.mean():.2%} ± {mape_cv.std():.2%}")
    print(f"  Train–test gap: {gap:.4f}", "⚠ overfitting" if gap > 0.15 else "✓")
    print(f"\n  Per-fold test R²:  {np.round(r2_test, 4).tolist()}")
    print(f"  Per-fold train R²: {np.round(r2_train, 4).tolist()}")

    if r2_test.mean() > 0.97:
        print("\n⚠  R² > 0.97 — check MODEL_FEATURES for post-arrival columns.")
    elif r2_test.mean() < 0.50:
        print("\n⚠  R² < 0.50 — insufficient shipments; accuracy will improve with more data.")
    else:
        print(f"\n✓  R² {r2_test.mean():.3f} — realistic range for pre-arrival estimation.")


if __name__ == "__main__":
    main()

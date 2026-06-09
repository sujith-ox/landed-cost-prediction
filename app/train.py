"""
train.py — Train and save the landed cost prediction model.

Steps:
    1. Load rmi, po, mrn from PostgreSQL
    2. Join tables and engineer features
    3. GroupKFold cross-validation (grouped by mrn_id)
    4. Train final pipeline on all data
    5. Compute anomaly threshold from held-out residuals
    6. Save artifact to disk

Usage:
    python train.py
    python train.py --schema rrmstock --anomaly-percentile 90
    python train.py --model-path /opt/models/landed_cost_v2.pkl
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold, cross_validate

from app.db import load_training_tables
from app.features import MODEL_FEATURES, TARGET_COLUMN, build_model_frame
from app.pipeline import (
    build_artifact,
    create_pipeline,
    regression_metrics,
    score_landed_cost,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--schema", default="rrmstock")
    p.add_argument("--model-path", default=os.path.join("models", "model.pkl"))
    p.add_argument(
        "--scored-path", default=os.path.join("models", "training_scores.csv")
    )
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--anomaly-percentile", type=float, default=90.0)
    p.add_argument("--n-cv-splits", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.scored_path) or ".", exist_ok=True)

    print("Loading data...")
    rmi, po, mrn = load_training_tables(schema=args.schema)

    dataset = build_model_frame(rmi, po=po, mrn=mrn)
    n_rows = len(dataset)
    n_shipments = dataset["mrn_id"].nunique()
    print(f"Rows: {n_rows}  |  Shipments: {n_shipments}")

    if n_rows < 30:
        raise ValueError(f"Too few rows ({n_rows}) to train. Check data quality.")

    X = dataset[MODEL_FEATURES].copy()
    y = dataset[TARGET_COLUMN].copy()
    groups = dataset["mrn_id"].fillna(-1)

    # ── Cross-validation ──────────────────────────────────────────────────────
    n_splits = min(args.n_cv_splits, groups.nunique())
    cv_results = cross_validate(
        create_pipeline(),
        X,
        y,
        cv=GroupKFold(n_splits=n_splits),
        groups=groups,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mape": "neg_mean_absolute_percentage_error",
            "r2": "r2",
        },
        n_jobs=-1,
        return_train_score=True,
    )

    cv_r2 = cv_results["test_r2"]
    cv_r2_std = cv_r2.std()
    cv_mae = float(-cv_results["test_mae"].mean())
    cv_mape = float(-cv_results["test_mape"].mean())

    print(f"\nGroupKFold CV  (n={n_splits}, grouped by mrn_id)")
    print(
        f"  R²   {cv_r2.mean():.4f} ± {cv_r2_std:.4f}   folds: {np.round(cv_r2, 3).tolist()}"
    )
    print(f"  MAE  ₹{cv_mae:,.0f}")
    print(f"  MAPE {cv_mape:.2%}")

    gap = cv_results["train_r2"].mean() - cv_r2.mean()
    print(f"  Train–test gap: {gap:.4f}", "⚠ overfitting" if gap > 0.15 else "✓")

    if cv_r2.mean() > 0.97:
        print("  ⚠  R² > 0.97 — check MODEL_FEATURES for post-arrival columns.")

    # ── Held-out test split (group-aware) ─────────────────────────────────────
    rng = np.random.default_rng(42)
    unique_groups = groups.unique()
    test_groups = set(
        rng.choice(
            unique_groups,
            size=max(1, int(len(unique_groups) * args.test_size)),
            replace=False,
        )
    )
    train_mask = ~groups.isin(test_groups)
    test_mask = groups.isin(test_groups)

    # ── Final training (all data) ─────────────────────────────────────────────
    print(f"\nTraining on all {n_rows} rows...")
    final_pipeline = create_pipeline(random_state=42)
    final_pipeline.fit(X, y)

    test_pred = final_pipeline.predict(X[test_mask])
    metrics = regression_metrics(y[test_mask], test_pred)
    print(
        f"  Held-out test — R²: {metrics['r2']:.4f}  MAE: ₹{metrics['mae']:,.0f}  MAPE: {metrics['mape']:.2%}"
    )

    # ── Anomaly threshold from held-out residuals ─────────────────────────────
    # Using held-out (not training) residuals avoids an optimistically tight threshold.
    residual_threshold = float(
        np.percentile(np.abs(y[test_mask].values - test_pred), args.anomaly_percentile)
    )

    # ── Save artifact ─────────────────────────────────────────────────────────
    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = build_artifact(
        pipeline=final_pipeline,
        residual_threshold=residual_threshold,
        metrics=metrics,
        cv_metrics={
            "r2": float(cv_r2.mean()),
            "r2_std": float(cv_r2_std),
            "mae": cv_mae,
            "mape": cv_mape,
            "n_splits": n_splits,
        },
        training_rows=n_rows,
        anomaly_percentile=args.anomaly_percentile,
        trained_at=trained_at,
    )
    joblib.dump(artifact, args.model_path)

    scored = score_landed_cost(dataset, artifact)
    scored.sort_values("absolute_residual", ascending=False).to_csv(
        args.scored_path, index=False
    )

    print(
        f"\n  Anomaly threshold (p{args.anomaly_percentile:.0f}): ₹{residual_threshold:,.0f}"
    )
    print(f"  Artifact → {args.model_path}")
    print(f"  Scores   → {args.scored_path}")

    if cv_r2.mean() > 0.97:
        print(
            "\n⚠  CV R² > 0.97 — verify no post-arrival columns are in MODEL_FEATURES."
        )
    elif cv_r2.mean() < 0.50:
        print("\n⚠  CV R² < 0.50 — not enough shipments to generalise reliably yet.")
    else:
        print(f"\n✓  CV R² {cv_r2.mean():.3f} — suitable for beta deployment.")


if __name__ == "__main__":
    main()

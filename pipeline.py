"""
pipeline.py — Sklearn pipeline and scoring utilities for the landed cost model.

Pipeline steps:
    1. LandedCostFeatureEngineer  — computes volume, density, area, etc.
    2. ColumnTransformer          — imputes and scales numeric; encodes categorical
    3. GradientBoostingRegressor  — 200 trees, lr=0.05, max_depth=4
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    LandedCostFeatureEngineer,
)


def create_pipeline(random_state: int = 42) -> Pipeline:
    """
    Return an untrained sklearn Pipeline.

    Numeric:     SimpleImputer(median) → RobustScaler
    Categorical: SimpleImputer(most_frequent) → OneHotEncoder(min_frequency=2)
                 min_frequency=2 groups rare vendors/brokers into an infrequent bucket.
                 handle_unknown='ignore' silently zeros unseen categories at inference.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,     NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=random_state,
    )
    return Pipeline([
        ("features",      LandedCostFeatureEngineer()),
        ("preprocessing", preprocessor),
        ("model",         model),
    ])


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def validate_model_input(df: pd.DataFrame) -> None:
    """Raise ValueError if required physical dimensions are missing or non-positive."""
    for col in ["height", "width", "length", "weight"]:
        if col not in df.columns:
            raise ValueError(f"Required field '{col}' is missing.")
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            raise ValueError(f"'{col}' contains non-numeric or missing values.")
        if (vals <= 0).any():
            raise ValueError(f"'{col}' must be greater than zero.")
    if "purchase_amount_per_uom_in_vendor_currency" in df.columns:
        vals = pd.to_numeric(df["purchase_amount_per_uom_in_vendor_currency"], errors="coerce")
        if (vals.dropna() <= 0).any():
            raise ValueError("purchase_amount_per_uom_in_vendor_currency must be positive.")


def build_artifact(
    pipeline:           Pipeline,
    residual_threshold: float,
    metrics:            Dict[str, float],
    cv_metrics:         Dict[str, float],
    training_rows:      int,
    anomaly_percentile: float,
    trained_at:         str,
) -> Dict[str, Any]:
    """Bundle the trained pipeline and metadata into a serialisable artifact dict."""
    return {
        "pipeline":             pipeline,
        "model_features":       MODEL_FEATURES,
        "numeric_features":     NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column":        "total_amount",
        "residual_threshold":   float(residual_threshold),
        "metrics":              metrics,
        "cv_metrics":           cv_metrics,
        "training_rows":        int(training_rows),
        "anomaly_percentile":   float(anomaly_percentile),
        "trained_at":           trained_at,
    }


def score_landed_cost(df: pd.DataFrame, artifact: Dict[str, Any]) -> pd.DataFrame:
    """
    Run predictions and append result columns to df.

    If total_amount is present in df, also computes residuals and anomaly flags.
    Columns added:
        predicted_landed_cost, actual_landed_cost, residual,
        absolute_residual, residual_pct, is_cost_anomaly
    """
    pipeline  = artifact["pipeline"]
    threshold = float(artifact["residual_threshold"])

    validate_model_input(df)

    result = df.copy()
    result["predicted_landed_cost"] = pipeline.predict(result)

    target_col = artifact.get("target_column", "total_amount")
    if target_col in result.columns:
        actual = pd.to_numeric(result[target_col], errors="coerce")
        result["actual_landed_cost"] = actual
        result["residual"]           = actual - result["predicted_landed_cost"]
        result["absolute_residual"]  = result["residual"].abs()
        result["residual_pct"]       = (
            result["absolute_residual"] / result["predicted_landed_cost"].replace(0, np.nan)
        )
        result["is_cost_anomaly"] = (result["absolute_residual"] >= threshold).astype(int)
    else:
        for col in ["actual_landed_cost", "residual", "absolute_residual", "residual_pct"]:
            result[col] = np.nan
        result["is_cost_anomaly"] = 0

    return result

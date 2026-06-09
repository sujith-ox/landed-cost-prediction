"""
predict_api.py — Total Landed Cost Prediction API
==================================================
Serves the trained landed cost model over HTTP.

Endpoints
---------
GET  /health              — model status, training metadata, CV metrics
GET  /features            — feature list (for ERP integration reference)
POST /predict             — predict landed cost for one slab
POST /predict/batch       — predict for multiple slabs (up to 500)
POST /score/from-db       — score full inventory from database, return anomalies

ERP integration notes
---------------------
Call POST /predict when a user adds a product line to a Purchase Order.
The response includes:
    predicted_landed_cost  — point estimate in INR
    lower_bound / upper_bound — ±15 % confidence band for display
    confidence_label       — "HIGH" / "MEDIUM" / "LOW" based on route data density
    is_cost_anomaly        — 1 if actual cost (when supplied) deviates from model

Run locally
-----------
    uvicorn predict_api:app --reload --port 8000

Environment variables (.env)
----------------------------
    LANDED_COST_MODEL_PATH   path to model.pkl  (default: models/model.pkl)
    DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.db import load_training_tables
from app.features import MODEL_FEATURES, build_model_frame
from app.pipeline import score_landed_cost

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("LANDED_COST_MODEL_PATH", os.path.join("models", "model.pkl"))
CONFIDENCE_BAND = 0.15  # ±15 % displayed in ERP as the estimate range
HIGH_CONF_ROUTES = {  # ports with ≥30 historical slabs — higher confidence
    "jebel ali",
    "mersin",
    "gemlik",
    "izmir",
    "aliaga",
    "venezia",
}


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RRM Landed Cost Prediction API",
    description=(
        "Predicts total landed cost (INR) for stone slabs at Purchase Order time. "
        "Uses only pre-arrival features: dimensions, vendor quoted price, "
        "origin port, vendor, and broker. "
        "No post-arrival cost columns (customs, clearing, freight) are required."
    ),
    version="3.0.0",
)


# ── Model cache ───────────────────────────────────────────────────────────────

_artifact: Dict[str, Any] | None = None


def load_artifact() -> Dict[str, Any]:
    global _artifact
    if _artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model not found at {MODEL_PATH!r}. "
                    "Run train.py first to train and save the model."
                ),
            )
        loaded = joblib.load(MODEL_PATH)
        required_keys = {"pipeline", "residual_threshold", "metrics", "training_rows"}
        missing = required_keys - set(loaded)
        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid model artifact — missing keys: {sorted(missing)}",
            )
        _artifact = loaded
    return _artifact


# ── Schemas ───────────────────────────────────────────────────────────────────


class SlabRequest(BaseModel):
    """
    Pre-arrival slab information.
    height, width, length, weight are required.
    All other fields improve prediction accuracy — supply as many as available.
    """

    # Required — physical dimensions
    height: float = Field(..., gt=0, description="Height in cm")
    width: float = Field(..., gt=0, description="Width in cm")
    length: float = Field(..., gt=0, description="Length in cm")
    weight: float = Field(..., gt=0, description="Weight in metric tons")

    # Strongly recommended — drives most of the prediction
    purchase_amount_per_uom_in_vendor_currency: Optional[float] = Field(
        None, gt=0, description="Vendor quoted price per ton in USD or EUR"
    )

    # Slab identity
    product_name: Optional[str] = Field(
        None, description="Stone type (e.g. RAINFOREST GREEN)"
    )
    color: Optional[str] = Field(None, description="Stone colour")
    cft: Optional[float] = Field(None, gt=0, description="Cubic feet")

    # Route context — significantly improves accuracy
    vendor_name: Optional[str] = Field(None, description="Stone supplier name")
    broker_name: Optional[str] = Field(None, description="Clearing/freight broker")
    vendor_currency: Optional[str] = Field(None, description="USD or EUR")
    port_of_shipment: Optional[str] = Field(
        None, description="Origin port (e.g. MERSIN)"
    )
    country_of_shipment: Optional[str] = Field(None, description="Origin country")
    port_of_discharge: Optional[str] = Field(None, description="Indian arrival port")
    country: Optional[str] = Field(None, description="Origin country (from PO)")

    # Optional: supply actual cost to trigger anomaly check
    total_amount: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Actual total landed cost if known — triggers anomaly detection. "
            "Supply after charges are finalised to validate entered data."
        ),
    )

    @model_validator(mode="after")
    def warn_missing_route_context(self) -> "SlabRequest":
        missing = []
        if not self.vendor_name:
            missing.append("vendor_name")
        if not self.port_of_shipment:
            missing.append("port_of_shipment")
        if not self.purchase_amount_per_uom_in_vendor_currency:
            missing.append("purchase_amount_per_uom_in_vendor_currency")
        # Store for response metadata — don't raise, just flag
        object.__setattr__(self, "_missing_context", missing)
        return self


class SlabResponse(BaseModel):
    predicted_landed_cost: float
    lower_bound: float
    upper_bound: float
    confidence_label: str  # HIGH / MEDIUM / LOW
    currency: str = "INR"
    actual_landed_cost: Optional[float] = None
    residual: Optional[float] = None
    residual_pct: Optional[float] = None
    is_cost_anomaly: int = 0
    anomaly_threshold_inr: float
    missing_context_fields: List[str] = []


class BatchRequest(BaseModel):
    slabs: List[SlabRequest] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    count: int
    results: List[SlabResponse]


class ScoreDbRequest(BaseModel):
    schema_name: str = Field("rrmstock", alias="schema")
    top_n: int = Field(25, ge=1, le=500)
    only_anomalies: bool = True

    model_config = {"populate_by_name": True}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _json_safe(v: Any) -> Any:
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def _confidence_label(port: Optional[str], n_training_rows: int) -> str:
    """
    HIGH  — port is well-represented in training data (≥30 slabs)
    MEDIUM— port seen in training but with fewer examples
    LOW   — port not seen in training (new route) or very sparse
    """
    if n_training_rows < 80:
        return "LOW"  # whole model is low-confidence until data accumulates
    if port and port.strip().lower() in HIGH_CONF_ROUTES:
        return "HIGH"
    return "MEDIUM"


def _predict_one(request: SlabRequest, artifact: Dict[str, Any]) -> SlabResponse:
    threshold = float(artifact["residual_threshold"])
    n_rows = int(artifact.get("training_rows", 0))

    input_df = pd.DataFrame([request.model_dump()])

    try:
        scored = score_landed_cost(input_df, artifact)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    row = scored.iloc[0]
    predicted = float(row["predicted_landed_cost"])
    conf = _confidence_label(request.port_of_shipment, n_rows)
    missing = getattr(request, "_missing_context", [])

    return SlabResponse(
        predicted_landed_cost=round(predicted, 2),
        lower_bound=round(predicted * (1 - CONFIDENCE_BAND), 2),
        upper_bound=round(predicted * (1 + CONFIDENCE_BAND), 2),
        confidence_label=conf,
        actual_landed_cost=_json_safe(row.get("actual_landed_cost")),
        residual=_json_safe(row.get("residual")),
        residual_pct=_json_safe(row.get("residual_pct")),
        is_cost_anomaly=int(row["is_cost_anomaly"]),
        anomaly_threshold_inr=threshold,
        missing_context_fields=missing,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> Dict[str, Any]:
    """Model status, training metadata, and CV metrics."""
    artifact = load_artifact()
    cv = artifact.get("cv_metrics", {})
    test = artifact.get("metrics", {})
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "trained_at": artifact.get("trained_at", "unknown"),
        "training_rows": artifact["training_rows"],
        "target_column": artifact.get("target_column", "total_amount"),
        "cv_metrics": {
            "r2": round(cv.get("r2", 0), 4),
            "mape": round(cv.get("mape", 0), 4),
            "mae_inr": round(cv.get("mae", 0), 2),
            "n_splits": cv.get("n_splits", "unknown"),
            "method": "GroupKFold on mrn_id (leak-free)",
        },
        "test_metrics": {
            "r2": round(test.get("r2", 0), 4),
            "mape": round(test.get("mape", 0), 4),
            "mae_inr": round(test.get("mae", 0), 2),
        },
        "anomaly_threshold_inr": artifact["residual_threshold"],
        "confidence_band": f"±{int(CONFIDENCE_BAND * 100)} %",
    }


@app.get("/features")
def feature_info() -> Dict[str, Any]:
    """Feature list consumed by the model — useful for ERP integration."""
    from app.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    return {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "note": (
            "All features are pre-arrival (known at PO creation time). "
            "Post-arrival cost columns (customs_duty, clearing_charges, ocean_fright, "
            "transportation_charges, amount) are intentionally excluded."
        ),
        "erp_integration_tip": (
            "Supply vendor_name, broker_name, port_of_shipment, and "
            "purchase_amount_per_uom_in_vendor_currency for highest accuracy. "
            "The model degrades gracefully when optional fields are absent."
        ),
    }


@app.post("/predict", response_model=SlabResponse)
def predict(request: SlabRequest) -> SlabResponse:
    """
    Predict total landed cost for a single slab.
    Call this when a user adds a product line to a Purchase Order in the ERP.
    """
    return _predict_one(request, load_artifact())


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    """
    Predict for up to 500 slabs in one request.
    Useful for bulk PO entry or re-scoring historical records.
    """
    artifact = load_artifact()
    results = [_predict_one(slab, artifact) for slab in request.slabs]
    return BatchResponse(count=len(results), results=results)


@app.post("/score/from-db")
def score_from_db(request: ScoreDbRequest) -> Dict[str, Any]:
    """
    Load all inventory from the database, score it, and return anomalies.
    Use this for periodic audits — run after all charge data has been entered
    for a batch of shipments to flag entries that deviate from model expectations.
    """
    artifact = load_artifact()

    try:
        rmi, po, mrn = load_training_tables(schema=request.schema_name)
        dataset = build_model_frame(rmi, po=po, mrn=mrn)
        scored = score_landed_cost(dataset, artifact).sort_values(
            "absolute_residual", ascending=False
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    total_rows = len(scored)
    anomaly_rows = int(scored["is_cost_anomaly"].sum())

    if request.only_anomalies:
        scored = scored[scored["is_cost_anomaly"] == 1]

    output_cols = [
        c
        for c in [
            "id",
            "mrn_id",
            "product_name",
            "color",
            "country",
            "vendor_name",
            "broker_name",
            "port_of_shipment",
            "weight",
            "purchase_amount_per_uom_in_vendor_currency",
            "actual_landed_cost",
            "predicted_landed_cost",
            "residual",
            "residual_pct",
            "is_cost_anomaly",
        ]
        if c in scored.columns
    ]

    rows = [
        {k: _json_safe(v) for k, v in row.items()}
        for row in scored.head(request.top_n)[output_cols].to_dict(orient="records")
    ]

    return {
        "summary": {
            "total_rows": total_rows,
            "anomaly_rows": anomaly_rows,
            "anomaly_rate": round(anomaly_rows / total_rows, 4) if total_rows else 0,
            "threshold_inr": artifact["residual_threshold"],
        },
        "results": rows,
    }

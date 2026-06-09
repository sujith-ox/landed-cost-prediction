"""
features.py — Feature definitions and dataset builder for the landed cost model.

Target column : total_amount  (INR per slab — vendor base cost + all import charges)

Excluded columns (post-arrival, unavailable at PO time):
    amount, amount_in_vendor_currency, purchase_amount_per_uom,
    total_raw_material_amount_without_tax, customs_duty, clearing_charges,
    ocean_fright, transportation_charges, other_charges, container_damage_charges
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


TARGET_COLUMN = "total_amount"

# ── Feature definitions ───────────────────────────────────────────────────────

# Raw numeric inputs from raw_material_inventory (all 100 % populated)
_RMI_NUMERIC = [
    "height",
    "width",
    "length",
    "weight",
    "cft",
    "purchase_amount_per_uom_in_vendor_currency",
]

# Categorical inputs from material_receive_note (via mrn_id join)
_MRN_CATEGORICAL = [
    "vendor_name",
    "broker_name",
    "vendor_currency",
    "port_of_shipment",
    "country_of_shipment",
    "port_of_discharge",
]

# Categorical inputs from raw_material_inventory
_RMI_CATEGORICAL = [
    "product_name",
    "color",
    "country",          # 44 % populated; falls back to country_of_shipment
]

# Engineered from raw inputs — all derivable at PO creation time
_ENGINEERED_NUMERIC = [
    "volume",                # height × width × length
    "density",               # weight / volume
    "area",                  # width × length
    "weight_x_vendor_price", # weight × vendor_price_per_uom_in_currency
    "price_per_cft",         # vendor_price_per_uom / cft
    "purchase_month",        # from MRN created_date
    "purchase_quarter",
]

NUMERIC_FEATURES     = _RMI_NUMERIC + _ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = _RMI_CATEGORICAL + _MRN_CATEGORICAL
MODEL_FEATURES       = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .replace("", "unknown")
    )


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = n / d
    return result.where((d != 0) & d.notna(), np.nan)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_model_frame(
    rmi: pd.DataFrame,
    po:  pd.DataFrame | None = None,
    mrn: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join rmi → mrn → po, engineer features, and return a training-ready DataFrame.

    Join path:
        rmi.mrn_id → mrn.id          (vendor, broker, port, currency)
        mrn.po_number → po.purchase_order_number  (currency confirmation)

    Returns a DataFrame containing MODEL_FEATURES + TARGET_COLUMN + mrn_id.
    Rows with a missing or zero target are dropped.
    """
    data = rmi.copy()
    data["mrn_id"] = pd.to_numeric(data["mrn_id"], errors="coerce")

    # Step 1: join MRN for route and vendor context
    if mrn is not None and not mrn.empty:
        mrn_work = mrn.copy()
        mrn_work["mrn_id"]           = pd.to_numeric(mrn_work["id"],        errors="coerce")
        mrn_work["po_number_from_mrn"] = pd.to_numeric(mrn_work["po_number"], errors="coerce")

        mrn_keep = [c for c in [
            "mrn_id", "po_number_from_mrn", "vendor_name", "broker_name",
            "vendor_currency", "port_of_shipment", "country_of_shipment",
            "port_of_discharge", "created_date",
        ] if c in mrn_work.columns]

        data = data.merge(
            mrn_work[mrn_keep].drop_duplicates("mrn_id"),
            on="mrn_id", how="left", suffixes=("", "_mrn"),
        )

        if "created_date_mrn" in data.columns:
            data["mrn_created_date"] = data["created_date_mrn"]
            data.drop(columns=["created_date_mrn"], inplace=True)
        elif "created_date" in data.columns:
            data["mrn_created_date"] = data["created_date"]
        else:
            data["mrn_created_date"] = pd.NaT
    else:
        for col in _MRN_CATEGORICAL:
            if col not in data.columns:
                data[col] = "unknown"
        data["po_number_from_mrn"] = np.nan
        data["mrn_created_date"]   = pd.NaT

    # Step 2: join PO to confirm/supplement vendor_currency
    if po is not None and not po.empty:
        po_work = po.copy()
        po_work["purchase_order_number"] = pd.to_numeric(
            po_work["purchase_order_number"], errors="coerce"
        )
        po_keep = [c for c in ["purchase_order_number", "currency"] if c in po_work.columns]

        data = data.merge(
            po_work[po_keep].drop_duplicates("purchase_order_number"),
            left_on="po_number_from_mrn",
            right_on="purchase_order_number",
            how="left", suffixes=("", "_po"),
        )
        if "currency" in data.columns and "vendor_currency" in data.columns:
            data["vendor_currency"] = data["vendor_currency"].fillna(data["currency"])
        data.drop(columns=["purchase_order_number"], inplace=True, errors="ignore")

    # Step 3: fill missing rmi.country from mrn.country_of_shipment
    if "country" in data.columns and "country_of_shipment" in data.columns:
        data["country"] = data["country"].fillna(data["country_of_shipment"])

    data = _add_features(data)

    _to_numeric(data, [TARGET_COLUMN])
    data = data.dropna(subset=[TARGET_COLUMN])
    data = data[data[TARGET_COLUMN] > 0].copy()
    return data


# ── Feature engineering ───────────────────────────────────────────────────────

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features in-place. Safe to call at training and inference time."""
    data = df.copy()

    for col in _RMI_NUMERIC:
        if col not in data.columns:
            data[col] = np.nan
    _to_numeric(data, _RMI_NUMERIC + [TARGET_COLUMN])

    for col in CATEGORICAL_FEATURES:
        if col not in data.columns:
            data[col] = "unknown"
        data[col] = _normalize_text(data[col])

    data["volume"]               = data["height"] * data["width"] * data["length"]
    data["density"]              = _safe_divide(data["weight"], data["volume"])
    data["area"]                 = data["width"] * data["length"]
    data["weight_x_vendor_price"] = data["weight"] * data["purchase_amount_per_uom_in_vendor_currency"]
    data["price_per_cft"]        = _safe_divide(data["purchase_amount_per_uom_in_vendor_currency"], data["cft"])

    dt = pd.to_datetime(
        data.get("mrn_created_date", pd.Series(pd.NaT, index=data.index)),
        errors="coerce",
    )
    data["purchase_month"]   = dt.dt.month.fillna(0).astype(int)
    data["purchase_quarter"] = dt.dt.quarter.fillna(0).astype(int)

    return data


# ── Sklearn transformer wrapper ───────────────────────────────────────────────

class LandedCostFeatureEngineer(BaseEstimator, TransformerMixin):
    """Wraps _add_features for use as a sklearn pipeline step."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return _add_features(X)


add_features = _add_features

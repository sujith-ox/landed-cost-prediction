"""
db.py — Database connection and table loading.

Join chain discovered from data analysis:
    raw_material_inventory.mrn_id  →  material_receive_note.id
    material_receive_note.po_number →  purchase_order.purchase_order_number

This is the ONLY working join path.  rmi.po_number is null on every row;
mrn.po_number is populated for shipments that have a completed PO.
"""

from __future__ import annotations

import os
import re
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


# ── Connection ────────────────────────────────────────────────────────────────

def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    host     = os.getenv("DB_HOST", "localhost")
    port     = os.getenv("DB_PORT", "5432")
    name     = os.getenv("DB_NAME")
    user     = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    missing = [
        k for k, v in
        {"DB_NAME": name, "DB_USER": user, "DB_PASSWORD": password}.items()
        if not v
    ]
    if missing:
        raise ValueError(
            f"Missing database credentials in .env: {', '.join(missing)}"
        )

    return (
        f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{name}"
    )


def get_engine():
    return create_engine(get_database_url(), pool_pre_ping=True)


# ── Safety ────────────────────────────────────────────────────────────────────

def validate_identifier(identifier: str) -> str:
    """Reject anything that is not a plain SQL identifier (prevents injection)."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")
    return identifier


def read_table(schema: str, table: str) -> pd.DataFrame:
    schema = validate_identifier(schema)
    table  = validate_identifier(table)
    query  = text(f"select * from {schema}.{table}")
    with get_engine().connect() as conn:
        return pd.read_sql_query(query, conn)


# ── Table loading ─────────────────────────────────────────────────────────────

def load_training_tables(
    schema: str = "rrmstock",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three tables needed to build a clean training frame.

    Returns
    -------
    rmi : raw_material_inventory  — one row per slab block
    po  : purchase_order          — vendor, broker, currency per PO
    mrn : material_receive_note   — shipment-level: port, route, po_number link
    """
    rmi = read_table(schema, "raw_material_inventory")
    po  = read_table(schema, "purchase_order")
    mrn = read_table(schema, "material_receive_note")
    return rmi, po, mrn
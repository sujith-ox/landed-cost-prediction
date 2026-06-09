# Landed Cost Prediction Model

A machine learning pipeline that predicts total landed cost (INR) for stone slab imports at Purchase Order time — using only pre-arrival features like dimensions, vendor quoted price, and shipment route.

---

## What it does

- Predicts **total landed cost per slab** before customs/freight charges are known
- Flags **cost anomalies** when actual charges deviate significantly from model expectations
- Serves predictions via a **REST API** (FastAPI) for ERP integration
- Uses **GroupKFold cross-validation** grouped by `mrn_id` (shipment) to prevent data leakage across folds
- Supports **Docker** for containerised deployment

Expected accuracy: R² 0.75–0.90 · MAPE 8–15%

---

## Stack

| Layer | Technology |
|---|---|
| Model | `GradientBoostingRegressor` (scikit-learn 1.4+) |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 16 via SQLAlchemy 2.0 |
| Serialisation | joblib |
| Containerisation | Docker + Docker Compose |

---

## Setup

### Local (pip)

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rrmstock
DB_USER=postgres
DB_PASSWORD=your_password

# Or set a full connection string instead of the above:
# DATABASE_URL=postgresql+psycopg2://user:password@host:5432/dbname

# Optional: override default model path
# LANDED_COST_MODEL_PATH=models/model.pkl
```

### Docker

```bash
docker compose up --build
```

This starts:
- `landed_cost_api` — the FastAPI server on port `8000`
- `landed_cost_db` — PostgreSQL 16 on port `5432` (volume-persisted)

> **Note:** Run `train.py` before starting the API — the `/predict` endpoints require a trained `model.pkl`.

---

## Usage

### 1. Train

```bash
python train.py

# With options:
python train.py \
  --schema rrmstock \
  --anomaly-percentile 90 \
  --model-path models/model.pkl \
  --n-cv-splits 5 \
  --test-size 0.15
```

Outputs:
- `models/model.pkl` — serialised pipeline + metadata artifact
- `models/training_scores.csv` — per-row predictions, residuals, and anomaly flags

### 2. Cross-validate only

```bash
python cross_validate_model.py

# With options:
python cross_validate_model.py --schema rrmstock --n-splits 5
```

Prints per-fold R², MAE, MAPE, and train–test gap. Use this to evaluate before committing to a full retrain.

### 3. Serve the API

```bash
uvicorn predict_api:app --reload --port 8000
```

Interactive docs available at `http://localhost:8000/docs`.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Model status, training metadata, CV + test metrics |
| `GET` | `/features` | Full feature list with ERP integration notes |
| `POST` | `/predict` | Predict landed cost for a single slab |
| `POST` | `/predict/batch` | Predict for up to 500 slabs in one request |
| `POST` | `/score/from-db` | Load full inventory from DB, score it, return anomalies |

### Predict — minimal request

```json
{
  "height": 180,
  "width": 90,
  "length": 2,
  "weight": 1.8,
  "purchase_amount_per_uom_in_vendor_currency": 420,
  "vendor_name": "MI STONE",
  "port_of_shipment": "MERSIN"
}
```

### Predict — full request (recommended for highest accuracy)

```json
{
  "height": 180,
  "width": 90,
  "length": 2,
  "weight": 1.8,
  "cft": 3.6,
  "purchase_amount_per_uom_in_vendor_currency": 420,
  "product_name": "RAINFOREST GREEN",
  "color": "green",
  "vendor_name": "MI STONE",
  "broker_name": "STAR LOGISTICS",
  "vendor_currency": "USD",
  "port_of_shipment": "MERSIN",
  "country_of_shipment": "TURKEY",
  "port_of_discharge": "NHAVA SHEVA",
  "country": "TURKEY"
}
```

### Predict response

```json
{
  "predicted_landed_cost": 142500.00,
  "lower_bound": 121125.00,
  "upper_bound": 163875.00,
  "confidence_label": "HIGH",
  "currency": "INR",
  "actual_landed_cost": null,
  "residual": null,
  "residual_pct": null,
  "is_cost_anomaly": 0,
  "anomaly_threshold_inr": 18400.00,
  "missing_context_fields": []
}
```

**Confidence labels:** `HIGH` for well-represented ports (Jebel Ali, Mersin, Gemlik, İzmir, Aliağa, Venezia with ≥30 historical slabs); `MEDIUM` for other known ports; `LOW` when the overall model has fewer than 80 training rows.

**Anomaly detection:** Supply `total_amount` (actual landed cost) in the request body to trigger a residual check against the model's anomaly threshold.

### Batch predict

```json
{
  "slabs": [
    { "height": 180, "width": 90, "length": 2, "weight": 1.8 },
    { "height": 200, "width": 100, "length": 2.2, "weight": 2.1 }
  ]
}
```

### Score from database

```json
{
  "schema": "rrmstock",
  "top_n": 25,
  "only_anomalies": true
}
```

Returns a summary (`total_rows`, `anomaly_rows`, `anomaly_rate`, `threshold_inr`) and the top-N anomalous rows sorted by absolute residual.

---

## Features

All features are available at PO creation time. Post-arrival columns (`customs_duty`, `clearing_charges`, `ocean_fright`, `transportation_charges`, `amount`, etc.) are intentionally excluded.

### Numeric

| Feature | Source | Notes |
|---|---|---|
| `height`, `width`, `length` | `raw_material_inventory` | Dimensions in cm |
| `weight` | `raw_material_inventory` | Metric tons |
| `cft` | `raw_material_inventory` | Cubic feet |
| `purchase_amount_per_uom_in_vendor_currency` | `raw_material_inventory` | Vendor quoted price |
| `volume` | Engineered | `height × width × length` |
| `density` | Engineered | `weight / volume` |
| `area` | Engineered | `width × length` |
| `weight_x_vendor_price` | Engineered | `weight × purchase_amount_per_uom_in_vendor_currency` |
| `price_per_cft` | Engineered | `purchase_amount_per_uom_in_vendor_currency / cft` |
| `purchase_month` | Engineered | From MRN `created_date` |
| `purchase_quarter` | Engineered | From MRN `created_date` |

### Categorical

| Feature | Source | Notes |
|---|---|---|
| `product_name`, `color` | `raw_material_inventory` | Stone type and colour |
| `country` | `raw_material_inventory` | 44% populated; falls back to `country_of_shipment` |
| `vendor_name`, `broker_name` | `material_receive_note` | Rare values grouped into an infrequent bucket |
| `vendor_currency` | `material_receive_note` / `purchase_order` | USD or EUR; PO value used as fallback |
| `port_of_shipment`, `country_of_shipment`, `port_of_discharge` | `material_receive_note` | Shipment route |

---

## Database schema

Expects a PostgreSQL schema (default: `rrmstock`) with three tables:

| Table | Role |
|---|---|
| `raw_material_inventory` | One row per slab — dimensions, product, cost columns |
| `material_receive_note` | Shipment-level — vendor, broker, port, currency, `created_date` |
| `purchase_order` | PO-level — currency confirmation |

**Join path (the only working path):**
```
raw_material_inventory.mrn_id
  → material_receive_note.id
  → material_receive_note.po_number
  → purchase_order.purchase_order_number
```

> `rmi.po_number` is null on every row. Always join via `mrn_id → mrn.id`.

---

## Model details

### Pipeline

```
LandedCostFeatureEngineer         # computes volume, density, area, etc.
  → ColumnTransformer
      numeric:     SimpleImputer(median) → RobustScaler
      categorical: SimpleImputer(most_frequent) → OneHotEncoder(min_frequency=2, handle_unknown='ignore')
  → GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8)
```

### Anomaly threshold

Computed as the Nth percentile (default: 90th) of absolute residuals on a held-out group-aware test split. Slabs whose `|actual − predicted|` exceeds this threshold are flagged `is_cost_anomaly = 1`.

### Overfitting check

Training prints the train–test R² gap. A gap > 0.15 triggers a warning. CV R² > 0.97 suggests a post-arrival column has leaked into `MODEL_FEATURES`.

---

## Project structure

```
├── train.py                  # Train and save model artifact
├── cross_validate_model.py   # Standalone GroupKFold CV evaluation
├── predict_api.py            # FastAPI prediction server
├── pipeline.py               # Sklearn pipeline, scoring utilities, artifact builder
├── features.py               # Feature definitions, dataset builder, sklearn transformer
├── db.py                     # DB connection, SQL identifier validation, table loading
├── models/
│   ├── model.pkl             # Saved artifact (generated by train.py)
│   └── training_scores.csv   # Per-row scores and anomaly flags
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | Full SQLAlchemy connection string (overrides individual DB_ vars) |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | — | Database name (required if `DATABASE_URL` not set) |
| `DB_USER` | — | Database user (required if `DATABASE_URL` not set) |
| `DB_PASSWORD` | — | Database password (required if `DATABASE_URL` not set) |
| `LANDED_COST_MODEL_PATH` | `models/model.pkl` | Path to the trained model artifact |

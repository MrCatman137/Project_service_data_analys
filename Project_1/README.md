# Vietnam Residential Property Valuation

Production-oriented Flask + scikit-learn/XGBoost application for instant residential property valuation using the **Vietnam Housing Dataset 2024**.

## Architecture

```text
vietnam_housing_dataset.csv
        |
        v
     train.py
        |
        v
   housing.training
        |
        +--> housing.config       paths and feature schemas
        +--> housing.data         cleaning, splits, and preprocessing
        +--> housing.models       estimator candidates and model helpers
        +--> housing.evaluation   grouped-CV and residual diagnostics
        +--> housing.components   pickle-safe transformers and regressors
                 +--> InputNormalizer
                 +--> HousingFeatureEngineer
                 +--> GroupedHierarchicalEncoder
                 +--> PriceTargetRegressor
        |
        +--> ColumnTransformer: imputation, location encoding, compact OHE
        |
        +--> 3-fold grouped-CV, eight-candidate compact search
        +--> OOF-selected non-negative blend (when it beats the best single model)
        |
        +--> models/vietnam_house_pipeline.pkl
        +--> models/feature_metadata.json
                         |
                         v
                    Flask app.py
                 /api/metadata
                 /predict/price
                         |
                         v
             Tailwind + Chart.js UI
```

## 1. Prerequisites

- Python 3.11+
- `vietnam_housing_dataset.csv` with the columns described in the assignment.
- Bash, PowerShell, or an equivalent shell.

Put the dataset in the repository root:

```text
vietnam_house_valuation/
├── vietnam_housing_dataset.csv
├── housing/
│   ├── config.py
│   ├── components.py
│   ├── data.py
│   ├── models.py
│   ├── evaluation.py
│   ├── training.py
│   └── eda/
│       ├── settings.py
│       ├── data.py
│       ├── plots.py
│       ├── holdout.py
│       └── report.py
├── train.py                 # CLI and backwards-compatible imports
├── pipeline_components.py   # legacy pickle/import compatibility
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── templates/
    └── index.html
```

## 2. Virtual environment

### Linux/macOS/Git Bash

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the dataset has another path:

```bash
export HOUSING_DATA=/absolute/path/to/vietnam_housing_dataset.csv
```

PowerShell:

```powershell
$env:HOUSING_DATA="C:\path\to\vietnam_housing_dataset.csv"
```

## 3. Train

```bash
python train.py
```

The script:

1. validates required columns and converts malformed numeric inputs to missing values;
2. removes only rows with missing/non-positive `Price` or `Area`; valid price and area tails remain available;
3. canonicalizes administrative prefixes and common province aliases, and separates street-like address heads from explicitly named developments;
4. creates train-fitted, frequency, size, room, frontage, road, and structure features with safe ratios;
5. keeps a deterministic address-disjoint holdout and selects among eight candidates using three-fold address-grouped cross-validation on development data only;
6. fits hierarchical target encodings with address-grouped out-of-fold values inside each training fold, including parsed street/project heads rather than one-hot encoding raw addresses;
7. compares a small randomized XGBoost candidate, raw/log/square-root targets, inverse-square-root and inverse-price weighting, an ablation without sparse direction fields, a train-only target-winsorized candidate, and an Extra Trees baseline;
8. optimizes non-negative blend weights from development OOF predictions, but keeps a blend only if it improves mean CV MAPE without reducing mean CV R² versus the best single estimator;
9. reports metrics and systematic residual slices on the untouched grouped test set after model selection;
10. writes the complete preprocessing + selected model pipeline to `models/vietnam_house_pipeline.pkl`;
11. writes evaluation metrics, categorical values, and feature importance to `models/feature_metadata.json`.

Tree-based regressors do not require z-score or robust scaling: monotonic scaling does not change their split ordering. The numeric pipeline imputes values without applying a scaler and adds missing-value indicators for numeric fields.

The script explicitly reports whether the selected model achieves the requested <=7% MAPE target. A target this specific cannot be honestly guaranteed without training on the supplied dataset.

## 4. Explore the data

Run the standalone exploratory analysis:

```bash
python eda.py
```

The script prints cleaning and distribution statistics, missingness, parsed location
coverage/cardinality, Spearman correlations, and price summaries by categorical
fields. If the saved model exists, it also recomputes diagnostics on the same
deterministic address-grouped holdout and prints the worst predictions and error
slices. It writes PNG charts and `eda_summary.json` under `reports/eda/`; set
`EDA_OUTPUT` to choose another output directory. This is descriptive analysis,
not a tuning step, and does not modify the model or metadata artifacts.

## 5. Run tests

Tests are organized under `tests/model/` and use Python's standard library:

```powershell
python -m unittest discover -s tests -v
```

The model tests load the persisted pipeline and metadata, so run `python train.py`
first if those artifacts are missing. Coverage includes prediction input formats,
finite/non-negative outputs, engineered location and room features, and Flask
inference endpoints.

## 6. Run locally

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

API metadata:

```bash
curl http://127.0.0.1:5000/api/metadata
```

Prediction example:

```bash
curl -X POST http://127.0.0.1:5000/predict/price \
  -H "Content-Type: application/json" \
  -d '{
    "Address": "Project, Ward, District, Ho Chi Minh City",
    "Area": 80,
    "Frontage": 5,
    "Access Road": 8,
    "House direction": "East",
    "Balcony direction": "South",
    "Floors": 3,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Legal status": "Red book",
    "Furniture state": "Fully furnished"
  }'
```

The persisted pipeline itself accepts a `dict`, a list of dict records, or a pandas DataFrame because `InputNormalizer` is the first pipeline stage.

Training and inference code lives in the `housing/` package: `config.py`
centralizes paths and feature schemas; `components.py` contains pickle-safe
transformers and estimators; `data.py` handles cleaning, preprocessing, and
splits; `models.py` builds and compares candidate estimators; `evaluation.py`
contains CV and residual diagnostics; and `training.py` orchestrates training
and artifact export. The root `train.py` remains the supported CLI, while
`pipeline_components.py` preserves imports used by previously saved pipelines.
EDA code is under `housing/eda/`: shared data helpers and output settings are
separate from descriptive plots, saved-model holdout diagnostics, and report
orchestration. Run it from the project directory with `python eda.py` as before.

## 7. Docker

Train first so that the model artifact exists:

```bash
python train.py
```

Build:

```bash
docker build -t vietnam-house-valuation .
```

Run:

```bash
docker run --rm -p 5000:5000 \
  -v "$(pwd)/models:/app/models:ro" \
  vietnam-house-valuation
```

Or with Compose:

```bash
docker compose build
docker compose up -d
```

Open:

```text
http://127.0.0.1:5000/
```

Stop:

```bash
docker compose down
```

## 8. API response

Successful prediction:

```json
{
  "status": "success",
  "predicted_price": 12.345,
  "price_range": {
    "min": 11.6043,
    "max": 13.0857
  },
  "currency": "Billion VND",
  "confidence": 94.2,
  "features_parsed": {
    "province_from_address": "Ho Chi Minh City",
    "address_parts": 4,
    "total_rooms": 5,
    "rooms_per_area": 0.0625
  }
}
```

`confidence` is intentionally described as a UI indicator derived from validation MAPE; it is **not** a calibrated probabilistic confidence interval. The displayed price range is the required adaptive ±6% interval.

## 9. Production notes

- Keep `models/` outside the application image when model artifacts are deployed independently.
- Pin dependency versions for reproducibility.
- Run behind a TLS-terminating reverse proxy in a real deployment.
- Do not interpret the ±6% display interval as a statistically calibrated prediction interval.
- Re-train and re-evaluate when the housing market or source-data distribution changes.
- The MAPE threshold must be verified against the actual dataset. If training reports `MAPE target: NOT MET`, the artifact is persisted for diagnostics but should not be represented as meeting the assignment's accuracy requirement.

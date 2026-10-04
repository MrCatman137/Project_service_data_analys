from __future__ import annotations

import json
import math
import os
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from housing.components import HousingFeatureEngineer
from housing.config import METADATA_PATH, MODEL_PATH

app = Flask(__name__)

pipeline = None
metadata: dict[str, Any] = {}


def load_artifacts() -> None:
    global pipeline, metadata
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}. Run train.py first.")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}. Run train.py first.")
    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def as_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if value is None or value == "":
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def as_text(payload: dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key, default)
    return default if value is None else str(value).strip()


def validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    area = as_float(payload, "Area", 0)
    if area <= 0:
        raise ValueError("Area must be greater than 0.")

    normalized = {
        "Address": as_text(payload, "Address"),
        "Area": area,
        "Frontage": as_float(payload, "Frontage", 0),
        "Access Road": as_float(payload, "Access Road", 0),
        "House direction": as_text(payload, "House direction", "Unknown"),
        "Balcony direction": as_text(payload, "Balcony direction", "Unknown"),
        "Floors": as_float(payload, "Floors", 1),
        "Bedrooms": as_float(payload, "Bedrooms", 0),
        "Bathrooms": as_float(payload, "Bathrooms", 0),
        "Legal status": as_text(payload, "Legal status", "Unknown"),
        "Furniture state": as_text(payload, "Furniture state", "Unknown"),
    }

    for key in ("Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"):
        if normalized[key] < 0:
            raise ValueError(f"{key} cannot be negative.")
    return normalized


def parsed_features(record: dict[str, Any]) -> dict[str, Any]:
    address_features = HousingFeatureEngineer.parse_address(record["Address"])
    bedrooms = record["Bedrooms"]
    bathrooms = record["Bathrooms"]
    area = record["Area"]
    return {
        "project_from_address": address_features["Project"],
        "ward_from_address": address_features["Ward"],
        "district_from_address": address_features["District"],
        "province_from_address": address_features["Province"],
        "address_parts": len(
            [part for part in record["Address"].split(",") if part.strip()]
        ),
        "total_rooms": round(bedrooms + bathrooms, 3),
        "rooms_per_area": round((bedrooms + bathrooms) / area, 6) if area > 0 else 0.0,
        "bed_bath_ratio": round(bedrooms / bathrooms, 6) if bathrooms > 0 else None,
    }


def finite_number(value: Any) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/metadata")
def api_metadata():
    return jsonify({
        "status": "success",
        "categorical_options": metadata.get("categorical_options", {}),
        "top_feature_importances": metadata.get("feature_importance", [])[:10],
        "model": metadata.get("selected_model"),
        "metrics": metadata.get("metrics", {}),
        "dataset": {
            "name": metadata.get("dataset"),
            "rows": metadata.get("rows_after_cleaning"),
        },
    })


@app.post("/predict/price")
def predict_price():
    payload = request.get_json(silent=True)
    record = validate_payload(payload)

    if pipeline is None:
        load_artifacts()

    frame = pd.DataFrame([record])
    prediction = pipeline.predict(frame)
    predicted = finite_number(prediction[0])
    if predicted < 0:
        predicted = 0.0

    interval = 0.06
    min_price = max(0.0, predicted * (1 - interval))
    max_price = predicted * (1 + interval)

    # Confidence is a UI-oriented indicator, not a calibrated statistical probability.
    mape = metadata.get("metrics", {}).get("MAPE_percent")
    confidence = 100.0 - float(mape) if isinstance(mape, (int, float)) else 94.0
    confidence = max(50.0, min(99.0, confidence))

    return jsonify({
        "status": "success",
        "predicted_price": round(predicted, 4),
        "price_range": {
            "min": round(min_price, 4),
            "max": round(max_price, 4),
        },
        "currency": "Billion VND",
        "confidence": round(confidence, 2),
        "features_parsed": parsed_features(record),
    })


@app.errorhandler(HTTPException)
def handle_http_error(error: HTTPException):
    return jsonify({
        "status": "error",
        "error": {"code": error.code, "message": error.description},
    }), error.code


@app.errorhandler(ValueError)
def handle_value_error(error: ValueError):
    return jsonify({
        "status": "error",
        "error": {"code": 400, "message": str(error)},
    }), 400


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    app.logger.exception("Unhandled application error")
    return jsonify({
        "status": "error",
        "error": {"code": 500, "message": "Internal server error."},
    }), 500


try:
    load_artifacts()
except FileNotFoundError as exc:
    # Keep the web process importable so the user can see the actionable error.
    app.logger.warning("%s", exc)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)

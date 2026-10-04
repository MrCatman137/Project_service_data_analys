"""Saved-model holdout diagnostics and error visualization."""
from __future__ import annotations

import json
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from housing.config import METADATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from housing.data import _balanced_group_split, metrics

from .data import MIN_CATEGORY_ROWS, _json_safe, _parse_locations, _sample_frame
from .plots import OUTPUT_DIR


def _model_holdout_diagnostics(
    X: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        print(f"\nNo saved model found at {MODEL_PATH}; skipping prediction residual plots.")
        return None
    test_metadata: dict[str, Any] = {}
    if METADATA_PATH.exists():
        test_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    _, test_idx = _balanced_group_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_test = X.iloc[test_idx]
    y_test = y[test_idx]
    model = joblib.load(MODEL_PATH)
    prediction = np.asarray(model.predict(X_test), dtype=float)
    if len(prediction) != len(y_test) or not np.isfinite(prediction).all():
        raise ValueError("Saved pipeline returned invalid holdout predictions.")

    holdout = pd.DataFrame({
        "actual_price": y_test,
        "predicted_price": prediction,
        "residual": y_test - prediction,
        "APE_percent": np.abs((y_test - prediction) / y_test) * 100,
        "Area": pd.to_numeric(X_test["Area"], errors="coerce").to_numpy(),
    }, index=X_test.index)
    parsed = _parse_locations(X_test)
    holdout["Province"] = parsed["Province"]
    holdout["District"] = parsed["District"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    sampled = _sample_frame(holdout)
    axes[0, 0].scatter(
        sampled["actual_price"], sampled["predicted_price"], s=10, alpha=0.35
    )
    limits = [
        min(sampled["actual_price"].min(), sampled["predicted_price"].min()),
        max(sampled["actual_price"].max(), sampled["predicted_price"].max()),
    ]
    axes[0, 0].plot(limits, limits, "r--")
    axes[0, 0].set(
        title="Actual versus predicted on grouped holdout",
        xlabel="Actual price (billion VND)",
        ylabel="Predicted price (billion VND)",
    )
    axes[0, 1].scatter(
        sampled["predicted_price"], sampled["residual"], s=10, alpha=0.35
    )
    axes[0, 1].axhline(0, color="red", linestyle="--")
    axes[0, 1].set(
        title="Residual versus predicted",
        xlabel="Predicted price (billion VND)",
        ylabel="Actual - predicted",
    )
    axes[1, 0].scatter(
        sampled["Area"], sampled["residual"], s=10, alpha=0.35, color="#478B74"
    )
    axes[1, 0].axhline(0, color="red", linestyle="--")
    axes[1, 0].set(title="Residual versus area", xlabel="Area (sqm)", ylabel="Actual - predicted")
    axes[1, 1].scatter(
        sampled["actual_price"], sampled["residual"], s=10, alpha=0.35, color="#9A7042"
    )
    axes[1, 1].axhline(0, color="red", linestyle="--")
    axes[1, 1].set(
        title="Residual versus actual price",
        xlabel="Actual price (billion VND)",
        ylabel="Actual - predicted",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_holdout_prediction_residuals.png", dpi=150)
    plt.close(fig)

    price_bucket = pd.qcut(holdout["actual_price"], 10, duplicates="drop")
    area_bucket = pd.qcut(holdout["Area"], 10, duplicates="drop")
    error_by_price = holdout.groupby(price_bucket, observed=True)["APE_percent"].mean()
    error_by_area = holdout.groupby(area_bucket, observed=True)["APE_percent"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    error_by_price.plot.bar(ax=axes[0], color="#B76A57")
    axes[0].set(title="Holdout MAPE by actual-price decile", ylabel="MAPE (%)", xlabel="Price bucket")
    error_by_area.plot.bar(ax=axes[1], color="#547F9B")
    axes[1].set(title="Holdout MAPE by area decile", ylabel="MAPE (%)", xlabel="Area bucket")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_holdout_error_by_decile.png", dpi=150)
    plt.close(fig)

    summary = {
        **metrics(y_test, prediction),
        "rows": int(len(y_test)),
        "metadata_selected_model": test_metadata.get("selected_model"),
        "metadata_test_mape_percent": test_metadata.get("metrics", {}).get("MAPE_percent"),
        "metadata_test_group_overlap_count": test_metadata.get("test_group_overlap_count"),
        "by_price_decile": [
            {"bucket": str(index), "MAPE_percent": float(value)}
            for index, value in error_by_price.items()
        ],
        "by_area_decile": [
            {"bucket": str(index), "MAPE_percent": float(value)}
            for index, value in error_by_area.items()
        ],
        "by_province": (
            holdout.assign(Province=parsed["Province"].to_numpy())
            .groupby("Province", observed=True)
            .agg(rows=("APE_percent", "size"), MAPE_percent=("APE_percent", "mean"))
            .query("rows >= @MIN_CATEGORY_ROWS")
            .sort_values("MAPE_percent", ascending=False)
            .head(20)
            .reset_index()
            .to_dict(orient="records")
        ),
        "by_district": (
            holdout.assign(District=parsed["District"].to_numpy())
            .groupby("District", observed=True)
            .agg(rows=("APE_percent", "size"), MAPE_percent=("APE_percent", "mean"))
            .query("rows >= @MIN_CATEGORY_ROWS")
            .sort_values("MAPE_percent", ascending=False)
            .head(20)
            .reset_index()
            .to_dict(orient="records")
        ),
        "worst_20": (
            holdout.assign(
                Province=parsed["Province"].to_numpy(),
                District=parsed["District"].to_numpy(),
                Address=X_test["Address"].to_numpy(),
                Bedrooms=X_test["Bedrooms"].to_numpy(),
                Bathrooms=X_test["Bathrooms"].to_numpy(),
                Floors=X_test["Floors"].to_numpy(),
            )
            .nlargest(20, "APE_percent")
            .reset_index(drop=True)
            .to_dict(orient="records")
        ),
    }
    return _json_safe(summary)
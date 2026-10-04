from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .components import HousingFeatureEngineer
from .config import MIN_GROUP_SAMPLES
from .data import metrics



def _fold_metric_summary(
    y_true: np.ndarray,
    prediction: np.ndarray,
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    fold_scores = [
        metrics(y_true[fold_ids == fold], prediction[fold_ids == fold])
        for fold in np.unique(fold_ids)
    ]
    mapes = [score["MAPE_percent"] for score in fold_scores]
    return {
        "mean_MAPE_percent": float(np.mean(mapes)),
        "std_MAPE_percent": float(np.std(mapes, ddof=1)) if len(mapes) > 1 else 0.0,
        "median_fold_MAPE_percent": float(np.median(mapes)),
        "best_fold_MAPE_percent": float(np.min(mapes)),
        "worst_fold_MAPE_percent": float(np.max(mapes)),
        "mean_RMSE": float(np.mean([score["RMSE"] for score in fold_scores])),
        "mean_MAE": float(np.mean([score["MAE"] for score in fold_scores])),
        "mean_R2": float(np.mean([score["R2"] for score in fold_scores])),
        "fold_metrics": fold_scores,
    }


def _residual_diagnostics(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    parsed = pd.DataFrame(
        X_test["Address"].map(HousingFeatureEngineer.parse_address).tolist(),
        index=X_test.index,
    )
    rows = pd.DataFrame({
        "actual_price": y_test,
        "predicted_price": prediction,
        "residual_actual_minus_predicted": y_test - prediction,
        "APE_percent": np.abs((y_test - prediction) / y_test) * 100.0,
        "Area": pd.to_numeric(X_test["Area"], errors="coerce").to_numpy(),
        "Bedrooms": X_test["Bedrooms"].to_numpy(),
        "Bathrooms": X_test["Bathrooms"].to_numpy(),
        "Floors": X_test["Floors"].to_numpy(),
        "Province": parsed["Province"].to_numpy(),
        "District": parsed["District"].to_numpy(),
        "Ward": parsed["Ward"].to_numpy(),
        "Address": X_test["Address"].to_numpy(),
    }, index=X_test.index)

    def summarize_group(column: str, minimum: int = 1) -> list[dict[str, Any]]:
        grouped = rows.groupby(column, dropna=False, observed=True).agg(
            count=("APE_percent", "size"),
            MAPE_percent=("APE_percent", "mean"),
            median_APE_percent=("APE_percent", "median"),
            mean_residual=("residual_actual_minus_predicted", "mean"),
        )
        grouped = grouped[grouped["count"] >= minimum]
        return [
            {
                "group": str(index),
                "count": int(row["count"]),
                "MAPE_percent": float(row["MAPE_percent"]),
                "median_APE_percent": float(row["median_APE_percent"]),
                "mean_residual": float(row["mean_residual"]),
            }
            for index, row in grouped.sort_values(
                "MAPE_percent", ascending=False
            ).iterrows()
        ]

    worst_rows = rows.nlargest(20, "APE_percent").astype(object)
    worst_rows = worst_rows.where(pd.notna(worst_rows), None)
    diagnostics = {
        "by_price_decile": summarize_group(
            pd.qcut(rows["actual_price"], 10, duplicates="drop")
        ),
        "by_area_decile": summarize_group(
            pd.qcut(rows["Area"], 10, duplicates="drop")
        ),
        "by_province": summarize_group("Province", MIN_GROUP_SAMPLES),
        "by_district": summarize_group("District", MIN_GROUP_SAMPLES),
        "actual_vs_predicted_correlation": float(
            np.corrcoef(y_test, prediction)[0, 1]
        ),
        "residual_vs_predicted_correlation": float(
            np.corrcoef(y_test - prediction, prediction)[0, 1]
        ),
        "residual_vs_area_correlation": float(
            np.corrcoef(y_test - prediction, rows["Area"])[0, 1]
        ),
        "residual_vs_price_correlation": float(
            np.corrcoef(y_test - prediction, y_test)[0, 1]
        ),
        "worst_20": worst_rows.reset_index(drop=True).to_dict(orient="records"),
    }
    return diagnostics


def _print_error_tables(diagnostics: dict[str, Any]) -> None:
    print("\nWorst 20 held-out predictions by absolute percentage error:")
    worst = pd.DataFrame(diagnostics["worst_20"])
    columns = [
        "actual_price", "predicted_price", "APE_percent", "Area", "Province",
        "District", "Bedrooms", "Bathrooms", "Floors", "Address",
    ]
    print(worst[columns].to_string(index=False, max_colwidth=60))
    for key in ("by_price_decile", "by_area_decile", "by_province", "by_district"):
        table = pd.DataFrame(diagnostics[key])
        print(f"\n{key}:")
        print(table.head(20).to_string(index=False))

"""EDA orchestration, terminal summaries, and report export."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from housing.config import DATA_PATH, RAW_FEATURES
from housing.data import clean_data

from .data import MIN_CATEGORY_ROWS, _json_safe, _number_summary, _parse_locations
from .holdout import _model_holdout_diagnostics
from .plots import (
    _save_correlation_plot, _save_location_plot, _save_missingness,
    _save_price_area_plot, _save_target_distributions,
)
from .settings import OUTPUT_DIR


def _print_section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    """Analyze source data, print diagnostics, and generate review plots."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    clean = clean_data(raw)
    parsed = _parse_locations(clean)
    numeric_columns = [
        column for column in (
            "Price", "Area", "Frontage", "Access Road", "Floors",
            "Bedrooms", "Bathrooms",
        )
    ]
    numeric = clean[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    source_correlations = _save_correlation_plot(clean, parsed)
    missing = _save_missingness(clean)
    cardinality = _save_location_plot(parsed)
    _save_target_distributions(clean)
    _save_price_area_plot(clean)

    _print_section("Dataset and cleaning")
    print(f"Source: {DATA_PATH}")
    print(f"Rows before cleaning: {len(raw):,}")
    print(f"Rows after cleaning:  {len(clean):,}")
    print(f"Rows removed:         {len(raw) - len(clean):,}")
    print("Only invalid/missing/non-positive Price and Area rows are excluded; no statistical tails are trimmed.")

    _print_section("Numeric summary")
    print(numeric.describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(3).to_string())
    numeric["Price_per_sqm"] = numeric["Price"] / numeric["Area"].where(numeric["Area"] > 0)
    print("\nPrice per square meter (billion VND/sqm):")
    print(numeric["Price_per_sqm"].describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).round(4).to_string())
    print("Note: Price per sqm is derived from the target and is descriptive only, not an input-feature correlation.")

    _print_section("Missingness")
    print(missing.sort_values(ascending=False).to_string(float_format=lambda value: f"{value:.1f}%"))

    _print_section("Address coverage")
    print(cardinality.to_string())
    for column in (
        "Province", "District", "Ward", "Project", "Street",
        "Address_Head", "Address_Head_Type",
    ):
        unknown = int((parsed[column] == "Unknown").sum())
        print(
            f"{column}: {parsed[column].nunique():,} unique; "
            f"{unknown:,} ({unknown / len(clean) * 100:.1f}%) unknown"
        )
        top = parsed[column].value_counts().head(8)
        print("  Most common: " + "; ".join(f"{key}={count:,}" for key, count in top.items()))

    _print_section("Spearman correlations with Price")
    print(source_correlations.head(15).to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nHigh correlation is association, not proof of causality; nonlinear location effects may not appear here.")

    _print_section("Category-level price spread (at least 30 listings)")
    for column in ("Legal status", "Furniture state", "House direction", "Balcony direction"):
        grouped = clean.assign(
            _category=clean[column].fillna("Missing").astype(str)
        ).groupby("_category", observed=True)["Price"].agg(["count", "median", "mean"])
        grouped = grouped[grouped["count"] >= MIN_CATEGORY_ROWS]
        print(f"\n{column}:")
        if grouped.empty:
            print("  No category meets the minimum sample count.")
        else:
            print(grouped.sort_values("median", ascending=False).head(12).round(3).to_string())

    model_summary = _model_holdout_diagnostics(
        clean[RAW_FEATURES], clean["Price"].to_numpy(dtype=float)
    )
    if model_summary is not None:
        _print_section("Saved model: address-grouped holdout")
        print(json.dumps({
            key: value for key, value in model_summary.items()
            if key not in (
                "worst_20", "by_price_decile", "by_area_decile",
                "by_province", "by_district",
            )
        }, ensure_ascii=False, indent=2))
        print("\nWorst 10 holdout errors:")
        worst = pd.DataFrame(model_summary["worst_20"])
        display_columns = [
            "actual_price", "predicted_price", "APE_percent", "Area",
            "Province", "District", "Bedrooms", "Bathrooms", "Floors", "Address",
        ]
        print(worst[display_columns].head(10).to_string(index=False, max_colwidth=55))
        print("\nHighest-error sufficiently sampled provinces:")
        print(pd.DataFrame(model_summary["by_province"]).head(12).to_string(index=False))
        print("\nHighest-error sufficiently sampled districts:")
        print(pd.DataFrame(model_summary["by_district"]).head(12).to_string(index=False))

    findings: list[str] = []
    for column, rate in missing.items():
        if rate >= 50:
            findings.append(
                f"{column} is missing in {rate:.1f}% of records, limiting its usefulness."
            )
    price_per_area = numeric["Price_per_sqm"].dropna()
    if len(price_per_area) and price_per_area.quantile(0.9) / max(
        price_per_area.quantile(0.1), 1e-9
    ) >= 3:
        findings.append(
            "Price per sqm varies by at least 3x between its 10th and 90th percentiles; "
            "area alone cannot explain value."
        )
    if parsed["Project"].nunique() / max(len(clean), 1) > 0.1:
        findings.append(
            "Project cardinality is high relative to row count, so many projects have limited examples."
        )
    street_share = float((parsed["Address_Head_Type"] == "street").mean())
    project_share = float(
        (parsed["Address_Head_Type"] == "named_project").mean()
    )
    findings.append(
        f"Address parsing classifies {street_share:.1%} of address heads as streets "
        f"and {project_share:.1%} as explicitly named projects; model them separately."
    )
    if model_summary and model_summary["MAPE_percent"] > 7:
        findings.append(
            f"The saved model's grouped-holdout MAPE is {model_summary['MAPE_percent']:.2f}%, "
            "so the 5-7% goal is not demonstrated by this dataset/model."
        )
    _print_section("Potential reasons for the accuracy ceiling")
    if findings:
        for finding in findings:
            print(f"- {finding}")
    else:
        print("No single obvious data-quality cause identified; inspect the error plots and category slices.")

    summary = {
        "source": str(DATA_PATH),
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(clean)),
        "rows_removed": int(len(raw) - len(clean)),
        "numeric_summary": {
            column: _number_summary(numeric[column]) for column in numeric_columns
        },
        "price_per_sqm": _number_summary(numeric["Price_per_sqm"]),
        "missing_percent": {str(key): float(value) for key, value in missing.items()},
        "location_cardinality": _json_safe(cardinality.to_dict(orient="index")),
        "address_head_type_counts": _json_safe(
            parsed["Address_Head_Type"].value_counts().to_dict()
        ),
        "spearman_correlation_with_price": _json_safe(
            source_correlations["spearman_correlation"].to_dict()
        ),
        "grouped_holdout_metrics": model_summary,
        "findings": findings,
        "plots": sorted(path.name for path in OUTPUT_DIR.glob("*.png")),
    }
    summary_path = OUTPUT_DIR / "eda_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {len(summary['plots'])} plots and summary:")
    for plot in summary["plots"]:
        print(f"  {OUTPUT_DIR / plot}")
    print(f"  {summary_path}")

if __name__ == "__main__":
    main()

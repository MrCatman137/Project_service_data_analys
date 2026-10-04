"""Plot generation for descriptive and model diagnostics."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from housing.config import RAW_FEATURES

from .data import _sample_frame
from .settings import OUTPUT_DIR


def _save_target_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].hist(df["Price"], bins=45, color="#3572A5", edgecolor="white")
    axes[0, 0].set(title="Price distribution", xlabel="Price (billion VND)", ylabel="Listings")
    axes[0, 1].hist(np.log1p(df["Price"]), bins=45, color="#5B9A8B", edgecolor="white")
    axes[0, 1].set(title="Log price distribution", xlabel="log1p(Price)", ylabel="Listings")
    axes[1, 0].hist(df["Area"], bins=45, color="#D89043", edgecolor="white")
    axes[1, 0].set(title="Land area distribution", xlabel="Area (sqm)", ylabel="Listings")
    axes[1, 1].hist(
        np.log1p(df["Area"]), bins=45, color="#9A79B8", edgecolor="white"
    )
    axes[1, 1].set(title="Log area distribution", xlabel="log1p(Area)", ylabel="Listings")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_target_and_area_distributions.png", dpi=150)
    plt.close(fig)


def _save_missingness(df: pd.DataFrame) -> pd.Series:
    missing = df[RAW_FEATURES + ["Price"]].isna().mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(11, 6))
    missing.plot.barh(ax=ax, color="#C66B5D")
    ax.set(title="Missing values by source field", xlabel="Missing rows (%)", ylabel="")
    ax.set_xlim(0, max(5.0, float(missing.max()) * 1.1))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_missing_values.png", dpi=150)
    plt.close(fig)
    return missing


def _save_correlation_plot(df: pd.DataFrame, parsed: pd.DataFrame) -> pd.DataFrame:
    numeric = df[["Price", "Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]].copy()
    numeric["Address_components"] = parsed["Address_Component_Count"]
    numeric["Address_length"] = parsed["Address_Length"]
    corr = numeric.corr(method="spearman", numeric_only=True)
    target_corr = corr["Price"].drop("Price").sort_values(key=np.abs, ascending=False)

    display_columns = ["Price", *target_corr.head(12).index.tolist()]
    display_corr = corr.loc[display_columns, display_columns]
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(display_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(display_columns)), display_columns, rotation=45, ha="right")
    ax.set_yticks(range(len(display_columns)), display_columns)
    ax.set_title("Spearman correlation among numeric fields")
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_spearman_correlations.png", dpi=150)
    plt.close(fig)
    return target_corr.to_frame("spearman_correlation")


def _save_price_area_plot(df: pd.DataFrame) -> None:
    sampled = _sample_frame(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    points = axes[0].scatter(
        sampled["Area"], sampled["Price"], c=sampled["Price"] / sampled["Area"],
        cmap="viridis", alpha=0.35, s=10,
    )
    axes[0].set(
        title="Price versus area",
        xlabel="Area (sqm)",
        ylabel="Price (billion VND)",
    )
    fig.colorbar(points, ax=axes[0], label="Billion VND per sqm")
    axes[1].scatter(
        sampled["Area"], sampled["Price"] / sampled["Area"],
        alpha=0.3, s=10, color="#A84A4A",
    )
    axes[1].set(
        title="Unit price varies by listing",
        xlabel="Area (sqm)",
        ylabel="Price per sqm (billion VND)",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_price_area_relationship.png", dpi=150)
    plt.close(fig)


def _save_location_plot(parsed: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "Province": parsed["Province"],
        "District": parsed["District"],
        "Ward": parsed["Ward"],
        "Project": parsed["Project"],
        "Street": parsed["Street"],
        "Address_Head": parsed["Address_Head"],
        "Address_Head_Type": parsed["Address_Head_Type"],
    })
    cardinality = pd.DataFrame({
        "unique_values": summary.nunique(),
        "unknown_rows": {
            column: int((summary[column] == "Unknown").sum())
            for column in summary
        },
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    top_provinces = summary["Province"].value_counts().head(15).sort_values()
    top_provinces.plot.barh(ax=axes[0], color="#438A8A")
    axes[0].set(title="Most frequent parsed provinces", xlabel="Listings", ylabel="")
    levels = cardinality["unique_values"]
    levels.plot.bar(ax=axes[1], color="#9772A8")
    axes[1].set(title="Parsed location cardinality", ylabel="Unique values", xlabel="")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_location_coverage.png", dpi=150)
    plt.close(fig)
    return cardinality
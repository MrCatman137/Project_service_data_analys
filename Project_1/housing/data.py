from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .components import GroupedHierarchicalEncoder, HousingFeatureEngineer, InputNormalizer
from .config import (
    CATEGORICAL_FEATURES, LOCATION_FEATURES, NUMERIC_FEATURES, RAW_FEATURES,
)



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean invalid target/area rows while retaining valid distribution tails."""
    required = set(RAW_FEATURES + ["Price"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    out = df.copy()
    for column in ("Price", "Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["Price", "Area"])
    return out[(out["Price"] > 0) & (out["Area"] > 0)].copy()


def distribution_summary(values: pd.Series) -> dict[str, float]:
    """Return compact quantile metadata for a numeric input distribution."""
    quantiles = values.quantile([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p01", "p10", "median", "p90", "p99", "max"),
            quantiles.to_numpy(),
        )
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate price-scale regression and business-threshold metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs((y_true - y_pred) / y_true) * 100.0
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_percent": float(np.mean(ape)),
        "Median_APE_percent": float(np.median(ape)),
        "within_5_percent": float(np.mean(ape <= 5.0) * 100.0),
        "within_7_percent": float(np.mean(ape <= 7.0) * 100.0),
        "within_10_percent": float(np.mean(ape <= 10.0) * 100.0),
    }


def build_preprocessor(include_directions: bool = True) -> Any:
    """Build fold-local imputers, grouped target encoders, and compact OHE."""
    from sklearn.compose import ColumnTransformer

    numeric = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
                add_indicator=True,
            ),
        ),
    ])
    location_columns = [*LOCATION_FEATURES, "Address"]
    location = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown",
                keep_empty_features=True,
            ).set_output(transform="pandas"),
        ),
        ("target_encoder", GroupedHierarchicalEncoder(smoothing=20.0, n_splits=5)),
    ])
    categorical_features = (
        CATEGORICAL_FEATURES
        if include_directions
        else ["Legal status", "Furniture state"]
    )
    categorical = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown",
                keep_empty_features=True,
            ),
        ),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("location", location, location_columns),
            ("cat", categorical, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(
    model: Any,
    preprocessor: Any,
    input_normalizer: InputNormalizer,
    feature_engineer: HousingFeatureEngineer,
) -> Pipeline:
    """Build the serving pipeline, accepting raw DataFrames or dict records."""
    return Pipeline([
        ("input", input_normalizer),
        ("features", feature_engineer),
        ("preprocess", preprocessor),
        ("model", model),
    ])


def transformed_feature_names(preprocessor: Any) -> list[str]:
    return preprocessor.get_feature_names_out().tolist()


def _price_bin_proportions(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges[1:-1], y, side="right")
    return np.bincount(bins, minlength=len(edges) - 1) / len(y)


def _balanced_group_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a deterministic address-disjoint split with similar price bins."""
    if X["Address"].nunique() < 2:
        raise ValueError("At least two distinct addresses are required for validation.")
    edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, 11)))
    if len(edges) < 3:
        edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, 6)))
    full_distribution = _price_bin_proportions(y, edges)
    splitter = GroupShuffleSplit(
        n_splits=30,
        test_size=test_size,
        random_state=random_state,
    )
    groups = X["Address"].map(HousingFeatureEngineer._clean_group_key).to_numpy()
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for train_idx, heldout_idx in splitter.split(X, y, groups):
        heldout_distribution = _price_bin_proportions(y[heldout_idx], edges)
        score = float(np.mean(np.abs(heldout_distribution - full_distribution)))
        score += 0.1 * abs(len(heldout_idx) / len(y) - test_size)
        if best is None or score < best[0]:
            best = score, train_idx, heldout_idx
    if best is None:
        raise RuntimeError("Unable to create an address-grouped data split.")
    return best[1], best[2]


def _prepare_features(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    include_directions: bool = True,
) -> tuple[InputNormalizer, HousingFeatureEngineer, Any, np.ndarray]:
    """Fit preprocessing on training rows only, including the target encoder."""
    normalizer = InputNormalizer().fit(X_fit)
    normalized = normalizer.transform(X_fit)
    engineer = HousingFeatureEngineer().fit(normalized, y_fit)
    engineered = engineer.transform(normalized)
    preprocessor = build_preprocessor(include_directions)
    transformed = preprocessor.fit_transform(engineered, y_fit)
    return normalizer, engineer, preprocessor, transformed


def _transform_features(
    X: pd.DataFrame,
    normalizer: InputNormalizer,
    engineer: HousingFeatureEngineer,
    preprocessor: Any,
) -> np.ndarray:
    normalized = normalizer.transform(X)
    return preprocessor.transform(engineer.transform(normalized))

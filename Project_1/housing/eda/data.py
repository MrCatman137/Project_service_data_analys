"""Shared EDA data helpers and serialization utilities."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from housing.components import HousingFeatureEngineer
from housing.config import RANDOM_STATE

MAX_SCATTER_POINTS = 5000
MIN_CATEGORY_ROWS = 30


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _number_summary(series: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if numeric.empty:
        return {"count": 0, "mean": None, "std": None, "min": None, "median": None, "max": None}
    return {
        "count": int(numeric.size),
        "mean": float(numeric.mean()),
        "std": float(numeric.std()),
        "min": float(numeric.min()),
        "median": float(numeric.median()),
        "max": float(numeric.max()),
    }


def _parse_locations(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["Address"].map(HousingFeatureEngineer.parse_address).apply(pd.Series)
    return parsed


def _sample_frame(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= MAX_SCATTER_POINTS:
        return df
    return df.sample(MAX_SCATTER_POINTS, random_state=RANDOM_STATE)
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from .components import PrefitWeightedRegressor, PriceTargetRegressor
from .config import RANDOM_STATE, XGB_SEARCH_CANDIDATES
from .data import metrics, transformed_feature_names


@dataclass(frozen=True)
class CandidateSpec:
    """Reproducible model family, target strategy, and search parameters."""

    name: str
    family: str
    target_strategy: str
    weighting: str
    target_clip_quantiles: tuple[float, float] | None
    parameters: dict[str, Any]
    include_directions: bool = True


def _random_xgb_candidates() -> list[CandidateSpec]:
    """Sample a bounded, deterministic hyperparameter search (not a grid)."""
    rng = np.random.default_rng(RANDOM_STATE)
    choices = {
        "n_estimators": [180, 240, 320],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08],
        "min_child_weight": [1, 3, 5, 10, 20],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "gamma": [0.0, 0.05, 0.1, 0.2],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    }
    strategies = ("raw", "log1p", "sqrt")
    weightings = ("none", "inverse_sqrt_price")
    candidates: list[CandidateSpec] = []
    seen: set[tuple[Any, ...]] = set()
    while len(candidates) < XGB_SEARCH_CANDIDATES:
        params = {
            name: values[int(rng.integers(0, len(values)))]
            for name, values in choices.items()
        }
        key = tuple(params.items())
        if key in seen:
            continue
        seen.add(key)
        index = len(candidates)
        strategy = strategies[index % len(strategies)]
        weighting = weightings[(index // len(strategies)) % len(weightings)]
        clip = (0.005, 0.995)
        name = f"xgb_{index + 1:02d}_{strategy}_{weighting}"
        candidates.append(CandidateSpec(
            name=name,
            family="xgboost",
            target_strategy=strategy,
            weighting=weighting,
            target_clip_quantiles=clip,
            parameters=params,
        ))
    return candidates


def _candidate_specs() -> list[CandidateSpec]:
    specs = _random_xgb_candidates()
    baseline_xgb_parameters = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    for target_strategy in ("raw", "log1p", "sqrt"):
        specs.append(CandidateSpec(
            f"xgb_baseline_{target_strategy}",
            "xgboost",
            target_strategy,
            "none",
            None,
            baseline_xgb_parameters.copy(),
        ))
    for weighting in ("inverse_sqrt_price",):
        specs.append(CandidateSpec(
            f"xgb_log_{weighting}",
            "xgboost",
            "log1p",
            weighting,
            None,
            baseline_xgb_parameters.copy(),
        ))
    specs.append(CandidateSpec(
        "xgb_log_inverse_price",
        "xgboost",
        "log1p",
        "inverse_price",
        None,
        baseline_xgb_parameters.copy(),
    ))
    specs.append(CandidateSpec(
        "xgb_log_no_directions",
        "xgboost",
        "log1p",
        "none",
        None,
        baseline_xgb_parameters.copy(),
        include_directions=False,
    ))
    specs.append(
        CandidateSpec(
            "extra_trees_log", "extra_trees", "log1p", "none", None,
            {"n_estimators": 120, "max_depth": 20, "min_samples_leaf": 2, "max_features": 0.8},
        )
    )
    return specs


def _new_estimator(spec: CandidateSpec) -> PriceTargetRegressor:
    """Create one candidate estimator with deterministic family parameters."""
    if spec.family == "xgboost":
        estimator = XGBRegressor(
            **spec.parameters,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            tree_method="hist",
            objective="reg:squarederror",
        )
    elif spec.family == "extra_trees":
        estimator = ExtraTreesRegressor(
            **spec.parameters,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unsupported model family: {spec.family}")
    return PriceTargetRegressor(
        estimator=estimator,
        target_strategy=spec.target_strategy,
        weighting=spec.weighting,
        target_clip_quantiles=spec.target_clip_quantiles,
    )


def _candidate_params(spec: CandidateSpec) -> dict[str, Any]:
    return {
        "family": spec.family,
        "target_strategy": spec.target_strategy,
        "weighting": spec.weighting,
        "target_clip_quantiles": spec.target_clip_quantiles,
        "include_directions": spec.include_directions,
        **spec.parameters,
    }


def _importance_group(feature_name: str) -> str:
    if "__" not in feature_name:
        return feature_name
    transformer, output = feature_name.split("__", 1)
    if transformer == "location":
        return output
    if transformer == "cat":
        return output.split("_", 1)[0]
    if output.startswith("missingindicator_"):
        output = output.removeprefix("missingindicator_")
    for prefix in (
        "Province_", "District_", "Ward_", "Project_", "District_Ward_",
    ):
        if output.startswith(prefix):
            return prefix.rstrip("_")
    return output


def extract_feature_importance(
    preprocessor: Any,
    model: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return original transformed importance plus aggregation by feature family."""
    estimators = (
        [estimator for _, estimator in model.estimators]
        if isinstance(model, PrefitWeightedRegressor)
        else [model]
    )
    importances = [
        getattr(estimator.estimator_, "feature_importances_", None)
        for estimator in estimators
    ]
    importances = [values for values in importances if values is not None]
    if not importances:
        return [], []
    values = np.mean(np.vstack(importances), axis=0)
    names = transformed_feature_names(preprocessor)
    if len(values) != len(names):
        return [], []
    grouped: dict[str, float] = {}
    details: list[dict[str, Any]] = []
    for name, importance in zip(names, values):
        importance_value = float(importance)
        grouped_name = _importance_group(name)
        grouped[grouped_name] = grouped.get(grouped_name, 0.0) + importance_value
        details.append({"feature": name, "importance": importance_value})
    total = sum(grouped.values())
    if total > 0:
        for item in details:
            item["importance"] /= total
    aggregated = [
        {"feature": name, "importance": value / total if total else 0.0}
        for name, value in grouped.items()
    ]
    return (
        sorted(aggregated, key=lambda item: item["importance"], reverse=True),
        sorted(details, key=lambda item: item["importance"], reverse=True),
    )


def _blend_weights(
    candidate_names: list[str],
    prediction_matrix: np.ndarray,
    y: np.ndarray,
) -> tuple[list[str], list[float], np.ndarray]:
    """Find a coarse non-negative blend using development OOF predictions only."""
    top_names = candidate_names[:3]
    if len(top_names) < 2:
        return top_names, [1.0], prediction_matrix[:, 0]
    top_predictions = prediction_matrix[:, : len(top_names)]
    best_score = float("inf")
    best_weights: tuple[float, ...] = (1.0,) + (0.0,) * (len(top_names) - 1)
    for values in itertools.product(np.linspace(0.0, 1.0, 11), repeat=len(top_names)):
        if not np.isclose(sum(values), 1.0):
            continue
        blended = top_predictions @ np.asarray(values)
        score = metrics(y, blended)["MAPE_percent"]
        if score < best_score:
            best_score = score
            best_weights = tuple(float(value) for value in values)
    return top_names, list(best_weights), top_predictions @ np.asarray(best_weights)

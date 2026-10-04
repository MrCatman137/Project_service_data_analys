from __future__ import annotations

import json
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from .components import (
    HousingFeatureEngineer, InputNormalizer, PrefitWeightedRegressor,
)
from .config import (
    CV_FOLDS, DATA_PATH, METADATA_PATH, MODEL_DIR, MODEL_PATH,
    RAW_FEATURES, RANDOM_STATE, TEST_SIZE,
)
from .data import (
    _balanced_group_split, _prepare_features, _transform_features,
    build_pipeline, clean_data, distribution_summary, metrics,
    transformed_feature_names,
)
from .evaluation import (
    _fold_metric_summary, _print_error_tables, _residual_diagnostics,
)
from .models import (
    CandidateSpec, _blend_weights, _candidate_params, _candidate_specs,
    _new_estimator, extract_feature_importance,
)



def _cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    specs: list[CandidateSpec],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, int],
]:
    """Evaluate candidates with leakage-safe address-grouped out-of-fold fits."""
    groups = X["Address"].map(HousingFeatureEngineer._clean_group_key).to_numpy()
    group_kfold = GroupKFold(n_splits=min(CV_FOLDS, len(np.unique(groups))))
    oof_predictions = {
        spec.name: np.full(len(X), np.nan, dtype=float) for spec in specs
    }
    fold_ids = np.full(len(X), -1, dtype=int)
    fold_rows: dict[str, int] = {}
    splits = list(group_kfold.split(X, y, groups))

    for fold_number, (train_idx, valid_idx) in enumerate(splits):
        print(
            f"\nGrouped CV fold {fold_number + 1}/{len(splits)}: "
            f"{len(train_idx)} training rows, {len(valid_idx)} validation rows"
        )
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_valid = X.iloc[valid_idx]
        train_groups = set(groups[train_idx])
        valid_groups = set(groups[valid_idx])
        if train_groups & valid_groups:
            raise RuntimeError("Address leakage detected inside a grouped CV fold.")
        fold_ids[valid_idx] = fold_number
        fold_rows[str(fold_number + 1)] = len(valid_idx)
        feature_sets: dict[
            bool,
            tuple[
                InputNormalizer,
                HousingFeatureEngineer,
                Any,
                np.ndarray,
                np.ndarray,
            ],
        ] = {}

        for candidate_index, spec in enumerate(specs):
            if spec.include_directions not in feature_sets:
                normalizer, engineer, preprocessor, train_features = _prepare_features(
                    X_train,
                    y_train,
                    spec.include_directions,
                )
                valid_features = _transform_features(
                    X_valid, normalizer, engineer, preprocessor
                )
                feature_sets[spec.include_directions] = (
                    normalizer,
                    engineer,
                    preprocessor,
                    train_features,
                    valid_features,
                )
            _, _, _, train_features, valid_features = feature_sets[
                spec.include_directions
            ]
            estimator = _new_estimator(spec)
            estimator.fit(train_features, y_train)
            prediction = estimator.predict(valid_features)
            oof_predictions[spec.name][valid_idx] = prediction
            if candidate_index < 2 or (candidate_index + 1) % 5 == 0:
                print(
                    f"  fitted candidate {candidate_index + 1}/{len(specs)} "
                    f"({spec.name})"
                )

    results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        prediction = oof_predictions[spec.name]
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Candidate {spec.name} has incomplete OOF predictions.")
        results[spec.name] = {
            **_fold_metric_summary(y, prediction, fold_ids),
            "parameters": _candidate_params(spec),
        }
    return results, oof_predictions, fold_ids, fold_rows


def _fit_final_pipeline(
    specs_by_name: dict[str, CandidateSpec],
    selected_names: list[str],
    weights: list[float],
    X_development: pd.DataFrame,
    y_development: np.ndarray,
) -> tuple[Pipeline, Any, Any]:
    if len({specs_by_name[name].include_directions for name in selected_names}) > 1:
        raise ValueError("Ensemble candidates must use the same feature schema.")
    normalizer, engineer, preprocessor, train_features = _prepare_features(
        X_development,
        y_development,
        specs_by_name[selected_names[0]].include_directions,
    )
    fitted_estimators: list[tuple[str, Any]] = []
    for name in selected_names:
        estimator = _new_estimator(specs_by_name[name])
        print(f"Refitting selected model {name} on all development rows...")
        estimator.fit(train_features, y_development)
        fitted_estimators.append((name, estimator))
    model: Any
    if len(fitted_estimators) == 1 or (
        len(weights) == 1 and weights[0] == 1.0
    ):
        model = fitted_estimators[0][1]
    else:
        model = PrefitWeightedRegressor(fitted_estimators, weights)
    pipeline = build_pipeline(model, preprocessor, normalizer, engineer)
    return pipeline, preprocessor, model


def main() -> None:
    """Train, compare, evaluate once on grouped holdout, and persist artifacts."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Place vietnam_housing_dataset.csv there "
            "or set HOUSING_DATA to its path."
        )
    raw = pd.read_csv(DATA_PATH)
    df = clean_data(raw)
    if len(df) < 100:
        raise ValueError("At least 100 valid housing records are required for training.")
    X = df[RAW_FEATURES].copy()
    y = df["Price"].astype(float).to_numpy()

    development_idx, test_idx = _balanced_group_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_development, X_test = X.iloc[development_idx], X.iloc[test_idx]
    y_development, y_test = y[development_idx], y[test_idx]
    development_groups = X_development["Address"].map(
        HousingFeatureEngineer._clean_group_key
    )
    test_groups = X_test["Address"].map(HousingFeatureEngineer._clean_group_key)
    overlap = set(development_groups) & set(test_groups)
    if overlap:
        raise RuntimeError("Address leakage detected between development and test sets.")

    specs = _candidate_specs()
    print(
        f"Cleaned {len(df):,}/{len(raw):,} rows. "
        f"Grouped test rows: {len(X_test):,}; CV candidates: {len(specs)}."
    )
    cv_results, oof_predictions, fold_ids, fold_rows = _cross_validate(
        X_development, y_development, specs
    )
    spec_lookup = {spec.name: spec for spec in specs}
    ordered_names = sorted(
        cv_results,
        key=lambda name: cv_results[name]["mean_MAPE_percent"],
    )
    single_best_name = ordered_names[0]
    top_feature_set = spec_lookup[ordered_names[0]].include_directions
    top_names = [
        name for name in ordered_names
        if spec_lookup[name].include_directions == top_feature_set
    ][:3]
    oof_matrix = np.column_stack([oof_predictions[name] for name in top_names])
    blend_names, blend_weights, blend_oof = _blend_weights(
        top_names, oof_matrix, y_development
    )
    blend_cv = _fold_metric_summary(y_development, blend_oof, fold_ids)
    best_single_cv = cv_results[single_best_name]["mean_MAPE_percent"]
    best_single_r2 = cv_results[single_best_name]["mean_R2"]
    if (
        blend_cv["mean_MAPE_percent"] < best_single_cv
        and blend_cv["mean_R2"] >= best_single_r2
    ):
        selected_names = blend_names
        selected_weights = blend_weights
        selected_name = "oof_blend"
        selected_cv = blend_cv
    else:
        selected_names = [single_best_name]
        selected_weights = [1.0]
        selected_name = single_best_name
        selected_cv = cv_results[single_best_name]

    pipeline, preprocessor, fitted_model = _fit_final_pipeline(
        spec_lookup,
        selected_names,
        selected_weights,
        X_development,
        y_development,
    )
    test_prediction = pipeline.predict(X_test)
    test_metrics = metrics(y_test, test_prediction)
    residuals = _residual_diagnostics(X_test, y_test, test_prediction)

    old_test_metrics = {
        "MAPE_percent": 18.753,
        "R2": 0.6285,
        "MAE": 0.9952,
        "within_7_percent": 26.44,
    }
    print("\nModel comparison (development grouped CV and one untouched test evaluation):")
    comparison_rows = []
    for name in ordered_names:
        result = cv_results[name]
        comparison_rows.append({
            "model": name,
            "CV MAPE": result["mean_MAPE_percent"],
            "CV std": result["std_MAPE_percent"],
            "TEST MAPE": "",
            "RMSE": "",
            "MAE": "",
            "R2": "",
        })
    comparison_rows.append({
        "model": selected_name,
        "CV MAPE": selected_cv["mean_MAPE_percent"],
        "CV std": selected_cv["std_MAPE_percent"],
        "TEST MAPE": test_metrics["MAPE_percent"],
        "RMSE": test_metrics["RMSE"],
        "MAE": test_metrics["MAE"],
        "R2": test_metrics["R2"],
    })
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    print(f"\nSelected model: {selected_name} ({', '.join(selected_names)})")
    print(f"MAPE: {test_metrics['MAPE_percent']:.3f}%")
    print(f"Median APE: {test_metrics['Median_APE_percent']:.3f}%")
    print(f"Within ±5%: {test_metrics['within_5_percent']:.2f}%")
    print(f"Within ±7%: {test_metrics['within_7_percent']:.2f}%")
    print(f"Within ±10%: {test_metrics['within_10_percent']:.2f}%")
    print(
        "MAPE TARGET 5–7%: "
        + ("PASS" if 5.0 <= test_metrics["MAPE_percent"] <= 7.0 else "NOT MET")
    )
    print("\nPrevious test baseline:")
    print(
        f"MAPE={old_test_metrics['MAPE_percent']:.3f}%, "
        f"R2={old_test_metrics['R2']:.3f}, "
        f"MAE={old_test_metrics['MAE']:.4f}, "
        f"within ±7%={old_test_metrics['within_7_percent']:.2f}%"
    )
    print(
        f"OLD TEST MAPE = {old_test_metrics['MAPE_percent']:.3f}%\n"
        f"NEW TEST MAPE = {test_metrics['MAPE_percent']:.3f}%\n"
        f"OLD TEST R² = {old_test_metrics['R2']:.3f}\n"
        f"NEW TEST R² = {test_metrics['R2']:.3f}\n"
        f"OLD TEST MAE = {old_test_metrics['MAE']:.4f}\n"
        f"NEW TEST MAE = {test_metrics['MAE']:.4f}\n"
        f"OLD WITHIN ±7% = {old_test_metrics['within_7_percent']:.2f}%\n"
        f"NEW WITHIN ±7% = {test_metrics['within_7_percent']:.2f}%"
    )
    _print_error_tables(residuals)

    grouped_importance, transformed_importance = extract_feature_importance(
        preprocessor, fitted_model
    )
    parsed_locations = pd.DataFrame(
        X["Address"].map(HousingFeatureEngineer.parse_address).tolist()
    )
    missing_rates = {
        column: float(df[column].isna().mean() * 100.0)
        for column in RAW_FEATURES
        if column != "Address"
    }
    model_parameters = {
        name: _candidate_params(spec_lookup[name]) for name in selected_names
    }
    metadata = {
        "dataset": DATA_PATH.name,
        "rows_raw": int(len(raw)),
        "rows_after_cleaning": int(len(df)),
        "rows_removed": int(len(raw) - len(df)),
        "outlier_strategy": "none; all finite positive Price and Area tails retained",
        "price_distribution_billion_vnd": distribution_summary(df["Price"]),
        "area_distribution_sqm": distribution_summary(df["Area"]),
        "training_rows": int(len(X_development)),
        "validation_rows_per_cv_fold": fold_rows,
        "test_rows": int(len(X_test)),
        "target": "Price",
        "target_unit": "Billion VND",
        "selected_model": selected_name,
        "selected_estimators": selected_names,
        "target_strategy": model_parameters,
        "hyperparameters": model_parameters,
        "sample_weighting": {
            name: spec_lookup[name].weighting for name in selected_names
        },
        "ensemble_weights": dict(zip(selected_names, selected_weights)),
        "metrics": test_metrics,
        "cv_metrics": selected_cv,
        "all_candidate_cv_metrics": cv_results,
        "old_test_baseline": old_test_metrics,
        "mape_target_percent": {"min": 5.0, "max": 7.0},
        "mape_target_met": bool(5.0 <= test_metrics["MAPE_percent"] <= 7.0),
        "selection_metric_policy": (
            "Use a blend only when it improves grouped CV MAPE without reducing "
            "mean grouped CV R2 versus the best single estimator."
        ),
        "validation_strategy": (
            f"{CV_FOLDS}-fold GroupKFold with normalized Address groups; "
            "final test is a disjoint address-group holdout"
        ),
        "test_group_overlap_count": 0,
        "feature_names": transformed_feature_names(preprocessor),
        "eda_guided_changes": [
            "Canonicalize administrative prefixes, punctuation, and common province aliases.",
            "Separate street-like address heads from explicitly named project heads.",
            "Compare direction-field ablation because both direction fields are mostly missing.",
            "Compare inverse-price weighting for the high relative-error price tails.",
        ],
        "feature_importance": grouped_importance[:50],
        "transformed_feature_importance": transformed_importance[:50],
        "categorical_options": {
            "provinces": sorted(parsed_locations["Province"].unique().tolist()),
            "legal_status": sorted(df["Legal status"].dropna().astype(str).unique().tolist()),
            "house_direction": sorted(df["House direction"].dropna().astype(str).unique().tolist()),
            "balcony_direction": sorted(df["Balcony direction"].dropna().astype(str).unique().tolist()),
            "furniture_state": sorted(df["Furniture state"].dropna().astype(str).unique().tolist()),
        },
        "location_cardinality": {
            column: int(parsed_locations[column].nunique())
            for column in (
                "Project", "Street", "Address_Head", "Ward", "District",
                "Province",
            )
        },
        "missing_feature_percent": missing_rates,
        "residual_diagnostics": residuals,
        "scaling": "None; selected estimators are tree-based",
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nPipeline: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()

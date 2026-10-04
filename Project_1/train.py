"""Backward-compatible command-line entry point for model training."""

from housing.config import (
    BASE_DIR, CV_FOLDS, DATA_PATH, METADATA_PATH, MODEL_DIR, MODEL_PATH,
    RAW_FEATURES, RANDOM_STATE, TEST_SIZE,
)
from housing.components import (
    GroupedHierarchicalEncoder, HousingFeatureEngineer, InputNormalizer,
    PriceTargetRegressor, PrefitWeightedRegressor,
)
from housing.data import (
    _balanced_group_split, _prepare_features, _price_bin_proportions,
    _transform_features, build_pipeline, build_preprocessor, clean_data,
    distribution_summary, metrics, transformed_feature_names,
)
from housing.evaluation import (
    _fold_metric_summary, _print_error_tables, _residual_diagnostics,
)
from housing.models import (
    CandidateSpec, _blend_weights, _candidate_params, _candidate_specs,
    _importance_group, _new_estimator, _random_xgb_candidates,
    extract_feature_importance,
)
from housing.training import _cross_validate, _fit_final_pipeline, main

__all__ = [
    "BASE_DIR", "CV_FOLDS", "DATA_PATH", "METADATA_PATH", "MODEL_DIR",
    "MODEL_PATH", "RAW_FEATURES", "RANDOM_STATE", "TEST_SIZE",
    "CandidateSpec", "GroupedHierarchicalEncoder", "HousingFeatureEngineer",
    "InputNormalizer", "PriceTargetRegressor", "PrefitWeightedRegressor",
    "_balanced_group_split", "_blend_weights", "_candidate_params",
    "_candidate_specs", "_cross_validate", "_fit_final_pipeline",
    "_fold_metric_summary", "_importance_group", "_new_estimator",
    "_prepare_features", "_price_bin_proportions", "_print_error_tables",
    "_random_xgb_candidates", "_residual_diagnostics", "_transform_features",
    "build_pipeline", "build_preprocessor", "clean_data",
    "distribution_summary", "extract_feature_importance", "main", "metrics",
    "transformed_feature_names",
]

if __name__ == "__main__":
    main()

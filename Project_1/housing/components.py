"""Reusable, pickle-safe preprocessing components for the Vietnam housing pipeline."""
from __future__ import annotations

import unicodedata
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import GroupKFold


class InputNormalizer(BaseEstimator, TransformerMixin):
    """Convert API-style dict/list records or DataFrames into a DataFrame."""

    def fit(self, X: Any, y: Any = None) -> InputNormalizer:
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, dict):
            return pd.DataFrame([X])
        if isinstance(X, (list, tuple)):
            if not X:
                return pd.DataFrame()
            if all(isinstance(item, dict) for item in X):
                return pd.DataFrame(list(X))
        raise TypeError("Input must be a pandas DataFrame, dict, or list of dict records.")


class GroupedHierarchicalEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target encoding with address-grouped out-of-fold training values."""

    def __init__(self, smoothing: float = 20.0, n_splits: int = 5) -> None:
        self.smoothing = smoothing
        self.n_splits = n_splits

    def fit(self, X: Any, y: Any) -> GroupedHierarchicalEncoder:
        frame = pd.DataFrame(X).copy()
        target = np.asarray(y, dtype=float)
        if "Address" not in frame:
            raise ValueError("GroupedHierarchicalEncoder requires Address group IDs.")
        self.feature_names_in_ = np.asarray(
            [column for column in frame.columns if column != "Address"], dtype=object
        )
        self.global_mean_ = float(np.mean(target))
        self.category_means_: dict[str, dict[str, float]] = {}
        for column in self.feature_names_in_:
            values = frame[column].fillna("Unknown").astype(str)
            stats = pd.DataFrame({"category": values, "target": target}).groupby(
                "category"
            )["target"].agg(["sum", "count"])
            smoothed = (
                stats["sum"] + self.smoothing * self.global_mean_
            ) / (stats["count"] + self.smoothing)
            self.category_means_[str(column)] = smoothed.to_dict()
        self.n_features_in_ = len(frame.columns)
        return self

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> np.ndarray:
        if y is None:
            raise ValueError("Target values are required for grouped target encoding.")
        frame = pd.DataFrame(X).copy()
        target = np.asarray(y, dtype=float)
        groups = frame["Address"].map(
            HousingFeatureEngineer._clean_group_key
        ).to_numpy()
        columns = [column for column in frame.columns if column != "Address"]
        encoded = np.full((len(frame), len(columns)), np.nan, dtype=float)
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            folds = min(self.n_splits, len(unique_groups))
            splitter = GroupKFold(n_splits=folds)
            for fit_idx, transform_idx in splitter.split(frame, target, groups):
                fold_encoder = GroupedHierarchicalEncoder(
                    smoothing=self.smoothing,
                    n_splits=self.n_splits,
                ).fit(frame.iloc[fit_idx], target[fit_idx])
                encoded[transform_idx] = fold_encoder.transform(
                    frame.iloc[transform_idx]
                )
        self.fit(frame, target)
        missing = ~np.isfinite(encoded)
        if missing.any():
            encoded[missing] = self.global_mean_
        return encoded

    def transform(self, X: Any) -> np.ndarray:
        frame = pd.DataFrame(X)
        columns = [column for column in frame.columns if column != "Address"]
        encoded_columns: list[np.ndarray] = []
        for column in columns:
            values = frame[column].fillna("Unknown").astype(str)
            mapping = self.category_means_.get(str(column), {})
            encoded_columns.append(
                values.map(mapping).fillna(self.global_mean_).to_numpy(dtype=float)
            )
        if not encoded_columns:
            return np.empty((len(frame), 0), dtype=float)
        return np.column_stack(encoded_columns)

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Extract address hierarchy and calculate shared property features."""

    _PROVINCE_PREFIXES = ("tỉnh ", "tp. ", "tp ", "thành phố ")
    _DISTRICT_PREFIXES = (
        "quận ", "huyện ", "thị xã ", "tx ", "tx. ", "thành phố ",
        "tp ", "tp. ",
    )
    _WARD_PREFIXES = ("phường ", "xã ", "thị trấn ")
    _STREET_PREFIXES = (
        "đường ", "đ. ", "phố ", "ngõ ", "ngách ", "hẻm ", "kiệt ",
        "đại lộ ", "tỉnh lộ ", "quốc lộ ",
    )
    _PROJECT_MARKERS = (
        "dự án ", "khu đô thị ", "khu dân cư ", "kdc ", "vinhomes ",
        "vincom ", "the ", "residence", "tower", "condominium",
    )
    _LOCATION_COUNT_COLUMNS = (
        "Province", "District", "Ward", "Project", "Address_Head",
        "District_Ward",
    )
    _ADMIN_PREFIXES = {
        "tỉnh ": "Tỉnh ",
        "tp.": "TP ",
        "tp. ": "TP ",
        "tp ": "TP ",
        "thành phố ": "TP ",
        "quận ": "Quận ",
        "huyện ": "Huyện ",
        "thị xã ": "Thị xã ",
        "tx. ": "Thị xã ",
        "tx ": "Thị xã ",
        "phường ": "Phường ",
        "xã ": "Xã ",
        "thị trấn ": "Thị trấn ",
        "đường ": "Đường ",
        "đ. ": "Đường ",
    }

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None or pd.isna(value):
            return "Unknown"
        text = unicodedata.normalize("NFC", str(value))
        text = " ".join(text.split()).strip(" \t\r\n,.;:")
        return text or "Unknown"

    @staticmethod
    def _clean_group_key(value: Any) -> str:
        if value is None or pd.isna(value):
            return "Unknown"
        text = unicodedata.normalize("NFC", str(value))
        return " ".join(text.split()) or "Unknown"

    @staticmethod
    def parse_address(value: Any) -> dict[str, str]:
        text = HousingFeatureEngineer._clean_text(value)
        parts = [
            HousingFeatureEngineer._normalize_admin_prefix(
                HousingFeatureEngineer._clean_text(part)
            )
            for part in text.split(",")
            if part.strip()
        ]
        province = "Unknown"
        district = "Unknown"
        ward = "Unknown"
        administrative_start = len(parts)

        if parts and len(parts) >= 2:
            province = parts[-1]
            administrative_start -= 1

        if administrative_start >= 2:
            district_component = parts[administrative_start - 1]
            district = district_component
            if HousingFeatureEngineer._has_prefix(
                district_component, HousingFeatureEngineer._DISTRICT_PREFIXES
            ):
                district = HousingFeatureEngineer._strip_admin_prefix(
                    district_component, HousingFeatureEngineer._DISTRICT_PREFIXES
                )
            administrative_start -= 1

        if administrative_start >= 2:
            ward_component = parts[administrative_start - 1]
            ward = ward_component
            if HousingFeatureEngineer._has_prefix(
                ward_component, HousingFeatureEngineer._WARD_PREFIXES
            ):
                ward = HousingFeatureEngineer._strip_admin_prefix(
                    ward_component, HousingFeatureEngineer._WARD_PREFIXES
                )
            administrative_start -= 1

        address_head = " ".join(parts[:administrative_start]) or (
            parts[0] if parts else "Unknown"
        )
        head_key = address_head.casefold()
        street_like = HousingFeatureEngineer._has_prefix(
            address_head, HousingFeatureEngineer._STREET_PREFIXES
        )
        named_project = any(
            marker in head_key for marker in HousingFeatureEngineer._PROJECT_MARKERS
        )
        project = address_head if named_project and not street_like else "Unknown"
        street = address_head if street_like else "Unknown"
        address_head_type = (
            "street"
            if street_like
            else "named_project"
            if named_project
            else "other"
            if address_head != "Unknown"
            else "unknown"
        )
        province = HousingFeatureEngineer._canonical_location(
            province, HousingFeatureEngineer._PROVINCE_PREFIXES
        )
        district = HousingFeatureEngineer._canonical_location(
            district, HousingFeatureEngineer._DISTRICT_PREFIXES
        )
        ward = HousingFeatureEngineer._canonical_location(
            ward, HousingFeatureEngineer._WARD_PREFIXES
        )
        last_token = parts[-1] if parts else "Unknown"
        count = len(parts)
        province_district = HousingFeatureEngineer._join_location(province, district)
        district_ward = HousingFeatureEngineer._join_location(district, ward)
        province_district_ward = HousingFeatureEngineer._join_location(
            province, district, ward
        )
        project_district = HousingFeatureEngineer._join_location(project, district)
        project_ward = HousingFeatureEngineer._join_location(project, ward)
        return {
            "Project": project,
            "Street": street,
            "Address_Head": address_head,
            "Address_Head_Type": address_head_type,
            "Ward": ward,
            "District": district,
            "Province": province,
            "Address_Last_Token": last_token,
            "Address_Component_Count": count,
            "Province_District": province_district,
            "District_Ward": district_ward,
            "Province_District_Ward": province_district_ward,
            "Project_District": project_district,
            "Project_Ward": project_ward,
            "Address_Length": len(text),
            "Address_Digit_Count": sum(character.isdigit() for character in text),
            "Has_Project": float(project != "Unknown"),
            "Has_Ward": float(ward != "Unknown"),
            "Has_District": float(district != "Unknown"),
        }

    @staticmethod
    def _normalize_admin_prefix(value: str) -> str:
        folded = value.casefold()
        for prefix, canonical in HousingFeatureEngineer._ADMIN_PREFIXES.items():
            if folded.startswith(prefix):
                return canonical + value[len(prefix):].strip()
        return value

    @staticmethod
    def _strip_admin_prefix(value: str, prefixes: Sequence[str]) -> str:
        folded = value.casefold()
        for prefix in prefixes:
            if folded.startswith(prefix):
                return value[len(prefix):].strip(" \t\r\n,.;:")
        return value

    @staticmethod
    def _canonical_location(value: str, prefixes: Sequence[str]) -> str:
        cleaned = HousingFeatureEngineer._clean_text(value)
        if cleaned == "Unknown":
            return cleaned
        stripped = HousingFeatureEngineer._strip_admin_prefix(cleaned, prefixes)
        normalized = " ".join(stripped.split()).strip(" \t\r\n,.;:")
        aliases = {
            "hồ chí minh city": "Hồ Chí Minh",
            "ho chi minh": "Hồ Chí Minh",
            "ho chi minh city": "Hồ Chí Minh",
            "hcm": "Hồ Chí Minh",
            "hồ chí minh": "Hồ Chí Minh",
            "sai gon": "Hồ Chí Minh",
            "sài gòn": "Hồ Chí Minh",
            "saigon": "Hồ Chí Minh",
            "ha noi": "Hà Nội",
            "hà nội": "Hà Nội",
            "da nang": "Đà Nẵng",
            "đà nẵng": "Đà Nẵng",
        }
        return aliases.get(normalized.casefold(), normalized or "Unknown")

    @staticmethod
    def _has_prefix(value: str, prefixes: Sequence[str]) -> bool:
        normalized = value.casefold()
        return any(normalized.startswith(prefix) for prefix in prefixes)

    @staticmethod
    def _join_location(*parts: str) -> str:
        return " | ".join(part for part in parts if part != "Unknown") or "Unknown"

    @staticmethod
    def _province(value: Any) -> str:
        """Backward-compatible helper for callers that only need the province."""
        return HousingFeatureEngineer.parse_address(value)["Province"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> HousingFeatureEngineer:
        addresses = X.get(
            "Address", pd.Series("", index=X.index, dtype="object")
        ).map(self.parse_address)
        parsed = pd.DataFrame(addresses.tolist(), index=X.index)
        self.location_counts_ = {
            column: parsed[column].value_counts().to_dict()
            for column in self._LOCATION_COUNT_COLUMNS
        }
        area = pd.to_numeric(X.get("Area"), errors="coerce")
        self.area_cap_ = float(area.quantile(0.995)) if area.notna().any() else np.inf
        self.area_bucket_edges_ = (
            np.unique(area.quantile([0.25, 0.5, 0.75]).to_numpy())
            if area.notna().any()
            else np.asarray([], dtype=float)
        )
        parsed["Address_Component_Count"] = pd.to_numeric(
            parsed["Address_Component_Count"], errors="coerce"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        address = df.get("Address", pd.Series("", index=df.index, dtype="object"))
        locations = address.map(self.parse_address).apply(pd.Series)
        parsed_columns = (
            "Project", "Street", "Address_Head", "Address_Head_Type",
            "Ward", "District", "Province", "Address_Last_Token",
            "Address_Component_Count", "Province_District", "District_Ward",
            "Province_District_Ward", "Project_District", "Project_Ward",
            "Address_Length", "Address_Digit_Count", "Has_Project",
            "Has_Ward", "Has_District",
        )
        for column in parsed_columns:
            df[column] = locations[column]
        for column in self._LOCATION_COUNT_COLUMNS:
            counts = getattr(self, "location_counts_", {}).get(column, {})
            is_known = locations[column] != "Unknown"
            df[f"{column}_Log_Frequency"] = (
                np.log1p(
                    locations[column].map(counts).where(is_known, 0).fillna(0).astype(float)
                )
            )
        df["Address_Component_Count"] = pd.to_numeric(
            locations["Address_Component_Count"], errors="coerce"
        )
        area = pd.to_numeric(
            df.get("Area", pd.Series(np.nan, index=df.index)), errors="coerce"
        )
        area_edges = getattr(self, "area_bucket_edges_", np.asarray([], dtype=float))
        area_buckets = pd.Series(
            np.searchsorted(area_edges, area.to_numpy(dtype=float), side="right"),
            index=df.index,
        ).map(lambda value: f"AreaBand{int(value)}")
        for level in ("Province", "District", "Project"):
            df[f"{level}_Area_Bucket"] = (
                locations[level].astype(str) + " | " + area_buckets.astype(str)
            )
        for column in ("House direction", "Balcony direction", "Legal status", "Furniture state"):
            if column in df:
                df[column] = df[column].map(self._clean_text)

        bedrooms = pd.to_numeric(
            df.get("Bedrooms", pd.Series(np.nan, index=df.index)), errors="coerce"
        )
        bathrooms = pd.to_numeric(
            df.get("Bathrooms", pd.Series(np.nan, index=df.index)), errors="coerce"
        )
        frontage = pd.to_numeric(
            df.get("Frontage", pd.Series(np.nan, index=df.index)), errors="coerce"
        )
        access_road = pd.to_numeric(
            df.get("Access Road", pd.Series(np.nan, index=df.index)), errors="coerce"
        )
        floors = pd.to_numeric(
            df.get("Floors", pd.Series(np.nan, index=df.index)), errors="coerce"
        )

        df["Total_Rooms"] = bedrooms + bathrooms
        df["Bedroom_Bathroom_sum"] = bedrooms + bathrooms
        df["Bedroom_Bathroom_difference"] = bedrooms - bathrooms
        df["Rooms_per_Area"] = df["Total_Rooms"].div(area.where(area > 0))
        df["Bed_Bath_Ratio"] = bedrooms.div(bathrooms.where(bathrooms != 0))
        df["Area_log1p"] = np.log1p(area.where(area >= 0))
        df["Frontage_log1p"] = np.log1p(frontage.where(frontage >= 0))
        df["Access_Road_log1p"] = np.log1p(access_road.where(access_road >= 0))
        area_cap = getattr(self, "area_cap_", np.inf)
        df["Area_squared"] = area.clip(upper=area_cap) ** 2
        df["Floors_squared"] = floors**2
        df["Bedrooms_squared"] = bedrooms**2
        df["Bathrooms_squared"] = bathrooms**2
        df["Frontage_to_Area"] = frontage.div(area.where(area > 0))
        df["Road_to_Frontage"] = access_road.div(frontage.where(frontage > 0))
        df["Rooms_per_100sqm"] = df["Total_Rooms"] * 100.0 / area.where(area > 0)
        df["Bedrooms_per_Area"] = bedrooms.div(area.where(area > 0))
        df["Bathrooms_per_Area"] = bathrooms.div(area.where(area > 0))
        df["Area_per_Bedroom"] = area.div(bedrooms.where(bedrooms > 0))
        df["Area_per_Room"] = area.div(df["Total_Rooms"].where(df["Total_Rooms"] > 0))
        df["Is_single_floor"] = (floors == 1).astype(float)
        df["Is_large_property"] = (area >= 150).astype(float)
        df["Is_wide_frontage"] = (frontage >= 6).astype(float)
        df["Is_wide_access_road"] = (access_road >= 6).astype(float)
        df["Area_x_Frontage"] = area * frontage
        df["Area_x_Floors"] = area * floors
        df["Frontage_x_Access_Road"] = frontage * access_road
        df["Bedrooms_x_Area"] = bedrooms * area
        df["Bathrooms_x_Area"] = bathrooms * area
        return df


class PrefitWeightedRegressor(RegressorMixin, BaseEstimator):
    """Blend predictions from already-fitted regressors without fitting copies."""

    def __init__(
        self,
        estimators: Sequence[tuple[str, Any]],
        weights: Sequence[float],
    ) -> None:
        if not estimators or len(estimators) != len(weights):
            raise ValueError("Estimators and weights must be non-empty and have equal lengths.")
        weight_array = np.asarray(weights, dtype=float)
        if not np.isfinite(weight_array).all() or (weight_array < 0).any():
            raise ValueError("Blend weights must be finite and non-negative.")
        if weight_array.sum() <= 0:
            raise ValueError("At least one blend weight must be positive.")
        self.estimators = estimators
        self.weights = weights
        self.estimators_ = tuple(estimators)
        self.weights_ = weight_array / weight_array.sum()

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def fit(self, X: Any, y: Any = None) -> PrefitWeightedRegressor:
        raise RuntimeError(
            "This blend contains pre-fitted estimators and cannot be fitted again."
        )

    def predict(self, X: Any) -> np.ndarray:
        predictions = np.vstack([
            np.asarray(estimator.predict(X), dtype=float)
            for _, estimator in self.estimators_
        ])
        return np.average(predictions, axis=0, weights=self.weights_)


class PriceTargetRegressor(RegressorMixin, BaseEstimator):
    """Fit a cloned regressor with a selectable price transform and weighting."""

    def __init__(
        self,
        estimator: Any,
        target_strategy: str = "log1p",
        weighting: str = "none",
        target_clip_quantiles: tuple[float, float] | None = None,
    ) -> None:
        self.estimator = estimator
        self.target_strategy = target_strategy
        self.weighting = weighting
        self.target_clip_quantiles = target_clip_quantiles

    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        if self.target_strategy == "raw":
            return y
        if self.target_strategy == "log1p":
            return np.log1p(y)
        if self.target_strategy == "sqrt":
            return np.sqrt(y)
        raise ValueError(f"Unsupported target strategy: {self.target_strategy}")

    def _inverse_target(self, prediction: np.ndarray) -> np.ndarray:
        if self.target_strategy == "raw":
            return prediction
        if self.target_strategy == "log1p":
            return np.expm1(prediction)
        if self.target_strategy == "sqrt":
            return np.square(np.maximum(prediction, 0.0))
        raise ValueError(f"Unsupported target strategy: {self.target_strategy}")

    def _sample_weights(self, y: np.ndarray) -> np.ndarray | None:
        if self.weighting == "none":
            return None
        if self.weighting == "inverse_price":
            weights = 1.0 / y
        elif self.weighting == "inverse_sqrt_price":
            weights = 1.0 / np.sqrt(y)
        elif self.weighting == "inverse_price_epsilon":
            epsilon = max(float(np.min(y)) * 0.01, 1e-6)
            weights = 1.0 / (y + epsilon)
        else:
            raise ValueError(f"Unsupported sample weighting: {self.weighting}")
        weights = weights / weights.mean()
        return np.clip(weights, 0.1, 10.0)

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> PriceTargetRegressor:
        target = np.asarray(y, dtype=float)
        if not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError("Training prices must be finite and strictly positive.")
        if self.target_clip_quantiles is not None:
            low, high = self.target_clip_quantiles
            if not 0 <= low < high <= 1:
                raise ValueError("Target clipping quantiles must satisfy 0 <= low < high <= 1.")
            self.target_clip_bounds_ = tuple(
                float(value) for value in np.quantile(target, [low, high])
            )
            fit_target = np.clip(target, *self.target_clip_bounds_)
        else:
            self.target_clip_bounds_ = None
            fit_target = target
        estimator = clone(self.estimator)
        weights = self._sample_weights(fit_target)
        if sample_weight is not None:
            supplied_weights = np.asarray(sample_weight, dtype=float)
            weights = (
                supplied_weights
                if weights is None
                else weights * supplied_weights
            )
        fit_parameters = {}
        if weights is not None:
            fit_parameters["sample_weight"] = weights
        estimator.fit(X, self._transform_target(fit_target), **fit_parameters)
        self.estimator_ = estimator
        self.n_features_in_ = getattr(estimator, "n_features_in_", X.shape[1])
        return self

    def predict(self, X: Any) -> np.ndarray:
        prediction = np.asarray(self.estimator_.predict(X), dtype=float)
        return np.maximum(self._inverse_target(prediction), 0.0)

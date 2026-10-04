from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("HOUSING_DATA", BASE_DIR / "vietnam_housing_dataset.csv"))
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "vietnam_house_pipeline.pkl"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", MODEL_DIR / "feature_metadata.json"))
RANDOM_STATE = 42
CV_FOLDS = 3
XGB_SEARCH_CANDIDATES = 1
TEST_SIZE = 0.15

RAW_FEATURES = [
    "Address", "Area", "Frontage", "Access Road", "House direction",
    "Balcony direction", "Floors", "Bedrooms", "Bathrooms",
    "Legal status", "Furniture state",
]
NUMERIC_FEATURES = [
    "Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms",
    "Total_Rooms", "Bedroom_Bathroom_sum", "Bedroom_Bathroom_difference",
    "Rooms_per_Area", "Bed_Bath_Ratio", "Area_log1p", "Frontage_log1p",
    "Access_Road_log1p", "Area_squared", "Floors_squared", "Bedrooms_squared",
    "Bathrooms_squared", "Frontage_to_Area", "Road_to_Frontage",
    "Rooms_per_100sqm", "Bedrooms_per_Area", "Bathrooms_per_Area",
    "Area_per_Bedroom", "Area_per_Room", "Is_single_floor",
    "Is_large_property", "Is_wide_frontage", "Is_wide_access_road",
    "Area_x_Frontage", "Area_x_Floors", "Frontage_x_Access_Road",
    "Bedrooms_x_Area", "Bathrooms_x_Area", "Address_Component_Count",
    "Address_Length", "Address_Digit_Count", "Has_Project", "Has_Ward",
    "Has_District", "Address_Head_Log_Frequency",
    "Province_Log_Frequency", "District_Log_Frequency", "Ward_Log_Frequency",
    "Project_Log_Frequency", "District_Ward_Log_Frequency",
]
LOCATION_FEATURES = [
    "Project", "Street", "Address_Head", "Address_Head_Type",
    "Ward", "District", "Province", "Address_Last_Token",
    "Province_District", "District_Ward", "Province_District_Ward",
    "Project_District", "Project_Ward", "Province_Area_Bucket",
    "District_Area_Bucket", "Project_Area_Bucket",
]
CATEGORICAL_FEATURES = [
    "House direction", "Balcony direction", "Legal status", "Furniture state",
]
MIN_GROUP_SAMPLES = 30

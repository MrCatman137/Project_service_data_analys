from __future__ import annotations

import json
import unittest

import joblib
import numpy as np
import pandas as pd

from housing.config import METADATA_PATH, MODEL_PATH, RAW_FEATURES


SAMPLE_PROPERTY = {
    "Address": "Dự án Vinhomes Central Park, Phường 22, Bình Thạnh, Hồ Chí Minh",
    "Area": 80.0,
    "Frontage": 5.0,
    "Access Road": 8.0,
    "House direction": "Đông",
    "Balcony direction": "Nam",
    "Floors": 3.0,
    "Bedrooms": 3.0,
    "Bathrooms": 2.0,
    "Legal status": "Sổ hồng",
    "Furniture state": "Đầy đủ",
}


class SavedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Saved model not found at {MODEL_PATH}; train it before running model tests."
            )
        if not METADATA_PATH.is_file():
            raise FileNotFoundError(
                f"Model metadata not found at {METADATA_PATH}; train it before running model tests."
            )
        cls.pipeline = joblib.load(MODEL_PATH)
        cls.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    def test_prediction_is_finite_and_non_negative(self) -> None:
        prediction = np.asarray(self.pipeline.predict(SAMPLE_PROPERTY), dtype=float)

        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreaterEqual(float(prediction[0]), 0.0)

    def test_supported_input_forms_produce_the_same_prediction(self) -> None:
        frame = pd.DataFrame([SAMPLE_PROPERTY], columns=RAW_FEATURES)
        predictions = [
            np.asarray(self.pipeline.predict(SAMPLE_PROPERTY), dtype=float),
            np.asarray(self.pipeline.predict([SAMPLE_PROPERTY]), dtype=float),
            np.asarray(self.pipeline.predict(frame), dtype=float),
        ]

        for prediction in predictions:
            self.assertEqual(prediction.shape, (1,))
            self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_allclose(predictions[0], predictions[1])
        np.testing.assert_allclose(predictions[0], predictions[2])

    def test_metadata_has_valid_evaluation_metrics(self) -> None:
        self.assertEqual(self.metadata["target"], "Price")
        self.assertTrue(self.metadata["selected_model"])
        for metric in ("RMSE", "MAE", "R2", "MAPE_percent"):
            with self.subTest(metric=metric):
                self.assertTrue(np.isfinite(self.metadata["metrics"][metric]))


if __name__ == "__main__":
    unittest.main()

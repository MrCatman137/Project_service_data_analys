from __future__ import annotations

import unittest

from app import app


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = app.test_client()

    def test_metadata_endpoint_returns_model_metrics(self) -> None:
        response = self.client.get("/api/metadata")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["model"])
        self.assertIn("MAPE_percent", payload["metrics"])

    def test_prediction_endpoint_returns_positive_price(self) -> None:
        response = self.client.post("/predict/price", json={
            "Address": "Dự án Vinhomes Central Park, Phường 22, Bình Thạnh, Hồ Chí Minh",
            "Area": 80,
            "Frontage": 5,
            "Access Road": 8,
            "Floors": 3,
            "Bedrooms": 3,
            "Bathrooms": 2,
            "House direction": "Đông",
            "Balcony direction": "Nam",
            "Legal status": "Sổ hồng",
            "Furniture state": "Đầy đủ",
        })

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertGreater(payload["predicted_price"], 0)

    def test_prediction_endpoint_rejects_non_positive_area(self) -> None:
        response = self.client.post("/predict/price", json={"Area": 0})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["status"], "error")


if __name__ == "__main__":
    unittest.main()

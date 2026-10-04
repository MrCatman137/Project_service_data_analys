from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from housing.components import HousingFeatureEngineer


class HousingFeatureEngineerTests(unittest.TestCase):
    def test_extracts_hierarchical_location_from_address(self) -> None:
        parsed = HousingFeatureEngineer.parse_address(
            "Dự án Vinhomes Central Park, Phường 22, Bình Thạnh, Hồ Chí Minh"
        )

        self.assertNotEqual(parsed["Project"], "Unknown")
        self.assertNotEqual(parsed["Ward"], "Unknown")
        self.assertNotEqual(parsed["District"], "Unknown")
        self.assertNotEqual(parsed["Province"], "Unknown")

    def test_engineers_room_and_area_interactions(self) -> None:
        row = pd.DataFrame([{
            "Address": "Đường Lê Lợi, Phường Bến Nghé, Quận 1, Hồ Chí Minh",
            "Area": 100.0,
            "Frontage": 5.0,
            "Access Road": 8.0,
            "Floors": 2.0,
            "Bedrooms": 3.0,
            "Bathrooms": 2.0,
            "House direction": "Đông",
            "Balcony direction": "Nam",
            "Legal status": "Sổ hồng",
            "Furniture state": "Đầy đủ",
        }])
        transformed = HousingFeatureEngineer().fit(row).transform(row)

        self.assertEqual(transformed.loc[0, "Total_Rooms"], 5.0)
        self.assertEqual(transformed.loc[0, "Rooms_per_Area"], 0.05)
        self.assertEqual(transformed.loc[0, "Bed_Bath_Ratio"], 1.5)
        self.assertTrue(
            np.isfinite(
                transformed[["Total_Rooms", "Rooms_per_Area", "Bed_Bath_Ratio"]]
                .to_numpy(dtype=float)
            ).all()
        )


if __name__ == "__main__":
    unittest.main()

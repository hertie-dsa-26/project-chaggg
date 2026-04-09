import os
import unittest
from pathlib import Path

import pandas as pd

from src.flask_app import create_app, knn_paths

KNN_OUTPUTS_READY = knn_paths.knn_artifacts_present()


class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        # Keep tests fast and independent of large local datasets.
        os.environ["CHAGGG_SKIP_DATA_LOAD"] = "1"
        app = create_app()
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_routes_render(self):
        routes = [
            "/",
            "/about",
            "/viz/placeholder",
            "/dashboards/time",
            "/dashboards/space",
            "/dashboards/types",
            "/viz/knn",
        ]
        for route in routes:
            with self.subTest(route=route):
                resp = self.client.get(route)
                self.assertEqual(resp.status_code, 200)

    @unittest.skipUnless(KNN_OUTPUTS_READY, "KNN parquet/CSV outputs not built")
    def test_api_knn_predict_json(self):
        test_path = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "knn"
            / "monthly_ca_test_2023_2024.parquet"
        )
        sample = pd.read_parquet(test_path, columns=["community_area", "month"])
        ca = float(sample.iloc[0]["community_area"])
        month = pd.Timestamp(sample.iloc[0]["month"]).strftime("%Y-%m-%d")
        resp = self.client.get(
            f"/api/knn/predict?community_area={ca}&month={month}&k=10&time_scale=0"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("predicted_crime_count", data)
        self.assertIn("actual_crime_count", data)


if __name__ == "__main__":
    unittest.main()

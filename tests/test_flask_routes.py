import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.flask_app import create_app, knn_paths


class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        # Keep tests fast and independent of large local datasets.
        os.environ["CHAGGG_SKIP_DATA_LOAD"] = "1"
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["CHAGGG_KNN_DIR"] = self._tmp.name

        knn_dir = Path(self._tmp.name)
        knn_dir.mkdir(parents=True, exist_ok=True)

        # Minimal monthly tables (train/test) for /api/knn/predict
        train = pd.DataFrame(
            {
                "community_area": [1.0, 2.0, 3.0],
                "month": pd.to_datetime(["2015-01-01", "2015-02-01", "2015-03-01"]),
                "crime_count": [10.0, 12.0, 14.0],
                "lat": [41.88, 41.89, 41.90],
                "lon": [-87.63, -87.62, -87.61],
                "month_index": [0.0, 1.0, 2.0],
            }
        )
        test = pd.DataFrame(
            {
                "community_area": [1.0],
                "month": pd.to_datetime(["2023-01-01"]),
                "crime_count": [11.0],
                "lat": [41.88],
                "lon": [-87.63],
                "month_index": [96.0],
            }
        )
        train.to_parquet(knn_dir / "monthly_ca_train_2015_2022.parquet", index=False)
        test.to_parquet(knn_dir / "monthly_ca_test_2023_2024.parquet", index=False)

        # Minimal prediction CSVs for /viz/knn
        pred = test.copy()
        pred["predicted_crime_count"] = [12.0]
        for k in (5, 10, 20):
            pred.to_csv(knn_dir / f"forecast_predictions_k{k}.csv", index=False)

        (knn_dir / "forecast_metrics.json").write_text(
            '{"metrics_by_k":{"10":{"mae":1.0,"rmse":2.0},"20":{"mae":1.0,"rmse":2.0},"5":{"mae":1.0,"rmse":2.0}}}',
            encoding="utf-8",
        )

        app = create_app()
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def tearDown(self):
        self._tmp.cleanup()

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

    def test_api_knn_predict_json(self):
        self.assertTrue(knn_paths.knn_artifacts_present())
        ca = 1.0
        month = "2023-01-01"
        resp = self.client.get(
            f"/api/knn/predict?community_area={ca}&month={month}&k=10&time_scale=0"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("predicted_crime_count", data)
        self.assertIn("actual_crime_count", data)

    def test_api_hotspots_missing_geojson(self):
        # In CI we don't ship geojson artifacts; we should get a controlled error.
        resp = self.client.get("/api/hotspots?variant=meters")
        self.assertIn(resp.status_code, (200, 404))


if __name__ == "__main__":
    unittest.main()

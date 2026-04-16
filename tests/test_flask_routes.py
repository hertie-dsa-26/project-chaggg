import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.flask_app import create_app, knn_paths


class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        os.environ["CHAGGG_SKIP_DATA_LOAD"] = "1"
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["CHAGGG_KNN_DIR"] = self._tmp.name

        knn_dir = Path(self._tmp.name)
        knn_dir.mkdir(parents=True, exist_ok=True)

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

        pred = test.copy()
        pred["predicted_crime_count"] = [12.0]
        for k in (5, 10, 20):
            pred.to_csv(knn_dir / f"forecast_predictions_k{k}.csv", index=False)

        (knn_dir / "forecast_metrics.json").write_text(
            '{"metrics_by_k":{"10":{"mae":1.0,"rmse":2.0},"20":{"mae":1.0,"rmse":2.0},"5":{"mae":1.0,"rmse":2.0}}}',
            encoding="utf-8",
        )

        self.app = create_app()
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self):
        os.environ.pop("CHAGGG_SKIP_DATA_LOAD", None)
        os.environ.pop("CHAGGG_KNN_DIR", None)
        self._tmp.cleanup()

    def test_routes_render(self):
        routes = [
            "/",
            "/about",
            "/viz/placeholder",
            "/dashboards/time",
            "/dashboards/space",
            "/dashboards/types",
            "/viz/hotspots",
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
        resp = self.client.get("/api/hotspots?variant=meters")
        self.assertIn(resp.status_code, (200, 404))

    def test_api_hotspots_unknown_variant(self):
        resp = self.client.get("/api/hotspots?variant=not_a_variant")
        self.assertEqual(resp.status_code, 400)

    def test_api_hotspots_returns_geojson_when_present(self):
        root = Path(self.app.root_path)
        gen = root / "static" / "geo" / "generated"
        gen.mkdir(parents=True, exist_ok=True)
        path = gen / "hotspots_meters.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
        try:
            resp = self.client.get("/api/hotspots?variant=meters")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.get_json()["type"], "FeatureCollection")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

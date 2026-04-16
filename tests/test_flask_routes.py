import json
import os
import unittest
from pathlib import Path

from src.flask_app import create_app


class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        os.environ["CHAGGG_SKIP_DATA_LOAD"] = "1"
        self.app = create_app()
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self):
        os.environ.pop("CHAGGG_SKIP_DATA_LOAD", None)

    def test_routes_render(self):
        routes = [
            "/",
            "/about",
            "/viz/placeholder",
            "/dashboards/time",
            "/dashboards/space",
            "/dashboards/types",
            "/viz/hotspots",
        ]
        for route in routes:
            with self.subTest(route=route):
                resp = self.client.get(route)
                self.assertEqual(resp.status_code, 200)

    def test_api_hotspots_missing_geojson(self):
        # 200 if repo/static ships GeoJSON; 404 in CI when artifacts are absent.
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

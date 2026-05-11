import unittest

from src.flask_app import create_app


class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_routes_render(self):
        routes = [
            "/",
            "/about",
            "/method",
            "/data-exploration",
            "/dashboards/time",
            "/dashboards/space",
            "/algorithm",
            "/codebook"
        ]
        for route in routes:
            with self.subTest(route=route):
                resp = self.client.get(route)
                self.assertEqual(resp.status_code, 200)


class TestApiPredict(unittest.TestCase):
    """Tests for POST /api/predict.

    Integration tests against the real route with real artifacts loaded
    at app startup. Uses 'homicide' (smallest dataset) to keep them fast.
    """

    @classmethod
    def setUpClass(cls):
        # Build the app once for the whole class — loading all 24 KNN
        # artifacts is slow, no point doing it per test.
        app = create_app()
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def _valid_payload(self):
        return {
            "algorithm": "knn_lrr",
            "crime_type": "homicide",
            "lat": 41.8781,
            "lon": -87.6298,
            "date": "2026-06-15",
            "hour": 14,
            "k": 5,
        }

    # --- Happy path ---------------------------------------------------------

    def test_happy_path_returns_200(self):
        resp = self.client.post("/api/predict", json=self._valid_payload())
        self.assertEqual(resp.status_code, 200)

    def test_response_contains_probability_in_unit_interval(self):
        resp = self.client.post("/api/predict", json=self._valid_payload())
        data = resp.get_json()
        self.assertIn("probability", data)
        self.assertGreaterEqual(data["probability"], 0.0)
        self.assertLessEqual(data["probability"], 1.0)

    def test_response_echoes_k_and_crime_type(self):
        payload = self._valid_payload()
        resp = self.client.post("/api/predict", json=payload)
        data = resp.get_json()
        self.assertEqual(data["k"], payload["k"])
        self.assertEqual(data["crime_type"], payload["crime_type"])

    def test_response_includes_n_total(self):
        resp = self.client.post("/api/predict", json=self._valid_payload())
        data = resp.get_json()
        self.assertIn("n_total", data)
        self.assertGreater(data["n_total"], 0)

    def test_derived_day_of_week_is_correct(self):
        # 2026-06-15 is a Monday → day_of_week == 0
        resp = self.client.post("/api/predict", json=self._valid_payload())
        data = resp.get_json()
        self.assertEqual(data["derived"]["day_of_week"], 0)

    # --- Validation errors --------------------------------------------------

    def test_unknown_crime_type_returns_400(self):
        payload = {**self._valid_payload(), "crime_type": "jaywalking"}
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unknown crime type", resp.get_json()["error"])

    def test_non_2026_date_returns_400(self):
        payload = {**self._valid_payload(), "date": "2025-06-15"}
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("2026", resp.get_json()["error"])

    def test_malformed_date_returns_400(self):
        payload = {**self._valid_payload(), "date": "not-a-date"}
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)

    def test_hour_out_of_range_returns_400(self):
        payload = {**self._valid_payload(), "hour": 25}
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("hour", resp.get_json()["error"])

    def test_k_too_large_returns_400(self):
        payload = {**self._valid_payload(), "k": 150}
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("k must be", resp.get_json()["error"])

    def test_k_zero_returns_400(self):
        payload = {**self._valid_payload(), "k": 0}
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)

    def test_missing_field_returns_400(self):
        payload = self._valid_payload()
        del payload["lat"]
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)

    # --- Algorithm dispatch -------------------------------------------------

    def test_naive_algorithm_returns_200(self):
        payload = {**self._valid_payload(), "algorithm": "naive_community_area"}
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 200)

    def test_naive_response_contains_probability_in_unit_interval(self):
        payload = {**self._valid_payload(), "algorithm": "naive_community_area"}
        resp = self.client.post("/api/predict", json=payload)
        data = resp.get_json()
        self.assertIn("probability", data)
        self.assertGreaterEqual(data["probability"], 0.0)
        self.assertLessEqual(data["probability"], 1.0)

    def test_naive_response_includes_derived_fallback(self):
        payload = {**self._valid_payload(), "algorithm": "naive_community_area"}
        resp = self.client.post("/api/predict", json=payload)
        data = resp.get_json()
        self.assertIn("derived", data)
        self.assertIn("fallback", data["derived"])

    def test_sklearn_algorithm_returns_200(self):
        payload = {**self._valid_payload(), "algorithm": "knn_sklearn"}
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 200)

    def test_unknown_algorithm_returns_400(self):
        payload = {**self._valid_payload(), "algorithm": "magic"}
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unknown algorithm", resp.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
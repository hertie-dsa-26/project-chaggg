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
            "/viz/placeholder",
            "/dashboards/time",
            "/dashboards/space",
            "/dashboards/types",
        ]
        for route in routes:
            with self.subTest(route=route):
                resp = self.client.get(route)
                self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()


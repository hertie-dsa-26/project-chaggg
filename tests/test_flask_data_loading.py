import os
import tempfile
import unittest


class TestFlaskDataLoading(unittest.TestCase):
    def test_app_boots_without_data_path(self):
        os.environ.pop("CHAGGG_DATA_PATH", None)
        from src.flask_app import create_app

        app = create_app()
        self.assertIn("CRIME_DF", app.config)

    def test_app_loads_csv_when_path_set(self):
        from src.flask_app import create_app

        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "tiny.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("date,latitude,longitude\n")
                f.write("2025-01-01T00:00:00.000,41.9,-87.7\n")

            os.environ["CHAGGG_DATA_PATH"] = csv_path
            app = create_app()
            df = app.config["CRIME_DF"]
            self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()


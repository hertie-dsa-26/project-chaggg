import numpy as np
import pytest
from pathlib import Path

from src.flask_app.load_crime_artifacts import load_knn_arrays, EXPECTED_KEYS


def make_fake_artifact(n: int = 100) -> dict:
    """Build a valid artifact dict matching what precompute_one produces."""
    rng = np.random.default_rng(0)
    return {
        "features_aug": rng.standard_normal((n, 11)).astype(np.float32),
        "label": rng.integers(0, 2, size=n).astype(np.int8),
        "features_mean": rng.standard_normal(10).astype(np.float32),
        "features_std": (rng.standard_normal(10).astype(np.float32) ** 2 + 1e-8),
    }


def write_artifact(path: Path, artifact: dict) -> None:
    np.savez(path, **artifact)


class TestLoadKnnArrays:
    def test_loads_single_artifact(self, tmp_path):
        write_artifact(tmp_path / "theft.npz", make_fake_artifact(100))

        artifacts = load_knn_arrays(artifact_dir=tmp_path)

        assert set(artifacts.keys()) == {"theft"}
        assert set(artifacts["theft"].keys()) == EXPECTED_KEYS

    def test_loads_multiple_artifacts(self, tmp_path):
        write_artifact(tmp_path / "theft.npz", make_fake_artifact(100))
        write_artifact(tmp_path / "battery.npz", make_fake_artifact(50))
        write_artifact(tmp_path / "motor_vehicle_theft.npz", make_fake_artifact(200))

        artifacts = load_knn_arrays(artifact_dir=tmp_path)

        assert set(artifacts.keys()) == {"theft", "battery", "motor_vehicle_theft"}

    def test_slug_is_filename_stem(self, tmp_path):
        write_artifact(tmp_path / "motor_vehicle_theft.npz", make_fake_artifact(100))

        artifacts = load_knn_arrays(artifact_dir=tmp_path)

        # The dict key is the filename without .npz, not the path
        assert "motor_vehicle_theft" in artifacts
        assert "motor_vehicle_theft.npz" not in artifacts

    def test_array_shapes_preserved(self, tmp_path):
        write_artifact(tmp_path / "theft.npz", make_fake_artifact(75))

        artifacts = load_knn_arrays(artifact_dir=tmp_path)

        theft = artifacts["theft"]
        assert theft["features_aug"].shape == (75, 11)
        assert theft["label"].shape == (75,)
        assert theft["features_mean"].shape == (10,)
        assert theft["features_std"].shape == (10,)

    def test_dtypes_preserved(self, tmp_path):
        write_artifact(tmp_path / "theft.npz", make_fake_artifact(100))

        artifacts = load_knn_arrays(artifact_dir=tmp_path)

        theft = artifacts["theft"]
        assert theft["features_aug"].dtype == np.float32
        assert theft["label"].dtype == np.int8
        assert theft["features_mean"].dtype == np.float32
        assert theft["features_std"].dtype == np.float32

    def test_values_round_trip(self, tmp_path):
        original = make_fake_artifact(100)
        write_artifact(tmp_path / "theft.npz", original)

        artifacts = load_knn_arrays(artifact_dir=tmp_path)

        loaded = artifacts["theft"]
        for key in EXPECTED_KEYS:
            np.testing.assert_array_equal(loaded[key], original[key])

    def test_arrays_are_in_memory_not_file_backed(self, tmp_path):
        """np.load is lazy by default; we copy out so file handles can close.
        After loading, deleting the file should not break access to the arrays."""
        path = tmp_path / "theft.npz"
        write_artifact(path, make_fake_artifact(100))

        artifacts = load_knn_arrays(artifact_dir=tmp_path)
        path.unlink()  # delete the file

        # Should still work since arrays are copies in memory
        assert artifacts["theft"]["features_aug"].shape == (100, 11)
        _ = artifacts["theft"]["features_aug"].sum()

    def test_ignores_non_npz_files(self, tmp_path):
        write_artifact(tmp_path / "theft.npz", make_fake_artifact(100))
        (tmp_path / "readme.txt").write_text("not an artifact")
        (tmp_path / "scaler.pkl").write_bytes(b"fake pickle")

        artifacts = load_knn_arrays(artifact_dir=tmp_path)

        assert set(artifacts.keys()) == {"theft"}


class TestLoadKnnArraysErrors:
    def test_missing_directory_raises(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_knn_arrays(artifact_dir=nonexistent)

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .npz files found"):
            load_knn_arrays(artifact_dir=tmp_path)

    def test_directory_with_only_other_files_raises(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not an artifact")
        with pytest.raises(FileNotFoundError, match="No .npz files found"):
            load_knn_arrays(artifact_dir=tmp_path)

    def test_artifact_missing_key_raises(self, tmp_path):
        # Build an artifact missing 'features_std'
        incomplete = make_fake_artifact(100)
        del incomplete["features_std"]
        write_artifact(tmp_path / "theft.npz", incomplete)

        with pytest.raises(KeyError, match="features_std"):
            load_knn_arrays(artifact_dir=tmp_path)

    def test_one_bad_artifact_in_many_raises(self, tmp_path):
        # Good artifact
        write_artifact(tmp_path / "theft.npz", make_fake_artifact(100))
        # Bad artifact missing a key
        incomplete = make_fake_artifact(100)
        del incomplete["label"]
        write_artifact(tmp_path / "battery.npz", incomplete)

        with pytest.raises(KeyError, match="label"):
            load_knn_arrays(artifact_dir=tmp_path)

    def test_error_message_points_to_precompute_script(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="precompute_knn_arrays"):
            load_knn_arrays(artifact_dir=nonexistent)
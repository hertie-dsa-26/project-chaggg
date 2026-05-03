"""Load precomputed per-crime-type KNN artifacts into memory at Flask startup."""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "data" / "precomputed" / "knn"
EXPECTED_KEYS = {"features_aug", "label", "features_mean", "features_std"}


def load_knn_arrays(artifact_dir: Path = ARTIFACT_DIR) -> dict:
    """Load all per-crime-type KNN artifacts from disk.

    Returns a dict mapping normalized crime type slug -> dict of arrays:
        {
            "theft": {
                "features_aug":  np.ndarray (n, 11) float32,
                "label":         np.ndarray (n,)   int8,
                "features_mean": np.ndarray (10,)  float32,
                "features_std":  np.ndarray (10,)  float32,
            },
            "battery": {...},
            ...
        }

    Raises:
        FileNotFoundError: if artifact_dir does not exist or contains no .npz files.
        KeyError: if any artifact is missing one of the expected arrays.
    """
    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"Artifact directory {artifact_dir} does not exist. "
            f"Run `uv run python -m scripts.precompute_knn_arrays` first."
        )

    artifact_paths = sorted(artifact_dir.glob("*.npz"))
    if not artifact_paths:
        raise FileNotFoundError(
            f"No .npz files found in {artifact_dir}. "
            f"Run `uv run python -m scripts.precompute_knn_arrays` first."
        )

    artifacts = {}
    for path in artifact_paths:
        slug = path.stem  # filename without .npz
        with np.load(path) as npz:
            missing = EXPECTED_KEYS - set(npz.files)
            if missing:
                raise KeyError(
                    f"Artifact {path} missing expected keys: {sorted(missing)}"
                )
            # Force eager copy out of the npz (np.load is lazy by default)
            artifacts[slug] = {key: npz[key].copy() for key in EXPECTED_KEYS}

    return artifacts
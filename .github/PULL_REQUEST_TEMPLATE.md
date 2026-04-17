## What does this PR do?

- Adds **from-scratch Spatiotemporal KNN** pipeline: monthly prep + train/test split + forecast (K=5/10/20) + metrics + demo script.
- Adds **Flask integration**: `/viz/knn` page and `/api/knn/predict` endpoint.
- Improves correctness/debuggability: **KD-tree vs brute-force** path + expanded unit tests.
- Adds **CI automation** (ruff + unit tests + coverage gate) + docs (architecture, contributing, rubric mapping).

## How can a reviewer test this?

**Fast (no data needed):**

```bash
uv sync --all-groups
uv run ruff check src tests
uv run ruff format --check src tests
CHAGGG_SKIP_DATA_LOAD=1 uv run python -m unittest discover -s tests -p "test_*.py" -q
CHAGGG_SKIP_DATA_LOAD=1 uv run coverage erase
CHAGGG_SKIP_DATA_LOAD=1 uv run coverage run -m unittest discover -s tests -p "test_*.py" -q
uv run coverage report
```

**Optional (with cleaned data):**

```bash
./scripts/demo_knn.sh
```

Then run Flask and open `/viz/knn`:

```bash
CHAGGG_SKIP_DATA_LOAD=1 uv run flask --app src.flask_app:create_app run
```

## Any edge cases to be aware of?

- KNN visualizations depend on artifacts under `outputs/knn/`. If missing, `/viz/knn` shows instructions and `/api/knn/predict` returns 503.
- Dataset snapshots may differ slightly (portal drift), which can change counts/metrics.
- `time_scale` grid tested; for the tested range, best was `time_scale=0` with `space=meters`.

## Checklist

- [x] Tests written
- [x] No direct push to main
- [ ] Reviewed by at least one teammate

## Rubric / quality (optional)

- [ ] Architecture: changes fit existing module boundaries (`docs/ARCHITECTURE.md`).
- [ ] Edge cases or limitations noted where relevant.
